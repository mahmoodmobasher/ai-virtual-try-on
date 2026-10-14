"""Runs a single try-on generation in an isolated, killable subprocess.

Local try-on inference (CatVTON/SD-based diffusion) is heavy enough that a slow
or stuck run can peg every CPU core and grow memory without bound, hanging the
whole machine. Running it in a subprocess gets us two things a same-process
thread can't: a hard kill switch (terminating the process instantly returns
all its memory to the OS) and a place to cap CPU thread usage before torch
initializes its thread pools.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...core.config import get_settings


def _cap_threads(thread_count: int) -> None:
    threads = str(thread_count)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = threads


def _worker_entry(request: dict, progress_queue: "mp.Queue", result_queue: "mp.Queue") -> None:
    settings = get_settings()
    _cap_threads(settings.effective_tryon_cpu_threads)

    import torch

    torch.set_num_threads(settings.effective_tryon_cpu_threads)

    from .tryon_pipeline import TryOnPipeline

    def progress_cb(current: int, total: int) -> None:
        try:
            progress_queue.put_nowait((current, total))
        except Exception:  # pragma: no cover
            pass

    try:
        pipeline = TryOnPipeline()
        output_path = pipeline.apply(
            Path(request["user_image_path"]),
            Path(request["garment_image_path"]) if request.get("garment_image_path") else None,
            Path(request["output_path"]),
            prompt=request.get("prompt"),
            mode=request.get("mode", "garment_swap"),
            category=request.get("category", "auto"),
            face_swaps=request.get("face_swaps"),
            progress_cb=progress_cb,
        )
        result_queue.put(("done", str(output_path)))
    except Exception as exc:  # pragma: no cover
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # pragma: no cover
            pass


def run_tryon_job(
    *,
    user_image_path: Path,
    garment_image_path: Optional[Path],
    output_path: Path,
    prompt: Optional[str] = None,
    mode: str = "garment_swap",
    category: str = "auto",
    face_swaps: Optional[list[dict]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    timeout_seconds: Optional[float] = None,
) -> Path:
    """Run one try-on generation to completion, or raise TimeoutError and kill the
    worker process if it overruns `timeout_seconds` (defaults to
    settings.tryon_timeout_seconds)."""
    settings = get_settings()
    timeout = timeout_seconds if timeout_seconds is not None else settings.tryon_timeout_seconds

    ctx = mp.get_context("spawn")
    progress_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    request = {
        "user_image_path": str(user_image_path),
        "garment_image_path": str(garment_image_path) if garment_image_path else None,
        "output_path": str(output_path),
        "prompt": prompt,
        "mode": mode,
        "category": category,
        "face_swaps": face_swaps,
    }

    process = ctx.Process(
        target=_worker_entry, args=(request, progress_queue, result_queue), daemon=True
    )
    process.start()

    deadline = time.monotonic() + timeout
    outcome: Optional[tuple[str, str]] = None
    try:
        while outcome is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError

            try:
                current, total = progress_queue.get(timeout=min(remaining, 0.5))
                if progress_cb:
                    progress_cb(current, total)
                continue
            except Exception:
                pass

            if not result_queue.empty():
                outcome = result_queue.get()
                break

            if not process.is_alive():
                outcome = (
                    "error",
                    "Try-on worker process exited unexpectedly (it may have run out of memory).",
                )
                break
    except TimeoutError:
        logger.warning("Try-on job exceeded {}s timeout; killing worker process.", timeout)
        process.kill()
        process.join(timeout=5)
        raise TimeoutError(f"Try-on generation exceeded the {timeout}s time limit and was stopped.")
    finally:
        if process.is_alive():
            process.join(timeout=1)

    status, payload = outcome
    if status == "error":
        raise RuntimeError(payload)
    return Path(payload)
