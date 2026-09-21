from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from ..core.config import get_settings
from ..services.video_generation_service import get_video_generation_service
from ..services.video_tryon_service import get_video_tryon_engine

router = APIRouter(prefix="/video", tags=["video"])
settings = get_settings()

# This video addon only drives the garment-swap try-on pipeline; face swap is
# intentionally excluded here.
SUPPORTED_TRYON_MODES = {"garment_swap"}


class VideoTryOnJob(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    frames_done: int = 0
    frames_total: int = 0
    result_video_path: str | None = None
    failure_reason: str | None = None
    created_at: float


# In-memory job registry. Fine for a single-process demo/addon; swap for the
# database-backed TryOnJob model if this needs to survive restarts or scale
# across workers.
_video_tryon_jobs: dict[str, VideoTryOnJob] = {}


@router.post("/clip", status_code=status.HTTP_202_ACCEPTED)
async def create_video_clip(
    user_image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    frames: int = Form(default=None),
    fps: int = Form(default=None),
    guidance_scale: float = Form(default=None),
    motion_bucket_id: int | None = Form(default=None),
    noise_aug_strength: float | None = Form(default=None),
    seed: int | None = Form(default=None),
) -> dict[str, object]:
    if user_image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a JPEG or PNG image.")

    temp_dir = settings.storage_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"video-input-{uuid4().hex}.png"

    try:
        contents = await user_image.read()
        temp_path.write_bytes(contents)
    finally:
        await user_image.close()

    clip_id = uuid4().hex
    output_path = settings.video_results_dir / f"clip-{clip_id}.mp4"

    service = get_video_generation_service()

    try:
        service.generate_clip(
            image_path=temp_path,
            output_path=output_path,
            frames=frames,
            fps=fps,
            prompt=prompt,
            guidance_scale=guidance_scale,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=seed,
        )
    except RuntimeError as exc:  # device/model issues
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    duration_seconds = (frames or settings.svd_default_frames) / (fps or settings.svd_default_fps)

    return {
        "result_video_path": f"/video-results/{output_path.name}",
        "frames": frames or settings.svd_default_frames,
        "fps": fps or settings.svd_default_fps,
        "duration_seconds": duration_seconds,
    }


def _run_video_tryon_job(
    job_id: str,
    video_path: Path,
    garment_path: Path,
    *,
    prompt: str | None,
    mode: str,
    category: str,
    fps: float | None,
    max_frames: int | None,
    frame_stride: int,
) -> None:
    job = _video_tryon_jobs[job_id]
    job.status = "processing"

    def on_progress(done: int, total: int) -> None:
        job.frames_done = done
        job.frames_total = total

    output_path = settings.video_results_dir / f"tryon-{job_id}.mp4"
    engine = get_video_tryon_engine()

    try:
        engine.run(
            video_path,
            garment_path,
            output_path,
            prompt=prompt,
            mode=mode,
            category=category,
            fps=fps,
            max_frames=max_frames,
            frame_stride=frame_stride,
            progress_cb=on_progress,
        )
        job.result_video_path = f"/video-results/{output_path.name}"
        job.status = "completed"
    except Exception as exc:  # pragma: no cover - background task error path
        logger.exception("Video try-on job %s failed: %s", job_id, exc)
        job.status = "failed"
        job.failure_reason = str(exc)
    finally:
        for path in (video_path, garment_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


@router.post("/tryon", response_model=VideoTryOnJob, status_code=status.HTTP_202_ACCEPTED)
async def create_video_tryon_job(
    background_tasks: BackgroundTasks,
    user_video: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    mode: str = Form(default="garment_swap"),  # only garment_swap is supported by this endpoint
    category: str = Form(default="auto"),
    fps: float | None = Form(default=None),
    max_frames: int | None = Form(default=None),
    frame_stride: int = Form(default=1),
) -> VideoTryOnJob:
    """Apply the garment try-on pipeline to every frame of an uploaded video."""
    if mode not in SUPPORTED_TRYON_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported mode '{mode}'. Use one of: {', '.join(sorted(SUPPORTED_TRYON_MODES))}.",
        )
    if not user_video.content_type or not user_video.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a video file.")

    temp_dir = settings.storage_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid4().hex
    video_suffix = Path(user_video.filename or "input.mp4").suffix or ".mp4"
    garment_suffix = Path(garment_image.filename or "garment.png").suffix or ".png"
    video_path = temp_dir / f"video-tryon-input-{job_id}{video_suffix}"
    garment_path = temp_dir / f"video-tryon-garment-{job_id}{garment_suffix}"

    try:
        video_path.write_bytes(await user_video.read())
        garment_path.write_bytes(await garment_image.read())
    finally:
        await user_video.close()
        await garment_image.close()

    job = VideoTryOnJob(job_id=job_id, status="pending", created_at=time.time())
    _video_tryon_jobs[job_id] = job

    background_tasks.add_task(
        _run_video_tryon_job,
        job_id,
        video_path,
        garment_path,
        prompt=prompt,
        mode=mode,
        category=category,
        fps=fps,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )

    return job


@router.get("/tryon/{job_id}", response_model=VideoTryOnJob)
async def get_video_tryon_job(job_id: str) -> VideoTryOnJob:
    job = _video_tryon_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video try-on job not found")
    return job
