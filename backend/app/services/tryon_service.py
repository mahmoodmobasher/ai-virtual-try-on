from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.config import get_settings
from ..core.database import db
from ..models import Garment, TryOnJob, TryOnStatus
from .pipelines.tryon_worker import run_tryon_job


class TryOnService:
    """Business logic around try-on jobs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # In-memory step progress per job, e.g. {job_id: (current_step, total_steps)}.
        # Ephemeral by design (mirrors the video job registries): it only needs to
        # survive for the lifetime of a single run, not a server restart.
        self._progress: dict[int, tuple[int, int]] = {}

    def get_progress(self, job_id: int) -> tuple[int, int] | None:
        return self._progress.get(job_id)

    async def submit_job(
        self,
        session: AsyncSession,
        *,
        garment: Garment | None = None,
        garment_image_bytes: Optional[bytes] = None,
        garment_filename: Optional[str] = None,
        image_bytes: bytes,
        filename: str | None,
        user_reference: Optional[str] = None,
        prompt: Optional[str] = None,
        mode: str = "garment_swap",
        category: str = "auto",
    ) -> TryOnJob:
        """Persist a pending job and return immediately; call `run_job` (typically via
        BackgroundTasks) to actually process it, so the caller can start polling for
        progress right away instead of blocking on the full generation."""
        if garment is None and garment_image_bytes is None:
            raise ValueError("Either a garment from the catalog or a garment image upload must be provided.")

        input_path = self._persist_upload(image_bytes, filename, self.settings.uploads_dir / "models")

        garment_path: Path | None = None
        if garment_image_bytes is not None and garment_filename:
            garment_path = self._persist_upload(
                garment_image_bytes,
                garment_filename,
                self.settings.uploads_dir / "garments",
            )
        elif garment is not None:
            garment_path = self._resolve_garment_image(garment)

        meta: dict[str, Optional[str]] = {}
        if prompt is not None:
            meta["prompt"] = prompt
        if garment_path:
            meta["garment_image_path"] = str(garment_path)
        meta["mode"] = mode  # Store the try-on mode
        meta["category"] = category

        job = TryOnJob(
            garment_id=garment.id if garment else None,
            garment_image_path=str(garment_path) if garment_path else None,
            input_image_path=str(input_path),
            status=TryOnStatus.pending,
            user_reference=user_reference,
            meta=meta or None,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)

        return job

    async def run_job(self, job_id: int) -> None:
        """Process a previously-created pending job. Opens its own DB session so it
        can run as a background task after the request that created the job returns."""
        async with db.session() as session:
            job = await session.get(TryOnJob, job_id)
            if job is None:
                logger.error("run_job called for missing try-on job %s", job_id)
                return
            await self._execute_job(session, job)

    async def _execute_job(self, session: AsyncSession, job: TryOnJob) -> None:
        job.status = TryOnStatus.processing
        await session.commit()

        def on_progress(current: int, total: int) -> None:
            self._progress[job.id] = (current, total)

        try:
            meta = job.meta or {}
            resolved_garment_path = Path(job.garment_image_path) if job.garment_image_path else None

            if resolved_garment_path is None and job.garment_id is not None:
                garment = await session.get(Garment, job.garment_id)
                if garment:
                    resolved_garment_path = self._resolve_garment_image(garment)

            if resolved_garment_path is None:
                raise FileNotFoundError("Garment image not available for try-on job.")

            output_path = Path(self.settings.results_dir) / f"{job.id}_result.png"

            # Run in an isolated subprocess (via a worker thread so we don't block the
            # event loop while waiting on it) so a stuck/slow generation can be killed
            # outright on timeout instead of hanging the machine.
            await asyncio.to_thread(
                run_tryon_job,
                user_image_path=Path(job.input_image_path),
                garment_image_path=resolved_garment_path,
                output_path=output_path,
                prompt=meta.get("prompt"),
                mode=meta.get("mode", "garment_swap"),
                category=meta.get("category", "auto"),
                progress_cb=on_progress,
            )
            job.result_image_path = f"/results/{output_path.name}"
            job.meta = {**meta, "output_path": str(output_path)}
            job.status = TryOnStatus.completed
        except Exception as exc:
            logger.exception("Failed to process job %s: %s", job.id, exc)
            job.status = TryOnStatus.failed
            job.failure_reason = str(exc)
        finally:
            self._progress.pop(job.id, None)
            job.touch()
            await session.commit()
            await session.refresh(job)

    def _persist_upload(self, image_bytes: bytes, filename: str | None, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "upload.png").suffix or ".png"
        file_path = target_dir / f"{uuid.uuid4().hex}{suffix}"
        with file_path.open("wb") as buffer:
            buffer.write(image_bytes)
        # Normalize orientation
        try:
            image = Image.open(file_path)
            image = image.convert("RGB")
            image.save(file_path)
        except Exception as exc:
            logger.warning("Failed to normalize upload image: %s", exc)
        return file_path

    def _resolve_garment_image(self, garment: Garment) -> Path:
        if garment.hero_image_path:
            candidate = Path(garment.hero_image_path)
            if not candidate.is_absolute():
                candidate = Path(self.settings.storage_dir) / candidate
            elif candidate.as_posix().startswith("/"):
                candidate = Path(self.settings.storage_dir) / garment.hero_image_path.strip("/")
            if candidate.exists():
                return candidate
        fallback = self.settings.catalog_dir / "placeholders" / "default-garment.png"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("Garment image not available; please set `hero_image_path`.")

