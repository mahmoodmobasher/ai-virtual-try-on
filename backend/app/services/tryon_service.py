from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.config import get_settings
from ..models import Garment, TryOnJob, TryOnStatus
from .pipelines.tryon_pipeline import TryOnPipeline


class TryOnService:
    """Business logic around try-on jobs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline = TryOnPipeline()

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
        mode: str = "face_swap",
    ) -> TryOnJob:
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

        logger.info("Processing try-on job %s synchronously for demo purposes.", job.id)
        await self._execute_job(session, job, garment_path)

        return job

    async def _execute_job(self, session: AsyncSession, job: TryOnJob, garment_path: Path | None) -> None:
        job.status = TryOnStatus.processing
        await session.commit()

        try:
            resolved_garment_path = garment_path
            if resolved_garment_path is None and job.garment_image_path:
                resolved_garment_path = Path(job.garment_image_path)

            if resolved_garment_path is None and job.garment_id is not None:
                garment = await session.get(Garment, job.garment_id)
                if garment:
                    resolved_garment_path = self._resolve_garment_image(garment)

            if resolved_garment_path is None:
                raise FileNotFoundError("Garment image not available for try-on job.")

            output_path = Path(self.settings.results_dir) / f"{job.id}_result.png"
            tryon_mode = job.meta.get("mode", "face_swap") if job.meta else "face_swap"
            
            self.pipeline.apply(
                Path(job.input_image_path),
                resolved_garment_path,
                output_path,
                prompt=job.meta.get("prompt") if job.meta else None,
                mode=tryon_mode,
            )
            job.result_image_path = f"/results/{output_path.name}"
            meta = job.meta or {}
            meta["output_path"] = str(output_path)
            job.meta = meta
            job.status = TryOnStatus.completed
        except Exception as exc:
            logger.exception("Failed to process job %s: %s", job.id, exc)
            job.status = TryOnStatus.failed
            job.failure_reason = str(exc)
        finally:
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

