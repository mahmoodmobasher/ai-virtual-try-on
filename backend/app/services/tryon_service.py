from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger
from PIL import Image, ImageOps
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.config import get_settings
from ..core.database import db
from ..models import Garment, TryOnJob, TryOnStatus
from .pipelines.tryon_worker import run_tryon_job


# Most faces one face-swap job may replace.
MAX_SWAP_FACES = 4
# Upper bound on faces offered in the picker, most prominent first.
MAX_SCANNED_FACES = 12

_UPLOAD_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_SCAN_FILE = "scan.json"


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

    # ------------------------------------------------------------------
    # Face swap: scan the photo for faces, then swap the ones the user picked
    # ------------------------------------------------------------------
    def _scan_dir(self, upload_id: str) -> Path:
        return self.settings.uploads_dir / "faceswap-image" / upload_id

    def scan_faces(self, image_bytes: bytes, filename: str | None) -> dict:
        """Find the faces in a photo so the caller can pick which to swap.

        The photo and each face's landmarks are kept on disk under the returned
        `upload_id`, so the swap job reuses exactly what the user was shown
        rather than re-uploading and re-detecting. Blocking (runs detection);
        call via a worker thread.
        """
        # Imported here so the API can start without the optional ML stack.
        from .segmentation_service import get_segmentation_service

        upload_id = uuid.uuid4().hex
        scan_dir = self._scan_dir(upload_id)
        try:
            image_path = self._persist_upload(image_bytes, filename, scan_dir)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("That photo could not be read. Please upload a valid JPEG or PNG image.")

            segmentation = get_segmentation_service()
            faces = segmentation.detect_faces(image)
            # Keep the most prominent faces, then list them left to right so the
            # numbering matches how the photo reads.
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
            del faces[MAX_SCANNED_FACES:]
            faces.sort(key=lambda f: f["bbox"][0])

            for index, face in enumerate(faces):
                cv2.imwrite(str(scan_dir / f"face_{index}.png"), segmentation.crop_face(image, face["bbox"]))
        except Exception:
            shutil.rmtree(scan_dir, ignore_errors=True)
            raise

        if not faces:
            shutil.rmtree(scan_dir, ignore_errors=True)

        height, width = image.shape[:2]
        scan = {"image": image_path.name, "width": width, "height": height, "faces": faces}
        if faces:
            (scan_dir / _SCAN_FILE).write_text(json.dumps(scan))
        logger.info("Face scan {}: {} face(s) found", upload_id, len(faces))
        return {"upload_id": upload_id, **scan}

    def load_scan(self, upload_id: str) -> dict | None:
        """The stored result of `scan_faces`, or None if there is no such scan."""
        if not _UPLOAD_ID_PATTERN.fullmatch(upload_id):
            return None
        scan_file = self._scan_dir(upload_id) / _SCAN_FILE
        if not scan_file.exists():
            return None
        return json.loads(scan_file.read_text())

    async def submit_face_swap_job(
        self,
        session: AsyncSession,
        *,
        upload_id: str,
        scan: dict,
        replacements: list[tuple[int, bytes, str | None]],
        user_reference: Optional[str] = None,
    ) -> TryOnJob:
        """Persist a pending face-swap job for a scanned photo.

        `replacements` pairs a face index from the scan with the bytes and
        filename of the photo replacing that face.
        """
        scan_dir = self._scan_dir(upload_id)
        face_swaps = []
        for face_index, image_bytes, filename in replacements:
            source_path = self._persist_upload(image_bytes, filename, scan_dir / "replacements")
            face_swaps.append(
                {
                    "face_index": face_index,
                    "kps": scan["faces"][face_index]["kps"],
                    "source_image_path": str(source_path),
                }
            )

        job = TryOnJob(
            input_image_path=str(scan_dir / scan["image"]),
            status=TryOnStatus.pending,
            user_reference=user_reference,
            meta={"mode": "face_swap", "upload_id": upload_id, "face_swaps": face_swaps},
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

            face_swaps = meta.get("face_swaps")
            if resolved_garment_path is None and not face_swaps:
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
                face_swaps=face_swaps,
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
        # Bake in EXIF orientation and normalize to RGB. Only re-encode when that
        # actually changes the pixels, and then at high quality: re-saving a JPEG
        # at the default quality softens the photo before any model sees it.
        try:
            with Image.open(file_path) as image:
                rotated = image.getexif().get(0x0112, 1) != 1
                if rotated or image.mode != "RGB":
                    normalized = ImageOps.exif_transpose(image).convert("RGB")
                    normalized.save(file_path, quality=95, subsampling=0)
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

