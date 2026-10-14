from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from ..core.dependencies import SessionDep
from ..models import Garment, TryOnJob, TryOnJobRead
from ..services.fashn_client import GARMENT_CATEGORIES
from ..services.garment_service import GarmentService
from ..services.tryon_service import MAX_SWAP_FACES, TryOnService

router = APIRouter(prefix="/tryon", tags=["try-on"])
SUPPORTED_MODES = {"garment_swap", "face_swap"}
service = TryOnService()


@router.post("/", response_model=TryOnJobRead, status_code=status.HTTP_202_ACCEPTED)
async def create_tryon_job(
    session: SessionDep,
    background_tasks: BackgroundTasks,
    garment_id: int | None = Form(None),
    user_reference: str | None = Form(None),
    prompt: str | None = Form(None),
    mode: str = Form("garment_swap"),  # "garment_swap" (clothes swap) or "face_swap"
    category: str = Form("auto"),  # clothes swap only: auto, tops, bottoms, one-pieces
    user_image: UploadFile = File(...),
    garment_image: UploadFile | None = File(None),
) -> TryOnJob:
    if mode not in SUPPORTED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported mode '{mode}'. Use one of: {', '.join(sorted(SUPPORTED_MODES))}.",
        )
    if category not in GARMENT_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported category '{category}'. Use one of: {', '.join(sorted(GARMENT_CATEGORIES))}.",
        )

    garment: Garment | None = None
    if garment_id is not None:
        garment = await GarmentService.get_garment(session, garment_id)
        if not garment:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")

    garment_image_bytes: bytes | None = None
    garment_filename: str | None = None
    if garment_image is not None:
        garment_image_bytes = await garment_image.read()
        garment_filename = garment_image.filename

    if garment is None and garment_image_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide either a garment_id from the catalog or upload a garment_image.",
        )

    image_bytes = await user_image.read()
    job = await service.submit_job(
        session,
        garment=garment,
        garment_image_bytes=garment_image_bytes,
        garment_filename=garment_filename,
        image_bytes=image_bytes,
        filename=user_image.filename,
        user_reference=user_reference,
        prompt=prompt,
        mode=mode,
        category=category,
    )
    background_tasks.add_task(service.run_job, job.id)
    return TryOnJobRead.model_validate(job)


# ----------------------------------------------------------------------
# Face swap on a still photo: the photo is scanned for faces first, the
# caller picks which of those people to replace (up to MAX_SWAP_FACES) and
# supplies a replacement photo for each. Mirrors the video face-swap flow in
# routers/video.py; the job itself is a regular TryOnJob, polled via
# GET /tryon/{job_id}.
# ----------------------------------------------------------------------


class DetectedFace(BaseModel):
    face_index: int
    bbox: list[float]
    det_score: float
    thumbnail_url: str


class FaceScanResult(BaseModel):
    upload_id: str
    image_width: int
    image_height: int
    max_swap_faces: int = MAX_SWAP_FACES
    faces: list[DetectedFace]


@router.post("/faceswap/scan", response_model=FaceScanResult)
async def scan_faceswap_faces(image: UploadFile = File(...)) -> FaceScanResult:
    """Find the faces in an uploaded photo so the caller can pick which to swap.

    The photo is kept server-side under the returned `upload_id`; pass that id
    to POST /tryon/faceswap instead of uploading the photo a second time.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a JPEG or PNG image.")

    try:
        image_bytes = await image.read()
    finally:
        await image.close()

    try:
        scan = await asyncio.to_thread(service.scan_faces, image_bytes, image.filename)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError) as exc:
        logger.exception("Face scan failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    upload_id = scan["upload_id"]
    return FaceScanResult(
        upload_id=upload_id,
        image_width=scan["width"],
        image_height=scan["height"],
        faces=[
            DetectedFace(
                face_index=index,
                bbox=face["bbox"],
                det_score=face["det_score"],
                thumbnail_url=f"/uploads/faceswap-image/{upload_id}/face_{index}.png",
            )
            for index, face in enumerate(scan["faces"])
        ],
    )


@router.post("/faceswap", response_model=TryOnJobRead, status_code=status.HTTP_202_ACCEPTED)
async def create_faceswap_job(
    session: SessionDep,
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    source_face_images: list[UploadFile] = File(...),
    target_face_indices: list[int] = Form(...),
    user_reference: str | None = Form(None),
) -> TryOnJob:
    """Replace the chosen faces in a previously scanned photo.

    `target_face_indices` selects which faces from POST /tryon/faceswap/scan to
    replace; `source_face_images` supplies the replacement face for each, in the
    same order. Everything else in the photo is preserved.
    """
    scan = service.load_scan(upload_id)
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That scan was not found. Re-upload the photo to scan it again.",
        )

    if len(source_face_images) != len(target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Each chosen face needs exactly one replacement photo.",
        )
    if not 1 <= len(target_face_indices) <= MAX_SWAP_FACES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Choose between 1 and {MAX_SWAP_FACES} faces to swap.",
        )
    if len(set(target_face_indices)) != len(target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The same face was chosen more than once.",
        )
    face_count = len(scan["faces"])
    if any(index < 0 or index >= face_count for index in target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A chosen face isn't part of that scan.",
        )
    for image in source_face_images:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload JPEG or PNG images.")

    replacements: list[tuple[int, bytes, str | None]] = []
    for target_index, image in zip(target_face_indices, source_face_images):
        try:
            replacements.append((target_index, await image.read(), image.filename))
        finally:
            await image.close()

    job = await service.submit_face_swap_job(
        session,
        upload_id=upload_id,
        scan=scan,
        replacements=replacements,
        user_reference=user_reference,
    )
    background_tasks.add_task(service.run_job, job.id)
    return TryOnJobRead.model_validate(job)


@router.get("/{job_id}", response_model=TryOnJobRead)
async def get_tryon_job(job_id: int, session: SessionDep) -> TryOnJob:
    job = await session.get(TryOnJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Try-on job not found")
    result = TryOnJobRead.model_validate(job)
    progress = service.get_progress(job_id)
    if progress:
        result.progress_current, result.progress_total = progress
    return result
