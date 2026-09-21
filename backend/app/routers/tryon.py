from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..core.dependencies import SessionDep
from ..models import Garment, TryOnJob, TryOnJobRead
from ..services.fashn_client import GARMENT_CATEGORIES
from ..services.garment_service import GarmentService
from ..services.tryon_service import TryOnService

router = APIRouter(prefix="/tryon", tags=["try-on"])
SUPPORTED_MODES = {"garment_swap", "face_swap"}
service = TryOnService()


@router.post("/", response_model=TryOnJobRead, status_code=status.HTTP_201_CREATED)
async def create_tryon_job(
    session: SessionDep,
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
    return TryOnJobRead.model_validate(job)


@router.get("/{job_id}", response_model=TryOnJobRead)
async def get_tryon_job(job_id: int, session: SessionDep) -> TryOnJob:
    job = await session.get(TryOnJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Try-on job not found")
    return TryOnJobRead.model_validate(job)
