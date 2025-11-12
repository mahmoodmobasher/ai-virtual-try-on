from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..core.dependencies import SessionDep
from ..models import Garment, TryOnJob, TryOnJobRead
from ..services.garment_service import GarmentService
from ..services.tryon_service import TryOnService

router = APIRouter(prefix="/tryon", tags=["try-on"])
service = TryOnService()


@router.post("/", response_model=TryOnJobRead, status_code=status.HTTP_201_CREATED)
async def create_tryon_job(
    session: SessionDep,
    garment_id: int | None = Form(None),
    user_reference: str | None = Form(None),
    prompt: str | None = Form(None),
    mode: str = Form("face_swap"),  # "face_swap" or "catvton_flux"
    user_image: UploadFile = File(...),
    garment_image: UploadFile | None = File(None),
) -> TryOnJob:
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
    )
    return TryOnJobRead.model_validate(job)


@router.get("/{job_id}", response_model=TryOnJobRead)
async def get_tryon_job(job_id: int, session: SessionDep) -> TryOnJob:
    job = await session.get(TryOnJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Try-on job not found")
    return TryOnJobRead.model_validate(job)

