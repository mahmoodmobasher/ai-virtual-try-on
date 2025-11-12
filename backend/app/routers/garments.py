from __future__ import annotations

from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, Response, status

from ..core.dependencies import SessionDep
from ..models import Garment, GarmentCreate, GarmentRead, GarmentUpdate
from ..services.garment_service import GarmentService

router = APIRouter(prefix="/garments", tags=["garments"])


@router.get("/", response_model=list[GarmentRead])
async def list_garments(session: SessionDep, category: str | None = None) -> Sequence[Garment]:
    garments = await GarmentService.list_garments(session, category=category)
    return [GarmentRead.model_validate(g) for g in garments]


@router.get("/{garment_id}", response_model=GarmentRead)
async def get_garment(garment_id: int, session: SessionDep) -> Garment:
    garment = await GarmentService.get_garment(session, garment_id)
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")
    return GarmentRead.model_validate(garment)


@router.post("/", response_model=GarmentRead, status_code=status.HTTP_201_CREATED)
async def create_garment(payload: GarmentCreate, session: SessionDep) -> Garment:
    garment = await GarmentService.create_garment(session, payload)
    return GarmentRead.model_validate(garment)


@router.patch("/{garment_id}", response_model=GarmentRead)
async def update_garment(garment_id: int, payload: GarmentUpdate, session: SessionDep) -> Garment:
    garment = await GarmentService.get_garment(session, garment_id)
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")
    garment = await GarmentService.update_garment(session, garment, payload)
    return GarmentRead.model_validate(garment)


@router.delete("/{garment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_garment(garment_id: int, session: SessionDep) -> Response:
    garment = await GarmentService.get_garment(session, garment_id)
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")
    await GarmentService.delete_garment(session, garment)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

