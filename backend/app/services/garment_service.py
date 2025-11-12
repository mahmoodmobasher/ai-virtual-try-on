from collections.abc import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..models import Garment, GarmentCreate, GarmentRead, GarmentUpdate


class GarmentService:
    """CRUD operations for garments."""

    @staticmethod
    async def list_garments(session: AsyncSession, *, category: str | None = None) -> Sequence[Garment]:
        statement = select(Garment)
        if category:
            statement = statement.where(Garment.category == category)
        result = await session.exec(statement)
        return result.all()

    @staticmethod
    async def get_garment(session: AsyncSession, garment_id: int) -> Garment | None:
        return await session.get(Garment, garment_id)

    @staticmethod
    async def get_by_sku(session: AsyncSession, sku: str) -> Garment | None:
        statement = select(Garment).where(Garment.sku == sku)
        result = await session.exec(statement)
        return result.first()

    @staticmethod
    async def create_garment(session: AsyncSession, payload: GarmentCreate) -> Garment:
        garment = Garment(**payload.model_dump())
        session.add(garment)
        await session.commit()
        await session.refresh(garment)
        return garment

    @staticmethod
    async def update_garment(session: AsyncSession, garment: Garment, payload: GarmentUpdate) -> Garment:
        update_data = payload.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(garment, field, value)
        garment.touch()
        session.add(garment)
        await session.commit()
        await session.refresh(garment)
        return garment

    @staticmethod
    async def delete_garment(session: AsyncSession, garment: Garment) -> None:
        await session.delete(garment)
        await session.commit()

