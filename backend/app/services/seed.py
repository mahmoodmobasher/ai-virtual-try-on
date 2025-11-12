from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..models import Garment, GarmentCreate
from .garment_service import GarmentService

DEMO_GARMENTS = [
    GarmentCreate(
        sku="AURORA-BOMBER",
        name="Aurora Satin Bomber",
        category="Outerwear",
        description="Luminous satin bomber with iridescent sheen.",
        brand="The New Black",
        price=428.0,
        colors=["Iridescent Navy", "Graphite"],
        sizes=["S", "M", "L", "XL"],
        hero_image_path="/catalog/aurora-bomber.png",
    ),
    GarmentCreate(
        sku="LYRA-SLIP",
        name="Lyra Bias Slip",
        category="Dresses",
        description="Liquid silk slip with bias cut drape.",
        brand="The New Black",
        price=348.0,
        colors=["Moonstone", "Obsidian"],
        sizes=["XS", "S", "M", "L"],
        hero_image_path="/catalog/lyra-slip.png",
    ),
    GarmentCreate(
        sku="NOVA-TROUSER",
        name="Nova Pleated Trouser",
        category="Bottoms",
        description="High-rise tailored trouser with razor pleats.",
        brand="The New Black",
        price=298.0,
        colors=["Black", "Ivory"],
        sizes=["0", "2", "4", "6", "8"],
        hero_image_path="/catalog/nova-trouser.png",
    ),
]


async def ensure_demo_catalog(session: AsyncSession) -> None:
    result = await session.exec(select(Garment))
    if result.first():
        return

    for garment in DEMO_GARMENTS:
        await GarmentService.create_garment(session, garment)

