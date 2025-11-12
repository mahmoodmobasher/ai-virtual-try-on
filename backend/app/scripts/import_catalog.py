from __future__ import annotations

import csv
from pathlib import Path

import typer
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.database import db
from ..models import Garment, GarmentCreate
from ..services.garment_service import GarmentService

app = typer.Typer(help="Catalog import utilities.")


@app.command()
def from_csv(path: Path) -> None:
    """Import garments from a CSV file."""

    async def _run() -> None:
        async with db.session() as session:  # type: AsyncSession
            await _import_csv(session, path)

    import asyncio

    asyncio.run(_run())


async def _import_csv(session: AsyncSession, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            payload = GarmentCreate(
                sku=row["sku"],
                name=row["name"],
                category=row["category"],
                description=row.get("description"),
                brand=row.get("brand"),
                price=float(row["price"]) if row.get("price") else None,
                colors=[c.strip() for c in row.get("colors", "").split("|") if c.strip()],
                sizes=[s.strip() for s in row.get("sizes", "").split("|") if s.strip()],
                hero_image_path=row.get("hero_image_path"),
                mask_image_path=row.get("mask_image_path"),
            )
            existing = await GarmentService.get_by_sku(session, payload.sku)
            if existing:
                logger.info("Skipping SKU %s (already exists)", payload.sku)
                continue
            garment = await GarmentService.create_garment(session, payload)
            logger.info("Imported garment %s (%s)", garment.name, garment.sku)


if __name__ == "__main__":
    app()

