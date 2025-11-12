from typing import Optional
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from .base import TimestampMixin


class Garment(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    sku: str = Field(index=True, unique=True)
    name: str
    category: str
    description: str | None = None
    brand: str | None = None
    price: float | None = None
    colors: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    sizes: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    hero_image_path: str | None = None
    mask_image_path: str | None = None
    embedding_vector: list[float] | None = Field(default=None, sa_column=Column(JSON, nullable=True))


class GarmentRead(SQLModel):
    id: int
    sku: str
    name: str
    category: str
    description: str | None = None
    brand: str | None = None
    price: float | None = None
    colors: list[str]
    sizes: list[str]
    hero_image_path: str | None = None
    mask_image_path: str | None = None


class GarmentCreate(SQLModel):
    sku: str
    name: str
    category: str
    description: str | None = None
    brand: str | None = None
    price: float | None = None
    colors: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    hero_image_path: str | None = None
    mask_image_path: str | None = None


class GarmentUpdate(SQLModel):
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    colors: Optional[list[str]] = None
    sizes: Optional[list[str]] = None
    hero_image_path: Optional[str] = None
    mask_image_path: Optional[str] = None

