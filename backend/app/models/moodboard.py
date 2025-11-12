from typing import Optional

from sqlalchemy import JSON, Column, ForeignKey
from sqlmodel import Field, Relationship, SQLModel

from .base import TimestampMixin


class Moodboard(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    owner: str | None = Field(default=None, index=True)
    description: str | None = None

    items: list["MoodboardItem"] = Relationship(back_populates="moodboard")


class MoodboardItem(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    moodboard_id: int = Field(foreign_key="moodboard.id")
    garment_id: int | None = Field(default=None, foreign_key="garment.id")
    asset_path: str | None = None
    caption: str | None = None
    meta: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))

    moodboard: Moodboard = Relationship(back_populates="items")

