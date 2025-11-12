from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

from .base import TimestampMixin


class AnalyticsEvent(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_type: str
    payload: dict = Field(sa_column=Column(JSON, nullable=False))

