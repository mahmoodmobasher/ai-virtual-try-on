from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import JSON, Column, Enum as SAEnum
from sqlmodel import Field, SQLModel

from .base import TimestampMixin


class TryOnStatus(str, PyEnum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TryOnJob(TimestampMixin, SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_reference: str | None = Field(default=None, index=True)
    garment_id: int | None = Field(default=None, foreign_key="garment.id", index=True)
    garment_image_path: str | None = None
    input_image_path: str
    result_image_path: str | None = None
    status: TryOnStatus = Field(
        default=TryOnStatus.pending,
        sa_column=Column(SAEnum(TryOnStatus, name="tryon_status"), nullable=False),
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None
    meta: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))


class TryOnJobRead(SQLModel):
    id: int
    garment_id: int | None
    garment_image_path: str | None = None
    status: TryOnStatus
    result_image_path: str | None = None
    failure_reason: str | None = None
    meta: dict | None = None
    created_at: datetime
    updated_at: datetime
    # Not persisted columns: filled in by the router from the service's in-memory
    # progress store, so the frontend can render a real progress bar while a job
    # is processing. Absent/None means "no step-wise progress available" (e.g. a
    # hosted engine, or before the first step has run).
    progress_current: int | None = None
    progress_total: int | None = None

