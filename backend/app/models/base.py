from datetime import datetime

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class TimestampMixin(SQLModel, table=False):
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"nullable": False})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"nullable": False})

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()

