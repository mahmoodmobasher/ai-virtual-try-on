from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker

from .config import get_settings


class Database:
    """Database connection manager."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine: AsyncEngine = create_async_engine(settings.database_url, echo=False, future=True)
        self._session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_models(self) -> None:
        """Create tables if missing."""

        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def dispose(self) -> None:
        """Close engine."""

        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency for FastAPI routes."""

        async with self._session_factory() as session:
            yield session


db = Database()

