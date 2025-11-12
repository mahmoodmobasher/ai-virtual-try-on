from collections.abc import AsyncIterator

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sqlmodel.ext.asyncio.session import AsyncSession

from .core.config import get_settings
from .core.database import db
from .routers import analytics, garments, llm, moodboard, tryon, video
from .services.seed import ensure_demo_catalog


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await db.init_models()
    async with db.session() as session:  # type: AsyncSession
        await ensure_demo_catalog(session)
    yield
    await db.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_prefix = settings.api_prefix
    app.include_router(garments.router, prefix=api_prefix)
    app.include_router(tryon.router, prefix=api_prefix)
    app.include_router(moodboard.router, prefix=api_prefix)
    app.include_router(analytics.router, prefix=api_prefix)
    app.include_router(video.router, prefix=api_prefix)
    app.include_router(llm.router, prefix=api_prefix)

    @app.get("/healthz", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.mount(
        "/results",
        StaticFiles(directory=settings.results_dir),
        name="results",
    )
    app.mount(
        "/uploads",
        StaticFiles(directory=settings.uploads_dir),
        name="uploads",
    )
    app.mount(
        "/catalog",
        StaticFiles(directory=settings.catalog_dir),
        name="catalog",
    )

    return app


app = create_app()

