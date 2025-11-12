from __future__ import annotations

from fastapi import APIRouter, status

from ..core.dependencies import SessionDep
from ..models import AnalyticsEvent

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_event(event: AnalyticsEvent, session: SessionDep) -> dict[str, str]:
    session.add(event)
    await session.commit()
    return {"status": "accepted"}

