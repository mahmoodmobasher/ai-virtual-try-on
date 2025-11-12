from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from ..core.dependencies import SessionDep
from ..models import Moodboard, MoodboardItem
from ..services.moodboard_service import MoodboardService

router = APIRouter(prefix="/moodboards", tags=["moodboards"])


@router.get("/", response_model=list[Moodboard])
async def list_boards(session: SessionDep, owner: str | None = None) -> list[Moodboard]:
    boards = await MoodboardService.list_boards(session, owner=owner)
    return list(boards)


@router.post("/", response_model=Moodboard, status_code=status.HTTP_201_CREATED)
async def create_board(session: SessionDep, name: str, owner: str | None = None, description: str | None = None) -> Moodboard:
    return await MoodboardService.create_board(session, name, owner, description)


@router.post("/{board_id}/items", response_model=MoodboardItem, status_code=status.HTTP_201_CREATED)
async def add_item(
    board_id: int,
    session: SessionDep,
    garment_id: int | None = None,
    asset_path: str | None = None,
    caption: str | None = None,
) -> MoodboardItem:
    board = await MoodboardService.get_board(session, board_id)
    if not board:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard not found")
    return await MoodboardService.add_item(
        session,
        board,
        garment_id=garment_id,
        asset_path=asset_path,
        caption=caption,
        meta=None,
    )


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int, session: SessionDep) -> Response:
    item = await session.get(MoodboardItem, item_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Moodboard item not found")
    await MoodboardService.delete_item(session, item)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

