from collections.abc import Sequence

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from ..models import Moodboard, MoodboardItem


class MoodboardService:
    """CRUD operations for moodboards and items."""

    @staticmethod
    async def list_boards(session: AsyncSession, *, owner: str | None = None) -> Sequence[Moodboard]:
        statement = select(Moodboard)
        if owner:
            statement = statement.where(Moodboard.owner == owner)
        result = await session.exec(statement)
        return result.all()

    @staticmethod
    async def get_board(session: AsyncSession, board_id: int) -> Moodboard | None:
        return await session.get(Moodboard, board_id)

    @staticmethod
    async def create_board(session: AsyncSession, name: str, owner: str | None, description: str | None) -> Moodboard:
        board = Moodboard(name=name, owner=owner, description=description)
        session.add(board)
        await session.commit()
        await session.refresh(board)
        return board

    @staticmethod
    async def add_item(
        session: AsyncSession,
        moodboard: Moodboard,
        *,
        garment_id: int | None,
        asset_path: str | None,
        caption: str | None,
        meta: dict | None,
    ) -> MoodboardItem:
        item = MoodboardItem(
            moodboard_id=moodboard.id,
            garment_id=garment_id,
            asset_path=asset_path,
            caption=caption,
            meta=meta,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return item

    @staticmethod
    async def delete_item(session: AsyncSession, item: MoodboardItem) -> None:
        await session.delete(item)
        await session.commit()

