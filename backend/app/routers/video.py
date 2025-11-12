from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..core.config import get_settings
from ..services.video_generation_service import get_video_generation_service

router = APIRouter(prefix="/video", tags=["video"])
settings = get_settings()


@router.post("/clip", status_code=status.HTTP_202_ACCEPTED)
async def create_video_clip(
    user_image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    frames: int = Form(default=None),
    fps: int = Form(default=None),
    guidance_scale: float = Form(default=None),
    motion_bucket_id: int | None = Form(default=None),
    noise_aug_strength: float | None = Form(default=None),
    seed: int | None = Form(default=None),
) -> dict[str, object]:
    if user_image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a JPEG or PNG image.")

    temp_dir = settings.storage_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"video-input-{uuid4().hex}.png"

    try:
        contents = await user_image.read()
        temp_path.write_bytes(contents)
    finally:
        await user_image.close()

    clip_id = uuid4().hex
    output_path = settings.video_results_dir / f"clip-{clip_id}.mp4"

    service = get_video_generation_service()

    try:
        service.generate_clip(
            image_path=temp_path,
            output_path=output_path,
            frames=frames,
            fps=fps,
            prompt=prompt,
            guidance_scale=guidance_scale,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=seed,
        )
    except RuntimeError as exc:  # device/model issues
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    duration_seconds = (frames or settings.svd_default_frames) / (fps or settings.svd_default_fps)

    return {
        "result_video_path": f"/video-results/{output_path.name}",
        "frames": frames or settings.svd_default_frames,
        "fps": fps or settings.svd_default_fps,
        "duration_seconds": duration_seconds,
    }
