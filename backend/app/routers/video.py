from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel

from ..core.config import get_settings
from ..services.segmentation_service import (
    IDENTITY_MATCH_THRESHOLD,
    FaceSwapSlot,
    get_segmentation_service,
)
from ..services.video_tryon_service import extract_frames, get_video_tryon_engine, probe_video

router = APIRouter(prefix="/video", tags=["video"])
settings = get_settings()

# This video addon only drives the garment-swap try-on pipeline; face swap is
# intentionally excluded here.
SUPPORTED_TRYON_MODES = {"garment_swap"}


class VideoTryOnJob(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    frames_done: int = 0
    frames_total: int = 0
    result_video_path: str | None = None
    failure_reason: str | None = None
    created_at: float


# In-memory job registry. Fine for a single-process demo/addon; swap for the
# database-backed TryOnJob model if this needs to survive restarts or scale
# across workers.
_video_tryon_jobs: dict[str, VideoTryOnJob] = {}


def _run_video_tryon_job(
    job_id: str,
    video_path: Path,
    garment_path: Path,
    *,
    prompt: str | None,
    mode: str,
    category: str,
    fps: float | None,
    max_frames: int | None,
    frame_stride: int,
) -> None:
    job = _video_tryon_jobs[job_id]
    job.status = "processing"

    def on_progress(done: int, total: int) -> None:
        job.frames_done = done
        job.frames_total = total

    output_path = settings.video_results_dir / f"tryon-{job_id}.mp4"
    engine = get_video_tryon_engine()

    try:
        engine.run(
            video_path,
            garment_path,
            output_path,
            prompt=prompt,
            mode=mode,
            category=category,
            fps=fps,
            max_frames=max_frames,
            frame_stride=frame_stride,
            progress_cb=on_progress,
        )
        job.result_video_path = f"/video-results/{output_path.name}"
        job.status = "completed"
    except Exception as exc:  # pragma: no cover - background task error path
        logger.exception("Video try-on job %s failed: %s", job_id, exc)
        job.status = "failed"
        job.failure_reason = str(exc)
    finally:
        for path in (video_path, garment_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


@router.post("/tryon", response_model=VideoTryOnJob, status_code=status.HTTP_202_ACCEPTED)
async def create_video_tryon_job(
    background_tasks: BackgroundTasks,
    user_video: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    mode: str = Form(default="garment_swap"),  # only garment_swap is supported by this endpoint
    category: str = Form(default="auto"),
    fps: float | None = Form(default=None),
    max_frames: int | None = Form(default=None),
    frame_stride: int = Form(default=1),
) -> VideoTryOnJob:
    """Apply the garment try-on pipeline to every frame of an uploaded video."""
    if mode not in SUPPORTED_TRYON_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported mode '{mode}'. Use one of: {', '.join(sorted(SUPPORTED_TRYON_MODES))}.",
        )
    if not user_video.content_type or not user_video.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a video file.")

    temp_dir = settings.storage_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid4().hex
    video_suffix = Path(user_video.filename or "input.mp4").suffix or ".mp4"
    garment_suffix = Path(garment_image.filename or "garment.png").suffix or ".png"
    video_path = temp_dir / f"video-tryon-input-{job_id}{video_suffix}"
    garment_path = temp_dir / f"video-tryon-garment-{job_id}{garment_suffix}"

    try:
        video_path.write_bytes(await user_video.read())
        garment_path.write_bytes(await garment_image.read())
    finally:
        await user_video.close()
        await garment_image.close()

    job = VideoTryOnJob(job_id=job_id, status="pending", created_at=time.time())
    _video_tryon_jobs[job_id] = job

    background_tasks.add_task(
        _run_video_tryon_job,
        job_id,
        video_path,
        garment_path,
        prompt=prompt,
        mode=mode,
        category=category,
        fps=fps,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )

    return job


@router.get("/tryon/{job_id}", response_model=VideoTryOnJob)
async def get_video_tryon_job(job_id: str) -> VideoTryOnJob:
    job = _video_tryon_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video try-on job not found")
    return job


# ----------------------------------------------------------------------
# Video face swap: a separate feature/section from garment video try-on
# above. The video is scanned for faces first, the caller picks which of
# those people to replace (up to MAX_SWAP_FACES) and supplies a replacement
# photo for each, then every frame is re-faced. Runs entirely on this
# machine via this project's own face-swap logic (segmentation_service.py,
# InsightFace); nothing is uploaded anywhere.
# ----------------------------------------------------------------------

MAX_SWAP_FACES = 2
# Frames sampled evenly across the video when scanning for faces. Enough to
# catch people who only appear partway through, without making the scan drag.
SCAN_SAMPLE_FRAMES = 12
# Upper bound on faces offered in the picker, most prominent first.
MAX_SCANNED_FACES = 10

_video_faceswap_jobs: dict[str, VideoTryOnJob] = {}
_video_faceswap_cancel_events: dict[str, threading.Event] = {}
# Scan results awaiting a job submit, keyed by upload_id. Holds the cached
# video plus each detected face's identity embedding so the caller doesn't
# have to re-upload the video when they submit. In-memory, matching the job
# registries above.
_video_faceswap_scans: dict[str, dict[str, Any]] = {}


class DetectedFace(BaseModel):
    face_index: int
    bbox: list[float]
    det_score: float
    thumbnail_url: str


class FaceScanResult(BaseModel):
    upload_id: str
    faces: list[DetectedFace]


def _scan_faces(scan_dir: Path, video_path: Path) -> list[dict]:
    """Find everyone who appears anywhere in the video.

    Samples frames spread across the whole clip rather than just looking at the
    first one, so someone who walks in halfway through still shows up in the
    picker. Faces are then grouped by identity — using the same similarity
    threshold the swap itself uses, so what the picker calls one person is
    exactly what the swap will track — leaving one entry per person instead of
    one per appearance.
    """
    sample_dir = scan_dir / "samples"
    info = probe_video(video_path)
    # Spread the samples over the running time; fall back to one per second
    # when ffprobe couldn't determine a duration.
    sample_fps = SCAN_SAMPLE_FRAMES / info.duration_seconds if info.duration_seconds > 0 else 1.0
    extract_frames(video_path, sample_dir, fps=sample_fps, max_frames=SCAN_SAMPLE_FRAMES)

    sample_frames = sorted(sample_dir.glob("*.png"))
    if not sample_frames:
        raise RuntimeError("Could not read any frames from that video.")

    segmentation = get_segmentation_service()
    people: list[dict] = []

    for sample_path in sample_frames:
        frame = cv2.imread(str(sample_path))
        if frame is None:
            continue
        for face in segmentation.detect_faces_with_embeddings(frame):
            embedding = face["embedding"]
            best: Optional[dict] = None
            best_score = IDENTITY_MATCH_THRESHOLD
            for person in people:
                score = float(np.dot(embedding, person["embedding"]))
                if score > best_score:
                    best, best_score = person, score

            if best is None:
                people.append(
                    {
                        "embedding": embedding,
                        "bbox": face["bbox"],
                        "det_score": face["det_score"],
                        "thumbnail": segmentation.crop_face(frame, face["bbox"]),
                        "appearances": 1,
                    }
                )
                continue

            best["appearances"] += 1
            # Keep the clearest sighting of this person as the thumbnail and
            # the reference embedding, since that's the most reliable one.
            if face["det_score"] > best["det_score"]:
                best.update(
                    embedding=embedding,
                    bbox=face["bbox"],
                    det_score=face["det_score"],
                    thumbnail=segmentation.crop_face(frame, face["bbox"]),
                )

    # Most-present people first: the person on screen throughout is far more
    # likely to be the one being swapped than a passer-by caught in one frame.
    people.sort(key=lambda p: (p["appearances"], p["det_score"]), reverse=True)
    del people[MAX_SCANNED_FACES:]

    for index, person in enumerate(people):
        cv2.imwrite(str(scan_dir / f"face_{index}.png"), person.pop("thumbnail"))

    logger.info("Face scan: {} frame(s) sampled, {} distinct face(s)", len(sample_frames), len(people))
    shutil.rmtree(sample_dir, ignore_errors=True)
    return people


@router.post("/faceswap/scan", response_model=FaceScanResult)
async def scan_video_faceswap_faces(video: UploadFile = File(...)) -> FaceScanResult:
    """Find the faces in an uploaded video so the caller can pick which to swap.

    The video is cached server-side under the returned `upload_id`; pass that
    id to POST /video/faceswap instead of uploading the video a second time.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a video file.")

    upload_id = uuid4().hex
    scan_dir = settings.uploads_dir / "faceswap" / upload_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    video_path = scan_dir / f"source{Path(video.filename or 'input.mp4').suffix or '.mp4'}"

    try:
        video_path.write_bytes(await video.read())
    finally:
        await video.close()

    try:
        faces = await asyncio.to_thread(_scan_faces, scan_dir, video_path)
    except Exception as exc:
        shutil.rmtree(scan_dir, ignore_errors=True)
        logger.exception("Face scan failed for upload %s: %s", upload_id, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not faces:
        shutil.rmtree(scan_dir, ignore_errors=True)
        return FaceScanResult(upload_id=upload_id, faces=[])

    _video_faceswap_scans[upload_id] = {
        "scan_dir": scan_dir,
        "video_path": video_path,
        "faces": faces,
        "created_at": time.time(),
    }

    return FaceScanResult(
        upload_id=upload_id,
        faces=[
            DetectedFace(
                face_index=index,
                bbox=face["bbox"],
                det_score=face["det_score"],
                thumbnail_url=f"/uploads/faceswap/{upload_id}/face_{index}.png",
            )
            for index, face in enumerate(faces)
        ],
    )


def _build_swap_slots(scan: dict[str, Any], face_paths: list[tuple[int, Path]]) -> list[FaceSwapSlot]:
    """Pair each chosen person in the video with the face replacing them."""
    segmentation = get_segmentation_service()
    slots: list[FaceSwapSlot] = []

    for position, (target_index, source_path) in enumerate(face_paths, start=1):
        source_image = cv2.imread(str(source_path))
        if source_image is None:
            raise ValueError(f"Could not read replacement photo #{position}.")
        source_face = segmentation.get_primary_face(source_image)
        if source_face is None:
            raise ValueError(f"No face found in replacement photo #{position}.")
        slots.append(
            FaceSwapSlot(
                target_ref_embedding=scan["faces"][target_index]["embedding"],
                source_face=source_face,
            )
        )

    return slots


def _run_video_faceswap_job(
    job_id: str,
    upload_id: str,
    target_video_path: Path,
    scan_dir: Path,
    slots: list[FaceSwapSlot],
    *,
    fps: float | None,
    max_frames: int | None,
    frame_stride: int,
    nose_ring: bool,
) -> None:
    job = _video_faceswap_jobs[job_id]
    job.status = "processing"
    cancel_event = _video_faceswap_cancel_events[job_id]

    def on_progress(done: int, total: int) -> None:
        job.frames_done = done
        job.frames_total = total

    output_path = settings.video_results_dir / f"faceswap-{job_id}.mp4"
    engine = get_video_tryon_engine()

    try:
        engine.run_face_swap_multi(
            target_video_path,
            output_path,
            slots,
            fps=fps,
            max_frames=max_frames,
            frame_stride=frame_stride,
            nose_ring=nose_ring,
            progress_cb=on_progress,
            should_stop=cancel_event.is_set,
        )
        job.result_video_path = f"/video-results/{output_path.name}"
        job.status = "cancelled" if cancel_event.is_set() else "completed"
    except Exception as exc:  # pragma: no cover - background task error path
        logger.exception("Video face swap job %s failed: %s", job_id, exc)
        job.status = "failed"
        job.failure_reason = str(exc)
    finally:
        _video_faceswap_cancel_events.pop(job_id, None)
        _video_faceswap_scans.pop(upload_id, None)
        shutil.rmtree(scan_dir, ignore_errors=True)


@router.post("/faceswap", response_model=VideoTryOnJob, status_code=status.HTTP_202_ACCEPTED)
async def create_video_faceswap_job(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    source_face_images: list[UploadFile] = File(...),
    target_face_indices: list[int] = Form(...),
    fps: float | None = Form(default=None),
    max_frames: int | None = Form(default=None),
    frame_stride: int = Form(default=1),
    nose_ring: bool = Form(default=False),
) -> VideoTryOnJob:
    """Replace the chosen people throughout a previously scanned video.

    `target_face_indices` selects which faces from POST /video/faceswap/scan to
    replace; `source_face_images` supplies the replacement face for each, in the
    same order. Body, pose and audio are preserved. `nose_ring` adds a gold
    hoop to each swapped person's left nostril.
    """
    scan = _video_faceswap_scans.get(upload_id)
    if scan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That scan has expired or was never made. Re-upload the video to scan it again.",
        )

    if len(source_face_images) != len(target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Each chosen face needs exactly one replacement photo.",
        )
    if not 1 <= len(target_face_indices) <= MAX_SWAP_FACES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Choose between 1 and {MAX_SWAP_FACES} faces to swap.",
        )
    if len(set(target_face_indices)) != len(target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The same face was chosen more than once.",
        )
    face_count = len(scan["faces"])
    if any(index < 0 or index >= face_count for index in target_face_indices):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A chosen face isn't part of that scan.",
        )
    for image in source_face_images:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload JPEG or PNG images.")

    scan_dir: Path = scan["scan_dir"]
    face_paths: list[tuple[int, Path]] = []
    for position, (target_index, image) in enumerate(zip(target_face_indices, source_face_images)):
        source_path = scan_dir / f"source_face_{position}{Path(image.filename or 'face.png').suffix or '.png'}"
        try:
            source_path.write_bytes(await image.read())
        finally:
            await image.close()
        face_paths.append((target_index, source_path))

    try:
        slots = await asyncio.to_thread(_build_swap_slots, scan, face_paths)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    job_id = uuid4().hex
    job = VideoTryOnJob(job_id=job_id, status="pending", created_at=time.time())
    _video_faceswap_jobs[job_id] = job
    _video_faceswap_cancel_events[job_id] = threading.Event()

    background_tasks.add_task(
        _run_video_faceswap_job,
        job_id,
        upload_id,
        scan["video_path"],
        scan_dir,
        slots,
        fps=fps,
        max_frames=max_frames,
        frame_stride=frame_stride,
        nose_ring=nose_ring,
    )

    return job


@router.get("/faceswap/{job_id}", response_model=VideoTryOnJob)
async def get_video_faceswap_job(job_id: str) -> VideoTryOnJob:
    job = _video_faceswap_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video face swap job not found")
    return job


@router.post("/faceswap/{job_id}/cancel", response_model=VideoTryOnJob)
async def cancel_video_faceswap_job(job_id: str) -> VideoTryOnJob:
    """Stop an in-progress face swap job as soon as possible; the frames already
    processed are still assembled into a (shorter) finished video."""
    job = _video_faceswap_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video face swap job not found")
    cancel_event = _video_faceswap_cancel_events.get(job_id)
    if cancel_event is not None and job.status in ("pending", "processing"):
        cancel_event.set()
    return job
