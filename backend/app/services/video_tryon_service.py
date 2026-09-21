"""Video try-on / video face-swap engine.

Applies the existing image try-on pipeline (see `pipelines/tryon_pipeline.py`)
to every frame of a source video, then reassembles the processed frames back
into a video and restores the original audio track. Used for both garment
try-on video (garment stays fixed, each frame is the person) and face-swap
video (source face stays fixed, each frame is the target to re-face).

The frame-extract -> per-frame-process -> merge -> restore-audio shape is a
standard video pipeline pattern (also used by tools like FaceFusion for
face-swap video), but this implementation is written from scratch against
ffmpeg/ffprobe directly and only carries this project's own garment-swap and
face-swap logic (`pipelines/tryon_pipeline.py`, `segmentation_service.py`).
It intentionally does not include any of FaceFusion's own code, models, or
content-moderation/guardrail modules.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from loguru import logger

from ..core.config import get_settings
from .pipelines.tryon_pipeline import TryOnPipeline

ProgressCallback = Callable[[int, int], None]


class VideoProcessingError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails or the input video is unusable."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    has_audio: bool


def _run(command: list[str]) -> subprocess.CompletedProcess:
    logger.debug("Running: {}", " ".join(command))
    return subprocess.run(command, capture_output=True, text=True)


def probe_video(video_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise VideoProcessingError(f"ffprobe failed to read '{video_path.name}': {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise VideoProcessingError(f"'{video_path.name}' does not contain a video stream.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    raw_fps = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    numerator, _, denominator = raw_fps.partition("/")
    fps = float(numerator) / float(denominator) if float(denominator or 0) else 0.0

    duration = float(payload.get("format", {}).get("duration", 0.0))
    frame_count = int(video_stream.get("nb_frames") or 0) or max(1, round(duration * fps))

    return VideoInfo(
        fps=fps or 25.0,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        frame_count=frame_count,
        duration_seconds=duration,
        has_audio=has_audio,
    )


def extract_frames(video_path: Path, frames_dir: Path, *, fps: float, max_frames: Optional[int] = None) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "%08d.png"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "1",
    ]
    if max_frames:
        command += ["-frames:v", str(max_frames)]
    command.append(str(frame_pattern))

    result = _run(command)
    if result.returncode != 0:
        raise VideoProcessingError(f"Frame extraction failed: {result.stderr.strip()[-500:]}")

    frame_count = len(list(frames_dir.glob("*.png")))
    if frame_count == 0:
        raise VideoProcessingError("Frame extraction produced no frames.")
    return frame_count


def merge_frames(frames_dir: Path, output_path: Path, *, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "%08d.png"

    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise VideoProcessingError(f"Frame merge failed: {result.stderr.strip()[-500:]}")


def mux_audio(silent_video_path: Path, audio_source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(audio_source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = _run(command)
    if result.returncode != 0:
        raise VideoProcessingError(f"Audio restore failed: {result.stderr.strip()[-500:]}")


class VideoTryOnEngine:
    """Runs the image try-on pipeline over every frame of a video."""

    def __init__(self, pipeline: Optional[TryOnPipeline] = None) -> None:
        self.settings = get_settings()
        self.pipeline = pipeline or TryOnPipeline()

    def run(
        self,
        video_path: Path,
        garment_path: Path,
        output_path: Path,
        *,
        prompt: Optional[str] = None,
        mode: str = "garment_swap",
        category: str = "auto",
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        frame_slot: str = "user_image",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Process `video_path` frame-by-frame and write the result to `output_path`.

        `garment_path` is the fixed reference image reused for every frame (the
        garment photo for garment_swap, or the single source-face photo for
        face_swap). `frame_slot` picks which of TryOnPipeline.apply's two image
        arguments the per-frame video content fills:
        - "user_image" (default, used by garment_swap): each frame is the person/
          body being dressed, `garment_path` is the garment.
        - "garment_image" (used by face_swap): each frame is the target whose
          face gets replaced, `garment_path` is the fixed source-face image.

        frame_stride > 1 processes every Nth frame through the (expensive) try-on
        pipeline and reuses the previous processed frame for the skipped ones,
        trading temporal detail for speed.
        """
        frame_stride = max(1, frame_stride)
        work_dir = self.settings.storage_dir / "tmp" / f"video-tryon-{uuid4().hex}"
        frames_in_dir = work_dir / "frames_in"
        frames_out_dir = work_dir / "frames_out"

        try:
            info = probe_video(video_path)
            effective_fps = fps or info.fps

            frame_count = extract_frames(video_path, frames_in_dir, fps=effective_fps, max_frames=max_frames)
            frames_out_dir.mkdir(parents=True, exist_ok=True)

            input_frames = sorted(frames_in_dir.glob("*.png"))
            last_output_frame: Path | None = None

            for index, frame_path in enumerate(input_frames):
                out_frame_path = frames_out_dir / frame_path.name

                if index % frame_stride == 0 or last_output_frame is None:
                    if frame_slot == "garment_image":
                        user_image_path, garment_image_path = garment_path, frame_path
                    else:
                        user_image_path, garment_image_path = frame_path, garment_path
                    self.pipeline.apply(
                        user_image_path,
                        garment_image_path,
                        out_frame_path,
                        prompt=prompt,
                        mode=mode,
                        category=category,
                    )
                    last_output_frame = out_frame_path
                else:
                    shutil.copyfile(last_output_frame, out_frame_path)

                if progress_cb:
                    progress_cb(index + 1, frame_count)

            silent_output_path = work_dir / "silent_output.mp4"
            merge_frames(frames_out_dir, silent_output_path, fps=effective_fps)

            if info.has_audio:
                mux_audio(silent_output_path, video_path, output_path)
            else:
                shutil.copyfile(silent_output_path, output_path)

            logger.info("Video try-on result saved to {}", output_path)
            return output_path
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


_engine: Optional[VideoTryOnEngine] = None


def get_video_tryon_engine() -> VideoTryOnEngine:
    global _engine
    if _engine is None:
        _engine = VideoTryOnEngine()
    return _engine
