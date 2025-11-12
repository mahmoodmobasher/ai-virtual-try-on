from __future__ import annotations

from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from loguru import logger
from PIL import Image

from ..core.config import get_settings

try:  # Optional heavy deps
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # Stable Video Diffusion pipeline
    from diffusers import StableVideoDiffusionPipeline
except Exception:  # pragma: no cover
    StableVideoDiffusionPipeline = None  # type: ignore


class StableVideoDiffusionService:
    """Wraps Stable Video Diffusion (image-to-video) for clip generation."""

    def __init__(self) -> None:
        self.settings = get_settings()

        if torch is None or StableVideoDiffusionPipeline is None:
            raise RuntimeError("Stable Video Diffusion requires PyTorch and diffusers")

        self.settings.video_results_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._select_device()
        logger.info("Loading Stable Video Diffusion pipeline (%s) on %s", self.settings.svd_model_id, self.device)

        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        pipeline_kwargs: dict[str, object] = {"torch_dtype": dtype}
        if dtype == torch.float16:
            pipeline_kwargs["variant"] = "fp16"

        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            self.settings.svd_model_id,
            **pipeline_kwargs,
        )

        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()
        elif self.device == "mps":
            self.pipeline.to("mps")
        else:
            self.pipeline.enable_sequential_cpu_offload()

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        logger.warning("Stable Video Diffusion running on CPU; generation will be very slow.")
        return "cpu"

    def generate_clip(
        self,
        image_path: Path,
        output_path: Path,
        *,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Path:
        if torch is None:
            raise RuntimeError("PyTorch not available for Stable Video Diffusion")

        frames = frames or self.settings.svd_default_frames
        fps = fps or self.settings.svd_default_fps
        guidance_scale = guidance_scale or self.settings.svd_guidance_scale
        motion_bucket_id = motion_bucket_id or self.settings.svd_motion_bucket_id
        noise_aug_strength = (
            self.settings.svd_noise_aug_strength if noise_aug_strength is None else noise_aug_strength
        )

        logger.info(
            "Generating video clip via Stable Video Diffusion (frames=%s, fps=%s, guidance=%s, motion_bucket=%s)",
            frames,
            fps,
            guidance_scale,
            motion_bucket_id,
        )

        pil_image = Image.open(image_path).convert("RGB")
        pil_image = pil_image.resize(
            (
                self.settings.svd_image_width,
                self.settings.svd_image_height,
            ),
            Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC,
        )

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        pipeline_kwargs = dict(
            image=pil_image,
            height=self.settings.svd_image_height,
            width=self.settings.svd_image_width,
            num_frames=frames,
            num_inference_steps=self.settings.svd_inference_steps,
            min_guidance_scale=guidance_scale,
            max_guidance_scale=max(guidance_scale, guidance_scale + 1.0),
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=min(frames, 8),
            output_type="pil",
        )
        if prompt:
            pipeline_kwargs["prompt"] = prompt
        if generator is not None:
            pipeline_kwargs["generator"] = generator

        result = self.pipeline(**pipeline_kwargs)
        clip_frames = result.frames[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            output_path,
            [np.array(frame) for frame in clip_frames],
            fps=fps,
            codec="libx264",
            quality=8,
        )

        logger.info("Stable Video Diffusion clip saved to %s", output_path)
        return output_path


_service: Optional[StableVideoDiffusionService] = None


def get_video_generation_service() -> StableVideoDiffusionService:
    global _service
    if _service is None:
        _service = StableVideoDiffusionService()
    return _service
