"""Local clothes swap engine: CatVTON + automatic clothing masks, runs on CUDA, Apple Silicon or CPU."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from loguru import logger
from PIL import Image, ImageFilter, ImageOps

from ...core.config import get_settings

CATEGORY_TO_PART = {"tops": "upper", "bottoms": "lower", "one-pieces": "overall"}


class LocalTryOnEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = self.settings.storage_dir / "cache" / "huggingface"
        self._pipeline = None
        self._masker = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch

            from .masker import ClothMasker
            from .pipeline import CatVTONPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Local clothes swap needs the ML environment. Run: make setup-ml ML_PYTHON=python3.11"
            ) from exc

        # Leave a couple of cores free for the OS/UI instead of pegging every
        # core, which is what makes the machine feel hung during inference.
        torch.set_num_threads(self.settings.effective_tryon_cpu_threads)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading CatVTON on {} (first run downloads ~4 GB of weights)...", device)
        self._masker = ClothMasker(device, self.cache_dir)
        self._pipeline = CatVTONPipeline(device, self.cache_dir)
        logger.info("CatVTON ready")

    def try_on(
        self,
        person_path: Path,
        garment_path: Path,
        output_path: Path,
        *,
        category: str = "auto",
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        with self._lock:  # one generation at a time; the model uses most of the GPU
            self._load()
            width, height = self.settings.local_tryon_width, self.settings.local_tryon_height

            original = ImageOps.exif_transpose(Image.open(person_path)).convert("RGB")
            person, box = _pad_to(original, (width, height))
            garment, _ = _pad_to(ImageOps.exif_transpose(Image.open(garment_path)).convert("RGB"), (width, height))

            part = CATEGORY_TO_PART.get(category) or self._masker.detect_part(garment)
            logger.info("Clothes swap: category={} -> mask part={}", category, part)
            mask = self._masker.mask(person, part).filter(ImageFilter.GaussianBlur(9))

            try:
                result = self._pipeline(
                    person,
                    garment,
                    mask,
                    num_inference_steps=self.settings.local_tryon_steps,
                    guidance_scale=self.settings.local_tryon_guidance_scale,
                    seed=self.settings.catvton_seed,
                    progress_cb=progress_cb,
                )
            finally:
                self._release_memory()

        # Paste only the repainted clothing back into the full-resolution photo so the
        # face and background stay pixel-identical.
        feather = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
        feather = feather.crop(box).resize(original.size, Image.LANCZOS)
        original.paste(result.crop(box).resize(original.size, Image.LANCZOS), (0, 0), feather)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        original.save(output_path)
        return output_path

    def _release_memory(self) -> None:
        """Diffusion inference leaves large intermediate tensors and allocator
        caches behind; without this, repeated calls in the same process (e.g.
        one per video frame) accumulate memory until the machine runs low."""
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _pad_to(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Fit the image inside `size` on a white canvas; return it with the box it occupies."""
    canvas = Image.new("RGB", size, (255, 255, 255))
    fitted = ImageOps.contain(image, size, Image.LANCZOS)
    left, top = (size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas, (left, top, left + fitted.width, top + fitted.height)


@lru_cache
def get_local_tryon_engine() -> LocalTryOnEngine:
    return LocalTryOnEngine()
