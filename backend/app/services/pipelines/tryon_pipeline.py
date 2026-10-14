from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from ..segmentation_service import get_segmentation_service
from ..body_segmentation_service import get_body_segmentation_service
from ..detectron2_service import get_detectron2_service
from ..ollama_client import get_ollama_client
from ..lm_studio_client import get_lm_studio_client
from ..catvton import get_local_tryon_engine
from ..fashn_client import get_fashn_client

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # CatVTON / FLUX dependencies
    from diffusers import FluxInpaintPipeline, FluxTransformer2DModel
    from diffusers.utils import load_image
except Exception:  # pragma: no cover
    FluxInpaintPipeline = None
    FluxTransformer2DModel = None
    load_image = None

try:
    from torchvision import transforms
except Exception:  # pragma: no cover
    transforms = None

from ...core.config import get_settings


class TryOnPipeline:
    """Pipeline orchestrator for clothes swap, face swap and CatVTON-FLUX try-on modes."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.cache_dir = cache_dir or (self.settings.storage_dir / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = (
            "cuda"
            if torch and torch.cuda.is_available()
            else "mps"
            if torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )

        self._segmentation = None
        self._body_segmentation = None
        self._detectron2 = None
        self._llm = None
        self._catvton_pipe = None
        self._catvton_dtype = None

    @property
    def segmentation(self):
        if self._segmentation is None:
            self._segmentation = get_segmentation_service()
        return self._segmentation

    @property
    def body_segmentation(self):
        if self._body_segmentation is None:
            self._body_segmentation = get_body_segmentation_service()
        return self._body_segmentation

    @property
    def detectron2(self):
        if self._detectron2 is None:
            self._detectron2 = get_detectron2_service()
        return self._detectron2

    @property
    def llm(self):
        if self._llm is None:
            if self.settings.llm_provider == "lm_studio":
                self._llm = get_lm_studio_client()
            else:
                self._llm = get_ollama_client()
        return self._llm

    def load(self) -> None:
        """Deprecated placeholder retained for backwards compatibility."""
        return

    def apply(
        self,
        user_image_path: Path,
        garment_image_path: Optional[Path],
        output_path: Path,
        *,
        prompt: str | None = None,
        mode: str = "garment_swap",
        category: str = "auto",
        face_swaps: Optional[Sequence[dict]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Generate try-on result with preprocessing.

        Args:
            user_image_path: Path to user's model photo.
            garment_image_path: Path to garment/reference photo.
            output_path: Where to save the result.
            prompt: Optional text prompt for CatVTON-FLUX / FASHN Try-On Max.
            mode: "garment_swap", "face_swap" or "catvton_flux" (legacy "viton_hd" accepted).
            category: Garment category for clothes swap: auto, tops, bottoms or one-pieces.
            face_swaps: face_swap only. Faces picked from a scan of `user_image_path`,
                each a dict with the face's `kps` and the `source_image_path` of the
                photo replacing it. When given, `user_image_path` is the photo being
                re-faced and `garment_image_path` is unused.
            progress_cb: Optional callback(step, total_steps) reporting diffusion steps
                (local garment_swap engine) or faces swapped (face_swap); other
                engines/modes don't have a comparable step-wise notion of progress.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Try-on mode: {}", mode)

        if mode == "garment_swap":
            if self.settings.garment_swap_engine == "fashn":
                if self.settings.enforce_local_only:
                    raise RuntimeError(
                        "garment_swap_engine='fashn' would upload images to the hosted FASHN API, "
                        "but enforce_local_only=True blocks that. Set TRYON_GARMENT_SWAP_ENGINE=local "
                        "or explicitly set TRYON_ENFORCE_LOCAL_ONLY=false to allow it."
                    )
                return get_fashn_client().try_on(
                    user_image_path, garment_image_path, output_path, category=category, prompt=prompt
                )
            return get_local_tryon_engine().try_on(
                user_image_path, garment_image_path, output_path, category=category, progress_cb=progress_cb
            )

        if mode in {"catvton_flux", "viton_hd"}:
            logger.info("Using CatVTON-FLUX mode for garment transfer")
            return self._apply_catvton_flux(user_image_path, garment_image_path, output_path, prompt)

        logger.info("Using Face Swap mode")
        if face_swaps:
            return self._apply_face_swaps(user_image_path, face_swaps, output_path, progress_cb)
        return self._apply_face_swap(user_image_path, garment_image_path, output_path)

    # ------------------------------------------------------------------
    # CatVTON helpers
    # ------------------------------------------------------------------
    def _resolve_catvton_dtype(self) -> Optional["torch.dtype"]:
        if not torch:
            return None

        if self.device == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        if self.device == "mps":
            return torch.float16

        return torch.float32

    def _load_catvton_flux(self) -> Optional["FluxInpaintPipeline"]:
        if self._catvton_pipe is not None:
            return self._catvton_pipe

        if not torch or FluxInpaintPipeline is None or FluxTransformer2DModel is None:
            logger.warning("CatVTON-FLUX dependencies missing; cannot load pipeline.")
            return None

        if transforms is None or load_image is None:
            logger.warning("torchvision / diffusers utilities missing; cannot run CatVTON-FLUX.")
            return None

        dtype = self._resolve_catvton_dtype()
        if dtype is None:
            logger.warning("Unable to determine dtype for CatVTON-FLUX; skipping load.")
            return None

        try:
            logger.info("Loading CatVTON-FLUX transformer weights (xiaozaa/catvton-flux-beta)...")
            transformer = FluxTransformer2DModel.from_pretrained(
                "xiaozaa/catvton-flux-beta",
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )

            base_model = "black-forest-labs/FLUX.1-Fill-dev"
            logger.info("Loading FLUX fill base pipeline %s...", base_model)
            pipe = FluxInpaintPipeline.from_pretrained(
                base_model,
                transformer=transformer,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )

            if self.device == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:  # pragma: no cover
                    logger.debug("xFormers not available; continuing without it.")

            pipe.to(self.device)
            self._catvton_pipe = pipe
            self._catvton_dtype = dtype
            logger.info("CatVTON-FLUX pipeline ready on %s with dtype %s", self.device, dtype)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load CatVTON-FLUX pipeline: %s", exc)
            self._catvton_pipe = None

        return self._catvton_pipe

    # ------------------------------------------------------------------
    # Face swap mode
    # ------------------------------------------------------------------
    def _apply_face_swap(
        self,
        user_image_path: Path,
        garment_image_path: Path,
        output_path: Path,
    ) -> Path:
        try:
            segmentation = self.segmentation
        except RuntimeError as exc:
            raise RuntimeError(
                "Face swap is unavailable because its ML runtime or model files are missing."
            ) from exc

        user_img_bgr = cv2.imread(str(user_image_path))
        garment_img_bgr = cv2.imread(str(garment_image_path))

        if user_img_bgr is None:
            raise ValueError("The source-face image could not be read. Please upload a valid JPEG or PNG image.")
        if garment_img_bgr is None:
            raise ValueError("The reference-outfit image could not be read. Please upload a valid JPEG or PNG image.")

        source_face = segmentation.get_primary_face(user_img_bgr)
        target_face = segmentation.get_primary_face(garment_img_bgr)
        logger.info("Face detection: source={}, reference={}", source_face is not None, target_face is not None)

        if source_face is None or target_face is None:
            missing = []
            if source_face is None:
                missing.append("source image")
            if target_face is None:
                missing.append("reference outfit image")
            error_msg = (
                f"No clear face was detected in the {' and '.join(missing)}. "
                "Use clear, front-facing photos; the reference image must show a person wearing the outfit."
            )
            logger.warning(error_msg)
            raise ValueError(error_msg)

        swapped = segmentation.swap_faces_image(garment_img_bgr, [(target_face.kps, source_face)])
        cv2.imwrite(str(output_path), swapped)
        logger.info("Face swap complete, saved to {}", output_path)
        return output_path

    def _apply_face_swaps(
        self,
        target_image_path: Path,
        face_swaps: Sequence[dict],
        output_path: Path,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Replace the faces the user picked from a scan of the target photo."""
        try:
            segmentation = self.segmentation
        except RuntimeError as exc:
            raise RuntimeError(
                "Face swap is unavailable because its ML runtime or model files are missing."
            ) from exc

        target_bgr = cv2.imread(str(target_image_path))
        if target_bgr is None:
            raise ValueError("The photo could not be read. Please upload a valid JPEG or PNG image.")

        assignments = []
        for position, swap in enumerate(face_swaps, start=1):
            source_bgr = cv2.imread(str(swap["source_image_path"]))
            if source_bgr is None:
                raise ValueError(f"Could not read replacement photo #{position}.")
            source_face = segmentation.get_primary_face(source_bgr)
            if source_face is None:
                raise ValueError(
                    f"No face found in replacement photo #{position}. Use a clear, front-facing photo."
                )
            assignments.append((np.asarray(swap["kps"], dtype=np.float32), source_face))

        swapped = segmentation.swap_faces_image(target_bgr, assignments, progress_cb=progress_cb)
        cv2.imwrite(str(output_path), swapped)
        logger.info("Face swap complete ({} face(s)), saved to {}", len(assignments), output_path)
        return output_path

    # ------------------------------------------------------------------
    # CatVTON mode
    # ------------------------------------------------------------------
    def _generate_person_mask(self, image_bgr: "np.ndarray", size: tuple[int, int]) -> Image.Image:
        if image_bgr is None:
            mask = np.ones(size[::-1], dtype=np.uint8) * 255
            return Image.fromarray(mask, mode="L").convert("RGB")

        mask = None
        try:
            _, detectron_mask = self.detectron2.segment_person(image_bgr, threshold=0.5, return_largest=True)
            mask = detectron_mask
            logger.debug("Using Detectron2 mask for CatVTON-FLUX.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Detectron2 mask generation failed: %s", exc)
            try:
                _, mediapipe_mask = self.body_segmentation.segment_person(image_bgr, threshold=0.5)
                mask = mediapipe_mask
                logger.debug("Using Mediapipe mask for CatVTON-FLUX.")
            except Exception as fallback_exc:  # pragma: no cover
                logger.warning("Mediapipe mask generation also failed: %s", fallback_exc)

        if mask is None:
            mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255

        binary_mask = (mask > 127).astype(np.uint8) * 255
        resized = cv2.resize(binary_mask, size[::-1], interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(resized, mode="L").convert("RGB")

    def _apply_catvton_flux(
        self,
        user_image_path: Path,
        garment_image_path: Path,
        output_path: Path,
        prompt: str | None = None,
    ) -> Path:
        if garment_image_path is None or not garment_image_path.exists():
            raise ValueError("CatVTON-FLUX mode requires a garment image upload.")

        if not torch:
            logger.warning("Torch not available; falling back to naive overlay for CatVTON-FLUX.")
            self._fallback_overlay(user_image_path, garment_image_path, output_path)
            return output_path

        pipe = self._load_catvton_flux()
        if pipe is None:
            logger.warning("CatVTON-FLUX pipeline unavailable; falling back to overlay.")
            self._fallback_overlay(user_image_path, garment_image_path, output_path)
            return output_path

        if transforms is None or load_image is None:
            logger.warning("Missing preprocessing utilities for CatVTON-FLUX; using fallback overlay.")
            self._fallback_overlay(user_image_path, garment_image_path, output_path)
            return output_path

        try:
            user_image = load_image(str(user_image_path)).convert("RGB")
            garment_image = load_image(str(garment_image_path)).convert("RGB")

            target_width = self.settings.catvton_width
            target_height = self.settings.catvton_height

            user_resized = user_image.resize((target_width, target_height), Image.BILINEAR)
            garment_resized = garment_image.resize((target_width, target_height), Image.BILINEAR)

            user_img_bgr = cv2.imread(str(user_image_path))
            mask_image = self._generate_person_mask(user_img_bgr, (target_width, target_height))

            transform_rgb = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
            transform_mask = transforms.Compose([transforms.ToTensor()])

            image_tensor = transform_rgb(user_resized)
            garment_tensor = transform_rgb(garment_resized)
            mask_tensor = transform_mask(mask_image)[:1]
            mask_tensor = (mask_tensor > 0.5).float()

            inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
            garment_blank = torch.zeros_like(mask_tensor)
            extended_mask = torch.cat([garment_blank, mask_tensor], dim=2)

            generator_device = "cuda" if self.device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(self.settings.catvton_seed)

            pipe_inputs = {
                "height": target_height,
                "width": target_width * 2,
                "image": inpaint_image.unsqueeze(0).to(self.device, dtype=self._catvton_dtype),
                "mask_image": extended_mask.unsqueeze(0).to(self.device, dtype=self._catvton_dtype),
                "num_inference_steps": self.settings.catvton_num_steps,
                "guidance_scale": self.settings.catvton_guidance_scale,
                "generator": generator,
                "max_sequence_length": 512,
                "prompt": prompt
                or "High resolution fashion editorial, garment fits naturally, realistic fabric and lighting.",
            }

            logger.info(
                "Running CatVTON-FLUX inference (steps=%s, guidance=%s, resolution=%sx%s)",
                self.settings.catvton_num_steps,
                self.settings.catvton_guidance_scale,
                target_width,
                target_height,
            )

            with torch.inference_mode():
                result_image = pipe(**pipe_inputs).images[0]

            result_tryon = result_image.crop((target_width, 0, target_width * 2, target_height))
            result_tryon.save(output_path)
            logger.info("CatVTON-FLUX inference complete; saved to %s", output_path)
            return output_path
        except Exception as exc:  # pragma: no cover
            logger.exception("CatVTON-FLUX inference failed: %s", exc)
            self._fallback_overlay(user_image_path, garment_image_path, output_path)
            return output_path

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _fallback_overlay(self, user_image_path: Path, garment_image_path: Path, output_path: Path) -> None:
        logger.warning("Running fallback overlay pipeline; output will be simplistic.")
        try:
            user_image = Image.open(user_image_path).convert("RGB")
            garment_image = Image.open(garment_image_path).convert("RGBA")
            garment_image = garment_image.resize((int(user_image.width * 0.35), int(user_image.height * 0.35)))
            user_image.paste(garment_image, (user_image.width - garment_image.width - 20, 20), garment_image)
            user_image.save(output_path)
        except Exception:
            shutil.copy(user_image_path, output_path)
