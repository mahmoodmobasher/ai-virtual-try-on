from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from ..segmentation_service import get_segmentation_service
from ..body_segmentation_service import get_body_segmentation_service
from ..detectron2_service import get_detectron2_service
from ..ollama_client import get_ollama_client
from ..lm_studio_client import get_lm_studio_client

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
    """Pipeline orchestrator for face swap and CatVTON-FLUX try-on modes."""

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

        self.segmentation = get_segmentation_service()
        self._body_segmentation = None
        self._detectron2 = None
        self._llm = None
        self._catvton_pipe = None
        self._catvton_dtype = None

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
        garment_image_path: Path,
        output_path: Path,
        *,
        prompt: str | None = None,
        mode: str = "face_swap",
    ) -> Path:
        """
        Generate try-on result with preprocessing.

        Args:
            user_image_path: Path to user's model photo.
            garment_image_path: Path to garment/reference photo.
            output_path: Where to save the result.
            prompt: Optional text prompt for CatVTON-FLUX.
            mode: "face_swap" or "catvton_flux" (legacy "viton_hd" accepted).
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Try-on mode: %s", mode)

        if mode in {"catvton_flux", "viton_hd"}:
            logger.info("Using CatVTON-FLUX mode for garment transfer")
            return self._apply_catvton_flux(user_image_path, garment_image_path, output_path, prompt)

        logger.info("Using Face Swap mode")
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
                local_files_only=False,
            )

            base_model = "black-forest-labs/FLUX.1-Fill-dev"
            logger.info("Loading FLUX fill base pipeline %s...", base_model)
            pipe = FluxInpaintPipeline.from_pretrained(
                base_model,
                transformer=transformer,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                local_files_only=False,
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
        preprocess_dir = self.cache_dir / "preprocessed"
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Preprocessing portrait with InsightFace + rembg...")
            preprocessed = self.segmentation.preprocess_portrait(
                user_image_path,
                preprocess_dir,
                align_face=False,
                remove_bg=True,
            )
            processed_user_path = (
                Path(preprocessed["segmented_path"]) if preprocessed.get("segmented_path") else user_image_path
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Preprocessing failed, using original image: %s", exc)
            processed_user_path = user_image_path

        user_img_bgr = cv2.imread(str(processed_user_path))
        garment_img_bgr = cv2.imread(str(garment_image_path))

        logger.info("Step 1: Segmenting body regions with Detectron2...")
        try:
            user_segmented, user_mask = self.detectron2.segment_person(user_img_bgr, threshold=0.5)
            garment_segmented, garment_mask = self.detectron2.segment_person(garment_img_bgr, threshold=0.5)
            debug_dir = self.cache_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "user_segmented_detectron2.png"), user_segmented)
            cv2.imwrite(str(debug_dir / "user_mask.png"), user_mask)
            cv2.imwrite(str(debug_dir / "garment_segmented_detectron2.png"), garment_segmented)
            cv2.imwrite(str(debug_dir / "garment_mask.png"), garment_mask)
        except Exception as exc:  # pragma: no cover
            logger.warning("Detectron2 segmentation failed, falling back to Mediapipe: %s", exc)
            try:
                user_segmented, user_mask = self.body_segmentation.segment_person(user_img_bgr, threshold=0.5)
                garment_segmented, garment_mask = self.body_segmentation.segment_person(garment_img_bgr, threshold=0.5)
            except Exception as fallback_exc:  # pragma: no cover
                logger.warning("Mediapipe segmentation also failed: %s", fallback_exc)

        user_faces = self.segmentation.face_app.get(user_img_bgr)
        garment_faces = self.segmentation.face_app.get(garment_img_bgr)
        has_user_face = len(user_faces) > 0
        has_garment_face = len(garment_faces) > 0
        logger.info("Face detection: User has face=%s, Garment has face=%s", has_user_face, has_garment_face)

        if has_user_face and has_garment_face:
            try:
                swapped = self.segmentation.swap_face(
                    source_image=user_img_bgr,
                    target_image=garment_img_bgr,
                )
                cv2.imwrite(str(output_path), swapped)
                logger.info("Face swap complete, saved to %s", output_path)
                return output_path
            except Exception as exc:  # pragma: no cover
                logger.exception("Face swap failed: %s", exc)
        else:
            error_msg = (
                "⚠️ Face swap requires BOTH images to have faces. "
                f"User has face: {has_user_face}, Garment has face: {has_garment_face}. "
                "Please upload a reference photo showing someone wearing the garment (not just the garment alone)."
            )
            logger.warning(error_msg)
            result = garment_img_bgr.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Garment needs face for swap"
            cv2.putText(result, text, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(output_path), result)
            raise ValueError(error_msg)

        logger.warning("Face swap pipeline unavailable; using fallback overlay result.")
        self._fallback_overlay(processed_user_path, garment_image_path, output_path)
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
