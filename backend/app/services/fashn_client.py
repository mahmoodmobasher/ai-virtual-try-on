from __future__ import annotations

import base64
import io
import time
from functools import lru_cache
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image, ImageOps

from ..core.config import get_settings

GARMENT_CATEGORIES = {"auto", "tops", "bottoms", "one-pieces"}
_PENDING_STATUSES = {"starting", "in_queue", "processing"}
_MAX_IMAGE_SIDE = 2048


class FashnError(RuntimeError):
    """Raised when the FASHN try-on API cannot produce a result."""


class FashnClient:
    """Minimal client for the FASHN virtual try-on API (https://docs.fashn.ai)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.fashn_api_key)

    def try_on(
        self,
        model_image_path: Path,
        garment_image_path: Path,
        output_path: Path,
        *,
        category: str = "auto",
        prompt: str | None = None,
    ) -> Path:
        if not self.is_configured:
            raise FashnError(
                "Clothes swap needs a FASHN API key. Set TRYON_FASHN_API_KEY in backend/.env "
                "(get one at https://app.fashn.ai/api)."
            )
        if category not in GARMENT_CATEGORIES:
            raise ValueError(f"Unsupported garment category '{category}'.")

        payload = {
            "model_name": self.settings.fashn_model,
            "inputs": self._build_inputs(model_image_path, garment_image_path, category, prompt),
        }
        headers = {"Authorization": f"Bearer {self.settings.fashn_api_key}"}

        with httpx.Client(base_url=self.settings.fashn_api_base, headers=headers, timeout=60) as client:
            response = client.post("/run", json=payload)
            data = self._json(response)
            prediction_id = data.get("id")
            if not prediction_id:
                raise FashnError(f"FASHN rejected the request: {self._error_message(data)}")
            logger.info("FASHN prediction {} started with model {}", prediction_id, self.settings.fashn_model)

            output_url = self._wait_for_output(client, prediction_id)
            result = client.get(output_url, timeout=60)
            result.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(result.content)).convert("RGB").save(output_path)
        logger.info("FASHN prediction {} saved to {}", prediction_id, output_path)
        return output_path

    def _build_inputs(
        self,
        model_image_path: Path,
        garment_image_path: Path,
        category: str,
        prompt: str | None,
    ) -> dict:
        model_image = _encode_image(model_image_path)
        garment_image = _encode_image(garment_image_path)

        # Try-On Max uses a different input schema from the tryon-v1.x models.
        if self.settings.fashn_model.startswith("tryon-max"):
            inputs = {
                "model_image": model_image,
                "product_image": garment_image,
                "generation_mode": self.settings.fashn_mode,
            }
            if prompt:
                inputs["prompt"] = prompt
            return inputs

        return {
            "model_image": model_image,
            "garment_image": garment_image,
            "category": category,
            "mode": self.settings.fashn_mode,
        }

    def _wait_for_output(self, client: httpx.Client, prediction_id: str) -> str:
        deadline = time.monotonic() + self.settings.fashn_timeout_seconds
        while time.monotonic() < deadline:
            data = self._json(client.get(f"/status/{prediction_id}"))
            status = data.get("status")
            if status == "completed":
                outputs = data.get("output") or []
                if not outputs:
                    raise FashnError("FASHN finished without returning an image.")
                return outputs[0]
            if status not in _PENDING_STATUSES:
                raise FashnError(f"Clothes swap failed: {self._error_message(data)}")
            time.sleep(2)
        raise FashnError("Clothes swap timed out waiting for FASHN.")

    @staticmethod
    def _json(response: httpx.Response) -> dict:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise FashnError(f"Unexpected FASHN response ({response.status_code}).")
        if response.status_code >= 400:
            raise FashnError(f"FASHN API error ({response.status_code}): {FashnClient._error_message(data)}")
        return data

    @staticmethod
    def _error_message(data: dict) -> str:
        error = data.get("error")
        if isinstance(error, dict):
            return error.get("message") or error.get("name") or "unknown error"
        return data.get("message") or error or "unknown error"


def _encode_image(path: Path) -> str:
    """Return a JPEG data URI, downscaled so uploads stay small."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache
def get_fashn_client() -> FashnClient:
    return FashnClient()
