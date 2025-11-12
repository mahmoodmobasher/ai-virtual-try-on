from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings


class LLMClient:
    """Thin client for LLM Studio HTTP API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.llm_studio_base_url
        self.api_key = self.settings.llm_studio_api_key

    @retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
    async def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.4) -> str:
        if not self.base_url:
            raise RuntimeError("LLM Studio base URL not configured.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.exception("Unexpected LLM Studio response: %s", data)
            raise RuntimeError("LLM Studio response parsing error") from exc
        return content.strip()

