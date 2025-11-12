"""LM Studio client for local LLM inference."""
from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings


class LMStudioClient:
    """Client for LM Studio local LLM server (OpenAI-compatible API)."""

    def __init__(self, base_url: str = "http://localhost:1234") -> None:
        self.settings = get_settings()
        self.base_url = base_url
        self.model = self.settings.llm_model or "gpt-oss-20b"

    @retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Chat with LM Studio model using OpenAI-compatible API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
            # Extract response from OpenAI-compatible format
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            logger.exception("LM Studio API error: %s", exc)
            raise RuntimeError(f"LM Studio API error: {exc}") from exc

    async def generate_outfit_description(
        self,
        model_description: str,
        garment_description: str
    ) -> str:
        """Generate a fashion description for the try-on result."""
        messages = [
            {
                "role": "system",
                "content": "You are a professional fashion stylist. Provide concise, stylish descriptions in 2-3 sentences."
            },
            {
                "role": "user",
                "content": f"Describe this outfit combination:\n"
                          f"Model: {model_description}\n"
                          f"Garment: {garment_description}"
            }
        ]
        
        return await self.chat(messages, temperature=0.8, max_tokens=200)

    async def suggest_styling_tips(self, outfit_type: str) -> str:
        """Get styling suggestions for an outfit."""
        messages = [
            {
                "role": "system",
                "content": "You are a fashion consultant. Give practical styling advice."
            },
            {
                "role": "user",
                "content": f"Give 3 quick styling tips for: {outfit_type}"
            }
        ]
        
        return await self.chat(messages, temperature=0.7, max_tokens=300)

    async def get_available_models(self) -> list[str]:
        """Get list of available LM Studio models."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                
            return [model["id"] for model in data.get("data", [])]
        except httpx.HTTPError as exc:
            logger.exception("Failed to fetch LM Studio models: %s", exc)
            return []


# Global singleton
_lm_studio_client: Optional[LMStudioClient] = None


def get_lm_studio_client() -> LMStudioClient:
    """Lazy-load the LM Studio client singleton."""
    global _lm_studio_client
    if _lm_studio_client is None:
        _lm_studio_client = LMStudioClient()
    return _lm_studio_client

