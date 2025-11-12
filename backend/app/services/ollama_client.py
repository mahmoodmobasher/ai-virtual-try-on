"""Ollama LLM client for local model inference."""
from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings


class OllamaClient:
    """Client for Ollama local LLM API."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = self.settings.llm_model

    @retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3))
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Chat with Ollama model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Convert messages to Ollama format
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
            return data.get("response", "").strip()
        except httpx.HTTPError as exc:
            logger.exception("Ollama API error: %s", exc)
            raise RuntimeError(f"Ollama API error: {exc}") from exc

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert ChatGPT-style messages to a single prompt."""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    async def generate_outfit_description(
        self,
        model_description: str,
        garment_description: str
    ) -> str:
        """Generate a fashion description for the try-on result."""
        messages = [
            {
                "role": "system",
                "content": "You are a professional fashion stylist. Provide concise, stylish descriptions."
            },
            {
                "role": "user",
                "content": f"Describe this outfit combination in 2-3 sentences:\n"
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
        """Get list of available Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as exc:
            logger.exception("Failed to fetch Ollama models: %s", exc)
            return []


# Global singleton
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Lazy-load the Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client

