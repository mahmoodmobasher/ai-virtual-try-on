from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services.llm_client import LLMClient
from ..services.ollama_client import get_ollama_client
from ..services.lm_studio_client import get_lm_studio_client
from ..core.config import get_settings

router = APIRouter(prefix="/assistant", tags=["assistant"])
settings = get_settings()

# Select LLM client based on configuration
if settings.llm_provider == "lm_studio":
    llm_client = get_lm_studio_client()
else:
    llm_client = get_ollama_client()

# Legacy client for backwards compatibility
legacy_client = LLMClient()


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    temperature: float | None = 0.4


class SuggestRequest(BaseModel):
    garment_name: str
    audience: str | None = None
    context: str | None = None


class OutfitDescriptionRequest(BaseModel):
    model_description: str = "Fashion model"
    garment_description: str = "Elegant outfit"


@router.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    try:
        content = await client.chat(request.messages, temperature=request.temperature or 0.4)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"response": content}


@router.post("/suggest")
async def suggest(request: SuggestRequest) -> dict[str, str]:
    system_prompt = (
        "You are a fashion stylist assisting users with virtual try-on moodboards for a premium DTC brand. "
        "Respond concisely with actionable advice."
    )
    user_prompt = (
        f"Create a styling suggestion for the garment '{request.garment_name}'. "
        f"Target audience: {request.audience or 'general consumer'}. "
        f"Context: {request.context or 'digital campaign'}."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        # Use configured LLM client (LM Studio or Ollama)
        content = await llm_client.chat(messages)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"suggestion": content}


@router.post("/describe-outfit")
async def describe_outfit(request: OutfitDescriptionRequest) -> dict[str, str]:
    """Generate a fashion description for the try-on result using local LLM."""
    try:
        description = await llm_client.generate_outfit_description(
            model_description=request.model_description,
            garment_description=request.garment_description
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"description": description}


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List available LLM models from configured provider."""
    try:
        models = await llm_client.get_available_models()
        provider = settings.llm_provider
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"models": models, "provider": provider, "active_model": settings.llm_model}

