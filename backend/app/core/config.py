from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "AI Virtual Try-On"
    environment: Literal["local", "dev", "prod"] = "local"
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3003", "http://127.0.0.1:3003"]
    storage_dir: Path = Path(__file__).resolve().parent.parent.parent / "storage"

    database_url: str = "sqlite+aiosqlite:///./tryon.db"
    redis_url: str = "redis://localhost:6379/0"

    # LLM Provider Selection
    llm_provider: Literal["lm_studio", "ollama"] = "lm_studio"

    # LM Studio configuration (OpenAI-compatible API)
    lm_studio_base_url: str = "http://localhost:1234"

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"

    # Model name (depends on provider)
    llm_model: str = "gpt-oss-20b"  # For LM Studio: gpt-oss-20b, For Ollama: mistral:latest

    # Segmentation service preference
    use_detectron2: bool = True  # Use Detectron2 for body segmentation (fallback to Mediapipe if fails)

    # Clothes swap engine: "local" runs CatVTON on this machine, "fashn" uses the hosted FASHN API
    garment_swap_engine: Literal["local", "fashn"] = "local"

    # When true (default), refuse to use any hosted/remote inference engine (e.g. the
    # FASHN API) even if garment_swap_engine is misconfigured to "fashn" — user images
    # must never leave this machine. Set to false only if you deliberately want the
    # hosted API and accept that images will be uploaded to a third party.
    enforce_local_only: bool = True
    local_tryon_width: int = 768
    local_tryon_height: int = 1024
    local_tryon_steps: int = 30
    local_tryon_guidance_scale: float = 2.5

    # FASHN hosted try-on API (clothes swap)
    fashn_api_key: Optional[str] = None
    fashn_api_base: str = "https://api.fashn.ai/v1"
    fashn_model: str = "tryon-v1.6"  # or "tryon-max"
    fashn_mode: str = "quality"  # tryon-v1.6: performance|balanced|quality; tryon-max: fast|balanced|quality
    fashn_timeout_seconds: int = 180

    # CatVTON-FLUX configuration
    catvton_width: int = 576
    catvton_height: int = 768
    catvton_num_steps: int = 30
    catvton_guidance_scale: float = 30.0
    catvton_seed: int = 1337

    # Stable Video Diffusion configuration (experimental)
    svd_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    svd_default_frames: int = 14
    svd_default_fps: int = 6
    svd_inference_steps: int = 25
    svd_guidance_scale: float = 1.2
    svd_motion_bucket_id: int = 127
    svd_noise_aug_strength: float = 0.02
    svd_image_width: int = 576
    svd_image_height: int = 576

    # Legacy LLM Studio config (deprecated)
    llm_studio_base_url: Optional[str] = None
    llm_studio_api_key: Optional[str] = None

    tryon_model: Literal["catvton-flux"] = "catvton-flux"
    segmentation_model: Literal["birefnet", "sam-hq", "mediapipe"] = "birefnet"
    pose_model: Literal["openpifpaf", "mediapipe"] = "openpifpaf"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="TRYON_",
        extra="ignore",
    )

    @property
    def uploads_dir(self) -> Path:
        return Path(self.storage_dir) / "uploads"

    @property
    def results_dir(self) -> Path:
        return Path(self.storage_dir) / "results"

    @property
    def catalog_dir(self) -> Path:
        return Path(self.storage_dir) / "catalog"

    @property
    def video_results_dir(self) -> Path:
        return Path(self.storage_dir) / "video-results"


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached settings instance."""

    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.catalog_dir.mkdir(parents=True, exist_ok=True)
    settings.video_results_dir.mkdir(parents=True, exist_ok=True)
    return settings
