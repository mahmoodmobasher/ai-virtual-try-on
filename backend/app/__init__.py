"""Application package for the AI Virtual Try-On backend."""

import os

# Force every ML dependency to stay fully local: no cache-metadata pings,
# version checks, or model downloads. Must run before transformers/diffusers/
# huggingface_hub are imported anywhere, which is why it lives at the very
# top of this package's __init__ (every submodule import runs this first).
# A missing/incomplete local model then fails loudly instead of silently
# reaching out to the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from .main import create_app

__all__ = ["create_app"]

