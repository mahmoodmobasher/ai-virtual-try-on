"""CatVTON inference, ported from https://github.com/Zheng-Chong/CatVTON.

CatVTON fine-tunes only the self-attention layers of Stable Diffusion 1.5 inpainting.
The person image and the garment image are concatenated side by side in latent space,
and the masked clothing region of the person is inpainted conditioned on the garment.

Model weights (zhengchong/CatVTON) are licensed CC BY-NC-SA 4.0: non-commercial use only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import load_file

BASE_MODEL_REPO = "booksforcharlie/stable-diffusion-inpainting"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
ATTN_REPO = "zhengchong/CatVTON"
ATTN_SUBFOLDER = "mix-48k-1024"


class SkipAttnProcessor(torch.nn.Module):
    """Replaces cross-attention: CatVTON has no text conditioning."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        return hidden_states


class CatVTONPipeline:
    def __init__(self, device: str, cache_dir: Path) -> None:
        self.device = device
        # fp16 UNet for speed; the VAE stays fp32 because it produces NaNs in fp16.
        self.unet_dtype = torch.float32 if device == "cpu" else torch.float16
        self.vae_dtype = torch.float32

        # local_files_only makes intent explicit at the call site: this must
        # only ever read the already-downloaded local cache, never dial out.
        # (HF_HUB_OFFLINE=1, set in app/__init__.py, enforces this globally
        # too; this is belt-and-suspenders for anyone importing this module
        # directly without going through the app package.)
        base_path = snapshot_download(
            BASE_MODEL_REPO,
            cache_dir=cache_dir,
            allow_patterns=["scheduler/*", "unet/config.json", "unet/diffusion_pytorch_model.*"],
            local_files_only=True,
        )
        attn_path = snapshot_download(
            ATTN_REPO,
            cache_dir=cache_dir,
            allow_patterns=[f"{ATTN_SUBFOLDER}/attention/*"],
            local_files_only=True,
        )

        self.scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler", local_files_only=True)
        self.vae = AutoencoderKL.from_pretrained(VAE_REPO, cache_dir=cache_dir, local_files_only=True).to(
            device, dtype=self.vae_dtype
        )
        self.unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", local_files_only=True).to(
            device, dtype=self.unet_dtype
        )

        self.unet.set_attn_processor(
            {
                name: AttnProcessor2_0() if name.endswith("attn1.processor") else SkipAttnProcessor()
                for name in self.unet.attn_processors
            }
        )
        self._load_attention(os.path.join(attn_path, ATTN_SUBFOLDER, "attention", "model.safetensors"))

    def _load_attention(self, checkpoint: str) -> None:
        # The checkpoint indexes self-attention blocks as they appeared in upstream's
        # ModuleList, which held 8 modules per block (attn1 plus its submodules and
        # processor). Re-key by block order so each block gets its own weights.
        blocks = [module for name, module in self.unet.named_modules() if name.endswith("attn1")]
        state = load_file(checkpoint)
        indices = sorted({int(key.split(".", 1)[0]) for key in state})
        if len(indices) != len(blocks):
            raise RuntimeError(f"CatVTON checkpoint has {len(indices)} attention blocks, UNet has {len(blocks)}.")
        for block, index in zip(blocks, indices):
            prefix = f"{index}."
            weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            block.load_state_dict(weights, strict=True)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        condition_image: Image.Image,
        mask: Image.Image,
        *,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = 42,
    ) -> Image.Image:
        """All three images must already share the same size (a multiple of 8)."""
        concat_dim = -2  # stack person above garment

        image_t = _to_tensor(image).to(self.device, dtype=self.vae_dtype)
        condition_t = _to_tensor(condition_image).to(self.device, dtype=self.vae_dtype)
        mask_t = torch.from_numpy((np.array(mask.convert("L")) >= 128).astype(np.float32))[None, None]
        mask_t = mask_t.to(self.device, dtype=self.vae_dtype)

        masked_latent = self._encode(image_t * (mask_t < 0.5))
        condition_latent = self._encode(condition_t)
        mask_latent = torch.nn.functional.interpolate(mask_t, size=masked_latent.shape[-2:], mode="nearest")

        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = randn_tensor(
            masked_latent_concat.shape, generator=generator, device=torch.device(self.device), dtype=self.unet_dtype
        )
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            unconditional = torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim)
            masked_latent_concat = torch.cat([unconditional, masked_latent_concat])
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)
        masked_latent_concat = masked_latent_concat.to(self.unet_dtype)
        mask_latent_concat = mask_latent_concat.to(self.unet_dtype)

        for t in self.scheduler.timesteps:
            model_input = torch.cat([latents] * 2) if do_cfg else latents
            model_input = self.scheduler.scale_model_input(model_input, t)
            model_input = torch.cat([model_input, mask_latent_concat, masked_latent_concat], dim=1)
            noise_pred = self.unet(model_input, t, encoder_hidden_states=None, return_dict=False)[0]
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, eta=1.0, generator=generator).prev_sample

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = latents.to(self.vae_dtype) / self.vae.config.scaling_factor
        decoded = (self.vae.decode(latents).sample / 2 + 0.5).clamp(0, 1)
        array = (decoded[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels).latent_dist.sample() * self.vae.config.scaling_factor


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None]
