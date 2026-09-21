"""Clothing masks for CatVTON.

Mirrors CatVTON's AutoMasker, but swaps DensePose + SCHP (which need a CUDA-built
detectron2) for a SegFormer human parser (ATR labels) and MediaPipe hand landmarks,
both of which run on Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

ClothPart = Literal["upper", "lower", "overall"]

PARSER_REPO = "mattmdjaga/segformer_b2_clothes"

ATR = {
    "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3, "Upper-clothes": 4, "Skirt": 5,
    "Pants": 6, "Dress": 7, "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
    "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15, "Bag": 16, "Scarf": 17,
}
ARMS = ["Left-arm", "Right-arm"]
LEGS = ["Left-leg", "Right-leg"]
ACCESSORIES = ["Hat", "Sunglasses", "Left-shoe", "Right-shoe", "Bag", "Scarf"]

PROTECT_BODY = {"upper": LEGS, "lower": [*ARMS, "Face"], "overall": []}
PROTECT_CLOTHES = {"upper": ["Skirt", "Pants"], "lower": ["Upper-clothes"], "overall": []}
MASK_CLOTHES = {
    "upper": ["Upper-clothes", "Dress"],
    "lower": ["Pants", "Skirt", "Dress"],
    "overall": ["Upper-clothes", "Dress", "Pants", "Skirt"],
}
# Body regions the new garment may cover (stands in for CatVTON's DensePose regions).
MASK_BODY_REGION = {
    "upper": ["Upper-clothes", "Dress", "Belt", *ARMS],
    "lower": ["Pants", "Skirt", "Belt", *LEGS],
    "overall": ["Upper-clothes", "Dress", "Pants", "Skirt", "Belt", *ARMS, *LEGS],
}


class ClothMasker:
    def __init__(self, device: str, cache_dir: Path) -> None:
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(
            PARSER_REPO, cache_dir=cache_dir, local_files_only=True
        )
        self.parser = SegformerForSemanticSegmentation.from_pretrained(
            PARSER_REPO, cache_dir=cache_dir, local_files_only=True
        )
        self.parser.to(device).eval()

    @torch.no_grad()
    def parse(self, image: Image.Image) -> np.ndarray:
        """Per-pixel ATR label map at the image's resolution."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        logits = self.parser(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    def detect_part(self, garment: Image.Image) -> ClothPart:
        """Guess which body region a garment photo covers."""
        parse = self.parse(garment)
        upper = np.count_nonzero(parse == ATR["Upper-clothes"])
        lower = np.count_nonzero(np.isin(parse, [ATR["Pants"], ATR["Skirt"]]))
        dress = np.count_nonzero(parse == ATR["Dress"])
        total = upper + lower + dress
        if total == 0:
            return "upper"
        if dress / total > 0.4 or (upper / total > 0.2 and lower / total > 0.2):
            return "overall"
        return "lower" if lower > upper else "upper"

    def mask(self, image: Image.Image, part: ClothPart) -> Image.Image:
        parse = self.parse(image)
        w, h = image.size

        dilate = np.ones((_odd(max(w, h) // 250),) * 2, np.uint8)
        blur_size = _odd(max(w, h) // 25)

        # Strong protection: face and hands are never repainted.
        strong_protect = _parts(parse, ["Face"]) | (_hand_mask(image, dilate) & _parts(parse, [*ARMS, *LEGS]))
        weak_protect = (
            strong_protect
            | _parts(parse, PROTECT_BODY[part] + PROTECT_CLOTHES[part] + ["Hair"] + ACCESSORIES)
        )
        strong_mask = _parts(parse, MASK_CLOTHES[part])
        background = _parts(parse, ["Background"])

        body_region = _parts(parse, MASK_BODY_REGION[part])
        body_region = cv2.resize(body_region, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        body_region = cv2.dilate(body_region, dilate, iterations=2)
        body_region = cv2.resize(body_region, (w, h), interpolation=cv2.INTER_NEAREST)

        area = ((1 - weak_protect) & (1 - background)) | body_region
        area = _hull(area) & (1 - weak_protect)
        area = cv2.GaussianBlur(area * 255, (blur_size, blur_size), 0)
        area = (area >= 25).astype(np.uint8)
        area = (area | strong_mask) & (1 - strong_protect)
        area = cv2.dilate(area, dilate, iterations=1)
        return Image.fromarray(area * 255)


def _parts(parse: np.ndarray, names: list[str]) -> np.ndarray:
    return np.isin(parse, [ATR[n] for n in names]).astype(np.uint8)


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _hull(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(mask)
    for contour in contours:
        cv2.fillPoly(hull, [cv2.convexHull(contour)], 1)
    return hull


def _hand_mask(image: Image.Image, dilate: np.ndarray) -> np.ndarray:
    w, h = image.size
    mask = np.zeros((h, w), np.uint8)
    if mp is None:
        return mask
    try:
        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
            result = hands.process(np.array(image.convert("RGB")))
    except Exception as exc:  # e.g. no OpenGL context in headless environments
        logger.warning("Hand detection unavailable; hands won't be protected from repainting: {}", exc)
        return mask
    for landmarks in result.multi_hand_landmarks or []:
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark], dtype=np.int32)
        cv2.fillPoly(mask, [cv2.convexHull(points)], 1)
    return cv2.dilate(mask, dilate, iterations=3)
