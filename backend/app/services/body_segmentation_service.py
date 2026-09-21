"""Body segmentation using Mediapipe Selfie Segmentation."""
from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from PIL import Image
from typing import Optional

try:
    import mediapipe as mp
except ImportError:  # Optional ML dependency
    mp = None


class BodySegmentationService:
    """Body segmentation using Mediapipe for virtual try-on."""

    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError(
                "Body segmentation requires the optional ML dependencies. "
                "Install them with `make setup-ml` using Python 3.11."
            )
        logger.info("Initializing Mediapipe Selfie Segmentation...")
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        # Model selection: 0 for general, 1 for landscape (higher accuracy)
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        logger.info("Mediapipe body segmentation ready.")

    def segment_person(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        blur_strength: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment person from background.
        
        Args:
            image: Input BGR image
            threshold: Segmentation threshold (0-1)
            blur_strength: Kernel size for mask smoothing
            
        Returns:
            Tuple of (segmented_image, mask)
        """
        # Convert BGR to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.segmenter.process(image_rgb)
        
        if results.segmentation_mask is None:
            logger.warning("No segmentation mask generated")
            return image, np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Create binary mask
        mask = (results.segmentation_mask > threshold).astype(np.uint8) * 255
        
        # Smooth the mask edges
        if blur_strength > 0:
            mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
        
        # Apply mask to image
        mask_3channel = cv2.merge([mask, mask, mask])
        segmented = cv2.bitwise_and(image, mask_3channel)
        
        return segmented, mask

    def extract_body_region(
        self,
        image: np.ndarray,
        region: str = "upper"
    ) -> np.ndarray:
        """
        Extract specific body region (upper/lower body).
        
        Args:
            image: Input BGR image
            region: "upper" for upper body, "lower" for lower body
            
        Returns:
            Masked image showing only the requested region
        """
        height, width = image.shape[:2]
        segmented, mask = self.segment_person(image)
        
        # Create region mask
        region_mask = np.zeros_like(mask)
        
        if region == "upper":
            # Top 60% of the image
            region_mask[:int(height * 0.6), :] = 255
        elif region == "lower":
            # Bottom 60% of the image
            region_mask[int(height * 0.4):, :] = 255
        else:
            # Full body
            region_mask[:, :] = 255
        
        # Combine with segmentation mask
        combined_mask = cv2.bitwise_and(mask, region_mask)
        mask_3channel = cv2.merge([combined_mask, combined_mask, combined_mask])
        
        return cv2.bitwise_and(image, mask_3channel)

    def replace_background(
        self,
        image: np.ndarray,
        background: Optional[np.ndarray] = None,
        bg_color: tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Replace background with solid color or image.
        
        Args:
            image: Input BGR image
            background: Optional background image (same size as input)
            bg_color: Background color if no background image provided
            
        Returns:
            Image with replaced background
        """
        segmented, mask = self.segment_person(image)
        
        # Create background
        if background is not None and background.shape[:2] == image.shape[:2]:
            bg = background
        else:
            bg = np.full_like(image, bg_color, dtype=np.uint8)
        
        # Invert mask for background
        mask_inv = cv2.bitwise_not(mask)
        mask_3channel = cv2.merge([mask, mask, mask])
        mask_inv_3channel = cv2.merge([mask_inv, mask_inv, mask_inv])
        
        # Combine foreground and background
        fg = cv2.bitwise_and(image, mask_3channel)
        bg = cv2.bitwise_and(bg, mask_inv_3channel)
        result = cv2.add(fg, bg)
        
        return result

    def get_body_bounding_box(self, image: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """
        Get bounding box of segmented person.
        
        Returns:
            Tuple of (x, y, width, height) or None if no person detected
        """
        _, mask = self.segment_person(image)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (main person)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)

    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'segmenter'):
            self.segmenter.close()


# Global singleton
_body_segmentation_service: Optional[BodySegmentationService] = None


def get_body_segmentation_service() -> BodySegmentationService:
    """Lazy-load the body segmentation service singleton."""
    global _body_segmentation_service
    if _body_segmentation_service is None:
        _body_segmentation_service = BodySegmentationService()
    return _body_segmentation_service
