"""Garment warping service - alternative to ClothFlow for warping garments to body pose."""
from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional, Tuple


class GarmentWarpingService:
    """
    Warps garment images to match target body pose.
    This is a simplified alternative to ClothFlow using OpenCV geometric transforms.
    """

    def __init__(self) -> None:
        logger.info("Initializing Garment Warping Service...")

    def warp_garment_to_pose(
        self,
        garment_image: np.ndarray,
        garment_mask: np.ndarray,
        target_image: np.ndarray,
        target_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp garment to match target body pose.
        
        Args:
            garment_image: Source garment image (BGR)
            garment_mask: Garment segmentation mask
            target_image: Target body image (BGR)
            target_mask: Target body segmentation mask
            
        Returns:
            Tuple of (warped_garment, warped_mask)
        """
        logger.info("Warping garment to match target pose...")
        
        # Step 1: Get bounding boxes for garment and target body
        garment_bbox = self._get_tight_bbox(garment_mask)
        target_bbox = self._get_tight_bbox(target_mask)
        
        if garment_bbox is None or target_bbox is None:
            logger.warning("Could not find bounding boxes, using original images")
            return garment_image, garment_mask
        
        g_x, g_y, g_w, g_h = garment_bbox
        t_x, t_y, t_w, t_h = target_bbox
        
        # Step 2: Calculate transformation to align garment with target
        # Scale factor to match target size
        scale_w = t_w / g_w
        scale_h = t_h / g_h
        scale = min(scale_w, scale_h) * 0.9  # Slightly smaller for better fit
        
        # Create transformation matrix
        # 1. Translate garment bbox to origin
        # 2. Scale to match target size
        # 3. Translate to target position
        
        # Center of garment
        g_center_x = g_x + g_w / 2
        g_center_y = g_y + g_h / 2
        
        # Center of target
        t_center_x = t_x + t_w / 2
        t_center_y = t_y + t_h / 2
        
        # Build affine transformation matrix
        M = np.array([
            [scale, 0, t_center_x - scale * g_center_x],
            [0, scale, t_center_y - scale * g_center_y]
        ], dtype=np.float32)
        
        # Apply transformation
        h, w = target_image.shape[:2]
        warped_garment = cv2.warpAffine(garment_image, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(garment_mask, M, (w, h), flags=cv2.INTER_LINEAR)
        
        logger.info("Garment warping complete")
        return warped_garment, warped_mask

    def _get_tight_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get tight bounding box around mask."""
        if mask.max() == 0:
            return None
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)

    def extract_garment_features(
        self,
        garment_image: np.ndarray,
        garment_mask: np.ndarray
    ) -> dict:
        """
        Extract garment features for better warping.
        
        Returns dict with:
        - bbox: Bounding box
        - contours: Garment contours
        - aspect_ratio: Width/height ratio
        """
        bbox = self._get_tight_bbox(garment_mask)
        
        if bbox is None:
            return {}
        
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Find contours for shape understanding
        contours, _ = cv2.findContours(garment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            "bbox": bbox,
            "contours": contours,
            "aspect_ratio": aspect_ratio,
            "width": w,
            "height": h
        }

    def apply_thin_plate_spline_warp(
        self,
        garment_image: np.ndarray,
        source_points: np.ndarray,
        target_points: np.ndarray
    ) -> np.ndarray:
        """
        Apply Thin-Plate Spline warping for more flexible deformation.
        This is a more advanced warping technique than simple affine transforms.
        
        Args:
            garment_image: Source garment
            source_points: Key points on garment
            target_points: Corresponding points on target
            
        Returns:
            Warped garment
        """
        # Placeholder for TPS implementation
        # For now, return original image
        logger.warning("TPS warping not yet implemented, using affine transform")
        return garment_image


# Global singleton
_garment_warping_service: Optional[GarmentWarpingService] = None


def get_garment_warping_service() -> GarmentWarpingService:
    """Lazy-load the garment warping service singleton."""
    global _garment_warping_service
    if _garment_warping_service is None:
        _garment_warping_service = GarmentWarpingService()
    return _garment_warping_service

