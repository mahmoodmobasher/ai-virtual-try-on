"""Body segmentation using Detectron2 instance segmentation."""
from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional

try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
except ImportError:  # Optional ML dependency
    model_zoo = None
    get_cfg = None
    DefaultPredictor = None


class Detectron2SegmentationService:
    """Body/person segmentation using Detectron2."""

    def __init__(self, model_name: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") -> None:
        if model_zoo is None or get_cfg is None or DefaultPredictor is None:
            raise RuntimeError("Detectron2 is not installed; another segmentation fallback will be used.")
        logger.info("Initializing Detectron2 for person segmentation...")
        
        # Configure Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_name))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        cfg.MODEL.DEVICE = "cpu"  # Use CPU on macOS
        
        # Create predictor
        self.predictor = DefaultPredictor(cfg)
        self.person_class_id = 0  # COCO class ID for "person"
        
        logger.info("Detectron2 ready with model: %s", model_name)

    def segment_person(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        return_largest: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment person(s) from the image using instance segmentation.
        
        Args:
            image: Input BGR image
            threshold: Confidence threshold for detection
            return_largest: If True, return only the largest person detected
            
        Returns:
            Tuple of (segmented_image, mask)
        """
        # Run detection
        outputs = self.predictor(image)
        
        # Get predictions
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        pred_masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        
        # Filter for person class and confidence threshold
        person_indices = np.where(
            (pred_classes == self.person_class_id) & (scores >= threshold)
        )[0]
        
        if len(person_indices) == 0:
            logger.warning("No person detected in image")
            return image, np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get person masks
        person_masks = pred_masks[person_indices]
        
        if return_largest:
            # Find the largest mask (most pixels)
            mask_areas = [mask.sum() for mask in person_masks]
            largest_idx = np.argmax(mask_areas)
            combined_mask = person_masks[largest_idx].astype(np.uint8) * 255
        else:
            # Combine all person masks
            combined_mask = np.any(person_masks, axis=0).astype(np.uint8) * 255
        
        # Apply mask to image
        mask_3channel = cv2.merge([combined_mask, combined_mask, combined_mask])
        segmented = cv2.bitwise_and(image, mask_3channel)
        
        return segmented, combined_mask

    def extract_body_region(
        self,
        image: np.ndarray,
        region: str = "upper"
    ) -> np.ndarray:
        """
        Extract specific body region using segmentation + spatial filtering.
        
        Args:
            image: Input BGR image
            region: "upper" for upper body, "lower" for lower body, "full" for whole person
            
        Returns:
            Masked image showing only the requested region
        """
        height, width = image.shape[:2]
        segmented, mask = self.segment_person(image)
        
        # Create spatial region mask
        region_mask = np.zeros_like(mask)
        
        if region == "upper":
            # Top 60% of the image
            region_mask[:int(height * 0.6), :] = 255
        elif region == "lower":
            # Bottom 60% of the image (overlaps with upper for better continuity)
            region_mask[int(height * 0.4):, :] = 255
        else:
            # Full body
            region_mask[:, :] = 255
        
        # Combine person mask with region mask
        combined_mask = cv2.bitwise_and(mask, region_mask)
        mask_3channel = cv2.merge([combined_mask, combined_mask, combined_mask])
        
        return cv2.bitwise_and(image, mask_3channel)

    def get_person_bounding_boxes(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> list[tuple[int, int, int, int]]:
        """
        Get bounding boxes for all detected persons.
        
        Returns:
            List of (x, y, width, height) tuples
        """
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        pred_classes = instances.pred_classes.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        
        # Filter for person class
        person_indices = np.where(
            (pred_classes == self.person_class_id) & (scores >= threshold)
        )[0]
        
        bboxes = []
        for idx in person_indices:
            x1, y1, x2, y2 = pred_boxes[idx]
            bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return bboxes

    def replace_background(
        self,
        image: np.ndarray,
        background: Optional[np.ndarray] = None,
        bg_color: tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Replace background using Detectron2 segmentation.
        
        Args:
            image: Input BGR image
            background: Optional background image
            bg_color: Background color if no image provided
            
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
        bg_masked = cv2.bitwise_and(bg, mask_inv_3channel)
        result = cv2.add(fg, bg_masked)
        
        return result


# Global singleton
_detectron2_service: Optional[Detectron2SegmentationService] = None


def get_detectron2_service() -> Detectron2SegmentationService:
    """Lazy-load the Detectron2 service singleton."""
    global _detectron2_service
    if _detectron2_service is None:
        _detectron2_service = Detectron2SegmentationService()
    return _detectron2_service
