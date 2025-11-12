"""Face detection, swapping & background removal using InsightFace + rembg."""
from __future__ import annotations

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from loguru import logger
from pathlib import Path
from PIL import Image
from rembg import remove
from typing import Optional


class SegmentationService:
    """Combines InsightFace face detection/swapping with rembg background removal."""

    def __init__(self) -> None:
        logger.info("Initializing InsightFace face analysis...")
        self.face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("Loading InsightFace inswapper model...")
        model_path = Path.home() / ".insightface" / "models" / "inswapper" / "inswapper_128.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Inswapper model not found at {model_path}. Please download it first.")
        self.face_swapper = get_model(str(model_path), providers=["CPUExecutionProvider"])
        logger.info("InsightFace ready with face swapping.")

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in the image and return bounding boxes + landmarks.
        
        Args:
            image: BGR numpy array from cv2.imread
            
        Returns:
            List of dicts with 'bbox', 'kps' (5-point landmarks), 'det_score'
        """
        faces = self.face_app.get(image)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
            })
        return results

    def align_face(self, image: np.ndarray, landmarks: np.ndarray, output_size: int = 512) -> np.ndarray:
        """
        Normalize/align face using 5-point landmarks.
        
        Args:
            image: BGR numpy array
            landmarks: (5, 2) array of facial keypoints
            output_size: Target square dimensions
            
        Returns:
            Aligned face crop as BGR numpy array
        """
        from insightface.utils import face_align
        aligned = face_align.norm_crop(image, landmarks, image_size=output_size)
        return aligned

    def remove_background(self, image_path: Path, output_path: Path) -> Path:
        """
        Remove background from portrait using rembg (U²-Net based segmentation).
        
        Args:
            image_path: Input image file
            output_path: Where to save the background-removed PNG
            
        Returns:
            Path to the output file
        """
        logger.info("Removing background from %s", image_path)
        with Image.open(image_path) as img:
            output = remove(img, post_process_mask=True)
            output.save(output_path)
        logger.info("Background removed, saved to %s", output_path)
        return output_path

    def preprocess_portrait(
        self,
        image_path: Path,
        output_dir: Path,
        align_face: bool = True,
        remove_bg: bool = True,
    ) -> dict:
        """
        Full preprocessing pipeline: face detection, alignment, background removal.
        
        Args:
            image_path: Input portrait
            output_dir: Directory to save intermediate results
            align_face: Whether to normalize face pose
            remove_bg: Whether to segment person from background
            
        Returns:
            Dict with 'aligned_path', 'segmented_path', 'faces' metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 1. Detect faces
        faces = self.face_app.get(img_bgr)
        logger.info("Detected %d face(s) in %s", len(faces), image_path.name)
        
        result = {
            "faces": [],
            "aligned_path": None,
            "segmented_path": None,
        }
        
        if faces:
            # Use the largest face
            primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            result["faces"].append({
                "bbox": primary_face.bbox.tolist(),
                "kps": primary_face.kps.tolist(),
                "det_score": float(primary_face.det_score),
            })
            
            # 2. Align face if requested
            if align_face:
                aligned = self.align_face(img_bgr, primary_face.kps, output_size=512)
                aligned_path = output_dir / f"{image_path.stem}_aligned.png"
                cv2.imwrite(str(aligned_path), aligned)
                result["aligned_path"] = str(aligned_path)
                logger.info("Aligned face saved to %s", aligned_path)
        
        # 3. Remove background if requested
        if remove_bg:
            segmented_path = output_dir / f"{image_path.stem}_nobg.png"
            self.remove_background(image_path, segmented_path)
            result["segmented_path"] = str(segmented_path)
        
        return result

    def swap_face(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
    ) -> np.ndarray:
        """
        Swap face from source image onto target image.
        
        Args:
            source_image: BGR numpy array containing the face to extract
            target_image: BGR numpy array to receive the swapped face
            
        Returns:
            Result BGR image with swapped face
        """
        # Detect faces in both images
        source_faces = self.face_app.get(source_image)
        target_faces = self.face_app.get(target_image)
        
        if not source_faces:
            logger.warning("No face detected in source image")
            return target_image
        
        if not target_faces:
            logger.warning("No face detected in target image")
            return target_image
        
        # Use the largest face from each image
        source_face = max(source_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        target_face = max(target_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Perform face swap
        result = self.face_swapper.get(target_image, target_face, source_face, paste_back=True)
        
        return result


# Global singleton
_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """Lazy-load the segmentation service singleton."""
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service

