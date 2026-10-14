"""Face detection, swapping & background removal using InsightFace + rembg."""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from PIL import Image
from typing import Any, Optional, Sequence

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    from insightface.utils import face_align
except ImportError:  # Optional ML dependency
    FaceAnalysis = None
    get_model = None
    face_align = None

try:
    from rembg import remove
except ImportError:  # Optional ML dependency
    remove = None

try:
    import onnxruntime
except ImportError:  # Optional ML dependency
    onnxruntime = None


def _resolve_execution_providers() -> list[str]:
    """Pick the fastest onnxruntime execution provider actually available.

    Falls back to CPU (last entry) whenever a preferred provider isn't
    present in this onnxruntime build/install, so this never hard-fails.
    """
    if onnxruntime is None:
        return ["CPUExecutionProvider"]

    available = set(onnxruntime.get_available_providers())
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider"]
    providers = [p for p in preferred if p in available]
    providers.append("CPUExecutionProvider")

    if len(providers) > 1:
        logger.info(f"onnxruntime execution providers: {providers}")
    else:
        logger.warning(
            "No GPU/CoreML onnxruntime execution provider available; "
            "falling back to CPUExecutionProvider (this will be slow)."
        )
    return providers


# Cosine-similarity floor for calling two embeddings the same person. ArcFace
# embeddings of the same identity typically score well above this even across
# pose/lighting changes, while different people sit near or below zero.
IDENTITY_MATCH_THRESHOLD = 0.25


# inswapper works on a 128px face crop, so a face that is larger than that in
# the photo comes back soft. For stills the crop is taken at up to this many
# times that size and swapped as interleaved 128px tiles ("pixel boost"), which
# recovers real detail up to 512px without needing another model.
MAX_PIXEL_BOOST = 4


@dataclass
class FaceSwapSlot:
    """One "replace this person with this face" assignment, tracked across frames.

    `target_ref_embedding` identifies *which* person in the video to replace and
    is what makes the swap stick to that person across frames instead of the
    old "whichever face is biggest in this frame" behaviour. `source_face` is
    the replacement identity, detected once from the uploaded photo and reused
    for every frame.
    """

    target_ref_embedding: np.ndarray
    source_face: Any


class SegmentationService:
    """Combines InsightFace face detection/swapping with rembg background removal."""

    def __init__(self) -> None:
        if FaceAnalysis is None or get_model is None or remove is None:
            raise RuntimeError(
                "Face analysis requires the optional ML dependencies. "
                "Install them with `make setup-ml` using Python 3.11."
            )
        providers = _resolve_execution_providers()

        logger.info("Initializing InsightFace face analysis...")
        # Only detection (bbox + 5-point landmarks) and recognition (identity
        # embeddings) are used anywhere here; skipping the 2d/3d landmark and
        # gender-age models cuts per-frame work and startup load time.
        self.face_app = FaceAnalysis(providers=providers, allowed_modules=["detection", "recognition"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        logger.info("Loading InsightFace inswapper model...")
        model_path = Path.home() / ".insightface" / "models" / "inswapper" / "inswapper_128.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Inswapper model not found at {model_path}. Please download it first.")
        self.face_swapper = get_model(str(model_path), providers=providers)
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

    def detect_faces_with_embeddings(self, image: np.ndarray) -> list[dict]:
        """Detect faces and include each one's identity embedding.

        Backs the "scan the video and let the user choose a face" flow: the
        embedding returned here is what later identifies the chosen person
        frame-by-frame during the swap.
        """
        faces = self.face_app.get(image)
        return [
            {
                "bbox": face.bbox.tolist(),
                "kps": face.kps.tolist(),
                "det_score": float(face.det_score),
                "embedding": face.normed_embedding,
            }
            for face in faces
        ]

    def get_primary_face(self, image: np.ndarray) -> Optional[Any]:
        """Largest detected face in `image`, or None. Used for source photos."""
        faces = self.face_app.get(image)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    @staticmethod
    def crop_face(image: np.ndarray, bbox: Sequence[float], padding: float = 0.35) -> np.ndarray:
        """Crop a padded square-ish thumbnail around `bbox` for the face picker."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_y))
        cx2 = int(min(width, x2 + pad_x))
        cy2 = int(min(height, y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return image
        return image[cy1:cy2, cx1:cx2]

    @staticmethod
    def _best_identity_match(
        faces: Sequence[Any],
        ref_embedding: np.ndarray,
        exclude_ids: frozenset[int] = frozenset(),
    ) -> Optional[Any]:
        """Face in `faces` whose identity best matches `ref_embedding`.

        Both embeddings are L2-normalized, so cosine similarity is a dot
        product. Returns None when nothing clears IDENTITY_MATCH_THRESHOLD,
        which means "that person isn't in this frame" — better to leave the
        frame untouched than to swap the wrong face.
        """
        best: Optional[Any] = None
        best_score = IDENTITY_MATCH_THRESHOLD
        for face in faces:
            if id(face) in exclude_ids:
                continue
            embedding = face.normed_embedding
            if embedding is None:
                continue
            score = float(np.dot(embedding, ref_embedding))
            if score > best_score:
                best, best_score = face, score
        return best

    def _swap_aligned(self, aimg: np.ndarray, latent: np.ndarray) -> np.ndarray:
        """Run the inswapper network on one aligned, model-sized BGR face crop."""
        swapper = self.face_swapper
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / swapper.input_std,
            swapper.input_size,
            (swapper.input_mean, swapper.input_mean, swapper.input_mean),
            swapRB=True,
        )
        pred = swapper.session.run(
            swapper.output_names,
            {swapper.input_names[0]: blob, swapper.input_names[1]: latent},
        )[0]
        return np.clip(255 * pred.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[:, :, ::-1]

    def _swap_forward(
        self, frame: np.ndarray, target_kps: np.ndarray, source_face: Any, boost: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the inswapper network for one face; returns the swapped crop and its transform.

        This is the expensive half of a swap (~65% of per-face cost) and touches
        no shared state, so several faces can run through it concurrently.

        With `boost` > 1 the face is aligned at `boost` times the model's 128px
        input and split into boost² interleaved 128px tiles (every boost-th
        pixel, one tile per sub-pixel offset). Each tile is a complete face the
        network can swap, and re-interleaving the results gives a crop with
        `boost` times the resolution at boost² times the inference cost.
        """
        swapper = self.face_swapper
        aimg, matrix = face_align.norm_crop2(frame, target_kps, swapper.input_size[0] * boost)
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, swapper.emap)
        latent /= np.linalg.norm(latent)

        if boost == 1:
            return self._swap_aligned(aimg, latent), matrix

        bgr_fake = np.empty_like(aimg)
        for dy in range(boost):
            for dx in range(boost):
                tile = np.ascontiguousarray(aimg[dy::boost, dx::boost])
                bgr_fake[dy::boost, dx::boost] = self._swap_aligned(tile, latent)
        return bgr_fake, matrix

    def _pixel_boost_for(self, target_kps: np.ndarray) -> int:
        """Smallest boost whose crop covers the face at its native resolution."""
        model_size = self.face_swapper.input_size[0]
        matrix = face_align.estimate_norm(np.asarray(target_kps, dtype=np.float32), model_size)
        # The transform's scale is model pixels per image pixel, so this is how
        # many image pixels the model's crop spans.
        face_span = model_size / float(np.hypot(matrix[0, 0], matrix[1, 0]))
        # Small tolerance so a face barely over a multiple doesn't pay for the next one.
        return int(np.clip(np.ceil(face_span / model_size - 0.15), 1, MAX_PIXEL_BOOST))

    def swap_faces_image(
        self,
        image: np.ndarray,
        assignments: Sequence[tuple[np.ndarray, Any]],
        progress_cb: Optional[Any] = None,
    ) -> np.ndarray:
        """Swap chosen faces in a still image at the best quality inswapper allows.

        `assignments` pairs the 5-point landmarks of a face in `image` with the
        detected face replacing it. Unlike the per-frame video path this spends
        extra inference on pixel boost, sized to each face.
        """
        swapped = []
        for done, (target_kps, source_face) in enumerate(assignments, start=1):
            target_kps = np.asarray(target_kps, dtype=np.float32)
            boost = self._pixel_boost_for(target_kps)
            logger.info("Swapping face {}/{} at {}x pixel boost", done, len(assignments), boost)
            # Every crop is taken from the untouched image, then composited in
            # order, so overlapping faces blend deterministically.
            swapped.append(self._swap_forward(image, target_kps, source_face, boost=boost))
            if progress_cb:
                progress_cb(done, len(assignments))

        result = image
        for bgr_fake, matrix in swapped:
            result = self._paste_swapped(result, bgr_fake, matrix)
        return result

    @staticmethod
    def _paste_swapped(frame: np.ndarray, bgr_fake: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Warp a swapped face back onto the frame with a feathered mask.

        Mirrors insightface's own paste-back blending, minus its `fake_diff`
        pass: upstream computes that mask (a full-frame warp, a dilate and a
        blur) and then never uses it, since the line that would apply it is
        commented out. Dropping it is free and changes no pixels.
        """
        inverse = cv2.invertAffineTransform(matrix)
        height, width = frame.shape[:2]

        warped_face = cv2.warpAffine(bgr_fake, inverse, (width, height), borderValue=0.0)
        mask = cv2.warpAffine(
            np.full(bgr_fake.shape[:2], 255, dtype=np.float32), inverse, (width, height), borderValue=0.0
        )
        mask[mask > 20] = 255

        rows, cols = np.where(mask == 255)
        if rows.size == 0:
            return frame
        mask_size = int(np.sqrt((rows.max() - rows.min()) * (cols.max() - cols.min())))

        erode_k = max(mask_size // 10, 10)
        mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8), iterations=1)
        blur_k = max(mask_size // 20, 5)
        mask = cv2.GaussianBlur(mask, (2 * blur_k + 1, 2 * blur_k + 1), 0)
        mask = (mask / 255).reshape(height, width, 1)

        return (mask * warped_face + (1 - mask) * frame.astype(np.float32)).astype(np.uint8)

    def swap_faces_multi(
        self, frame: np.ndarray, slots: Sequence[FaceSwapSlot]
    ) -> tuple[np.ndarray, list[Optional[np.ndarray]]]:
        """Swap every slot's assigned person in `frame`, matching them by identity.

        One detection pass serves every slot, and each slot claims the face
        whose identity matches the person the user picked — so a slot follows
        that person across frames regardless of where they move or how large
        they appear. Faces nobody claimed are left untouched.

        Also returns each slot's 5-point landmarks in this frame (None where
        that person wasn't found), for overlays drawn after the swap.
        """
        slot_kps: list[Optional[np.ndarray]] = [None] * len(slots)
        detected = self.face_app.get(frame)
        if not detected:
            return frame, slot_kps

        claimed: set[int] = set()
        matches: list[tuple[FaceSwapSlot, Any]] = []
        for slot_index, slot in enumerate(slots):
            # Matched against `detected`, captured before any swapping, so
            # embeddings still describe the original people even after an
            # earlier slot has already altered the frame.
            match = self._best_identity_match(detected, slot.target_ref_embedding, frozenset(claimed))
            if match is None:
                # That person isn't in this frame (turned away, occluded, out
                # of shot) — leaving the frame alone beats swapping a stranger.
                continue
            claimed.add(id(match))
            matches.append((slot, match))
            slot_kps[slot_index] = match.kps.copy()

        if not matches:
            return frame, slot_kps

        # Every face's network pass runs first, then compositing happens in slot
        # order, so overlapping faces blend deterministically. This stays
        # single-threaded on purpose: callers processing a video run whole
        # frames concurrently, and adding a second layer of threads here only
        # contends for the same accelerator.
        swapped = [self._swap_forward(frame, match.kps, slot.source_face) for slot, match in matches]

        result = frame
        for bgr_fake, matrix in swapped:
            result = self._paste_swapped(result, bgr_fake, matrix)

        return result, slot_kps

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


# Global singleton
_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """Lazy-load the segmentation service singleton."""
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service
