"""Gold hoop nose ring overlay, placed from InsightFace's 5-point landmarks.

No model involved: the hoop is a procedurally drawn RGBA sprite, positioned on
the person's left nostril wing every frame from the eye and nose-tip
keypoints, scaled by the eye distance and rotated with the eye line. Callers
processing video should smooth the keypoints over time first (see
`smooth_keypoint_track`) — raw per-frame landmarks wobble by a pixel or two,
which makes a rigid overlay shimmer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

import cv2
import numpy as np

# InsightFace kps order: [image-left eye, image-right eye, nose tip,
# image-left mouth corner, image-right mouth corner]. For an unmirrored
# frontal face, the image-right side is the person's own left.
_EYE_L, _EYE_R, _NOSE = 0, 1, 2

# Geometry, in units of the (projected) distance between the eyes.
_PIERCE_LATERAL = 0.22  # nose tip -> piercing point, along the eye line
_PIERCE_UP = 0.09  # above the tip, up on the nostril wing
_HOOP_DIAMETER = 0.17

# Yaw proxy: how far the nose tip sits from the eye midpoint along the eye
# line, as a fraction of eye distance. Positive = nose pointing toward the
# person's left, i.e. the pierced side turning away from the camera, where it
# ends up behind the nose. The ring fades out across this range.
_YAW_FADE_START = 0.12
_YAW_FADE_END = 0.24

_SPRITE_SIZE = 256  # sprite resolution; downscaled per frame, so keep it generous
_SMOOTH_RADIUS = 2  # frames either side for the temporal moving average


@lru_cache(maxsize=1)
def _hoop_sprite() -> tuple[np.ndarray, tuple[float, float], float]:
    """Render the hoop once: BGRA float sprite, piercing point, hoop diameter (px).

    The hoop hangs from its piercing point at the top. It's drawn as a
    slightly narrow ellipse because the ring sits roughly along the side of
    the nose and so is seen a bit edge-on. A small gap at the top is where
    the wire passes through the nostril.
    """
    size = _SPRITE_SIZE
    center = (size / 2, size / 2)
    radius_y = size * 0.36
    radius_x = radius_y * 0.82
    wire = size * 0.045  # half-thickness of the wire

    # Supersampled wire mask, then distance-to-edge gives a round tube profile.
    scale = 4
    big = np.zeros((size * scale, size * scale), np.uint8)
    cv2.ellipse(
        big,
        (int(center[0] * scale), int(center[1] * scale)),
        (int(radius_x * scale), int(radius_y * scale)),
        0,
        -90 + 16,  # leave a gap either side of the top (the piercing)
        270 - 16,
        255,
        thickness=int(2 * wire * scale),
        lineType=cv2.LINE_AA,
    )
    dist = cv2.distanceTransform((big > 127).astype(np.uint8), cv2.DIST_L2, 5) / scale
    dist = cv2.resize(dist, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    # Tube shading: bright ridge down the middle of the wire, dark at its edges.
    profile = np.clip(dist / wire, 0, 1)
    ridge = np.sqrt(profile)

    # Light from the upper left: the part of the ring facing it catches more.
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = np.arctan2(ys - center[1], xs - center[0])
    facing = 0.5 + 0.5 * np.cos(angle - np.deg2rad(-135))

    shade = 0.35 + 0.65 * ridge * (0.55 + 0.45 * facing)
    dark_gold = np.array([18, 88, 150], np.float32)  # BGR
    light_gold = np.array([95, 205, 245], np.float32)
    color = dark_gold + (light_gold - dark_gold) * shade[..., None]

    # Specular glint on the lit side of the ridge.
    glint = np.clip((ridge - 0.8) / 0.2, 0, 1) * np.clip((facing - 0.75) / 0.25, 0, 1)
    color = color + (255 - color) * (0.85 * glint)[..., None]

    sprite = np.dstack([color, alpha * 255]).astype(np.float32)
    pierce_point = (center[0], center[1] - radius_y)
    return sprite, pierce_point, 2 * radius_y


def smooth_keypoint_track(track: Sequence[Optional[np.ndarray]]) -> list[Optional[np.ndarray]]:
    """Centered moving average of one person's keypoints across frames.

    Entries are None where the person wasn't found; those stay None, and the
    average only uses neighbours where they were present and nearby, so the
    ring never drifts toward a stale position across a gap or a scene cut.
    """
    smoothed: list[Optional[np.ndarray]] = []
    for index, kps in enumerate(track):
        if kps is None:
            smoothed.append(None)
            continue
        kps = np.asarray(kps, dtype=np.float32)
        max_jump = 0.25 * float(np.linalg.norm(kps[_EYE_R] - kps[_EYE_L]))
        window = [
            np.asarray(track[j], dtype=np.float32)
            for j in range(max(0, index - _SMOOTH_RADIUS), min(len(track), index + _SMOOTH_RADIUS + 1))
            if track[j] is not None
        ]
        window = [other for other in window if np.linalg.norm(other[_NOSE] - kps[_NOSE]) <= max_jump]
        smoothed.append(np.mean(window, axis=0))
    return smoothed


def draw_nose_ring(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Composite a gold hoop onto the person's left nostril; returns a new frame."""
    kps = np.asarray(kps, dtype=np.float32)
    eye_vec = kps[_EYE_R] - kps[_EYE_L]
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 8:  # face too small to see a ring on anyway
        return frame
    along = eye_vec / eye_dist  # toward the person's left
    down = np.array([-along[1], along[0]], np.float32)

    eye_mid = (kps[_EYE_L] + kps[_EYE_R]) / 2
    yaw = float(np.dot(kps[_NOSE] - eye_mid, along)) / eye_dist
    visibility = float(np.clip((_YAW_FADE_END - yaw) / (_YAW_FADE_END - _YAW_FADE_START), 0, 1))
    if visibility <= 0:
        return frame

    # Turning the pierced side toward the camera spreads the nostril wing out
    # from the tip; turning it away slides the wing in behind the tip, faster.
    lateral = max(_PIERCE_LATERAL - (0.4 if yaw < 0 else 0.9) * yaw, 0.05)
    pierce = kps[_NOSE] + along * (lateral * eye_dist) - down * (_PIERCE_UP * eye_dist)

    sprite, sprite_pierce, sprite_diameter = _hoop_sprite()
    scale = _HOOP_DIAMETER * eye_dist / sprite_diameter
    roll = float(np.degrees(np.arctan2(along[1], along[0])))
    # Rotate/scale about the sprite's piercing point, then move that point
    # onto the nostril. Negative roll: OpenCV's angle is counter-clockwise.
    matrix = cv2.getRotationMatrix2D(sprite_pierce, -roll, scale)
    matrix[:, 2] += pierce - np.array(sprite_pierce, np.float32)

    height, width = frame.shape[:2]
    half = int(np.ceil(_HOOP_DIAMETER * eye_dist)) + 4
    x0, y0 = max(int(pierce[0]) - half, 0), max(int(pierce[1]) - half, 0)
    x1, y1 = min(int(pierce[0]) + half, width), min(int(pierce[1]) + half, height)
    if x1 <= x0 or y1 <= y0:
        return frame

    # Warp only into the small region around the nose.
    matrix[:, 2] -= (x0, y0)
    region_size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(sprite, matrix, region_size, flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = warped[..., 3:4] / 255 * visibility
    if not alpha.any():
        return frame

    region = frame[y0:y1, x0:x1].astype(np.float32)

    # Match the scene lighting: a dim shot shouldn't get a glowing ring.
    skin = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    exposure = float(np.clip(np.median(skin) / 150.0, 0.55, 1.15))
    ring = np.clip(warped[..., :3] * exposure, 0, 255)

    # Soft contact shadow just below/right of the wire.
    shadow = cv2.GaussianBlur(alpha[..., 0], (0, 0), max(eye_dist * 0.012, 0.8))
    offset = max(eye_dist * 0.01, 1.0)
    shift = np.float32([[1, 0, offset * 0.6], [0, 1, offset]])
    shadow = cv2.warpAffine(shadow, shift, region_size)[..., None]
    region *= 1 - 0.35 * shadow

    region = alpha * ring + (1 - alpha) * region
    result = frame.copy()
    result[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    return result
