from __future__ import annotations

import numpy as np
import cv2

from .image_utils import ensure_gray


def _identity_transform() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def _warp(frame: np.ndarray, transform: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _phase_fallback(
    reference_gray: np.ndarray,
    moving_gray: np.ndarray,
    response_threshold: float,
) -> tuple[np.ndarray, str]:
    try:
        shift, response = cv2.phaseCorrelate(
            reference_gray.astype(np.float32),
            moving_gray.astype(np.float32),
        )
    except cv2.error:
        return _identity_transform(), "identity"

    if not np.isfinite(shift[0]) or not np.isfinite(shift[1]) or not np.isfinite(response) or response < response_threshold:
        return _identity_transform(), "identity"

    transform = np.array(
        [[1.0, 0.0, -float(shift[0])], [0.0, 1.0, -float(shift[1])]],
        dtype=np.float32,
    )
    return transform, "phase"


def register_frame(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    orb_features: int = 800,
    min_matches: int = 8,
    ransac_reproj_threshold: float = 3.0,
    phase_response_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Register moving frame to reference using ORB/RANSAC, then phase/identity fallback."""
    reference_gray = ensure_gray(reference)
    moving_gray = ensure_gray(moving)
    if reference_gray.shape != moving_gray.shape:
        raise ValueError("Registration frames must have equal dimensions")
    shape = reference_gray.shape

    orb = cv2.ORB_create(nfeatures=int(orb_features))
    ref_keypoints, ref_desc = orb.detectAndCompute(reference_gray, None)
    mov_keypoints, mov_desc = orb.detectAndCompute(moving_gray, None)

    if ref_desc is not None and mov_desc is not None and len(ref_keypoints) >= min_matches and len(mov_keypoints) >= min_matches:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(mov_desc, ref_desc)
        matches = sorted(matches, key=lambda item: item.distance)
        if len(matches) >= min_matches:
            limited = matches[: min(len(matches), 80)]
            src = np.float32([mov_keypoints[m.queryIdx].pt for m in limited]).reshape(-1, 1, 2)
            dst = np.float32([ref_keypoints[m.trainIdx].pt for m in limited]).reshape(-1, 1, 2)
            transform, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_reproj_threshold),
            )
            if (transform is not None and transform.shape == (2, 3) and np.isfinite(transform).all()
                    and inliers is not None and int(inliers.sum()) >= min_matches // 2):
                transform = transform.astype(np.float32)
                return _warp(moving_gray, transform, shape), transform, "orb"

    transform, method = _phase_fallback(reference_gray, moving_gray, float(phase_response_threshold))
    return _warp(moving_gray, transform, shape), transform, method
