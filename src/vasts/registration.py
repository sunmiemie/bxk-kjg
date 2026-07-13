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


def valid_registration_mask(shape: tuple[int, int], transform: np.ndarray) -> np.ndarray:
    """Pixels fully supported by the previous frame's bilinear sample footprint."""
    if len(shape) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("registration mask shape must contain positive height and width")
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (2, 3) or not np.isfinite(transform).all():
        raise ValueError("registration transform must be a finite 2x3 matrix")
    height, width = shape
    coverage = cv2.warpAffine(np.ones(shape, dtype=np.float32), transform, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return coverage >= 1.0 - 1e-6


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

    for name, value in (("orb_features", orb_features), ("min_matches", min_matches)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(ransac_reproj_threshold) or ransac_reproj_threshold < 0:
        raise ValueError("ransac_reproj_threshold must be finite and nonnegative")
    if not np.isfinite(phase_response_threshold) or not 0 <= phase_response_threshold <= 1:
        raise ValueError("phase_response_threshold must be finite and in [0, 1]")
    try:
        transform = _orb_transform(reference_gray, moving_gray, orb_features, min_matches, ransac_reproj_threshold)
    except cv2.error:
        transform = None
    if transform is not None:
        return _warp(moving_gray, transform, shape), transform, "orb"
    transform, method = _phase_fallback(reference_gray, moving_gray, float(phase_response_threshold))
    return _warp(moving_gray, transform, shape), transform, method


def _orb_transform(reference_gray, moving_gray, orb_features, min_matches, ransac_reproj_threshold):
    # Partial affine estimation needs at least two correspondences.
    min_matches = max(2, min_matches)

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
                    and inliers is not None and int(inliers.sum()) >= max(2, (min_matches + 1) // 2)):
                with np.errstate(over="ignore"):
                    transform = transform.astype(np.float32)
                if np.isfinite(transform).all() and abs(np.linalg.det(transform[:, :2].astype(np.float64))) > 1e-8:
                    return transform
    return None
