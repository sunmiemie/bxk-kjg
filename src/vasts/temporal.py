from __future__ import annotations

import numpy as np
import cv2
from .image_utils import ensure_gray


def registered_difference(reference_gray: np.ndarray, registered_moving_gray: np.ndarray) -> np.ndarray:
    reference = ensure_gray(reference_gray)
    moving = ensure_gray(registered_moving_gray)
    if reference.shape != moving.shape:
        raise ValueError("difference images must have equal shapes")
    return cv2.absdiff(reference, moving)


def compute_fmhi(
    previous_mhi: np.ndarray,
    diff: np.ndarray,
    *,
    threshold: int = 35,
    decay: int = 25,
) -> np.ndarray:
    if any(not isinstance(image, np.ndarray) or image.ndim != 2 for image in (previous_mhi, diff)):
        raise ValueError("FMHI inputs must be grayscale 2D arrays")
    previous_u8, diff_u8 = ensure_gray(previous_mhi), ensure_gray(diff)
    if previous_u8.shape != diff_u8.shape:
        raise ValueError("FMHI inputs must have equal shapes")
    for name, value in (("threshold", threshold), ("decay", decay)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f"{name} must be an integer in [0, 255]")
    previous = previous_u8.astype(np.int16)
    diff_i = diff_u8.astype(np.int16)
    quiet = np.maximum(0, previous + diff_i - int(decay))
    quiet = np.minimum(255, quiet)
    current = quiet.astype(np.uint8)
    current[diff_u8 >= threshold] = 255
    return current


def motion_regions(
    fmhi: np.ndarray,
    *,
    candidate_threshold: int = 1,
    min_area: int = 6,
    max_area_ratio: float = 0.20,
    border_margin: int = 0,
) -> list[list[int]]:
    if not isinstance(fmhi, np.ndarray) or fmhi.ndim != 2:
        raise ValueError("motion history must be a 2D array")
    fmhi = ensure_gray(fmhi)
    for name, value, minimum in (("candidate_threshold", candidate_threshold, 0), ("min_area", min_area, 1), ("border_margin", border_margin, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if candidate_threshold > 255 or not np.isfinite(max_area_ratio) or not 0 <= max_area_ratio <= 1:
        raise ValueError("threshold must be <=255 and max_area_ratio must be in [0,1]")
    binary = ((fmhi > 0) & (fmhi >= candidate_threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = fmhi.shape[:2]
    image_area = float(height * width)
    boxes: list[list[int]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if int(area) < int(min_area):
            continue
        if float(area) > image_area * float(max_area_ratio):
            continue
        if border_margin > 0:
            if x < border_margin or y < border_margin:
                continue
            if x + w > width - border_margin or y + h > height - border_margin:
                continue
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes
