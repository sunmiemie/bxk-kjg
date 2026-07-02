from __future__ import annotations

import numpy as np
import cv2


def registered_difference(reference_gray: np.ndarray, registered_moving_gray: np.ndarray) -> np.ndarray:
    return cv2.absdiff(reference_gray.astype(np.uint8), registered_moving_gray.astype(np.uint8))


def compute_fmhi(
    previous_mhi: np.ndarray,
    diff: np.ndarray,
    *,
    threshold: int = 35,
    decay: int = 25,
) -> np.ndarray:
    previous = previous_mhi.astype(np.int16, copy=False)
    diff_i = diff.astype(np.int16, copy=False)
    quiet = np.maximum(0, previous + diff_i - int(decay))
    quiet = np.minimum(255, quiet)
    current = quiet.astype(np.uint8)
    current[diff >= int(threshold)] = 255
    return current


def motion_regions(
    fmhi: np.ndarray,
    *,
    candidate_threshold: int = 1,
    min_area: int = 6,
    max_area_ratio: float = 0.20,
    border_margin: int = 0,
) -> list[list[int]]:
    binary = (fmhi >= int(candidate_threshold)).astype(np.uint8)
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
        if w <= 1 or h <= 1:
            continue
        if border_margin > 0:
            if x <= border_margin or y <= border_margin:
                continue
            if x + w >= width - border_margin or y + h >= height - border_margin:
                continue
        boxes.append([int(x), int(y), int(w), int(h)])
    return boxes
