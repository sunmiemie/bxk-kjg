from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .image_utils import ensure_gray


def detect_static_dark_blobs(
    frame: np.ndarray,
    roi_mask: np.ndarray,
    *,
    threshold_offset: float = 28.0,
    min_area: int = 2,
    max_area: int = 250,
    min_width: int = 1,
    min_height: int = 1,
    max_width: int = 50,
    max_height: int = 30,
) -> list[dict[str, Any]]:
    gray = ensure_gray(frame)
    if not isinstance(roi_mask, np.ndarray) or roi_mask.shape != gray.shape:
        raise ValueError("roi_mask must have the same 2D shape as the grayscale frame")
    if not np.isfinite(threshold_offset) or threshold_offset <= 0:
        raise ValueError("threshold_offset must be finite and positive")
    valid = roi_mask.astype(bool)
    if not np.any(valid):
        return []

    roi_values = gray[valid]
    background = float(np.median(roi_values))
    threshold = background - float(threshold_offset)
    if threshold < 0:
        # A clipped zero threshold would turn a uniformly black ROI into a blob
        # despite there being no contrast against the estimated background.
        return []
    binary = ((gray <= threshold) & valid).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < int(min_area) or area > int(max_area):
            continue
        if w < int(min_width) or h < int(min_height):
            continue
        if w > int(max_width) or h > int(max_height):
            continue

        region = labels[y : y + h, x : x + w] == label
        values = gray[y : y + h, x : x + w][region]
        if values.size == 0:
            continue
        contrast = max(0.0, background - float(values.mean()))
        score = min(1.0, contrast / max(float(threshold_offset), 1.0))
        boxes.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "score": round(score, 4),
                "source": "static",
            }
        )
    boxes.sort(key=lambda item: float(item["score"]), reverse=True)
    return boxes
