from __future__ import annotations

import math

import numpy as np
import cv2


def expand_box(box: list[int] | tuple[int, int, int, int], *, factor: float, image_shape: tuple[int, int]) -> list[int]:
    x, y, w, h = [int(v) for v in box]
    height, width = image_shape[:2]
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    new_w = max(1.0, w * float(factor))
    new_h = max(1.0, h * float(factor))
    left = max(0, int(math.floor(center_x - new_w / 2.0)))
    top = max(0, int(math.floor(center_y - new_h / 2.0)))
    right = min(width, int(math.ceil(center_x + new_w / 2.0)))
    bottom = min(height, int(math.ceil(center_y + new_h / 2.0)))
    return [left, top, max(0, right - left), max(0, bottom - top)]


def box_slice(box: list[int] | tuple[int, int, int, int]) -> tuple[slice, slice]:
    x, y, w, h = [int(v) for v in box]
    return slice(y, y + h), slice(x, x + w)


def boxes_from_mask(
    mask: np.ndarray,
    saliency: np.ndarray,
    *,
    min_area: int = 6,
    max_area_ratio: float = 0.20,
) -> list[dict[str, float | int]]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape[:2]
    max_area = float(height * width) * float(max_area_ratio)
    boxes: list[dict[str, float | int]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if int(area) < int(min_area) or float(area) > max_area:
            continue
        region = labels[y : y + h, x : x + w] == label
        scores = saliency[y : y + h, x : x + w][region]
        score = float(scores.mean()) if scores.size else 0.0
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": round(score, 4)})
    boxes.sort(key=lambda item: (item["y"], item["x"]))
    return boxes


def _box_center(box: dict) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0


def deduplicate_boxes(boxes: list[dict], *, center_distance: float = 8.0) -> list[dict]:
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        cx, cy = _box_center(box)
        duplicate = False
        for existing in kept:
            ex, ey = _box_center(existing)
            if math.hypot(cx - ex, cy - ey) <= float(center_distance):
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    kept.sort(key=lambda item: (item["y"], item["x"]))
    return kept
