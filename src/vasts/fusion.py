from __future__ import annotations

import numpy as np
import cv2

from .image_utils import normalize01


def fuse_saliency(
    temporal: np.ndarray,
    pixel: np.ndarray,
    region: np.ndarray,
    *,
    temporal_weight: float = 0.31,
    pixel_weight: float = 0.42,
    region_weight: float = 0.27,
) -> np.ndarray:
    fused = (
        float(temporal_weight) * normalize01(temporal)
        + float(pixel_weight) * normalize01(pixel)
        + float(region_weight) * normalize01(region)
    )
    return normalize01(fused)


def threshold_saliency(
    saliency: np.ndarray,
    *,
    method: str = "otsu",
    fixed: float = 0.45,
    percentile: float = 97.5,
) -> np.ndarray:
    sal = normalize01(saliency)
    method = method.lower()
    if method == "fixed":
        mask = sal >= float(fixed)
    elif method == "percentile":
        values = sal[sal > 0]
        source = values if values.size else sal.reshape(-1)
        threshold = float(np.percentile(source, float(percentile)))
        mask = sal >= threshold if threshold > 0 else sal > 0
    elif method == "otsu":
        scaled = np.clip(sal * 255.0, 0, 255).astype(np.uint8)
        if int(scaled.max()) == int(scaled.min()):
            mask = scaled > 0
        else:
            _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = binary > 0
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return (mask.astype(np.uint8) * 255)


def postprocess_mask(mask: np.ndarray, *, morph_kernel: int = 3) -> np.ndarray:
    kernel_size = max(1, int(morph_kernel))
    if kernel_size <= 1:
        return mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed
