from __future__ import annotations

import math

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
    maps = (temporal, pixel, region)
    if any(not isinstance(image, np.ndarray) or image.ndim != 2 for image in maps) or any(image.shape != temporal.shape for image in maps):
        raise ValueError("fusion maps must be 2D arrays with equal shapes")
    for weight in (temporal_weight, pixel_weight, region_weight):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("fusion weights must be finite and nonnegative")
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
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not isinstance(saliency, np.ndarray) or saliency.ndim != 2:
        raise ValueError("saliency must be a 2D array")
    if not isinstance(method, str):
        raise ValueError("threshold method must be a string")
    if not math.isfinite(fixed) or not 0 <= fixed <= 1:
        raise ValueError("fixed threshold must be finite and in [0, 1]")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be finite and in [0, 100]")
    if valid_mask is None:
        valid = np.ones(saliency.shape, dtype=bool)
    elif not isinstance(valid_mask, np.ndarray) or valid_mask.shape != saliency.shape or valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must be a boolean array matching saliency")
    else:
        valid = valid_mask
    sal = np.zeros(saliency.shape, dtype=np.float32)
    if np.any(valid):
        sal[valid] = normalize01(saliency[valid])
    method = method.lower()
    if method not in {"fixed", "percentile", "otsu"}:
        raise ValueError(f"Unknown threshold method: {method}")
    if not sal.size or not np.any(valid):
        return np.zeros(sal.shape, dtype=np.uint8)
    if method == "fixed":
        mask = sal >= float(fixed)
    elif method == "percentile":
        values = sal[valid & (sal > 0)]
        source = values if values.size else sal[valid]
        threshold = float(np.percentile(source, float(percentile)))
        mask = sal >= threshold if threshold > 0 else sal > 0
    elif method == "otsu":
        scaled = np.clip(sal * 255.0, 0, 255).astype(np.uint8)
        samples = scaled[valid].reshape(-1, 1)
        if int(samples.max()) == int(samples.min()):
            mask = scaled > 0
        else:
            _, binary = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = np.zeros(sal.shape, dtype=bool)
            mask[valid] = binary.reshape(-1) > 0
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return ((mask & valid).astype(np.uint8) * 255)


def postprocess_mask(mask: np.ndarray, *, morph_kernel: int = 3) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or (mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.number)) or np.iscomplexobj(mask):
        raise ValueError("mask must be a real numeric or boolean 2D array")
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError("mask values must be finite and nonnegative")
    if isinstance(morph_kernel, bool) or not isinstance(morph_kernel, int) or morph_kernel <= 0 or morph_kernel % 2 == 0:
        raise ValueError("morph_kernel must be a positive odd integer")
    binary = (mask > 0).astype(np.uint8) * 255
    kernel_size = morph_kernel
    if not binary.size:
        return binary
    if kernel_size <= 1:
        return binary
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed
