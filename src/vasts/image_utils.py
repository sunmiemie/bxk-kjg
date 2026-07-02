from __future__ import annotations

import numpy as np
import cv2


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    return gray.astype(np.uint8, copy=False)


def normalize01(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)


def to_uint8(image: np.ndarray) -> np.ndarray:
    arr = normalize01(image)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def colorize_saliency(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(to_uint8(image), cv2.COLORMAP_INFERNO)
