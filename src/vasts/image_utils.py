from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path


def ensure_gray(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or not np.issubdtype(frame.dtype, np.number) or np.iscomplexobj(frame):
        raise ValueError("frame must be a real numeric NumPy array in the 0..255 intensity range")
    if frame.size == 0 or not np.isfinite(frame).all() or np.min(frame) < 0 or np.max(frame) > 255:
        raise ValueError("frame must be nonempty with finite intensities in 0..255")
    # Convert before cvtColor so float64 and integer inputs follow one convention.
    frame = frame.astype(np.uint8, copy=False)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    return gray.astype(np.uint8, copy=False)


def normalize01(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
        raise ValueError("saliency must be a real numeric NumPy array")
    arr = image.astype(np.float64, copy=False)
    if arr.size == 0:
        return arr.astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("saliency values must be finite")
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    scale = max(abs(min_value), abs(max_value))
    scaled = arr / scale
    return ((scaled - min_value / scale) / (max_value / scale - min_value / scale)).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    arr = normalize01(image)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def colorize_saliency(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(to_uint8(image), cv2.COLORMAP_INFERNO)


def write_png(path, image: np.ndarray) -> None:
    """Encode in memory and use Python paths, including Windows Unicode names."""
    try:
        success, encoded = cv2.imencode(".png", image)
    except cv2.error as error:
        raise RuntimeError(f"Could not encode PNG map: {path}") from error
    if not success:
        raise RuntimeError(f"Could not encode PNG map: {path}")
    Path(path).write_bytes(encoded.tobytes())
