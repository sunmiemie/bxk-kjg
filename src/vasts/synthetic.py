from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import cv2


def _base_background(width: int, height: int, seed: int = 42) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = 70 + 60 * x + 30 * y
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, size=(height, width)).astype(np.float32)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    for px in range(8, width, 17):
        cv2.line(image, (px, 0), (px, height - 1), 110, 1)
    for py in range(6, height, 19):
        cv2.line(image, (0, py), (width - 1, py), 95, 1)
    return image


def make_synthetic_video(
    video_path: str | Path,
    *,
    frame_count: int = 20,
    size: tuple[int, int] = (160, 120),
    target_size: tuple[int, int] = (10, 8),
    fps: float = 10.0,
    seed: int = 42,
) -> Path:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    for name, values in (("size", size), ("target_size", target_size)):
        if not isinstance(values, (tuple, list)) or len(values) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"{name} must contain two positive integers")
    width, height = size
    target_w, target_h = target_size
    if width % 2 or height % 2:
        raise ValueError("MP4 synthetic dimensions must be even to avoid codec cropping")
    if target_w > width - 4 or target_h > height - 4:
        raise ValueError("target must fit inside the image with a two-pixel margin")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    background = _base_background(width + 32, height + 32, seed)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer: {path}")

    try:
        for index in range(int(frame_count)):
            cam_x = index % 7
            cam_y = (index * 2) % 7
            gray = background[cam_y : cam_y + height, cam_x : cam_x + width].copy()
            target_x = 14 + index * 4
            target_y = 20 + index * 2
            target_x = min(width - target_w - 2, target_x)
            target_y = min(height - target_h - 2, target_y)
            cv2.rectangle(gray, (target_x, target_y), (target_x + target_w - 1, target_y + target_h - 1), 235, -1)
            cv2.rectangle(gray, (target_x, target_y), (target_x + target_w - 1, target_y + target_h - 1), 255, 1)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()
    return path
