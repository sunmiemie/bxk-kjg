from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2


def _base_background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = 70 + 60 * x + 30 * y
    rng = np.random.default_rng(42)
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
) -> Path:
    if int(frame_count) <= 0:
        raise ValueError("frame_count must be positive")
    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(size[0]), int(size[1])
    target_w, target_h = int(target_size[0]), int(target_size[1])
    background = _base_background(width + 32, height + 32)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
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
            cv2.rectangle(gray, (target_x, target_y), (target_x + target_w, target_y + target_h), 235, -1)
            cv2.rectangle(gray, (target_x, target_y), (target_x + target_w, target_y + target_h), 255, 1)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()
    return path
