from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .lite_detector import V3LiteDetector
from .pipeline import _create_writer, _draw_overlay, _open_capture


@dataclass
class LitePipelineResult:
    output_dir: Path
    frames_processed: int
    boxes: list[dict[str, Any]]
    boxes_path: Path


def run_lite_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    save_video: bool = False,
    config: dict[str, Any] | None = None,
    source_mode: str = "input",
) -> LitePipelineResult:
    detector = V3LiteDetector(config)
    cfg = detector.config

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(input_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    writer = None
    frames: list[dict[str, Any]] = []

    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Input video has no frames: {input_path}")
        height, width = frame.shape[:2]
        if save_video:
            writer = _create_writer(out_dir / "overlay_lite.mp4", fps, (width, height))

        while ok:
            if max_frames is not None and detector.frame_index >= int(max_frames):
                break

            result = detector.update(frame)
            frames.append({"frame_index": result.frame_index, "boxes": result.boxes})
            if writer is not None:
                writer.write(_draw_overlay(frame, result.boxes, result.frame_index))

            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    payload = {
        "schema_version": 1,
        "mode": "v3-lite",
        "source": {"mode": source_mode, "path": str(input_path), "max_frames": max_frames},
        "config": cfg,
        "summary": {
            "frames_processed": len(frames),
            "processed_detection_frames": detector.processed_detection_frames,
            "frames_with_boxes": sum(1 for item in frames if item["boxes"]),
            "total_boxes": sum(len(item["boxes"]) for item in frames),
            "held_boxes": detector.held_boxes,
        },
        "frames": frames,
    }
    boxes_path = out_dir / "boxes_lite.json"
    boxes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LitePipelineResult(output_dir=out_dir, frames_processed=len(frames), boxes=frames, boxes_path=boxes_path)
