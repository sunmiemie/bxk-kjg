from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, deep_update
from .image_utils import ensure_gray
from .pipeline import _create_writer, _draw_overlay, _open_capture
from .regions import deduplicate_boxes
from .static_target import detect_static_dark_blobs


@dataclass
class LitePipelineResult:
    output_dir: Path
    frames_processed: int
    boxes: list[dict[str, Any]]
    boxes_path: Path


def _roi_bounds(shape: tuple[int, int], roi_cfg: dict[str, Any]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x_min = max(0, min(width, int(roi_cfg.get("x_min", 0))))
    y_min = max(0, min(height, int(roi_cfg.get("y_min", 0))))
    x_max_cfg = roi_cfg.get("x_max")
    y_max_cfg = roi_cfg.get("y_max")
    x_max = width if x_max_cfg is None else max(0, min(width, int(x_max_cfg)))
    y_max = height if y_max_cfg is None else max(0, min(height, int(y_max_cfg)))
    if not bool(roi_cfg.get("enabled", False)):
        return 0, 0, width, height
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("ROI must have positive width and height")
    return x_min, y_min, x_max, y_max


def _local_roi_mask(shape: tuple[int, int], roi_cfg: dict[str, Any], offset: tuple[int, int]) -> np.ndarray:
    height, width = shape[:2]
    offset_x, offset_y = offset
    mask = np.ones((height, width), dtype=bool)
    for rect in roi_cfg.get("ignore_rectangles", []):
        rx0 = int(rect["x"]) - offset_x
        ry0 = int(rect["y"]) - offset_y
        rx1 = rx0 + int(rect["w"])
        ry1 = ry0 + int(rect["h"])
        left = max(0, min(width, rx0))
        top = max(0, min(height, ry0))
        right = max(0, min(width, rx1))
        bottom = max(0, min(height, ry1))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = False
    return mask


def _map_boxes_to_full_frame(boxes: list[dict[str, Any]], offset: tuple[int, int]) -> list[dict[str, Any]]:
    offset_x, offset_y = offset
    mapped = []
    for box in boxes:
        mapped_box = dict(box)
        mapped_box["x"] = int(mapped_box["x"]) + offset_x
        mapped_box["y"] = int(mapped_box["y"]) + offset_y
        mapped.append(mapped_box)
    return mapped


def _motion_boxes(
    gray: np.ndarray,
    prev_gray: np.ndarray | None,
    roi_mask: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    max_area: int,
) -> list[dict[str, Any]]:
    if prev_gray is None:
        return []
    diff = cv2.absdiff(gray, prev_gray)
    binary = ((diff >= float(threshold)) & roi_mask.astype(bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < int(min_area) or area > int(max_area):
            continue
        region = labels[y : y + h, x : x + w] == label
        values = diff[y : y + h, x : x + w][region]
        score = float(values.mean()) / 255.0 if values.size else 0.0
        boxes.append({"x": x, "y": y, "w": w, "h": h, "score": round(min(1.0, score), 4), "source": "motion"})
    return boxes


def _filter_boxes(boxes: list[dict[str, Any]], filter_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = []
    max_width = filter_cfg.get("max_width")
    max_height = filter_cfg.get("max_height")
    for box in boxes:
        if int(box["w"]) < int(filter_cfg["min_width"]) or int(box["h"]) < int(filter_cfg["min_height"]):
            continue
        if max_width is not None and int(box["w"]) > int(max_width):
            continue
        if max_height is not None and int(box["h"]) > int(max_height):
            continue
        if float(box["score"]) < float(filter_cfg["min_score"]):
            continue
        filtered.append(box)
    return filtered


def _held_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    held = []
    for box in boxes:
        held_box = dict(box)
        held_box["held"] = True
        held.append(held_box)
    return held


def run_lite_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    save_video: bool = False,
    config: dict[str, Any] | None = None,
    source_mode: str = "input",
) -> LitePipelineResult:
    cfg = deep_update(DEFAULT_CONFIG, config or {})
    lite_cfg = cfg["lite"]
    frame_stride = max(1, int(lite_cfg["frame_stride"]))
    hold_missing_frames = max(0, int(lite_cfg["hold_missing_frames"]))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(input_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    writer = None
    frames: list[dict[str, Any]] = []
    processed_detection_frames = 0
    held_count = 0
    previous_roi_gray = None
    last_boxes: list[dict[str, Any]] = []
    last_detection_frame = -10**9

    try:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Input video has no frames: {input_path}")
        height, width = frame.shape[:2]
        roi_x0, roi_y0, roi_x1, roi_y1 = _roi_bounds((height, width), cfg["roi"])
        if save_video:
            writer = _create_writer(out_dir / "overlay_lite.mp4", fps, (width, height))

        frame_index = 0
        while ok:
            if max_frames is not None and frame_index >= int(max_frames):
                break

            roi_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]
            roi_gray = ensure_gray(roi_frame)
            roi_mask = _local_roi_mask(roi_gray.shape, cfg["roi"], (roi_x0, roi_y0))
            boxes: list[dict[str, Any]] = []
            should_detect = frame_index % frame_stride == 0
            if should_detect:
                if bool(cfg["static_target"]["enabled"]):
                    static_cfg = {key: value for key, value in cfg["static_target"].items() if key != "enabled"}
                    boxes.extend(detect_static_dark_blobs(roi_gray, roi_mask, **static_cfg))
                if bool(lite_cfg["motion_enabled"]):
                    boxes.extend(
                        _motion_boxes(
                            roi_gray,
                            previous_roi_gray,
                            roi_mask,
                            threshold=lite_cfg["motion_threshold"],
                            min_area=lite_cfg["motion_min_area"],
                            max_area=lite_cfg["motion_max_area"],
                        )
                    )
                boxes = _filter_boxes(boxes, cfg["box_filter"])
                boxes = deduplicate_boxes(boxes, center_distance=float(lite_cfg["dedupe_center_distance"]))
                boxes = _map_boxes_to_full_frame(boxes, (roi_x0, roi_y0))
                processed_detection_frames += 1
                previous_roi_gray = roi_gray.copy()
                if boxes:
                    last_boxes = boxes
                    last_detection_frame = frame_index
            elif last_boxes and frame_index - last_detection_frame <= hold_missing_frames:
                boxes = _held_boxes(last_boxes)
                held_count += len(boxes)

            frames.append({"frame_index": frame_index, "boxes": boxes})
            if writer is not None:
                writer.write(_draw_overlay(frame, boxes, frame_index))

            frame_index += 1
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
            "processed_detection_frames": processed_detection_frames,
            "frames_with_boxes": sum(1 for item in frames if item["boxes"]),
            "total_boxes": sum(len(item["boxes"]) for item in frames),
            "held_boxes": held_count,
        },
        "frames": frames,
    }
    boxes_path = out_dir / "boxes_lite.json"
    boxes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LitePipelineResult(output_dir=out_dir, frames_processed=len(frames), boxes=frames, boxes_path=boxes_path)
