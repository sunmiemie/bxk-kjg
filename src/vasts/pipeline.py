from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .config import DEFAULT_CONFIG, deep_update, _validate_config
from .fusion import fuse_saliency, postprocess_mask, threshold_saliency
from .image_utils import colorize_saliency, ensure_gray, to_uint8
from .pixel_saliency import pixel_saliency_map
from .region_saliency import region_saliency_map
from .regions import boxes_from_mask, deduplicate_boxes
from .registration import register_frame
from .static_target import detect_static_dark_blobs
from .temporal import compute_fmhi, motion_regions, registered_difference


@dataclass
class PipelineResult:
    output_dir: Path
    frames_processed: int
    boxes: list[dict[str, Any]]
    boxes_path: Path


def _open_capture(input_path: str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open input video: {input_path}")
    return capture


def _create_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _capture_fps(capture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return fps if math.isfinite(fps) and fps > 0 else 10.0


def _draw_overlay(frame: np.ndarray, boxes: list[dict[str, Any]], frame_index: int) -> np.ndarray:
    overlay = frame.copy()
    for box in boxes:
        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"{float(box['score']):.2f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(overlay, f"frame {frame_index}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def _save_maps(
    maps_dir: Path,
    frame_index: int,
    *,
    diff: np.ndarray,
    fmhi: np.ndarray,
    pixel: np.ndarray,
    region: np.ndarray,
    fused: np.ndarray,
    mask: np.ndarray,
) -> None:
    maps_dir.mkdir(parents=True, exist_ok=True)
    prefix = maps_dir / f"frame_{frame_index:05d}"
    outputs = {
        "diff": diff,
        "fmhi": fmhi,
        "pixel": to_uint8(pixel),
        "region": to_uint8(region),
        "fused": to_uint8(fused),
        "mask": mask,
    }
    for suffix, image in outputs.items():
        path = str(prefix) + f"_{suffix}.png"
        if not cv2.imwrite(path, image):
            raise RuntimeError(f"Could not write map image: {path}")


def _roi_mask(shape: tuple[int, int], roi_cfg: dict[str, Any]) -> np.ndarray:
    height, width = shape[:2]
    if not bool(roi_cfg.get("enabled", False)):
        return np.ones((height, width), dtype=bool)
    x_min = max(0, min(width, int(roi_cfg.get("x_min", 0))))
    y_min = max(0, min(height, int(roi_cfg.get("y_min", 0))))
    x_max_cfg = roi_cfg.get("x_max")
    y_max_cfg = roi_cfg.get("y_max")
    x_max = width if x_max_cfg is None else max(0, min(width, int(x_max_cfg)))
    y_max = height if y_max_cfg is None else max(0, min(height, int(y_max_cfg)))
    mask = np.zeros((height, width), dtype=bool)
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = True
    for rect in roi_cfg.get("ignore_rectangles", []):
        rx = max(0, min(width, int(rect["x"])))
        ry = max(0, min(height, int(rect["y"])))
        rw = max(0, int(rect["w"]))
        rh = max(0, int(rect["h"]))
        mask[ry : min(height, ry + rh), rx : min(width, rx + rw)] = False
    return mask


def _filter_boxes(boxes: list[dict[str, Any]], filter_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
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


def run_pipeline(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    save_maps: bool = False,
    save_video: bool = False,
    config: dict[str, Any] | None = None,
    source_mode: str = "input",
) -> PipelineResult:
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dictionary or None")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer or None")
    cfg = _validate_config(deep_update(DEFAULT_CONFIG, config or {}))
    out_dir = Path(output_dir)
    input_resolved = Path(input_path).resolve()
    for name in ("overlay.mp4", "saliency.mp4", "boxes.json"):
        if input_resolved == (out_dir / name).resolve():
            raise ValueError("Input video must not be one of the output files")
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = _open_capture(input_path)

    overlay_writer = None
    saliency_writer = None
    try:
        fps = _capture_fps(capture)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Input video has no frames: {input_path}")

        height, width = frame.shape[:2]
        prev_gray = ensure_gray(frame)
        valid_roi = _roi_mask((height, width), cfg["roi"])
        fmhi = np.zeros_like(prev_gray, dtype=np.uint8)
        frames: list[dict[str, Any]] = [{"frame_index": 0, "boxes": []}]
        if save_video:
            overlay_writer = _create_writer(out_dir / "overlay.mp4", fps, (width, height))
            saliency_writer = _create_writer(out_dir / "saliency.mp4", fps, (width, height))
            overlay_writer.write(_draw_overlay(frame, [], 0))
            saliency_writer.write(np.zeros_like(frame))

        if save_maps:
            maps_dir = out_dir / "maps"
            zero = np.zeros_like(prev_gray)
            _save_maps(maps_dir, 0, diff=zero, fmhi=zero, pixel=zero, region=zero, fused=zero, mask=zero)

        frames_processed = 1
        frame_index = 1
        while True:
            if max_frames is not None and frames_processed >= int(max_frames):
                break
            ok, frame = capture.read()
            if not ok:
                break

            gray = ensure_gray(frame)
            if gray.shape != prev_gray.shape:
                raise ValueError("Input video dimensions changed during processing")
            registered_prev, _, _ = register_frame(gray, prev_gray, **cfg["registration"])
            diff = registered_difference(gray, registered_prev)
            fmhi = compute_fmhi(
                fmhi,
                diff,
                threshold=cfg["temporal"]["threshold"],
                decay=cfg["temporal"]["decay"],
            )
            fmhi = np.where(valid_roi, fmhi, 0).astype(np.uint8)
            motion_cfg = {key: cfg["temporal"][key] for key in ["candidate_threshold", "min_area", "max_area_ratio", "border_margin"]}
            candidates = motion_regions(fmhi, **motion_cfg)

            spatial_cfg = cfg["spatial"]
            pixel = (
                pixel_saliency_map(
                    gray,
                    candidates,
                    cr_factor=spatial_cfg["cr_factor"],
                    er_factor=spatial_cfg["er_factor"],
                )
                if spatial_cfg["pixel_enabled"]
                else np.zeros_like(fmhi, dtype=np.float32)
            )
            region = (
                region_saliency_map(
                    gray,
                    candidates,
                    cr_factor=spatial_cfg["cr_factor"],
                    quantization_levels=spatial_cfg["region_quantization_levels"],
                    hist_bins=spatial_cfg["region_hist_bins"],
                    grid_size=spatial_cfg["region_grid_size"],
                    min_patch_area=spatial_cfg["min_patch_area"],
                )
                if spatial_cfg["region_enabled"]
                else np.zeros_like(fmhi, dtype=np.float32)
            )
            fused = fuse_saliency(fmhi, pixel, region, **cfg["fusion"])
            fused = np.where(valid_roi, fused, 0.0).astype(np.float32)
            mask = threshold_saliency(fused, **cfg["thresholding"])
            mask = postprocess_mask(mask, morph_kernel=cfg["postprocess"]["morph_kernel"])
            boxes = boxes_from_mask(
                mask,
                fused,
                min_area=cfg["postprocess"]["min_area"],
                max_area_ratio=cfg["postprocess"]["max_area_ratio"],
            )
            boxes = _filter_boxes(boxes, cfg["box_filter"])
            if cfg["static_target"]["enabled"]:
                static_boxes = detect_static_dark_blobs(gray, valid_roi, **{k: v for k, v in cfg["static_target"].items() if k != "enabled"})
                boxes = boxes + _filter_boxes(static_boxes, cfg["box_filter"])
            boxes = deduplicate_boxes(boxes, center_distance=10.0)

            frames.append({"frame_index": frame_index, "boxes": boxes})
            if save_maps:
                _save_maps(out_dir / "maps", frame_index, diff=diff, fmhi=fmhi, pixel=pixel, region=region, fused=fused, mask=mask)
            if overlay_writer is not None and saliency_writer is not None:
                overlay_writer.write(_draw_overlay(frame, boxes, frame_index))
                saliency_writer.write(colorize_saliency(fused))

            prev_gray = gray
            frame_index += 1
            frames_processed += 1
    finally:
        capture.release()
        if overlay_writer is not None:
            overlay_writer.release()
        if saliency_writer is not None:
            saliency_writer.release()

    payload = {
        "schema_version": 1,
        "source": {"mode": source_mode, "path": str(input_path), "max_frames": max_frames},
        "config": cfg,
        "summary": {
            "frames_processed": frames_processed,
            "frames_with_boxes": sum(1 for item in frames if item["boxes"]),
            "total_boxes": sum(len(item["boxes"]) for item in frames),
        },
        "frames": frames,
    }
    boxes_path = out_dir / "boxes.json"
    boxes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PipelineResult(output_dir=out_dir, frames_processed=frames_processed, boxes=frames, boxes_path=boxes_path)
