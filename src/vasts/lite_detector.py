from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, deep_update
from .image_utils import ensure_gray
from .lite_tracker import TrajectoryFilter
from .regions import deduplicate_boxes
from .static_target import detect_static_dark_blobs


@dataclass(frozen=True)
class LiteFrameResult:
    frame_index: int
    boxes: list[dict[str, Any]]
    detection_performed: bool
    held_boxes: int


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
    binary = ((diff >= float(threshold)) & roi_mask).astype(np.uint8)
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


class V3LiteDetector:
    """Stateful v3-lite detector for images supplied one frame at a time."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deep_update(DEFAULT_CONFIG, config or {})
        self._frame_stride = max(1, int(self.config["lite"]["frame_stride"]))
        self._hold_missing_frames = max(0, int(self.config["lite"]["hold_missing_frames"]))
        self.reset()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def processed_detection_frames(self) -> int:
        return self._processed_detection_frames

    @property
    def held_boxes(self) -> int:
        return self._held_count

    def reset(self) -> None:
        self._frame_index = 0
        self._processed_detection_frames = 0
        self._held_count = 0
        self._frame_shape: tuple[int, int] | None = None
        self._roi_bounds: tuple[int, int, int, int] | None = None
        self._roi_mask: np.ndarray | None = None
        self._previous_roi_gray: np.ndarray | None = None
        self._last_boxes: list[dict[str, Any]] = []
        self._last_detection_frame = -10**9
        self._trajectory_filter = self._create_trajectory_filter()

    def update(self, frame: np.ndarray) -> LiteFrameResult:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy array")
        if frame.ndim not in {2, 3}:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
        if self._frame_shape is None:
            self._initialize_geometry(frame_shape)
        elif frame_shape != self._frame_shape:
            raise ValueError(
                f"Frame size changed from {self._frame_shape[1]}x{self._frame_shape[0]} "
                f"to {frame_shape[1]}x{frame_shape[0]}; call reset() before changing resolution"
            )

        assert self._roi_bounds is not None
        assert self._roi_mask is not None
        roi_x0, roi_y0, roi_x1, roi_y1 = self._roi_bounds
        roi_gray = ensure_gray(frame[roi_y0:roi_y1, roi_x0:roi_x1])
        frame_index = self._frame_index
        boxes: list[dict[str, Any]] = []
        should_detect = frame_index % self._frame_stride == 0

        if should_detect:
            boxes = self._detect_candidates(roi_gray)
            boxes = _map_boxes_to_full_frame(boxes, (roi_x0, roi_y0))
            if self._trajectory_filter is not None:
                boxes = self._trajectory_filter.update(boxes)
                self._held_count += self._count_trajectory_held(boxes)
            self._processed_detection_frames += 1
            self._previous_roi_gray = roi_gray.copy()
            if boxes:
                self._last_boxes = boxes
                self._last_detection_frame = frame_index
        elif self._trajectory_filter is not None:
            boxes = self._trajectory_filter.update([])
            self._held_count += self._count_trajectory_held(boxes)
        elif self._last_boxes and frame_index - self._last_detection_frame <= self._hold_missing_frames:
            boxes = _held_boxes(self._last_boxes)
            self._held_count += len(boxes)

        result = LiteFrameResult(
            frame_index=frame_index,
            boxes=boxes,
            detection_performed=should_detect,
            held_boxes=sum(1 for box in boxes if box.get("held") or box.get("trajectory_held")),
        )
        self._frame_index += 1
        return result

    def _initialize_geometry(self, frame_shape: tuple[int, int]) -> None:
        self._frame_shape = frame_shape
        self._roi_bounds = _roi_bounds(frame_shape, self.config["roi"])
        roi_x0, roi_y0, roi_x1, roi_y1 = self._roi_bounds
        roi_shape = (roi_y1 - roi_y0, roi_x1 - roi_x0)
        self._roi_mask = _local_roi_mask(roi_shape, self.config["roi"], (roi_x0, roi_y0))

    def _detect_candidates(self, roi_gray: np.ndarray) -> list[dict[str, Any]]:
        assert self._roi_mask is not None
        boxes: list[dict[str, Any]] = []
        if bool(self.config["static_target"]["enabled"]):
            static_cfg = {key: value for key, value in self.config["static_target"].items() if key != "enabled"}
            boxes.extend(detect_static_dark_blobs(roi_gray, self._roi_mask, **static_cfg))
        if bool(self.config["lite"]["motion_enabled"]):
            boxes.extend(
                _motion_boxes(
                    roi_gray,
                    self._previous_roi_gray,
                    self._roi_mask,
                    threshold=self.config["lite"]["motion_threshold"],
                    min_area=self.config["lite"]["motion_min_area"],
                    max_area=self.config["lite"]["motion_max_area"],
                )
            )
        boxes = _filter_boxes(boxes, self.config["box_filter"])
        return deduplicate_boxes(boxes, center_distance=float(self.config["lite"]["dedupe_center_distance"]))

    def _create_trajectory_filter(self) -> TrajectoryFilter | None:
        tracker_cfg = self.config["trajectory_filter"]
        if not bool(tracker_cfg["enabled"]):
            return None
        return TrajectoryFilter(
            min_hits=int(tracker_cfg["min_hits"]),
            max_missing_frames=int(tracker_cfg["max_missing_frames"]),
            max_center_distance=float(tracker_cfg["max_center_distance"]),
            max_area_ratio_change=float(tracker_cfg["max_area_ratio_change"]),
            single_target=bool(tracker_cfg["single_target"]),
        )

    @staticmethod
    def _count_trajectory_held(boxes: list[dict[str, Any]]) -> int:
        return sum(1 for box in boxes if bool(box.get("trajectory_held", False)))
