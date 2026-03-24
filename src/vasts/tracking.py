from __future__ import annotations

import copy
import math
from typing import Any


def _center(box: dict[str, Any]) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _copy_track_box(box: dict[str, Any], *, track_id: int, interpolated: bool = False) -> dict[str, Any]:
    return {
        **copy.deepcopy(box),
        "x": int(round(float(box["x"]))),
        "y": int(round(float(box["y"]))),
        "w": max(1, int(round(float(box["w"])))),
        "h": max(1, int(round(float(box["h"])))),
        "score": round(float(box.get("score", 0.0)), 4),
        "track_id": track_id,
        "interpolated": interpolated,
    }


def _interpolate_box(left: dict[str, Any], right: dict[str, Any], alpha: float, *, track_id: int) -> dict[str, Any]:
    box = {}
    for key in ["x", "y", "w", "h", "score"]:
        box[key] = float(left.get(key, 0.0)) + (float(right.get(key, 0.0)) - float(left.get(key, 0.0))) * alpha
    return _copy_track_box(box, track_id=track_id, interpolated=True)


def _best_box(boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda item: float(item.get("score", 0.0)))


def _best_observations(frames: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    observations = [(int(frame["frame_index"]), _best_box(frame.get("boxes", []))) for frame in frames]
    return [(frame_index, box) for frame_index, box in observations if box is not None]


def _validate_payload(payload: dict[str, Any], max_gap: int, max_distance: float) -> None:
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    if isinstance(max_distance, bool) or not isinstance(max_distance, (int, float)) or not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError("payload must contain a frames array")
    previous_index = -1
    for frame in payload["frames"]:
        if not isinstance(frame, dict):
            raise ValueError("each frame must be an object")
        index = frame.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index <= previous_index:
            raise ValueError("frame indices must be nonnegative, unique and strictly increasing")
        previous_index = index
        if not isinstance(frame.get("boxes", []), list):
            raise ValueError("frame boxes must be an array")
        for box in frame.get("boxes", []):
            if not isinstance(box, dict):
                raise ValueError("each box must be an object")
            for key in ("x", "y", "w", "h", "score"):
                value = box.get(key, 0.0 if key == "score" else None)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"box {key} must be finite")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"box {key} must be finite")
                if key in ("w", "h") and value <= 0:
                    raise ValueError("box width and height must be positive")
                if key == "score" and not 0 <= value <= 1:
                    raise ValueError("box score must be in [0, 1]")


def fill_single_track_gaps(
    payload: dict[str, Any],
    *,
    max_gap: int = 10,
    max_distance: float = 80.0,
) -> dict[str, Any]:
    """Keep the strongest box per supplied frame and fill short represented gaps.

    Missing frame rows are not created. This is a single-target heuristic, not
    multi-object tracking; frame and observed-box metadata are preserved.
    """
    _validate_payload(payload, max_gap, max_distance)
    frames = copy.deepcopy(payload.get("frames", []))
    observations = _best_observations(frames)

    output = copy.deepcopy(payload)
    by_frame: dict[int, dict[str, Any]] = {
        frame_index: _copy_track_box(box, track_id=1, interpolated=False) for frame_index, box in observations
    }
    represented_frames = {frame["frame_index"] for frame in frames}
    for (left_frame, left_box), (right_frame, right_box) in zip(observations, observations[1:]):
        gap = right_frame - left_frame - 1
        if gap <= 0 or gap > max_gap:
            continue
        if _distance(left_box, right_box) > max_distance:
            continue
        for step in range(1, gap + 1):
            frame_index = left_frame + step
            if frame_index not in represented_frames:
                continue
            alpha = step / float(gap + 1)
            by_frame[frame_index] = _interpolate_box(left_box, right_box, alpha, track_id=1)

    tracked_frames = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        box = by_frame.get(frame_index)
        tracked_frames.append({**frame, "frame_index": frame_index, "boxes": [box] if box is not None else []})

    output["frames"] = tracked_frames
    output["summary"] = {
        "frames_processed": len(tracked_frames),
        "frames_with_boxes": sum(1 for frame in tracked_frames if frame["boxes"]),
        "total_boxes": sum(len(frame["boxes"]) for frame in tracked_frames),
        "interpolated_boxes": sum(box["interpolated"] for frame in tracked_frames for box in frame["boxes"]),
        "track_observations": len(observations),
        "max_gap": int(max_gap),
        "max_distance": float(max_distance),
    }
    return output
