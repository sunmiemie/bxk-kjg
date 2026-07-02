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
        box[key] = float(left[key]) + (float(right[key]) - float(left[key])) * alpha
    return _copy_track_box(box, track_id=track_id, interpolated=True)


def _best_box(boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda item: float(item.get("score", 0.0)))


def _best_observations(frames: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    observations = [(int(frame["frame_index"]), _best_box(frame.get("boxes", []))) for frame in frames]
    return [(frame_index, box) for frame_index, box in observations if box is not None]


def fill_single_track_gaps(
    payload: dict[str, Any],
    *,
    max_gap: int = 10,
    max_distance: float = 80.0,
) -> dict[str, Any]:
    frames = copy.deepcopy(payload.get("frames", []))
    observations = _best_observations(frames)

    output = copy.deepcopy(payload)
    if not observations:
        output["frames"] = [{"frame_index": int(frame["frame_index"]), "boxes": []} for frame in frames]
        output["summary"] = {
            "frames_processed": len(frames),
            "frames_with_boxes": 0,
            "total_boxes": 0,
            "interpolated_boxes": 0,
        }
        return output

    by_frame: dict[int, dict[str, Any]] = {
        frame_index: _copy_track_box(box, track_id=1, interpolated=False) for frame_index, box in observations
    }
    interpolated_count = 0
    for (left_frame, left_box), (right_frame, right_box) in zip(observations, observations[1:]):
        gap = right_frame - left_frame - 1
        if gap <= 0 or gap > max_gap:
            continue
        if _distance(left_box, right_box) > max_distance:
            continue
        for step in range(1, gap + 1):
            frame_index = left_frame + step
            alpha = step / float(gap + 1)
            by_frame[frame_index] = _interpolate_box(left_box, right_box, alpha, track_id=1)
            interpolated_count += 1

    tracked_frames = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        box = by_frame.get(frame_index)
        tracked_frames.append({"frame_index": frame_index, "boxes": [box] if box is not None else []})

    output["frames"] = tracked_frames
    output["summary"] = {
        "frames_processed": len(tracked_frames),
        "frames_with_boxes": sum(1 for frame in tracked_frames if frame["boxes"]),
        "total_boxes": sum(len(frame["boxes"]) for frame in tracked_frames),
        "interpolated_boxes": interpolated_count,
        "track_observations": len(observations),
        "max_gap": int(max_gap),
        "max_distance": float(max_distance),
    }
    return output
