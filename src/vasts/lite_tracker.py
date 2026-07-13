from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _center(box: dict[str, Any]) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2.0, float(box["y"]) + float(box["h"]) / 2.0


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return math.hypot(lx - rx, ly - ry)


def _area(box: dict[str, Any]) -> float:
    return max(1.0, float(box["w"]) * float(box["h"]))


def _area_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_area = _area(left)
    right_area = _area(right)
    return max(left_area, right_area) / min(left_area, right_area)


def _copy_box(box: dict[str, Any]) -> dict[str, Any]:
    copied = dict(box)
    copied["x"] = int(round(float(copied["x"])))
    copied["y"] = int(round(float(copied["y"])))
    copied["w"] = max(1, int(round(float(copied["w"]))))
    copied["h"] = max(1, int(round(float(copied["h"]))))
    copied["score"] = round(float(copied.get("score", 0.0)), 4)
    return copied


@dataclass
class _Track:
    track_id: int
    box: dict[str, Any]
    hits: int = 1
    missing: int = 0


class TrajectoryFilter:
    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_missing_frames: int = 3,
        max_center_distance: float = 18.0,
        max_area_ratio_change: float = 4.0,
        single_target: bool = False,
    ) -> None:
        self.min_hits = max(1, int(min_hits))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.max_center_distance = float(max_center_distance)
        self.max_area_ratio_change = float(max_area_ratio_change)
        self.single_target = bool(single_target)
        self._next_track_id = 1
        self._tracks: list[_Track] = []

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        detections = [_copy_box(box) for box in detections]
        original_track_count = len(self._tracks)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        ranked_detections = sorted(
            enumerate(detections),
            key=lambda item: float(item[1].get("score", 0.0)),
            reverse=True,
        )
        for detection_index, detection in ranked_detections:
            best_track_index = None
            best_distance = float("inf")
            for track_index, track in enumerate(self._tracks):
                if track_index in matched_tracks:
                    continue
                distance = _distance(track.box, detection)
                if distance > self.max_center_distance:
                    continue
                if _area_ratio(track.box, detection) > self.max_area_ratio_change:
                    continue
                if distance < best_distance:
                    best_track_index = track_index
                    best_distance = distance
            if best_track_index is None:
                continue
            track = self._tracks[best_track_index]
            track.box = detection
            track.hits += 1
            track.missing = 0
            matched_tracks.add(best_track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            self._tracks.append(_Track(track_id=self._next_track_id, box=detection))
            self._next_track_id += 1

        for track_index in range(original_track_count):
            if track_index not in matched_tracks:
                self._tracks[track_index].missing += 1
        self._tracks = [track for track in self._tracks if track.missing <= self.max_missing_frames]

        outputs = [self._to_output_box(track) for track in self._tracks if track.hits >= self.min_hits]
        if self.single_target and outputs:
            outputs = [max(outputs, key=lambda box: (int(box["trajectory_hits"]), float(box["score"]), -int(box["trajectory_missing"])))]
        return outputs

    def _to_output_box(self, track: _Track) -> dict[str, Any]:
        output = _copy_box(track.box)
        output["track_id"] = track.track_id
        output["trajectory_hits"] = track.hits
        output["trajectory_missing"] = track.missing
        output["trajectory_confirmed"] = True
        if track.missing > 0:
            output["trajectory_held"] = True
        return output
