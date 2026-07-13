import json

import cv2
import numpy as np

from vasts.lite_pipeline import run_lite_pipeline


def _write_hover_video(path, *, frames=8, size=(160, 100), roi=(60, 30, 50, 30)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    x, y, _, _ = roi
    try:
        for _ in range(frames):
            frame = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
            cv2.rectangle(frame, (x + 20, y + 12), (x + 30, y + 15), (35, 35, 35), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_motion_video(path, *, frames=8, size=(160, 100), roi=(60, 30, 50, 30)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    x, y, _, _ = roi
    try:
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
            cv2.rectangle(frame, (x + 8 + index, y + 12), (x + 16 + index, y + 15), (40, 40, 40), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_flicker_and_stable_video(path, *, frames=7, size=(160, 100), roi=(60, 30, 50, 30)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    assert writer.isOpened()
    x, y, _, _ = roi
    try:
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
            cv2.rectangle(frame, (x + 20, y + 12), (x + 30, y + 15), (35, 35, 35), -1)
            if index == 0:
                cv2.rectangle(frame, (x + 42, y + 5), (x + 46, y + 8), (35, 35, 35), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_lite_pipeline_detects_static_dark_target_and_maps_to_full_frame(tmp_path):
    video = tmp_path / "hover.mp4"
    _write_hover_video(video)

    result = run_lite_pipeline(
        input_path=video,
        output_dir=tmp_path / "out",
        config={
            "roi": {"enabled": True, "x_min": 60, "y_min": 30, "x_max": 110, "y_max": 60},
            "static_target": {"enabled": True, "threshold_offset": 25, "min_area": 4, "max_area": 80, "min_width": 2, "min_height": 1, "max_width": 20, "max_height": 12},
            "lite": {"frame_stride": 1, "hold_missing_frames": 0, "motion_enabled": False},
        },
    )

    assert result.frames_processed == 8
    payload = json.loads(result.boxes_path.read_text(encoding="utf-8"))
    boxed = [frame for frame in payload["frames"] if frame["boxes"]]
    assert len(boxed) >= 7
    first_box = boxed[0]["boxes"][0]
    assert first_box["source"] == "static"
    assert 78 <= first_box["x"] <= 82
    assert 41 <= first_box["y"] <= 43


def test_lite_pipeline_detects_motion_target_with_simple_frame_difference(tmp_path):
    video = tmp_path / "motion.mp4"
    _write_motion_video(video)

    result = run_lite_pipeline(
        input_path=video,
        output_dir=tmp_path / "out",
        config={
            "roi": {"enabled": True, "x_min": 60, "y_min": 30, "x_max": 110, "y_max": 60},
            "static_target": {"enabled": False},
            "lite": {"frame_stride": 1, "hold_missing_frames": 0, "motion_enabled": True, "motion_threshold": 8, "motion_min_area": 2, "motion_max_area": 100},
        },
    )

    payload = json.loads(result.boxes_path.read_text(encoding="utf-8"))
    motion_frames = [frame for frame in payload["frames"] if frame["boxes"]]
    assert len(motion_frames) >= 5
    assert all(frame["boxes"][0]["source"] == "motion" for frame in motion_frames[:3])


def test_lite_pipeline_frame_stride_holds_recent_detection(tmp_path):
    video = tmp_path / "hover.mp4"
    _write_hover_video(video, frames=9)

    result = run_lite_pipeline(
        input_path=video,
        output_dir=tmp_path / "out",
        config={
            "roi": {"enabled": True, "x_min": 60, "y_min": 30, "x_max": 110, "y_max": 60},
            "static_target": {"enabled": True, "threshold_offset": 25, "min_area": 4, "max_area": 80, "min_width": 2, "min_height": 1, "max_width": 20, "max_height": 12},
            "lite": {"frame_stride": 3, "hold_missing_frames": 2, "motion_enabled": False},
        },
    )

    payload = json.loads(result.boxes_path.read_text(encoding="utf-8"))
    assert result.frames_processed == 9
    assert payload["summary"]["processed_detection_frames"] == 3
    assert payload["summary"]["held_boxes"] >= 4
    assert len([frame for frame in payload["frames"] if frame["boxes"]]) >= 7


def test_lite_pipeline_trajectory_filter_confirms_stable_target_and_drops_flicker(tmp_path):
    video = tmp_path / "stable_with_flicker.mp4"
    _write_flicker_and_stable_video(video)

    result = run_lite_pipeline(
        input_path=video,
        output_dir=tmp_path / "out",
        config={
            "roi": {"enabled": True, "x_min": 60, "y_min": 30, "x_max": 110, "y_max": 60},
            "static_target": {"enabled": True, "threshold_offset": 25, "min_area": 4, "max_area": 80, "min_width": 2, "min_height": 1, "max_width": 20, "max_height": 12},
            "lite": {"frame_stride": 1, "hold_missing_frames": 0, "motion_enabled": False, "dedupe_center_distance": 3},
            "trajectory_filter": {"enabled": True, "min_hits": 3, "max_missing_frames": 2, "max_center_distance": 8, "max_area_ratio_change": 4, "single_target": True},
        },
    )

    payload = json.loads(result.boxes_path.read_text(encoding="utf-8"))
    boxed = [frame for frame in payload["frames"] if frame["boxes"]]
    assert len(boxed) >= 4
    assert payload["frames"][0]["boxes"] == []
    first_box = boxed[0]["boxes"][0]
    assert first_box["trajectory_confirmed"] is True
    assert first_box["trajectory_hits"] >= 3
    assert 78 <= first_box["x"] <= 82
    assert first_box["track_id"] >= 1
