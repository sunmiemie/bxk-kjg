import cv2
import numpy as np
import pytest

from vasts.lite_detector import V3LiteDetector


def _config(*, motion_enabled=False):
    return {
        "roi": {"enabled": True, "x_min": 60, "y_min": 30, "x_max": 110, "y_max": 60},
        "box_filter": {"min_width": 1, "min_height": 1, "max_width": 20, "max_height": 12, "min_score": 0.0},
        "static_target": {
            "enabled": not motion_enabled,
            "threshold_offset": 25,
            "min_area": 4,
            "max_area": 80,
            "min_width": 2,
            "min_height": 1,
            "max_width": 20,
            "max_height": 12,
        },
        "lite": {
            "frame_stride": 1,
            "hold_missing_frames": 0,
            "motion_enabled": motion_enabled,
            "motion_threshold": 8,
            "motion_min_area": 2,
            "motion_max_area": 100,
            "dedupe_center_distance": 3,
        },
    }


def _gray_frame(*, target_x=80):
    frame = np.full((100, 160), 180, dtype=np.uint8)
    cv2.rectangle(frame, (target_x, 42), (target_x + 10, 45), 35, -1)
    return frame


def test_realtime_detector_accepts_grayscale_frame_without_video_io():
    detector = V3LiteDetector(_config())

    result = detector.update(_gray_frame())

    assert result.frame_index == 0
    assert result.detection_performed is True
    assert len(result.boxes) == 1
    assert result.boxes[0]["source"] == "static"
    assert 79 <= result.boxes[0]["x"] <= 81
    assert 41 <= result.boxes[0]["y"] <= 43


def test_realtime_detector_keeps_previous_frame_for_motion_detection():
    detector = V3LiteDetector(_config(motion_enabled=True))

    first = detector.update(_gray_frame(target_x=70))
    second = detector.update(_gray_frame(target_x=74))

    assert first.boxes == []
    assert second.frame_index == 1
    assert second.boxes
    assert all(box["source"] == "motion" for box in second.boxes)


def test_realtime_detector_reset_clears_temporal_state():
    detector = V3LiteDetector(_config(motion_enabled=True))
    detector.update(_gray_frame(target_x=70))
    assert detector.update(_gray_frame(target_x=74)).boxes

    detector.reset()
    after_reset = detector.update(_gray_frame(target_x=74))

    assert after_reset.frame_index == 0
    assert after_reset.boxes == []
    assert detector.processed_detection_frames == 1


def test_realtime_detector_rejects_resolution_change_until_reset():
    detector = V3LiteDetector(_config())
    detector.update(_gray_frame())

    with pytest.raises(ValueError, match=r"call reset\(\)"):
        detector.update(np.full((120, 180), 180, dtype=np.uint8))


def test_realtime_detector_accepts_bgr_camera_frame():
    detector = V3LiteDetector(_config())
    bgr = cv2.cvtColor(_gray_frame(), cv2.COLOR_GRAY2BGR)

    result = detector.update(bgr)

    assert result.boxes
