import numpy as np
import pytest

from vasts import pipeline
from vasts.static_target import detect_static_dark_blobs


def test_uniform_black_small_roi_has_no_zero_contrast_detection():
    frame = np.zeros((8, 8), dtype=np.uint8)
    assert detect_static_dark_blobs(frame, np.ones_like(frame, dtype=bool)) == []


def test_invalid_static_roi_and_threshold_are_actionable():
    frame = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="same 2D shape"):
        detect_static_dark_blobs(frame, np.zeros((8, 1), dtype=bool))
    with pytest.raises(ValueError, match="finite and positive"):
        detect_static_dark_blobs(frame, np.ones_like(frame), threshold_offset=np.nan)


def test_single_frame_video_runs_enabled_static_detector(tmp_path, monkeypatch):
    frame = np.full((32, 32, 3), 180, dtype=np.uint8)
    frame[14:18, 14:20] = 30
    class Capture:
        read_once = False
        def get(self, _):
            return 10
        def read(self):
            if self.read_once:
                return False, None
            self.read_once = True
            return True, frame.copy()
        def release(self):
            pass
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    result = pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path,
                                   config={"static_target": {"enabled": True}})
    assert result.frames_processed == 1
    assert len(result.boxes[0]["boxes"]) == 1
    assert result.boxes[0]["boxes"][0]["source"] == "static"
