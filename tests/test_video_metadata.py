import json
import cv2
import numpy as np
from vasts import pipeline


def test_source_metadata_and_estimated_timestamps_are_explicit(tmp_path, monkeypatch):
    class Capture:
        reads = 0
        def get(self, property):
            return 99 if property == cv2.CAP_PROP_FRAME_COUNT else float("nan")
        def read(self):
            self.reads += 1
            return (True, np.zeros((24, 32, 3), dtype=np.uint8)) if self.reads <= 2 else (False, None)
        def release(self): pass
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    result = pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path)
    payload = json.loads(result.boxes_path.read_text())
    assert payload["source"]["reported_frame_count"] == 99
    assert payload["summary"]["frames_processed"] == 2
    assert payload["source"]["width"] == 32
    assert payload["source"]["fps_fallback_used"] is True
    assert payload["source"]["reported_fps"] is None
    assert payload["frames"][1]["timestamp_seconds"] == .1
    assert sum(payload["summary"]["registration_methods"].values()) == 1
