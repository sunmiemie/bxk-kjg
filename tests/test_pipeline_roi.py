import numpy as np

from vasts import pipeline


def test_threshold_and_morphology_cannot_refill_ignored_roi_pixels(tmp_path, monkeypatch):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    class Capture:
        remaining = 2
        def get(self, _):
            return 10
        def read(self):
            self.remaining -= 1
            return (True, frame.copy()) if self.remaining >= 0 else (False, None)
        def release(self):
            pass
    masks = []
    def boxes(mask, *args, **kwargs):
        masks.append(mask.copy())
        return []
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    monkeypatch.setattr(pipeline, "boxes_from_mask", boxes)
    pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path,
                          config={"roi": {"enabled": True, "x_min": 4, "y_min": 4, "x_max": 28, "y_max": 28,
                                          "ignore_rectangles": [{"x": 15, "y": 15, "w": 1, "h": 1}]},
                                  "thresholding": {"method": "fixed", "fixed": 0}})
    assert len(masks) == 1
    assert masks[0][10, 10] == 255
    assert masks[0][15, 15] == 0
    assert not masks[0][:4].any()
    assert not masks[0][:, :4].any()
    assert not masks[0][28:].any()
    assert not masks[0][:, 28:].any()
