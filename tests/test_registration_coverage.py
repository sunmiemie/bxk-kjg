import numpy as np
import cv2
from vasts import pipeline
from vasts.registration import valid_registration_mask


def test_translation_and_subpixel_support_masks():
    mask = valid_registration_mask((8, 10), np.array([[1, 0, 2], [0, 1, 0]], dtype=float))
    assert not mask[:, :2].any() and mask[:, 2:].all()
    subpixel = valid_registration_mask((8, 10), np.array([[1, 0, .5], [0, 1, 0]], dtype=float))
    assert not subpixel[:, 0].any() and subpixel[:, 1:].all()


def test_uncovered_edges_do_not_enter_frame_difference(tmp_path, monkeypatch):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    class Capture:
        remaining = 2
        def get(self, _): return 10
        def read(self):
            self.remaining -= 1
            return (True, frame) if self.remaining >= 0 else (False, None)
        def release(self): pass
    captured = []
    def update(previous, diff, **kwargs):
        captured.append(diff.copy())
        return np.zeros_like(previous)
    transform = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32)
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    monkeypatch.setattr(pipeline, "register_frame", lambda reference, moving, **kwargs: (np.full((16, 16), 255, dtype=np.uint8), transform, "phase"))
    monkeypatch.setattr(pipeline, "compute_fmhi", update)
    pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path)
    assert not captured[0][:, :2].any()
    assert (captured[0][:, 2:] == 255).all()


def test_morphology_cannot_refill_unsupported_rotated_borders(tmp_path, monkeypatch):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    class Capture:
        remaining = 2
        def get(self, _): return 10
        def read(self):
            self.remaining -= 1
            return (True, frame) if self.remaining >= 0 else (False, None)
        def release(self): pass
    transform = cv2.getRotationMatrix2D((15.5, 15.5), 5, 1)
    valid = valid_registration_mask((32, 32), transform)
    masks = []
    def boxes(mask, *args, **kwargs):
        masks.append(mask.copy())
        return []
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    monkeypatch.setattr(pipeline, "register_frame", lambda reference, moving, **kwargs: (moving, transform, "orb"))
    monkeypatch.setattr(pipeline, "boxes_from_mask", boxes)
    pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path,
                          config={"thresholding": {"method": "fixed", "fixed": 0}})
    assert masks and masks[0].any()
    assert not masks[0][~valid].any()
