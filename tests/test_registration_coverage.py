import numpy as np
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
