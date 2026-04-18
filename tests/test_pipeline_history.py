import numpy as np

from vasts import pipeline


def test_fmhi_is_warped_to_current_camera_coordinates(tmp_path, monkeypatch):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    class Capture:
        remaining = 3
        def get(self, _):
            return 10
        def read(self):
            self.remaining -= 1
            return (True, frame.copy()) if self.remaining >= 0 else (False, None)
        def release(self):
            pass
    histories = []
    def update(previous, diff, **kwargs):
        histories.append(previous.copy())
        result = np.zeros_like(previous)
        result[8, 5] = 200
        return result
    transform = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32)
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: Capture())
    monkeypatch.setattr(pipeline, "register_frame", lambda reference, moving, **kwargs: (moving, transform, "phase"))
    monkeypatch.setattr(pipeline, "compute_fmhi", update)
    pipeline.run_pipeline(input_path="synthetic.mp4", output_dir=tmp_path)
    assert len(histories) == 2
    assert histories[1][8, 7] == 200
    assert histories[1][8, 5] == 0
    assert not histories[1][:, :2].any()
