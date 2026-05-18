import numpy as np
import pytest
from vasts import synthetic


@pytest.mark.parametrize("options", [{"frame_count": True}, {"frame_count": 1.5}, {"size": (31, 24)},
                                     {"size": (10, 8)}, {"target_size": (0, 2)}, {"fps": np.nan}])
def test_invalid_synthetic_settings_fail_before_output(tmp_path, options):
    with pytest.raises(ValueError):
        synthetic.make_synthetic_video(tmp_path / "output" / "video.mp4", **options)
    assert not (tmp_path / "output").exists()


def test_target_dimensions_do_not_gain_an_inclusive_pixel(tmp_path, monkeypatch):
    frames = []
    class Writer:
        def isOpened(self): return True
        def write(self, frame): frames.append(frame.copy())
        def release(self): pass
    monkeypatch.setattr(synthetic.cv2, "VideoWriter", lambda *args: Writer())
    synthetic.make_synthetic_video(tmp_path / "video.mp4", frame_count=1, size=(32, 32), target_size=(4, 3))
    ys, xs = np.where(frames[0][:, :, 0] >= 235)
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (4, 3)
