import cv2
import numpy as np
import pytest
from vasts import image_utils
from vasts.pipeline import _save_maps


def test_saved_maps_round_trip_in_unicode_directory(tmp_path):
    directory = tmp_path / "地图输出"
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    _save_maps(directory, 1, diff=image, fmhi=image, pixel=image, region=image, fused=image, mask=image)
    path = directory / "frame_00001_diff.png"
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(decoded, image)
    assert len(list(directory.glob("*.png"))) == 6


def test_failed_encode_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(image_utils.cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="encode PNG"):
        image_utils.write_png(tmp_path / "map.png", np.zeros((2, 2), dtype=np.uint8))
    assert not (tmp_path / "map.png").exists()
