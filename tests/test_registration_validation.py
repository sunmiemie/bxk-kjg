import numpy as np
import pytest

from vasts import registration


@pytest.mark.parametrize("shift,response", [((0, 0), np.nan), ((0, 0), np.inf), ((np.nan, 0), 1), ((1, 1), 0)])
def test_unreliable_phase_results_use_identity(monkeypatch, shift, response):
    monkeypatch.setattr(registration.cv2, "phaseCorrelate", lambda *args: (shift, response))
    image = np.zeros((16, 16), dtype=np.uint8)
    transform, method = registration._phase_fallback(image, image, .01)
    assert method == "identity"
    np.testing.assert_array_equal(transform, [[1, 0, 0], [0, 1, 0]])


def test_mismatched_frame_sizes_fail_before_feature_extraction():
    with pytest.raises(ValueError, match="equal dimensions"):
        registration.register_frame(np.zeros((16, 16), dtype=np.uint8), np.zeros((16, 8), dtype=np.uint8))


def test_valid_phase_translation_uses_inverse_shift(monkeypatch):
    monkeypatch.setattr(registration.cv2, "phaseCorrelate", lambda *args: ((3, -2), .9))
    image = np.zeros((16, 16), dtype=np.uint8)
    transform, method = registration._phase_fallback(image, image, .01)
    assert method == "phase"
    np.testing.assert_array_equal(transform, [[1, 0, -3], [0, 1, 2]])
