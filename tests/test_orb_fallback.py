from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from vasts import registration


def test_orb_backend_failure_uses_phase_fallback(monkeypatch):
    def fail(*args, **kwargs): raise cv2.error("simulated ORB failure")
    monkeypatch.setattr(registration.cv2, "ORB_create", fail)
    monkeypatch.setattr(registration.cv2, "phaseCorrelate", lambda *args: ((0, 0), .9))
    image = np.zeros((16, 16), dtype=np.uint8)
    _, transform, method = registration.register_frame(image, image)
    assert method == "phase" and np.isfinite(transform).all()


def test_one_match_never_attempts_affine_estimation(monkeypatch):
    class Orb:
        def detectAndCompute(self, *args):
            return [SimpleNamespace(pt=(1, 1))], np.zeros((1, 32), dtype=np.uint8)
    monkeypatch.setattr(registration.cv2, "ORB_create", lambda **kwargs: Orb())
    monkeypatch.setattr(registration.cv2, "estimateAffinePartial2D", lambda *args, **kwargs: pytest.fail("insufficient correspondences"))
    image = np.zeros((16, 16), dtype=np.uint8)
    assert registration._orb_transform(image, image, 10, 1, 3) is None


@pytest.mark.parametrize("options", [{"orb_features": 1.5}, {"min_matches": True}, {"phase_response_threshold": np.nan}])
def test_invalid_registration_options_fail(options):
    image = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError):
        registration.register_frame(image, image, **options)
