import numpy as np

from vasts.registration import register_frame


def test_registration_falls_back_on_blank_frames_and_reports_fallback():
    reference = np.zeros((48, 64), dtype=np.uint8)
    moving = np.zeros((48, 64), dtype=np.uint8)
    moving[20:24, 30:34] = 255

    registered, transform, method = register_frame(reference, moving)

    assert registered.shape == reference.shape
    assert transform.shape == (2, 3)
    assert method in {"phase", "identity"}
    assert np.isfinite(transform).all()
