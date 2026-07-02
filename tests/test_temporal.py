import numpy as np

from vasts.temporal import compute_fmhi


def test_fmhi_sets_motion_pixels_and_decays_quiet_pixels():
    previous = np.array([[100, 20], [0, 250]], dtype=np.uint8)
    diff = np.array([[10, 40], [0, 34]], dtype=np.uint8)

    fmhi = compute_fmhi(previous, diff, threshold=35, decay=25)

    expected = np.array([[85, 255], [0, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(fmhi, expected)
