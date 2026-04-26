import numpy as np
import pytest

from vasts.fusion import fuse_saliency, threshold_saliency
from vasts.image_utils import ensure_gray, normalize01


@pytest.mark.parametrize("array", [np.array([[-1]]), np.array([[256]]), np.array([[np.nan]]),
                                    np.array([[np.inf]]), np.empty((0, 2)), np.array([[1j]])])
def test_invalid_frame_values_fail_instead_of_wrapping(array):
    with pytest.raises(ValueError):
        ensure_gray(array)


def test_float64_bgr_is_converted_before_opencv():
    frame = np.full((2, 3, 3), 128.0, dtype=np.float64)
    result = ensure_gray(frame)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.full((2, 3), 128, dtype=np.uint8))


def test_extreme_finite_saliency_normalizes_without_overflow():
    result = normalize01(np.array([[-1e308, 0, 1e308]]))
    np.testing.assert_allclose(result, [[0, .5, 1]])
    assert result.dtype == np.float32


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_saliency_fails(value):
    with pytest.raises(ValueError, match="finite"):
        normalize01(np.array([[0, value]]))


def test_fusion_rejects_broadcast_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        fuse_saliency(np.zeros((3, 3)), np.zeros((3, 1)), np.zeros((3, 3)))


@pytest.mark.parametrize("method", ["otsu", "fixed", "percentile"])
def test_empty_threshold_maps_remain_empty(method):
    result = threshold_saliency(np.empty((0, 0)), method=method)
    assert result.shape == (0, 0) and result.dtype == np.uint8


def test_threshold_and_fusion_reject_nonfinite_parameters():
    zeros = np.zeros((2, 2))
    with pytest.raises(ValueError):
        threshold_saliency(zeros, percentile=np.nan)
    with pytest.raises(ValueError):
        fuse_saliency(zeros, zeros, zeros, pixel_weight=np.nan)
