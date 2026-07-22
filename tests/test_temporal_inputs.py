import numpy as np
import pytest
from vasts.temporal import compute_fmhi, registered_difference


def test_temporal_arithmetic_preserves_inputs_and_saturates():
    previous = np.array([[250, 1]], dtype=np.uint8)
    diff = np.array([[20, 0]], dtype=np.uint8)
    result = compute_fmhi(previous, diff, threshold=255, decay=2)
    np.testing.assert_array_equal(result, [[255, 0]])
    np.testing.assert_array_equal(previous, [[250, 1]])
    np.testing.assert_array_equal(diff, [[20, 0]])


@pytest.mark.parametrize("options", [{"threshold": -1}, {"threshold": 1.5}, {"decay": 256}, {"decay": True}])
def test_invalid_temporal_parameters_fail(options):
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError): compute_fmhi(image, image, **options)


def test_shapes_and_intensities_are_checked_before_arithmetic():
    image = np.zeros((2, 2), dtype=np.uint8)
    for function in (compute_fmhi, registered_difference):
        with pytest.raises(ValueError): function(image, np.zeros((1, 2), dtype=np.uint8))
        with pytest.raises(ValueError): function(image, np.full((2, 2), -1))
