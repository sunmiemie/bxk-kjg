import numpy as np
import pytest
from vasts.fusion import threshold_saliency


@pytest.mark.parametrize("method", ["fixed", "percentile", "otsu"])
def test_excluded_pixels_do_not_change_valid_threshold_decisions(method):
    saliency = np.array([[0, 1, 100], [2, 3, 100]], dtype=float)
    valid = np.array([[True, True, False], [True, True, False]])
    first = threshold_saliency(saliency, valid_mask=valid, method=method, percentile=50)
    saliency[:, 2] = -999
    second = threshold_saliency(saliency, valid_mask=valid, method=method, percentile=50)
    np.testing.assert_array_equal(first, second)
    assert not first[:, 2].any()
    assert first[1, 1] == 255


def test_empty_roi_stays_empty_even_at_zero_threshold():
    assert not threshold_saliency(np.ones((2, 2)), method="fixed", fixed=0, valid_mask=np.zeros((2, 2), dtype=bool)).any()


def test_wrong_mask_shape_fails():
    with pytest.raises(ValueError):
        threshold_saliency(np.ones((2, 2)), valid_mask=np.ones((1, 2), dtype=bool))
