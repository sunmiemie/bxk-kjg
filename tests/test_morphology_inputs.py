import numpy as np
import pytest
from vasts.config import load_config
from vasts.fusion import postprocess_mask


def test_binary_masks_have_consistent_uint8_output_and_no_mutation():
    mask = np.array([[True, False], [False, True]])
    result = postprocess_mask(mask, morph_kernel=1)
    np.testing.assert_array_equal(result, [[255, 0], [0, 255]])
    assert mask.dtype == np.bool_
    assert postprocess_mask(np.empty((0, 0)), morph_kernel=3).size == 0


@pytest.mark.parametrize("kernel", [0, 2, 1.5, True])
def test_ambiguous_kernel_settings_fail(kernel):
    with pytest.raises(ValueError):
        postprocess_mask(np.ones((4, 4)), morph_kernel=kernel)


def test_nonfinite_masks_and_even_config_fail():
    with pytest.raises(ValueError):
        postprocess_mask(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="odd"):
        load_config('{"postprocess":{"morph_kernel":2}}')
