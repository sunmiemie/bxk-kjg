import numpy as np
import pytest
from vasts.regions import expand_box, boxes_from_mask


def test_outside_expansion_has_in_bounds_empty_origin():
    assert expand_box([30, 30, 2, 2], factor=2, image_shape=(10, 10)) == [10, 10, 0, 0]
    assert expand_box([-30, -30, 2, 2], factor=2, image_shape=(10, 10)) == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        expand_box([1, 1, -2, 3], factor=2, image_shape=(10, 10))


def test_component_scores_exclude_background_and_empty_masks_are_safe():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1:3] = 255
    saliency = np.zeros((4, 4), dtype=float)
    saliency[1, 1:3] = [.6, .8]
    assert boxes_from_mask(mask, saliency, min_area=2, max_area_ratio=1) == [{"x": 1, "y": 1, "w": 2, "h": 1, "score": .7}]
    assert boxes_from_mask(np.empty((0, 0)), np.empty((0, 0))) == []


def test_mismatched_components_and_nonfinite_scores_fail():
    with pytest.raises(ValueError):
        boxes_from_mask(np.ones((2, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError):
        boxes_from_mask(np.ones((2, 2)), np.full((2, 2), np.nan))
