import numpy as np
import pytest
from vasts.temporal import motion_regions


def test_single_pixel_and_thin_targets_follow_area_rule():
    history = np.zeros((10, 10), dtype=np.uint8)
    history[1, 1] = 255
    history[4:8, 5] = 255
    assert motion_regions(history, min_area=1, max_area_ratio=1, border_margin=1) == [[1, 1, 1, 1], [5, 4, 1, 4]]
    assert motion_regions(history, min_area=2, max_area_ratio=1) == [[5, 4, 1, 4]]


def test_zero_threshold_does_not_invent_motion_from_zero_history():
    history = np.zeros((4, 4), dtype=np.uint8)
    assert motion_regions(history, candidate_threshold=0, min_area=1, max_area_ratio=1) == []


def test_invalid_component_filters_fail():
    with pytest.raises(ValueError):
        motion_regions(np.zeros((4, 4), dtype=np.uint8), max_area_ratio=np.nan)
