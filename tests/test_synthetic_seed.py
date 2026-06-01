import json
import numpy as np
import pytest
from vasts.config import load_config
from vasts.synthetic import _base_background, make_synthetic_video


def test_seed_repeats_without_mutating_global_rng():
    np.random.seed(17)
    before = np.random.get_state()
    first = _base_background(40, 32, seed=12)
    np.testing.assert_array_equal(first, _base_background(40, 32, seed=12))
    assert not np.array_equal(first, _base_background(40, 32, seed=13))
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])


def test_seed_is_configurable_and_checked(tmp_path):
    assert load_config(json.dumps({"synthetic": {"seed": 9}}))["synthetic"]["seed"] == 9
    with pytest.raises(ValueError):
        make_synthetic_video(tmp_path / "video.mp4", seed=-1)
