import json

import numpy as np
import pytest

from vasts.config import DEFAULT_CONFIG, deep_update, load_config
from vasts.region_saliency import _quantized_components


@pytest.mark.parametrize("override", [
    {"registration": {"ransac_reproj_threshold": float("nan")}},
    {"fusion": {"pixel_weight": float("inf")}},
    {"fusion": {"pixel_weight": 1e308, "temporal_weight": 1e308}},
    {"thresholding": {"percentile": 10 ** 400}},
    {"synthetic": {"width": 32.5}},
    {"roi": {"x_max": 10, "x_min": 20}},
    {"roi": {"ignore_rectangles": [{"x": 0, "y": 0, "w": float("nan"), "h": 2}]}},
    {"static_target": {"min_area": 20, "max_area": 10}},
    {"spatial": {"er_factor": 1, "cr_factor": 2}},
    {"spatial": {"region_quantization_levels": 257}},
    {"fusion": {"temporal_weight": 0, "pixel_weight": 0, "region_weight": 0}},
    {"temproal": {}}, {"temporal": {"threshod": 4}},
])
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(json.dumps(override))


def test_load_does_not_share_nested_lists_with_defaults_or_overrides():
    first, second = load_config(), load_config()
    first["roi"]["ignore_rectangles"].append({"x": 0})
    assert second["roi"]["ignore_rectangles"] == []
    assert DEFAULT_CONFIG["roi"]["ignore_rectangles"] == []
    override = {"roi": {"ignore_rectangles": [{"x": 1}]}}
    merged = deep_update(DEFAULT_CONFIG, override)
    merged["roi"]["ignore_rectangles"][0]["x"] = 5
    assert override["roi"]["ignore_rectangles"][0]["x"] == 1


def test_config_file_bom_and_unreadable_input(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"temporal":{"threshold":12}}', encoding="utf-8-sig")
    assert load_config(path)["temporal"]["threshold"] == 12
    with pytest.raises(ValueError, match="Could not read configuration"):
        load_config(tmp_path)
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="Could not read configuration"):
        load_config(path)


def test_256_level_quantization_keeps_black_and_white_separate():
    gray = np.array([[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    patches = _quantized_components(gray, 256, 32, 1)
    assert len(patches) == 2
    assert sorted(p.area for p in patches) == [4, 4]
