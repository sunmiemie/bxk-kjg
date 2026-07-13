from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "registration": {
        "orb_features": 800,
        "min_matches": 8,
        "ransac_reproj_threshold": 3.0,
        "phase_response_threshold": 0.01,
    },
    "temporal": {
        "threshold": 35,
        "decay": 25,
        "candidate_threshold": 1,
        "min_area": 6,
        "max_area_ratio": 0.20,
        "border_margin": 0,
    },
    "spatial": {
        "pixel_enabled": True,
        "region_enabled": True,
        "cr_factor": 2.0,
        "er_factor": 4.0,
        "region_quantization_levels": 8,
        "region_hist_bins": 32,
        "region_grid_size": 8,
        "min_patch_area": 4,
    },
    "fusion": {
        "temporal_weight": 0.31,
        "pixel_weight": 0.42,
        "region_weight": 0.27,
    },
    "thresholding": {
        "method": "otsu",
        "fixed": 0.45,
        "percentile": 97.5,
    },
    "postprocess": {
        "morph_kernel": 3,
        "min_area": 6,
        "max_area_ratio": 0.20,
    },
    "roi": {
        "enabled": False,
        "x_min": 0,
        "y_min": 0,
        "x_max": None,
        "y_max": None,
        "ignore_rectangles": [],
    },
    "box_filter": {
        "min_width": 1,
        "min_height": 1,
        "max_width": None,
        "max_height": None,
        "min_score": 0.0,
    },
    "static_target": {
        "enabled": False,
        "threshold_offset": 28,
        "min_area": 2,
        "max_area": 250,
        "min_width": 1,
        "min_height": 1,
        "max_width": 50,
        "max_height": 30,
    },
    "lite": {
        "frame_stride": 1,
        "hold_missing_frames": 0,
        "motion_enabled": True,
        "motion_threshold": 8,
        "motion_min_area": 2,
        "motion_max_area": 120,
        "dedupe_center_distance": 10,
    },
    "trajectory_filter": {
        "enabled": False,
        "min_hits": 2,
        "max_missing_frames": 3,
        "max_center_distance": 18.0,
        "max_area_ratio_change": 4.0,
        "single_target": False,
    },
    "synthetic": {
        "frame_count": 20,
        "width": 160,
        "height": 120,
        "fps": 10.0,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_number(config: dict[str, Any], dotted: str, *, min_value: float | None = None, max_value: float | None = None) -> None:
    current: Any = config
    for part in dotted.split("."):
        current = current[part]
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        raise ValueError(f"{dotted} must be a number")
    if min_value is not None and float(current) < min_value:
        raise ValueError(f"{dotted} must be >= {min_value:g}")
    if max_value is not None and float(current) > max_value:
        raise ValueError(f"{dotted} must be <= {max_value:g}")


def _require_optional_number(
    config: dict[str, Any],
    dotted: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    current: Any = config
    for part in dotted.split("."):
        current = current[part]
    if current is None:
        return
    _require_number(config, dotted, min_value=min_value, max_value=max_value)


def _require_bool(config: dict[str, Any], dotted: str) -> None:
    current: Any = config
    for part in dotted.split("."):
        current = current[part]
    if not isinstance(current, bool):
        raise ValueError(f"{dotted} must be true or false")


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required_sections = [
        "registration",
        "temporal",
        "spatial",
        "fusion",
        "thresholding",
        "postprocess",
        "roi",
        "box_filter",
        "static_target",
        "lite",
        "trajectory_filter",
        "synthetic",
    ]
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a JSON object")

    _require_number(config, "registration.orb_features", min_value=1)
    _require_number(config, "registration.min_matches", min_value=1)
    _require_number(config, "registration.ransac_reproj_threshold", min_value=0)
    _require_number(config, "registration.phase_response_threshold", min_value=0, max_value=1)
    _require_number(config, "temporal.threshold", min_value=0, max_value=255)
    _require_number(config, "temporal.decay", min_value=0, max_value=255)
    _require_number(config, "temporal.candidate_threshold", min_value=0, max_value=255)
    _require_number(config, "temporal.min_area", min_value=1)
    _require_number(config, "temporal.max_area_ratio", min_value=0, max_value=1)
    _require_number(config, "temporal.border_margin", min_value=0)
    _require_bool(config, "spatial.pixel_enabled")
    _require_bool(config, "spatial.region_enabled")
    _require_number(config, "spatial.cr_factor", min_value=1)
    _require_number(config, "spatial.er_factor", min_value=1)
    _require_number(config, "spatial.region_quantization_levels", min_value=2)
    _require_number(config, "spatial.region_hist_bins", min_value=2)
    _require_number(config, "spatial.region_grid_size", min_value=2)
    _require_number(config, "spatial.min_patch_area", min_value=1)
    _require_number(config, "fusion.temporal_weight", min_value=0)
    _require_number(config, "fusion.pixel_weight", min_value=0)
    _require_number(config, "fusion.region_weight", min_value=0)
    method = config["thresholding"].get("method")
    if method not in {"otsu", "percentile", "fixed"}:
        raise ValueError("thresholding.method must be one of: otsu, percentile, fixed")
    _require_number(config, "thresholding.fixed", min_value=0, max_value=1)
    percentile = config["thresholding"].get("percentile")
    if not isinstance(percentile, (int, float)) or isinstance(percentile, bool) or not 0 <= float(percentile) <= 100:
        raise ValueError("thresholding.percentile must be between 0 and 100")
    _require_number(config, "postprocess.morph_kernel", min_value=1)
    _require_number(config, "postprocess.min_area", min_value=1)
    _require_number(config, "postprocess.max_area_ratio", min_value=0, max_value=1)
    _require_bool(config, "roi.enabled")
    _require_number(config, "roi.x_min", min_value=0)
    _require_number(config, "roi.y_min", min_value=0)
    _require_optional_number(config, "roi.x_max", min_value=0)
    _require_optional_number(config, "roi.y_max", min_value=0)
    if not isinstance(config["roi"].get("ignore_rectangles"), list):
        raise ValueError("roi.ignore_rectangles must be a list")
    for index, rect in enumerate(config["roi"].get("ignore_rectangles", [])):
        if not isinstance(rect, dict):
            raise ValueError(f"roi.ignore_rectangles[{index}] must be a JSON object")
        for key in ["x", "y", "w", "h"]:
            if key not in rect or not isinstance(rect[key], (int, float)) or isinstance(rect[key], bool) or float(rect[key]) < 0:
                raise ValueError(f"roi.ignore_rectangles[{index}].{key} must be a non-negative number")
    _require_number(config, "box_filter.min_width", min_value=1)
    _require_number(config, "box_filter.min_height", min_value=1)
    _require_optional_number(config, "box_filter.max_width", min_value=1)
    _require_optional_number(config, "box_filter.max_height", min_value=1)
    _require_number(config, "box_filter.min_score", min_value=0, max_value=1)
    _require_bool(config, "static_target.enabled")
    _require_number(config, "static_target.threshold_offset", min_value=1, max_value=255)
    _require_number(config, "static_target.min_area", min_value=1)
    _require_number(config, "static_target.max_area", min_value=1)
    _require_number(config, "static_target.min_width", min_value=1)
    _require_number(config, "static_target.min_height", min_value=1)
    _require_number(config, "static_target.max_width", min_value=1)
    _require_number(config, "static_target.max_height", min_value=1)
    _require_number(config, "lite.frame_stride", min_value=1)
    _require_number(config, "lite.hold_missing_frames", min_value=0)
    _require_bool(config, "lite.motion_enabled")
    _require_number(config, "lite.motion_threshold", min_value=0, max_value=255)
    _require_number(config, "lite.motion_min_area", min_value=1)
    _require_number(config, "lite.motion_max_area", min_value=1)
    _require_number(config, "lite.dedupe_center_distance", min_value=0)
    _require_bool(config, "trajectory_filter.enabled")
    _require_number(config, "trajectory_filter.min_hits", min_value=1)
    _require_number(config, "trajectory_filter.max_missing_frames", min_value=0)
    _require_number(config, "trajectory_filter.max_center_distance", min_value=0)
    _require_number(config, "trajectory_filter.max_area_ratio_change", min_value=1)
    _require_bool(config, "trajectory_filter.single_target")
    _require_number(config, "synthetic.frame_count", min_value=1)
    _require_number(config, "synthetic.width", min_value=16)
    _require_number(config, "synthetic.height", min_value=16)
    _require_number(config, "synthetic.fps", min_value=1)
    return config


def load_config(config_arg: str | Path | None = None) -> dict[str, Any]:
    if not config_arg:
        return _validate_config(deep_update(DEFAULT_CONFIG, {}))

    raw = str(config_arg)
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            updates = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid inline JSON config: {exc}") from exc
    else:
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"Configuration file not found: {path}")
        try:
            updates = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config file {path}: {exc}") from exc
    if not isinstance(updates, dict):
        raise ValueError("Configuration JSON must be an object.")
    return _validate_config(deep_update(DEFAULT_CONFIG, updates))
