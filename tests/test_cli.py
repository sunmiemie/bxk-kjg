import json
import os
import subprocess
import sys
from pathlib import Path

import cv2


def _run_cli(tmp_path, *args):
    output_dir = tmp_path / "cli_out"
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")

    return output_dir, subprocess.run(
        [
            sys.executable,
            "-m",
            "vasts.cli",
            *args,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _assert_video_readable(path: Path, expected_frames: int):
    assert path.exists()
    assert path.stat().st_size > 0
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count == expected_frames
        ok, frame = capture.read()
        assert ok
        assert frame is not None and frame.size > 0
    finally:
        capture.release()


def test_cli_synthetic_writes_boxes_schema_videos_and_maps(tmp_path):
    output_dir, completed = _run_cli(
        tmp_path,
        "--synthetic",
        "--output-dir",
        str(tmp_path / "cli_out"),
        "--max-frames",
        "10",
        "--save-video",
        "--save-maps",
    )

    assert completed.returncode == 0, completed.stderr
    boxes_path = output_dir / "boxes.json"
    assert boxes_path.exists()
    _assert_video_readable(output_dir / "overlay.mp4", expected_frames=10)
    _assert_video_readable(output_dir / "saliency.mp4", expected_frames=10)
    for suffix in ["diff", "fmhi", "pixel", "region", "fused", "mask"]:
        assert (output_dir / "maps" / f"frame_00001_{suffix}.png").exists()

    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source"]["mode"] == "synthetic"
    assert payload["source"]["max_frames"] == 10
    assert payload["summary"]["frames_processed"] == 10
    assert isinstance(payload["summary"]["total_boxes"], int)
    assert "config" in payload and "fusion" in payload["config"]
    assert isinstance(payload["frames"], list)
    assert payload["frames"]
    for frame in payload["frames"]:
        assert isinstance(frame["frame_index"], int)
        assert isinstance(frame["boxes"], list)

    boxed_frames = [frame for frame in payload["frames"] if frame["boxes"]]
    assert boxed_frames, "Synthetic run should produce at least one detection box"
    first_box = boxed_frames[0]["boxes"][0]
    assert {"x", "y", "w", "h", "score"} <= first_box.keys()
    assert all(isinstance(first_box[key], int) for key in ["x", "y", "w", "h"])
    assert isinstance(first_box["score"], float)


def test_cli_rejects_synthetic_and_input_together(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"not a real video")
    _, completed = _run_cli(
        tmp_path,
        "--synthetic",
        "--input",
        str(dummy_video),
        "--output-dir",
        str(tmp_path / "cli_out"),
    )

    assert completed.returncode != 0
    assert "not allowed with argument" in completed.stderr


def test_cli_rejects_non_positive_max_frames(tmp_path):
    _, completed = _run_cli(
        tmp_path,
        "--synthetic",
        "--output-dir",
        str(tmp_path / "cli_out"),
        "--max-frames",
        "0",
    )

    assert completed.returncode != 0
    assert "--max-frames must be a positive integer" in completed.stderr


def test_cli_reports_missing_config_file(tmp_path):
    _, completed = _run_cli(
        tmp_path,
        "--synthetic",
        "--output-dir",
        str(tmp_path / "cli_out"),
        "--config",
        str(tmp_path / "missing.json"),
    )

    assert completed.returncode != 0
    assert "Configuration file not found" in completed.stderr


def test_cli_reports_invalid_config_range(tmp_path):
    _, completed = _run_cli(
        tmp_path,
        "--synthetic",
        "--output-dir",
        str(tmp_path / "cli_out"),
        "--config",
        '{"thresholding":{"method":"percentile","percentile":150}}',
    )

    assert completed.returncode != 0
    assert "thresholding.percentile must be between 0 and 100" in completed.stderr
