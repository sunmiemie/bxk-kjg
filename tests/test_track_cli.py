import json

import pytest

from vasts import track_cli


def arguments(tmp_path):
    return ["--input-video", str(tmp_path / "video.mp4"), "--boxes-json", str(tmp_path / "boxes.json"),
            "--output-dir", str(tmp_path / "output")]


@pytest.mark.parametrize("contents", [None, "bad json", "[]", '{"frames": "wrong"}'])
def test_invalid_boxes_are_reported_before_rendering(tmp_path, monkeypatch, capsys, contents):
    if contents is not None:
        (tmp_path / "boxes.json").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(track_cli, "render_tracked_overlay", lambda *args: pytest.fail("must not render"))
    with pytest.raises(SystemExit) as error:
        track_cli.main(arguments(tmp_path))
    assert error.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("distance", ["nan", "inf", "0", "-1"])
def test_invalid_distance_is_cli_error(tmp_path, capsys, distance):
    with pytest.raises(SystemExit) as error:
        track_cli.main(arguments(tmp_path) + ["--max-distance", distance])
    assert error.value.code == 2
    assert "finite and > 0" in capsys.readouterr().err


def test_failed_render_does_not_publish_boxes_result(tmp_path, monkeypatch):
    (tmp_path / "boxes.json").write_text('{"frames": []}', encoding="utf-8-sig")
    def fail(*args):
        raise RuntimeError("writer unavailable")
    monkeypatch.setattr(track_cli, "render_tracked_overlay", fail)
    with pytest.raises(SystemExit):
        track_cli.main(arguments(tmp_path))
    assert not (tmp_path / "output" / "boxes_tracked.json").exists()


def test_successful_render_publishes_result(tmp_path, monkeypatch):
    (tmp_path / "boxes.json").write_text('{"frames": []}', encoding="utf-8-sig")
    monkeypatch.setattr(track_cli, "render_tracked_overlay", lambda *args: None)
    assert track_cli.main(arguments(tmp_path)) == 0
    payload = json.loads((tmp_path / "output" / "boxes_tracked.json").read_text())
    assert payload["summary"]["frames_processed"] == 0


def test_renderer_rejects_input_output_collision_before_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(track_cli, "_open_capture", lambda *args: pytest.fail("must not open"))
    with pytest.raises(ValueError, match="overwrite"):
        track_cli.render_tracked_overlay(tmp_path / "video.mp4", [], tmp_path / "video.mp4")
