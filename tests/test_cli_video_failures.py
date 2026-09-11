import json
import cv2
import numpy as np
import pytest
from vasts import cli, track_cli
from vasts.synthetic import make_synthetic_video


def test_detector_cli_reports_opencv_failure_without_traceback(tmp_path, monkeypatch, capsys):
    def fail(**kwargs): raise cv2.error("simulated decode backend failure")
    monkeypatch.setattr(cli, "run_pipeline", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["--input", "unused.mp4", "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "decode backend failure" in message and "Traceback" not in message


def test_tracking_rejects_box_indices_beyond_real_video(tmp_path, capsys):
    video = make_synthetic_video(tmp_path / "input.mp4", frame_count=2, size=(32, 32))
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps({"frames": [{"frame_index": 3, "boxes": []}]}))
    output = tmp_path / "result"
    with pytest.raises(SystemExit) as error:
        track_cli.main(["--input-video", str(video), "--boxes-json", str(boxes), "--output-dir", str(output)])
    assert error.value.code == 2
    assert "exceed the decoded" in capsys.readouterr().err
    assert not (output / "boxes_tracked.json").exists()


def test_tracking_cannot_overwrite_input_json(tmp_path, monkeypatch):
    boxes = tmp_path / "boxes_tracked.json"
    boxes.write_text('{"frames": []}')
    monkeypatch.setattr(track_cli, "render_tracked_overlay", lambda *args: pytest.fail("must not render"))
    with pytest.raises(SystemExit):
        track_cli.main(["--input-video", "unused.mp4", "--boxes-json", str(boxes), "--output-dir", str(tmp_path)])
    assert boxes.read_text() == '{"frames": []}'


def test_tracking_releases_resources_on_dimension_change(tmp_path, monkeypatch):
    class Capture:
        released = False
        count = 0
        def get(self, _): return 10
        def read(self):
            self.count += 1
            return True, np.zeros((16 + self.count - 1, 16, 3), dtype=np.uint8)
        def release(self): self.released = True
    class Writer:
        released = False
        def write(self, frame): pass
        def release(self): self.released = True
    capture, writer = Capture(), Writer()
    monkeypatch.setattr(track_cli, "_open_capture", lambda _: capture)
    monkeypatch.setattr(track_cli, "_create_writer", lambda *args: writer)
    with pytest.raises(ValueError, match="dimensions changed"):
        track_cli.render_tracked_overlay("unused.mp4", [], tmp_path / "overlay.mp4")
    assert capture.released and writer.released
