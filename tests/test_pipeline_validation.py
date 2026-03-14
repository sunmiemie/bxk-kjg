from pathlib import Path

import pytest

from vasts import pipeline
from vasts.cli import main


@pytest.mark.parametrize("max_frames", [0, -1, True, 1.5])
def test_pipeline_rejects_invalid_limit_before_opening_video(tmp_path, monkeypatch, max_frames):
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: pytest.fail("must not open video"))
    with pytest.raises(ValueError, match="max_frames"):
        pipeline.run_pipeline(input_path="unused.mp4", output_dir=tmp_path / "out", max_frames=max_frames)
    assert not (tmp_path / "out").exists()


def test_direct_pipeline_uses_configuration_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_open_capture", lambda _: pytest.fail("must not open video"))
    with pytest.raises(ValueError, match="finite"):
        pipeline.run_pipeline(input_path="unused.mp4", output_dir=tmp_path / "out",
                              config={"fusion": {"pixel_weight": float("nan")}})


def test_input_cannot_be_overwritten_by_output(tmp_path):
    path = tmp_path / "overlay.mp4"
    path.write_bytes(b"existing input")
    with pytest.raises(ValueError, match="output files"):
        pipeline.run_pipeline(input_path=path, output_dir=tmp_path, save_video=True)
    assert path.read_bytes() == b"existing input"


@pytest.mark.parametrize("fps", [0, -5, float("nan"), float("inf"), 29.97])
def test_invalid_capture_fps_has_finite_fallback(fps):
    class Capture:
        def get(self, _):
            return fps
    assert pipeline._capture_fps(Capture()) == (29.97 if fps == 29.97 else 10.0)


def test_failed_capture_and_writer_are_released(monkeypatch, tmp_path):
    class Resource:
        released = False
        def isOpened(self):
            return False
        def release(self):
            self.released = True
    capture, writer = Resource(), Resource()
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(pipeline.cv2, "VideoWriter", lambda *args: writer)
    with pytest.raises(FileNotFoundError):
        pipeline._open_capture("missing.mp4")
    with pytest.raises(RuntimeError):
        pipeline._create_writer(tmp_path / "output.mp4", 10, (32, 32))
    assert capture.released and writer.released


def test_cli_reports_config_error_before_making_output_directory(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--synthetic", "--output-dir", str(tmp_path / "out"), "--config", str(tmp_path / "missing.json")])
    assert error.value.code == 2
    assert "Configuration file not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_reports_pipeline_failure_without_traceback(tmp_path, monkeypatch, capsys):
    import vasts.cli as cli
    def fail(**kwargs):
        raise FileNotFoundError("Could not open input video")
    monkeypatch.setattr(cli, "run_pipeline", fail)
    with pytest.raises(SystemExit) as error:
        main(["--input", "missing.mp4", "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Could not open input video" in message
    assert "Traceback" not in message
