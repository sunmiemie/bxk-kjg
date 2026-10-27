from pathlib import Path

import cv2
import numpy as np
import pytest

from vasts import track_cli
from vasts.synthetic import make_synthetic_video


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_tracking_range_does_not_publish_or_replace_video(tmp_path, existing):
    source = make_synthetic_video(tmp_path / "input.mp4", frame_count=2, size=(32, 32))
    output = tmp_path / "overlay_tracked.mp4"
    previous = b"previous completed overlay"
    if existing:
        output.write_bytes(previous)
    original_files = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="exceed the decoded"):
        track_cli.render_tracked_overlay(source, [{"frame_index": 3, "boxes": []}], output)
    assert set(tmp_path.iterdir()) == original_files
    if existing:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()


@pytest.mark.parametrize("failure", ["create", "write", "release"])
def test_codec_failures_preserve_destination_and_clean_temporary_files(tmp_path, monkeypatch, failure):
    output = tmp_path / "overlay.mp4"
    previous = b"previous completed overlay"
    output.write_bytes(previous)

    class Capture:
        released = False
        count = 0

        def get(self, _): return 10
        def read(self):
            self.count += 1
            return (True, np.zeros((32, 32, 3), dtype=np.uint8)) if self.count == 1 else (False, None)
        def release(self): self.released = True

    class Writer:
        released = False

        def write(self, _):
            if failure == "write": raise cv2.error("simulated write failure")
        def release(self):
            self.released = True
            if failure == "release": raise cv2.error("simulated release failure")

    capture, writer = Capture(), Writer()

    def create_writer(path, *_):
        Path(path).write_bytes(b"incomplete encoding")
        if failure == "create": raise cv2.error("simulated create failure")
        return writer

    monkeypatch.setattr(track_cli, "_open_capture", lambda _: capture)
    monkeypatch.setattr(track_cli, "_create_writer", create_writer)
    with pytest.raises(cv2.error, match="simulated"):
        track_cli.render_tracked_overlay(tmp_path / "input.mp4", [], output)
    assert capture.released
    assert writer.released == (failure != "create")
    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {output}


def test_failed_replacement_keeps_previous_video(tmp_path, monkeypatch):
    source = make_synthetic_video(tmp_path / "input.mp4", frame_count=2, size=(32, 32))
    output = tmp_path / "overlay.mp4"
    previous = b"previous completed overlay"
    output.write_bytes(previous)

    def deny_replace(*_):
        raise PermissionError("destination is locked")

    monkeypatch.setattr("os.replace", deny_replace)
    with pytest.raises(PermissionError, match="locked"):
        track_cli.render_tracked_overlay(source, [], output)
    assert output.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {source, output}


def test_successful_render_replaces_video_with_complete_decodable_output(tmp_path):
    source = make_synthetic_video(tmp_path / "input.mp4", frame_count=3, size=(32, 32))
    output = tmp_path / "overlay.mp4"
    output.write_bytes(b"previous completed overlay")
    track_cli.render_tracked_overlay(source, [], output)
    capture = cv2.VideoCapture(str(output))
    count = 0
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (32, 32, 3)
            count += 1
    finally:
        capture.release()
    assert count == 3
    assert set(tmp_path.iterdir()) == {source, output}
