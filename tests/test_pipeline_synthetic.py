from pathlib import Path

from vasts.pipeline import run_pipeline
from vasts.synthetic import make_synthetic_video


def test_synthetic_video_end_to_end_detects_moving_target(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    make_synthetic_video(video_path, frame_count=12, size=(96, 72), target_size=(8, 6))

    result = run_pipeline(
        input_path=video_path,
        output_dir=tmp_path / "run",
        max_frames=12,
        save_maps=False,
        save_video=False,
    )

    assert result.frames_processed >= 8
    frames_with_boxes = [frame for frame in result.boxes if frame["boxes"]]
    assert len(frames_with_boxes) >= 3
    first_box = frames_with_boxes[0]["boxes"][0]
    assert {"x", "y", "w", "h", "score"} <= first_box.keys()
    assert first_box["w"] > 0 and first_box["h"] > 0
