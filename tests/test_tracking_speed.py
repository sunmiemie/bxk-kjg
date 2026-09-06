import pytest
from vasts.tracking import fill_single_track_gaps


def payload(last=2):
    return {"frames": [{"frame_index": 0, "boxes": [{"x": 0, "y": 0, "w": 2, "h": 2}]},
                       {"frame_index": 1, "boxes": []},
                       {"frame_index": last, "boxes": [{"x": 10, "y": 0, "w": 2, "h": 2}]}]}


def test_speed_limit_records_rejected_gap_without_changing_observations():
    result = fill_single_track_gaps(payload(), max_speed_pixels_per_frame=4)
    assert result["frames"][1]["boxes"] == []
    assert result["summary"]["gap_decisions"] == {"speed": 1}
    assert fill_single_track_gaps(payload(), max_speed_pixels_per_frame=5)["summary"]["interpolated_boxes"] == 1


def test_large_sparse_gap_does_not_iterate_absent_frames():
    result = fill_single_track_gaps(payload(10**9), max_gap=10**9)
    assert result["summary"]["interpolated_boxes"] == 1
    assert len(result["frames"]) == 3


def test_nonfinite_speed_rejected():
    with pytest.raises(ValueError):
        fill_single_track_gaps(payload(), max_speed_pixels_per_frame=float("nan"))
