from vasts.tracking import fill_single_track_gaps


def test_fill_single_track_gaps_interpolates_short_missing_run():
    payload = {
        "schema_version": 1,
        "source": {},
        "config": {},
        "summary": {},
        "frames": [
            {"frame_index": 0, "boxes": [{"x": 10, "y": 20, "w": 4, "h": 2, "score": 0.9}]},
            {"frame_index": 1, "boxes": []},
            {"frame_index": 2, "boxes": [{"x": 14, "y": 22, "w": 4, "h": 2, "score": 0.8}]},
        ],
    }

    tracked = fill_single_track_gaps(payload, max_gap=2, max_distance=20)

    assert tracked["summary"]["frames_processed"] == 3
    assert tracked["summary"]["frames_with_boxes"] == 3
    assert tracked["summary"]["interpolated_boxes"] == 1
    assert tracked["frames"][1]["boxes"] == [
        {"x": 12, "y": 21, "w": 4, "h": 2, "score": 0.85, "track_id": 1, "interpolated": True}
    ]


def test_fill_single_track_gaps_does_not_bridge_large_jump():
    payload = {
        "frames": [
            {"frame_index": 0, "boxes": [{"x": 10, "y": 10, "w": 2, "h": 2, "score": 0.9}]},
            {"frame_index": 1, "boxes": []},
            {"frame_index": 2, "boxes": [{"x": 200, "y": 200, "w": 2, "h": 2, "score": 0.9}]},
        ],
    }

    tracked = fill_single_track_gaps(payload, max_gap=2, max_distance=20)

    assert tracked["summary"]["interpolated_boxes"] == 0
    assert tracked["summary"]["frames_with_boxes"] == 2
    assert tracked["frames"][1]["boxes"] == []
