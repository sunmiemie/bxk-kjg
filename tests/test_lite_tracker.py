from vasts.lite_tracker import TrajectoryFilter


def test_trajectory_filter_suppresses_single_frame_candidate():
    tracker = TrajectoryFilter(min_hits=2, max_missing_frames=1, max_center_distance=10)

    first = tracker.update([{"x": 10, "y": 10, "w": 4, "h": 2, "score": 0.8}])
    second = tracker.update([])
    third = tracker.update([])

    assert first == []
    assert second == []
    assert third == []


def test_trajectory_filter_confirms_consistent_candidate_and_holds_short_gap():
    tracker = TrajectoryFilter(min_hits=2, max_missing_frames=2, max_center_distance=10)

    assert tracker.update([{"x": 10, "y": 10, "w": 4, "h": 2, "score": 0.8}]) == []
    confirmed = tracker.update([{"x": 12, "y": 11, "w": 4, "h": 2, "score": 0.9}])
    held = tracker.update([])

    assert len(confirmed) == 1
    assert confirmed[0]["track_id"] == 1
    assert confirmed[0]["trajectory_hits"] == 2
    assert "trajectory_held" not in confirmed[0]
    assert held[0]["track_id"] == 1
    assert held[0]["trajectory_held"] is True
    assert held[0]["trajectory_missing"] == 1


def test_trajectory_filter_rejects_large_jump_as_new_unconfirmed_track():
    tracker = TrajectoryFilter(min_hits=2, max_missing_frames=1, max_center_distance=10)

    assert tracker.update([{"x": 10, "y": 10, "w": 4, "h": 2, "score": 0.8}]) == []
    jumped = tracker.update([{"x": 80, "y": 80, "w": 4, "h": 2, "score": 0.9}])

    assert jumped == []
