import copy

import pytest

from vasts.tracking import fill_single_track_gaps


def box(x=0):
    return {"x": x, "y": 0, "w": 2, "h": 2}


def test_optional_scores_interpolate_and_preserve_metadata():
    first = {**box(), "source": "static"}
    payload = {"frames": [{"frame_index": 0, "timestamp": .1, "boxes": [first]},
                          {"frame_index": 1, "timestamp": .2, "boxes": []},
                          {"frame_index": 2, "timestamp": .3, "boxes": [box(4)]}]}
    before = copy.deepcopy(payload)
    result = fill_single_track_gaps(payload)
    assert result["frames"][1]["boxes"][0]["x"] == 2
    assert result["frames"][1]["boxes"][0]["score"] == 0
    assert result["frames"][0]["boxes"][0]["source"] == "static"
    assert result["frames"][1]["timestamp"] == .2
    assert payload == before


def test_sparse_rows_only_count_interpolations_present_in_output():
    payload = {"frames": [{"frame_index": 0, "boxes": [box()]},
                          {"frame_index": 2, "boxes": []},
                          {"frame_index": 4, "boxes": [box(8)]}]}
    result = fill_single_track_gaps(payload, max_gap=3)
    assert result["summary"]["interpolated_boxes"] == 1
    assert len(result["frames"]) == 3


@pytest.mark.parametrize("indices", [[0, 0], [2, 1], [-1], [True], [1.5]])
def test_invalid_frame_indices_fail(indices):
    with pytest.raises(ValueError, match="frame indices"):
        fill_single_track_gaps({"frames": [{"frame_index": i, "boxes": []} for i in indices]})


@pytest.mark.parametrize("patch", [{"w": 0}, {"x": float("nan")}, {"score": 2}, {"h": True}])
def test_invalid_boxes_fail(patch):
    with pytest.raises(ValueError):
        fill_single_track_gaps({"frames": [{"frame_index": 0, "boxes": [{**box(), **patch}]}]})


def test_empty_result_keeps_metadata_and_summary_schema():
    result = fill_single_track_gaps({"frames": [{"frame_index": 0, "timestamp": 1, "boxes": []}]})
    assert result["frames"][0]["timestamp"] == 1
    assert result["summary"]["track_observations"] == 0
    assert result["summary"]["max_gap"] == 10


@pytest.mark.parametrize("kwargs", [{"max_gap": -1}, {"max_gap": 1.5}, {"max_distance": float("nan")}, {"max_distance": 0}])
def test_invalid_tracking_options_fail(kwargs):
    with pytest.raises(ValueError):
        fill_single_track_gaps({"frames": []}, **kwargs)
