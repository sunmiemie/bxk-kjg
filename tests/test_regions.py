from vasts.regions import deduplicate_boxes, expand_box


def test_cr_er_expansion_clips_to_image_boundaries():
    image_shape = (40, 50)
    motion_box = [2, 3, 8, 5]

    cr = expand_box(motion_box, factor=2.0, image_shape=image_shape)
    er = expand_box(motion_box, factor=4.0, image_shape=image_shape)

    assert cr == [0, 0, 14, 11]
    assert er == [0, 0, 22, 16]
    assert cr[0] >= 0 and cr[1] >= 0
    assert er[2] <= image_shape[1] and er[3] <= image_shape[0]


def test_expansion_clips_right_and_bottom_boundaries():
    image_shape = (40, 50)
    motion_box = [44, 34, 5, 5]

    cr = expand_box(motion_box, factor=2.0, image_shape=image_shape)
    er = expand_box(motion_box, factor=4.0, image_shape=image_shape)

    assert cr == [41, 31, 9, 9]
    assert er == [36, 26, 14, 14]
    assert cr[0] + cr[2] == image_shape[1]
    assert cr[1] + cr[3] == image_shape[0]
    assert er[0] + er[2] == image_shape[1]
    assert er[1] + er[3] == image_shape[0]


def test_deduplicate_boxes_keeps_best_nearby_box():
    boxes = [
        {"x": 100, "y": 100, "w": 2, "h": 2, "score": 0.7},
        {"x": 96, "y": 98, "w": 12, "h": 6, "score": 0.95, "source": "static"},
        {"x": 200, "y": 100, "w": 3, "h": 2, "score": 0.8},
    ]

    deduped = deduplicate_boxes(boxes, center_distance=12)

    assert len(deduped) == 2
    assert deduped[0]["source"] == "static"
    assert deduped[1]["x"] == 200
