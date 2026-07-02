import numpy as np

from vasts.static_target import detect_static_dark_blobs


def test_detect_static_dark_blob_in_roi():
    gray = np.full((80, 120), 180, dtype=np.uint8)
    gray[30:34, 55:64] = 45
    roi = np.zeros_like(gray, dtype=bool)
    roi[10:60, 30:90] = True

    boxes = detect_static_dark_blobs(
        gray,
        roi,
        threshold_offset=35,
        min_area=4,
        max_area=100,
        min_width=2,
        min_height=2,
        max_width=20,
        max_height=12,
    )

    assert len(boxes) == 1
    box = boxes[0]
    assert box["x"] <= 55
    assert box["y"] <= 30
    assert box["x"] + box["w"] >= 64
    assert box["y"] + box["h"] >= 34
    assert box["source"] == "static"
    assert box["score"] > 0.5


def test_detect_static_dark_blob_respects_roi():
    gray = np.full((80, 120), 180, dtype=np.uint8)
    gray[30:34, 55:64] = 45
    roi = np.zeros_like(gray, dtype=bool)
    roi[0:20, 0:40] = True

    boxes = detect_static_dark_blobs(gray, roi)

    assert boxes == []
