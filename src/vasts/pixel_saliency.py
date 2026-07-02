from __future__ import annotations

import numpy as np

from .image_utils import ensure_gray
from .regions import box_slice, expand_box


_VALUES = np.arange(256, dtype=np.float32)
_DISTANCE_TABLE = np.abs(_VALUES[:, None] - _VALUES[None, :])


def _hist256(values: np.ndarray) -> np.ndarray:
    hist = np.bincount(values.reshape(-1).astype(np.uint8), minlength=256).astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def pixel_saliency_map(
    frame: np.ndarray,
    motion_boxes: list[list[int]],
    *,
    cr_factor: float = 2.0,
    er_factor: float = 4.0,
) -> np.ndarray:
    gray = ensure_gray(frame)
    saliency = np.zeros(gray.shape, dtype=np.float32)
    for box in motion_boxes:
        cr = expand_box(box, factor=cr_factor, image_shape=gray.shape)
        er = expand_box(box, factor=er_factor, image_shape=gray.shape)
        if cr[2] <= 0 or cr[3] <= 0 or er[2] <= 0 or er[3] <= 0:
            continue

        cr_values = gray[box_slice(cr)]
        er_values = gray[box_slice(er)]
        hist_cr = _hist256(cr_values)
        hist_er = _hist256(er_values)
        hdiff = np.abs(hist_cr - hist_er)
        hdiff_max = float(hdiff.max())
        w_rc = np.maximum(0.0, hdiff_max - hdiff)
        s_hc = _DISTANCE_TABLE.dot(hist_cr)

        local = w_rc[cr_values] * s_hc[cr_values]
        if float(local.max()) == 0.0 and float(s_hc.max()) > 0.0:
            local = s_hc[cr_values]
        y_slice, x_slice = box_slice(cr)
        saliency[y_slice, x_slice] = np.maximum(saliency[y_slice, x_slice], local.astype(np.float32))
    return saliency
