from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from .image_utils import ensure_gray
from .regions import box_slice, expand_box


@dataclass
class Patch:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    hist: np.ndarray


def _histogram(values: np.ndarray, bins: int) -> np.ndarray:
    scaled = values.reshape(-1).astype(np.int32) * int(bins) // 256
    hist = np.bincount(np.clip(scaled, 0, bins - 1), minlength=bins).astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def _patch_from_mask(mask: np.ndarray, gray: np.ndarray, bins: int) -> Patch | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    values = gray[mask]
    return Patch(
        mask=mask,
        bbox=(x0, y0, x1, y1),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=int(xs.size),
        hist=_histogram(values, bins),
    )


def _quantized_components(gray: np.ndarray, levels: int, bins: int, min_patch_area: int) -> list[Patch]:
    quantized = np.clip(gray.astype(np.int32) * int(levels) // 256, 0, int(levels) - 1).astype(np.uint8)
    patches: list[Patch] = []
    for level in range(int(levels)):
        mask = (quantized == level).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_patch_area):
                continue
            patch = _patch_from_mask(labels == label, gray, bins)
            if patch is not None:
                patches.append(patch)
    return patches


def _grid_patches(gray: np.ndarray, grid_size: int, bins: int) -> list[Patch]:
    height, width = gray.shape[:2]
    patches: list[Patch] = []
    step = max(2, int(grid_size))
    for y in range(0, height, step):
        for x in range(0, width, step):
            mask = np.zeros_like(gray, dtype=bool)
            mask[y : min(height, y + step), x : min(width, x + step)] = True
            patch = _patch_from_mask(mask, gray, bins)
            if patch is not None:
                patches.append(patch)
    return patches


def _spatial_weight(patch: Patch, width: int, height: int) -> float:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    cx, cy = patch.centroid
    denom_x = max(width / 2.0, 1.0)
    denom_y = max(height / 2.0, 1.0)
    centroid_offset = max(abs(cx - center_x) / denom_x, abs(cy - center_y) / denom_y)
    centroid_offset = float(np.clip(centroid_offset, 0.0, 1.0))

    x0, y0, x1, y1 = patch.bbox
    boundary_offset = min(min(x0, width - x1) / max(width, 1), min(y0, height - y1) / max(height, 1))
    boundary_offset = float(np.clip(boundary_offset, 0.0, 1.0))
    return boundary_offset**2 + (1.0 - centroid_offset**2)


def _local_region_saliency(
    local_gray: np.ndarray,
    *,
    quantization_levels: int,
    hist_bins: int,
    grid_size: int,
    min_patch_area: int,
) -> np.ndarray:
    patches = _quantized_components(local_gray, quantization_levels, hist_bins, min_patch_area)
    if len(patches) < 2:
        patches = _grid_patches(local_gray, grid_size, hist_bins)
    if len(patches) < 2:
        return np.zeros(local_gray.shape, dtype=np.float32)

    total_area = float(sum(p.area for p in patches))
    area_weights = np.array([p.area / total_area for p in patches], dtype=np.float32)
    scores = np.zeros(len(patches), dtype=np.float32)
    height, width = local_gray.shape[:2]
    for i, patch_i in enumerate(patches):
        contrast = 0.0
        for j, patch_j in enumerate(patches):
            if i == j:
                continue
            distance = 0.5 * float(np.sum(np.abs(patch_i.hist - patch_j.hist)))
            contrast += float(area_weights[j]) * distance
        scores[i] = float(_spatial_weight(patch_i, width, height)) * contrast

    local = np.zeros(local_gray.shape, dtype=np.float32)
    for patch, score in zip(patches, scores):
        local[patch.mask] = max(float(score), float(local[patch.mask].max(initial=0.0)))
    return local


def region_saliency_map(
    frame: np.ndarray,
    motion_boxes: list[list[int]],
    *,
    cr_factor: float = 2.0,
    quantization_levels: int = 8,
    hist_bins: int = 32,
    grid_size: int = 8,
    min_patch_area: int = 4,
) -> np.ndarray:
    gray = ensure_gray(frame)
    saliency = np.zeros(gray.shape, dtype=np.float32)
    for box in motion_boxes:
        cr = expand_box(box, factor=cr_factor, image_shape=gray.shape)
        if cr[2] <= 0 or cr[3] <= 0:
            continue
        y_slice, x_slice = box_slice(cr)
        local = _local_region_saliency(
            gray[y_slice, x_slice],
            quantization_levels=quantization_levels,
            hist_bins=hist_bins,
            grid_size=grid_size,
            min_patch_area=min_patch_area,
        )
        saliency[y_slice, x_slice] = np.maximum(saliency[y_slice, x_slice], local)
    return saliency
