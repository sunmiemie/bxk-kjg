# Visible Aerial Spatiotemporal Saliency

This project is a lightweight Python/OpenCV reproduction of the paper **"Moving object detection in aerial video based on spatiotemporal saliency"** by Shen Hao et al., Chinese Journal of Aeronautics, 2013, DOI: [10.1016/j.cja.2013.07.038](https://doi.org/10.1016/j.cja.2013.07.038).

No official reusable implementation was found during web search by title, DOI, author names, `FMHI`, GitHub, Gitee, and Code Ocean keywords. Two related repositories were found, but they implement different video saliency methods rather than this aerial FMHI + local pixel/region saliency pipeline:

- [ZhigangTU/Spatiotemporal-Salient-Object-Detection-in-Video](https://github.com/ZhigangTU/Spatiotemporal-Salient-Object-Detection-in-Video)
- [sridarah/spa_tem_saliency](https://github.com/sridarah/spa_tem_saliency)

## Implemented Paper Flow

1. Read consecutive visible-light video frames.
2. Register adjacent frames with ORB feature matching and RANSAC affine estimation.
3. Fall back to phase correlation, then identity, when feature matching is insufficient.
4. Compute registered frame difference.
5. Build forward motion history image (FMHI):
   `M_k(x,y)=255` when `D_k(x,y)>=T`; otherwise `max(0, M_{k-1}(x,y)+D_k(x,y)-d)`.
6. Extract candidate motion regions (MR) by connected components over FMHI.
7. Expand each MR to a center region (CR, 2x) and extended region (ER, 4x), clipped to image boundaries.
8. Compute pixel saliency `S_P=w_RC*S_HC` in CR.
9. Compute region saliency in CR with an OpenCV/NumPy patch approximation.
10. Normalize temporal, pixel, and region maps to `0..1`, then fuse with weights `0.31/0.42/0.27`.
11. Threshold fused saliency by Otsu, percentile, or fixed threshold.
12. Apply morphology and connected components to produce boxes.

## Parameters

| Area | Parameter | Default | Notes |
|---|---:|---:|---|
| Registration | `orb_features` | `800` | ORB keypoints. |
| Registration | `min_matches` | `8` | Below this, use fallback. |
| Registration | `ransac_reproj_threshold` | `3.0` | Affine RANSAC threshold. |
| Temporal | `threshold` | `35` | Paper FMHI value `T`. |
| Temporal | `decay` | `25` | Paper FMHI decay `d`. |
| Temporal | `candidate_threshold` | `1` | Binary threshold for FMHI candidates. |
| Temporal | `min_area` | `6` | Candidate MR area filter. |
| Spatial | `cr_factor` | `2.0` | MR to CR expansion. |
| Spatial | `er_factor` | `4.0` | MR to ER expansion. |
| Region | `region_quantization_levels` | `8` | Patch approximation. |
| Region | `region_hist_bins` | `32` | Patch histogram bins. |
| Fusion | `temporal_weight` | `0.31` | Paper fusion weight. |
| Fusion | `pixel_weight` | `0.42` | Paper fusion weight. |
| Fusion | `region_weight` | `0.27` | Paper fusion weight. |
| Thresholding | `method` | `otsu` | `otsu`, `percentile`, or `fixed`. |
| Thresholding | `fixed` | `0.45` | Used only by fixed threshold. |
| Thresholding | `percentile` | `97.5` | Used only by percentile threshold. |

Override parameters with a JSON file or inline JSON:

```powershell
$env:PYTHONPATH = "D:\Users\Desktop\bxk\visible_aerial_spatiotemporal_saliency\src"
python -m vasts.cli --synthetic --output-dir runs\demo --config '{"thresholding":{"method":"percentile","percentile":98}}'
```

## Run

From this directory, either install the local package:

```powershell
python -m pip install -e .
vasts --synthetic --output-dir runs\synthetic --max-frames 20 --save-video --save-maps
vasts --input D:\path\to\video.mp4 --output-dir runs\video --max-frames 200 --save-video
```

Or run without installation by setting `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "D:\Users\Desktop\bxk\visible_aerial_spatiotemporal_saliency\src"
python -m vasts.cli --synthetic --output-dir runs\synthetic --max-frames 20 --save-video --save-maps
python -m vasts.cli --input D:\path\to\video.mp4 --output-dir runs\video --max-frames 200 --save-video
```

## Outputs

Every run writes:

- `boxes.json`: schema version, source info, config, summary, and per-frame boxes.

With `--save-video`:

- `overlay.mp4`: input video with red detection boxes.
- `saliency.mp4`: fused saliency heatmap.

With `--save-maps`:

- `maps/frame_XXXXX_diff.png`
- `maps/frame_XXXXX_fmhi.png`
- `maps/frame_XXXXX_pixel.png`
- `maps/frame_XXXXX_region.png`
- `maps/frame_XXXXX_fused.png`
- `maps/frame_XXXXX_mask.png`

`boxes.json` schema:

```json
{
  "schema_version": 1,
  "source": {"mode": "synthetic", "path": "...", "max_frames": 20},
  "config": {},
  "summary": {"frames_processed": 20, "frames_with_boxes": 10, "total_boxes": 12},
  "frames": [
    {"frame_index": 0, "boxes": []},
    {"frame_index": 1, "boxes": [{"x": 10, "y": 12, "w": 8, "h": 6, "score": 0.73}]}
  ]
}
```

## Known Deviations And Approximations

- The paper uses a graph-based image segmentation method for region saliency. This implementation avoids `skimage` and uses OpenCV/NumPy brightness quantization connected components, with grid patches as fallback. This preserves the "patch contrast + area weight + spatial weight" logic but is not an exact Felzenszwalb graph segmentation reproduction.
- The paper gives `w_RC=Hdiff_max-Hdiff(pixel_value)`. This implementation follows that formula using normalized CR/ER gray histograms. If the CR/ER histogram difference is zero, it falls back to the histogram contrast term so the map remains usable on flat synthetic regions.
- The paper uses visible aerial video and does not publish a fixed final threshold. This implementation exposes `otsu`, `percentile`, and `fixed`.
- Fusion weights are the paper defaults. The paper says they were obtained offline, but no training details are provided.
- Registration is affine only. Strong parallax, rolling shutter, heavy blur, low texture, or very large camera motion can still cause false positives.
- Output boxes are derived independently per frame; no tracker is applied.

## NSTSM Transfer Understanding

The visible-light aerial paper and the infrared NSTSM paper share the same high-level design: compute a temporal saliency cue, compute a spatial saliency cue, fuse them, and threshold the fused map. The implementation keeps that migration logic at the pipeline level only.

It does not reuse infrared NSTSM formulas. NSTSM uses spatial variance saliency (SVSM), temporal gray saliency (TGSM), and multiplication fusion for low-altitude slow-small infrared targets. This project uses the visible-light paper's registered frame difference/FMHI, CR/ER pixel contrast, patch region contrast, and linear weighted fusion.

More detail is in `docs/nstsm_transfer_notes.md`.

## Tests

```powershell
python -m pytest
```

The tests use only synthetic data and local temporary files.
