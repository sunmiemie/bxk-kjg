# NSTSM Transfer Notes

This note explains how the visible-light aerial detection paper maps conceptually to the earlier infrared NSTSM paper without mixing their formulas.

## Shared Skeleton

Both methods are training-free spatiotemporal saliency pipelines:

1. Build a temporal cue from image sequences.
2. Build a spatial cue from local target/background contrast.
3. Fuse temporal and spatial saliency.
4. Segment the fused saliency map to extract targets.

That shared skeleton is the only migration used in this codebase.

## Visible-Light Aerial Paper

The visible-light paper targets moving objects in UAV/aerial video where the camera platform moves. Its temporal cue is FMHI after adjacent-frame registration:

- ORB/RANSAC affine registration in this implementation.
- Registered frame difference.
- FMHI update with `T=35` and `d=25`.
- Connected components over FMHI create candidate motion regions.

Its spatial cue is computed only inside candidate regions:

- MR is expanded to CR by 2x and ER by 4x.
- Pixel saliency uses CR histogram contrast weighted by CR/ER histogram distribution difference.
- Region saliency uses local patches, patch histogram distance, area weights, and spatial weights.
- Fusion is weighted linear summation with defaults `0.31`, `0.42`, and `0.27`.

## Infrared NSTSM Paper

NSTSM targets low-altitude slow-small infrared targets. It uses different assumptions and formulas:

- Spatial variance saliency map (SVSM) from a three-layer local contrast window.
- Temporal gray saliency map (TGSM) from three frames `(k-n, k, k+n)`.
- Fusion saliency map (FSM) by multiplication.
- Adaptive threshold `mean + k * std`.

Those infrared formulas are not implemented here because the visible-light paper uses a different temporal cue, different spatial descriptors, and different fusion.

## Implementation Migration

The code expresses the transferable idea as:

```text
temporal_saliency -> spatial_saliency -> fusion -> threshold -> boxes
```

For the visible-light implementation:

```text
FMHI -> pixel CR/ER saliency + region patch saliency -> weighted sum -> Otsu/percentile/fixed -> boxes
```

For NSTSM, a separate implementation would be:

```text
TGSM -> SVSM -> multiplication -> adaptive mean/std threshold -> boxes
```

Keeping these as separate formula families avoids accidentally mixing infrared-specific variance/gray assumptions into the visible-light aerial detector.
