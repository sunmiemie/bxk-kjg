# Input and temporal-state maintenance

These changes were actually implemented and checked on **2026-10-21**. Any historical Git author dates assigned to these commits organize contribution history; they do not mean this implementation or its tests existed on those earlier dates. The committer date and this note retain the actual implementation context.

## Configuration and input behavior

Configuration loading now rejects unknown settings, nonfinite numbers, fractional integer fields, inverted bounds, zero total fusion weights, and invalid region expansion relationships. JSON files may have a UTF-8 BOM. File errors are reported by the command line before creating a synthetic video. Nested configuration lists are copied so one run cannot mutate defaults or another run's settings.

Direct `run_pipeline` calls apply the same validation and require a positive integer `max_frames` when provided. Input videos cannot also be named output artifacts in the same directory. Capture and writer failures release their resources. Missing or invalid FPS values use 10 FPS for output encoding. Video dimensions must remain constant.

Frame helpers accept real numeric NumPy arrays with intensities in `0..255`; they reject empty, nonfinite, negative, or out-of-range pixels instead of wrapping them into `uint8`. Floating values are converted to `uint8` before BGR conversion; normalized floating inputs are not automatically rescaled from `0..1`. Saliency normalization rejects nonfinite values and avoids range overflow. Fusion requires equal 2D map shapes. Empty threshold maps remain empty.

## Detection state and geometry

- Motion history is warped with the same previous-to-current registration transform before adding the next difference image. Newly uncovered history pixels are zero. This is an explicit camera-coordinate convention; the broader algorithm is still an approximation of the referenced paper.
- ROI exclusions are applied both before and after morphology, so closing cannot refill an ignored region or leak beyond the ROI.
- Registration requires equal input dimensions and rejects nonfinite affine/phase results. Unreliable phase results fall back to identity.
- Static detection runs on the first frame when enabled, including one-frame videos. It does not treat a uniformly black ROI as a dark target when the requested contrast is impossible. Initial motion maps remain zero because no preceding frame is available; static boxes may still be present.
- Brightness quantization uses wider integer arithmetic, preserving separate black and white regions at 256 quantization levels.

## Tracking behavior

`vasts-track` expects strictly increasing, nonnegative, unique frame indices. Supplied frames can be sparse; it interpolates only rows that already exist, and its summary counts only boxes actually returned. Endpoint geometry and scores must be finite, widths/heights positive, and scores in `[0, 1]`. A missing score defaults to zero consistently, including interpolation. Observed-box and frame metadata are retained without mutating input data.

The heuristic chooses one strongest box per frame. It does not establish multi-target identity, recover missing video frames, or prove that an interpolated target was visible. `interpolated: true` distinguishes generated boxes. Invalid JSON, unreadable files, and invalid tracking options produce command-line errors. A new `boxes_tracked.json` is written only after overlay rendering succeeds; a failed render may leave a partial video file which should not be used as a completed result.

All validation uses synthetic arrays, temporary files, or local generated videos. No real-data accuracy claim, training result, or paper-level reproduction validation is added.

## Further local validation and output reliability

The following follow-up work was also actually implemented on 2026-10-21:

- Synthetic MP4 dimensions must be even so codecs do not silently crop an edge. Target sizes are exact pixel extents, must fit with a two-pixel margin, and no longer gain an inclusive endpoint pixel. Frame counts, FPS and dimensions are validated before output creation. `synthetic.seed` controls a local random generator (default 42) and is included in saved configuration.
- Source JSON records image dimensions, reported frame count, effective/reported FPS and whether fallback was needed. Frame timestamps are explicitly estimated as `frame_index / fps`; these are not container presentation timestamps or support for variable-frame-rate timing. Registration methods and their totals are recorded for diagnosis.
- JSON results are serialized with finite values and replaced atomically from a temporary file in the same directory. Existing result JSON survives failed replacement. Videos themselves are not transactional; failed runs may leave partial videos. PNG maps use in-memory encoding and Python file paths for Windows Unicode paths.
- ORB/OpenCV failures and degenerate affine estimates fall back to phase/identity registration. Pixels lacking complete previous-frame interpolation support are excluded from temporal difference and history. This prevents border replication from creating apparent motion but can miss genuine objects newly entering the view; static detection remains separate.
- Temporal APIs reject incompatible shapes and invalid intensity/range parameters. One-pixel-wide or one-pixel targets now follow the requested area filters rather than an implicit width/height veto. Border margins use the exact inner rectangle, and zero history never creates candidates at a zero candidate threshold.
- Threshold statistics and normalization use only the supplied valid ROI. Excluded pixels cannot move the selected threshold. Morphology accepts explicit positive odd kernels and canonicalizes boolean/nonzero masks to 0/255. Region expansion clips even wholly outside boxes to valid empty origins; component scores require normalized saliency and safe matching shapes.
- Tracking accepts optional `--max-speed` in pixels per frame. It records why gaps were skipped, visits only represented rows even across extremely large frame-index gaps, and retains existing endpoint observations. This remains a single-target interpolation heuristic. Tracked JSON may not overwrite its input, frames must fit within the decoded video, and changing video dimensions or OpenCV backend errors produce clear CLI failures with resources released.

These changes have offline regression and generated-video coverage. They do not establish real-video detection accuracy, paper reproduction equivalence, or training results.
