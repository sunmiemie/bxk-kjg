from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from .pipeline import _create_writer, _draw_overlay, _open_capture, _capture_fps
from .tracking import fill_single_track_gaps
from .json_output import write_json_atomic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill short missed detections with a single-target trajectory model")
    parser.add_argument("--input-video", required=True, help="Original input video path")
    parser.add_argument("--boxes-json", required=True, help="Detector boxes.json path")
    parser.add_argument("--output-dir", required=True, help="Directory for boxes_tracked.json and overlay_tracked.mp4")
    parser.add_argument("--max-gap", type=int, default=10, help="Maximum missing-frame gap to interpolate")
    parser.add_argument("--max-distance", type=float, default=80.0, help="Maximum center distance for linking boxes")
    parser.add_argument("--max-speed", type=float, default=None, help="Optional maximum interpolation speed in pixels per frame")
    return parser


def _draw_padded_overlay(frame, boxes, frame_index: int):
    padded = []
    height, width = frame.shape[:2]
    for box in boxes:
        pad = 6 if box.get("interpolated") else 4
        x = max(0, int(box["x"]) - pad)
        y = max(0, int(box["y"]) - pad)
        right = min(width, int(box["x"]) + int(box["w"]) + pad)
        bottom = min(height, int(box["y"]) + int(box["h"]) + pad)
        padded_box = dict(box)
        padded_box["x"] = x
        padded_box["y"] = y
        padded_box["w"] = max(1, right - x)
        padded_box["h"] = max(1, bottom - y)
        padded.append(padded_box)
    return _draw_overlay(frame, padded, frame_index)


def render_tracked_overlay(input_video: str | Path, frames: list[dict], output_path: str | Path) -> None:
    if Path(input_video).resolve() == Path(output_path).resolve():
        raise ValueError("Tracked overlay must not overwrite the input video")
    capture = _open_capture(input_video)
    writer = None
    try:
        fps = _capture_fps(capture)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Input video has no frames: {input_video}")
        height, width = frame.shape[:2]
        writer = _create_writer(Path(output_path), fps, (width, height))
        frame_lookup = {int(item["frame_index"]): item.get("boxes", []) for item in frames}
        frame_index = 0
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError("Input video dimensions changed during tracked rendering")
            writer.write(_draw_padded_overlay(frame, frame_lookup.get(frame_index, []), frame_index))
            frame_index += 1
            ok, frame = capture.read()
        if frame_lookup and max(frame_lookup) >= frame_index:
            raise ValueError("Tracking frame indices exceed the decoded input video")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_gap < 0:
        parser.error("--max-gap must be >= 0")
    if not math.isfinite(args.max_distance) or args.max_distance <= 0:
        parser.error("--max-distance must be finite and > 0")
    try:
        return _run(args)
    except (OSError, UnicodeError, ValueError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


def _run(args) -> int:

    boxes_path = Path(args.boxes_json)
    payload = json.loads(boxes_path.read_text(encoding="utf-8-sig"))
    tracked = fill_single_track_gaps(payload, max_gap=args.max_gap, max_distance=args.max_distance, max_speed_pixels_per_frame=args.max_speed)

    out_dir = Path(args.output_dir)
    if boxes_path.resolve() == (out_dir / "boxes_tracked.json").resolve():
        raise ValueError("Tracking output must not overwrite its input boxes JSON")
    out_dir.mkdir(parents=True, exist_ok=True)
    tracked_path = out_dir / "boxes_tracked.json"
    render_tracked_overlay(args.input_video, tracked["frames"], out_dir / "overlay_tracked.mp4")
    write_json_atomic(tracked_path, tracked)

    print(f"Wrote {tracked_path}")
    print(f"Wrote {out_dir / 'overlay_tracked.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
