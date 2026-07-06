from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .lite_pipeline import run_lite_pipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Board-oriented v3-lite UAV detector")
    parser.add_argument("--input", required=True, dest="input_path", help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Directory for boxes_lite.json and optional overlay_lite.mp4")
    parser.add_argument("--max-frames", type=_positive_int, default=None, help="Maximum frames to process")
    parser.add_argument("--save-video", action="store_true", help="Save overlay_lite.mp4 for debugging")
    parser.add_argument("--config", default=None, help="JSON config path or inline JSON object")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_lite_pipeline(
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames,
        save_video=args.save_video,
        config=config,
    )
    print(f"Processed {result.frames_processed} frames")
    print(f"Wrote {result.boxes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
