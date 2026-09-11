from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from .config import load_config
from .pipeline import run_pipeline
from .synthetic import make_synthetic_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visible aerial spatiotemporal saliency detector")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Generate and process a synthetic aerial video")
    source.add_argument("--input", dest="input_path", help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Directory for boxes.json, videos, and optional maps")
    parser.add_argument("--max-frames", type=_positive_int, default=None, help="Maximum frames to process")
    parser.add_argument("--save-maps", action="store_true", help="Save diff/FMHI/pixel/region/fused/mask images")
    parser.add_argument("--save-video", action="store_true", help="Save overlay.mp4 and saliency.mp4")
    parser.add_argument("--config", default=None, help="JSON config path or inline JSON object")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-frames must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.error(str(exc))


def _run(args) -> int:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        synth_cfg = config["synthetic"]
        frame_count = args.max_frames if args.max_frames is not None else int(synth_cfg["frame_count"])
        input_path = make_synthetic_video(
            output_dir / "synthetic_input.mp4",
            frame_count=frame_count,
            size=(int(synth_cfg["width"]), int(synth_cfg["height"])),
            fps=float(synth_cfg["fps"]),
            seed=synth_cfg["seed"],
        )
        source_mode = "synthetic"
    elif args.input_path:
        input_path = Path(args.input_path)
        source_mode = "input"
    else:
        raise ValueError("Provide --synthetic or --input VIDEO")

    result = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        max_frames=args.max_frames,
        save_maps=args.save_maps,
        save_video=args.save_video,
        config=config,
        source_mode=source_mode,
    )
    print(f"Processed {result.frames_processed} frames")
    print(f"Wrote {result.boxes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
