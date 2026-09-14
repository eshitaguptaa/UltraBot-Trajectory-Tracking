"""CLI for the existing UltraBot YOLOv8 + LSTM HUD demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultrabot.pipeline import DEFAULT_OUTPUT, DEFAULT_VIDEO, LSTM_CHECKPOINT, YOLO_WEIGHTS, run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run UltraBot YOLOv8 + LSTM demo")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--lstm", type=Path, default=LSTM_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--show", action="store_true", help="Ignored; kept for compatibility. Use the Streamlit app for a live view.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = run_pipeline(
            video_path=args.video,
            output_path=args.output,
            weights_path=args.weights,
            lstm_path=args.lstm,
            max_frames=args.max_frames,
            progress_cb=lambda cur, total, det: (
                print(f"  frame {cur}/{total} detections={det}")
                if cur == 1 or cur % 20 == 0 or cur == total
                else None
            ),
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Saved {result.frames_processed} frames to {result.output_video}")
    print(f"Associated detections: {result.detections}/{result.frames_processed}")
    print(f"Primary track: {result.primary_track_id}")
    print(f"Tracks: {result.tracks}")
    if args.show:
        print("Live OpenCV window is not used here. Open the dashboard with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
