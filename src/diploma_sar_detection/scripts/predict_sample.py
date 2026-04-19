from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import configure_ultralytics, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a sample image or directory.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model checkpoint.")
    parser.add_argument("--source", required=True, type=Path, help="Path to an image or directory with images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory to save rendered predictions.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not model_path.is_file():
        print(f"Model checkpoint not found: {model_path}", file=sys.stderr)
        return 2
    if not source_path.exists():
        print(f"Source path not found: {source_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    config_dir = configure_ultralytics(args.yolo_config_dir)

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(source_path),
            imgsz=args.imgsz,
            device=device,
            conf=args.conf,
            save=True,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
        )
    except ImportError:
        print("Ultralytics is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print("\nInference finished.")
    print(f"Processed items: {len(results)}")
    print(f"YOLO config directory: {config_dir}")
    print(f"Predictions saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
