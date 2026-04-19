from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import configure_ultralytics, extract_detection_metrics, resolve_device, resolve_save_dir, timestamp_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone validation for a trained YOLO model.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model checkpoint.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--split", default="val", help="Dataset split to validate: val or test.")
    parser.add_argument("--project", default="runs", help="Project directory for validation artifacts.")
    parser.add_argument("--name", help="Optional run name for validation artifacts.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def register_custom_backbones_if_available() -> None:
    """
    Best-effort registration of local custom backbones.
    Required for checkpoints that depend on repository-local modules (e.g. Swin-T wrapper).
    """

    try:
        from custom_models import register_swin_t_backbone

        register_swin_t_backbone()
    except Exception:
        # Keep baseline validation working even if local custom modules are not available.
        return


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()

    if not model_path.is_file():
        print(f"Model checkpoint not found: {model_path}", file=sys.stderr)
        return 2
    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    config_dir = configure_ultralytics(args.yolo_config_dir)
    project_dir = Path(args.project).expanduser().resolve()
    run_name = args.name or f"val_{model_path.stem}_{args.split}_{timestamp_tag()}"

    try:
        from ultralytics import YOLO

        register_custom_backbones_if_available()
        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(data_path),
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            split=args.split,
            project=str(project_dir),
            name=run_name,
        )
    except ImportError:
        print("Ultralytics is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    main_metrics = extract_detection_metrics(metrics)
    print("\nValidation metrics")
    print("------------------")
    for metric_name, value in main_metrics.items():
        formatted = f"{value:.4f}" if value is not None else "n/a"
        print(f"{metric_name}: {formatted}")

    print(f"YOLO config directory: {config_dir}")
    save_dir = resolve_save_dir(metrics, project_dir / run_name)
    print(f"\nValidation artifacts saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
