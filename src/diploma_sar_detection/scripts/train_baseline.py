from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from utils import configure_ultralytics, resolve_device, resolve_save_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline YOLO object detection model.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics model name or path to checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--project", default="runs", help="Project directory for Ultralytics outputs.")
    parser.add_argument("--name", default="baseline_yolo26", help="Run name inside project directory.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", choices=("ram", "disk"), help="Optional dataset caching mode.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Automatic Mixed Precision.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the provided checkpoint.")
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save checkpoint every N epochs. Use -1 to disable periodic checkpoints.",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use for a smoke test or quick run.",
    )
    return parser.parse_args()


def _print_run_configuration(config: dict[str, Any]) -> None:
    print("\nTraining configuration")
    print("----------------------")
    for key, value in config.items():
        print(f"{key}: {value}")


def _resolve_best_checkpoint(model: Any, fallback_save_dir: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best:
        return Path(best)

    return fallback_save_dir / "weights" / "best.pt"


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    set_seed(args.seed)
    config_dir = configure_ultralytics(args.yolo_config_dir)

    project_dir = Path(args.project).expanduser().resolve()
    fallback_save_dir = project_dir / args.name
    run_config = {
        "data": str(data_path),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": str(project_dir),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "cache": args.cache,
        "amp": args.amp,
        "resume": args.resume,
        "save_period": args.save_period,
        "exist_ok": args.exist_ok,
        "yolo_config_dir": str(config_dir),
        "fraction": args.fraction,
    }
    _print_run_configuration(run_config)

    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
        train_kwargs: dict[str, Any] = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": device,
            "project": str(project_dir),
            "name": args.name,
            "workers": args.workers,
            "patience": args.patience,
            "seed": args.seed,
            "fraction": args.fraction,
            "resume": args.resume,
            "save_period": args.save_period,
            "exist_ok": args.exist_ok,
        }
        if args.cache is not None:
            train_kwargs["cache"] = args.cache
        if args.amp is not None:
            train_kwargs["amp"] = args.amp

        model.train(
            **train_kwargs,
        )
    except ImportError:
        print("Ultralytics is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    save_dir = resolve_save_dir(getattr(model, "trainer", None), fallback_save_dir)
    best_checkpoint = _resolve_best_checkpoint(model, save_dir)
    print("\nTraining finished.")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
