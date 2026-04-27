from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import register_backbone
from utils import configure_ultralytics, resolve_device, resolve_save_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom YOLO26 model with a Swin-based custom backbone scaffold."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument(
        "--model",
        default="models/yolo26_cnn_swin_t.yaml",
        help="Path to the custom model YAML or checkpoint to resume from.",
    )
    parser.add_argument(
        "--backbone-variant",
        default="auto",
        choices=(
            "auto",
            "swin_t",
            "cnn_swin_t",
            "wavevit_s",
            "wavevit_b",
            "wavevit_l",
            "original_wavevit_s",
            "original_wavevit_b",
            "original_wavevit_l",
            "official_wavevit_s",
            "official_wavevit_b",
            "official_wavevit_l",
        ),
        help="Backbone registration variant. Use auto to infer from --model path.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--project", default="runs", help="Project directory for Ultralytics outputs.")
    parser.add_argument("--name", default="yolo26_cnn_swin_t", help="Run name inside project directory.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=("SGD", "Adam", "AdamW", "RMSProp", "auto"),
        help="Optional optimizer override for Ultralytics train().",
    )
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
    parser.add_argument(
        "--pretrained-backbone",
        type=Path,
        help="Optional local checkpoint path for custom backbone pretrained weights.",
    )
    parser.add_argument("--lr0", type=float, help="Optional initial learning rate override for Ultralytics train().")
    parser.add_argument("--lrf", type=float, help="Optional final LR fraction override for Ultralytics train().")
    parser.add_argument("--weight-decay", type=float, help="Optional weight decay override for Ultralytics train().")
    parser.add_argument("--mosaic", type=float, help="Optional mosaic augmentation probability override.")
    return parser.parse_args()


def _print_run_configuration(config: dict[str, Any]) -> None:
    print("\nSwin-based training configuration")
    print("--------------------------------")
    for key, value in config.items():
        print(f"{key}: {value}")


def _resolve_backbone_variant(arg_variant: str, model_path: Path) -> str:
    if arg_variant != "auto":
        return arg_variant

    model_name = model_path.name.lower()
    if "official_wavevit_l" in model_name:
        return "official_wavevit_l"
    if "official_wavevit_b" in model_name:
        return "official_wavevit_b"
    if "official_wavevit" in model_name:
        return "official_wavevit_s"
    if "original_wavevit_l" in model_name:
        return "original_wavevit_l"
    if "original_wavevit_b" in model_name:
        return "original_wavevit_b"
    if "original_wavevit" in model_name:
        return "original_wavevit_s"
    if "cnn_swin" in model_name:
        return "cnn_swin_t"
    if "swin" in model_name:
        return "swin_t"
    if "wavevit_l" in model_name:
        return "wavevit_l"
    if "wavevit_b" in model_name:
        return "wavevit_b"
    if "wavevit" in model_name:
        return "wavevit_s"
    return "cnn_swin_t"


def _resolve_best_checkpoint(model: Any, fallback_save_dir: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best:
        return Path(best)

    return fallback_save_dir / "weights" / "best.pt"


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    model_path = Path(args.model).expanduser()

    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2
    if model_path.suffix in {".yaml", ".yml"} and not model_path.resolve().is_file():
        print(f"Model config not found: {model_path.resolve()}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    set_seed(args.seed)
    config_dir = configure_ultralytics(args.yolo_config_dir)
    project_dir = Path(args.project).expanduser().resolve()
    fallback_save_dir = project_dir / args.name
    backbone_variant = _resolve_backbone_variant(args.backbone_variant, model_path)

    run_config = {
        "data": str(data_path),
        "model": str(model_path),
        "backbone_variant": backbone_variant,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "project": str(project_dir),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "cache": args.cache,
        "amp": args.amp,
        "resume": args.resume,
        "save_period": args.save_period,
        "exist_ok": args.exist_ok,
        "yolo_config_dir": str(config_dir),
        "fraction": args.fraction,
        "pretrained_backbone": args.pretrained_backbone,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "mosaic": args.mosaic,
    }
    _print_run_configuration(run_config)

    try:
        from ultralytics import YOLO

        register_backbone(backbone_variant)
        model = YOLO(str(model_path))
        if args.pretrained_backbone is not None:
            backbone = model.model.model[0]
            load_pretrained = getattr(backbone, "load_pretrained", None)
            if not callable(load_pretrained):
                raise RuntimeError(
                    f"Backbone `{type(backbone).__name__}` does not support --pretrained-backbone loading."
                )
            load_pretrained(args.pretrained_backbone)
            model.ckpt = {"model": model.model, "epoch": -1, "optimizer": None}
            model.ckpt_path = str(args.pretrained_backbone)
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
        if args.optimizer is not None:
            train_kwargs["optimizer"] = args.optimizer
        if args.amp is not None:
            train_kwargs["amp"] = args.amp
        if args.lr0 is not None:
            train_kwargs["lr0"] = args.lr0
        if args.lrf is not None:
            train_kwargs["lrf"] = args.lrf
        if args.weight_decay is not None:
            train_kwargs["weight_decay"] = args.weight_decay
        if args.mosaic is not None:
            train_kwargs["mosaic"] = args.mosaic

        model.train(**train_kwargs)
    except ImportError as exc:
        print(f"Missing dependency for custom Swin-based training: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Swin-based training failed: {exc}", file=sys.stderr)
        return 1

    save_dir = resolve_save_dir(getattr(model, "trainer", None), fallback_save_dir)
    best_checkpoint = _resolve_best_checkpoint(model, save_dir)
    print("\nSwin-based training finished.")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
