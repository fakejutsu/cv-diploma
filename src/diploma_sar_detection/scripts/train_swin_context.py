from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import register_context_modules
from utils import configure_ultralytics, resolve_device, resolve_save_dir, set_seed


_BASELINE_TO_CONTEXT_LAYER_REMAP = {
    **{index: index for index in range(0, 11)},
    11: 14,
    12: 15,
    13: 16,
    14: 17,
    15: 18,
    16: 19,
    17: 20,
    18: 21,
    19: 22,
    20: 23,
    21: 24,
    22: 25,
    23: 26,
}

_BASELINE_TO_GATED_P4_P5_LAYER_REMAP = {
    **{index: index for index in range(0, 11)},
    11: 13,
    12: 14,
    13: 15,
    14: 16,
    15: 17,
    16: 18,
    17: 19,
    18: 20,
    19: 21,
    20: 22,
    21: 23,
    22: 24,
    23: 25,
}

_WEIGHT_TRANSFER_STRATEGIES = {
    "exact": None,
    "context_shift3": _BASELINE_TO_CONTEXT_LAYER_REMAP,
    "gated_shift2": _BASELINE_TO_GATED_P4_P5_LAYER_REMAP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n with a Swin-style context block on P5.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument(
        "--model",
        default="models/yolo26n_swin_context_p5.yaml",
        help="Path to the context-augmented model YAML.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional checkpoint to load into the model before training, e.g. yolo26n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--project", default="runs", help="Project directory for Ultralytics outputs.")
    parser.add_argument("--name", default="yolo26n_swin_context_p5", help="Run name inside project directory.")
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
    return parser.parse_args()


def _print_run_configuration(config: dict[str, Any]) -> None:
    print("\nSwin-context training configuration")
    print("---------------------------------")
    for key, value in config.items():
        print(f"{key}: {value}")


def _resolve_best_checkpoint(model: Any, fallback_save_dir: Path) -> Path:
    trainer = getattr(model, "trainer", None)
    best = getattr(trainer, "best", None)
    if best:
        return Path(best)

    return fallback_save_dir / "weights" / "best.pt"


def _remap_model_key(key: str, index_remap: dict[int, int]) -> str:
    parts = key.split(".")
    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
        layer_index = int(parts[1])
        if layer_index in index_remap:
            parts[1] = str(index_remap[layer_index])
    return ".".join(parts)


def _collect_compatible_weights(
    source_state_dict: dict[str, Any],
    target_state_dict: dict[str, Any],
    *,
    index_remap: dict[int, int] | None = None,
) -> tuple[dict[str, Any], Counter[int]]:
    compatible: dict[str, Any] = {}
    counts: Counter[int] = Counter()

    for source_key, value in source_state_dict.items():
        target_key = _remap_model_key(source_key, index_remap or {}) if index_remap else source_key
        target_value = target_state_dict.get(target_key)
        if target_value is None or getattr(target_value, "shape", None) != getattr(value, "shape", None):
            continue

        compatible[target_key] = value
        parts = target_key.split(".")
        if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
            counts[int(parts[1])] += 1

    return compatible, counts


def _load_pretrained_weights(model: Any, weights_path: Path) -> None:
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(weights_path))
    pretrained_model = checkpoint.get("ema") or checkpoint["model"]
    source_state_dict = pretrained_model.float().state_dict()
    target_state_dict = model.model.state_dict()

    selected_weights: dict[str, Any] = {}
    selected_counts: Counter[int] = Counter()
    strategy = "exact"

    for strategy_name, index_remap in _WEIGHT_TRANSFER_STRATEGIES.items():
        candidate_weights, candidate_counts = _collect_compatible_weights(
            source_state_dict,
            target_state_dict,
            index_remap=index_remap,
        )
        if len(candidate_weights) > len(selected_weights):
            selected_weights = candidate_weights
            selected_counts = candidate_counts
            strategy = strategy_name

    model.model.load_state_dict(selected_weights, strict=False)

    detect_layer = model.model.model[-1]
    detect_matches = selected_counts.get(int(getattr(detect_layer, "i", len(model.model.model) - 1)), 0)
    print(
        f"Pretrained weight transfer: strategy={strategy}, "
        f"matched={len(selected_weights)}/{len(target_state_dict)}, detect_matches={detect_matches}"
    )


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    model_path = Path(args.model).expanduser()
    weights_path = args.weights.expanduser().resolve() if args.weights else None

    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2
    if model_path.suffix in {".yaml", ".yml"} and not model_path.resolve().is_file():
        print(f"Model config not found: {model_path.resolve()}", file=sys.stderr)
        return 2
    if weights_path is not None and not weights_path.is_file():
        print(f"Weights checkpoint not found: {weights_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    set_seed(args.seed)
    config_dir = configure_ultralytics(args.yolo_config_dir)
    project_dir = Path(args.project).expanduser().resolve()
    fallback_save_dir = project_dir / args.name

    run_config = {
        "data": str(data_path),
        "model": str(model_path),
        "weights": str(weights_path) if weights_path else None,
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
    }
    _print_run_configuration(run_config)

    try:
        from ultralytics import YOLO

        register_context_modules()
        model = YOLO(str(model_path))
        if weights_path is not None:
            _load_pretrained_weights(model, weights_path)

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

        model.train(**train_kwargs)
    except ImportError as exc:
        print(f"Missing dependency for Swin-context training: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Swin-context training failed: {exc}", file=sys.stderr)
        return 1

    save_dir = resolve_save_dir(getattr(model, "trainer", None), fallback_save_dir)
    best_checkpoint = _resolve_best_checkpoint(model, save_dir)
    print("\nSwin-context training finished.")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
