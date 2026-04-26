from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import AdaptiveDetailGatedSwinFusion, GatedSwinFusion, register_context_modules
from utils import configure_ultralytics, resolve_device, resolve_save_dir, set_seed


FusionModule = GatedSwinFusion | AdaptiveDetailGatedSwinFusion


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
    "adaptive_gated_shift2": _BASELINE_TO_GATED_P4_P5_LAYER_REMAP,
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
    parser.add_argument("--lr0", type=float, help="Optional initial learning rate override for Ultralytics train().")
    parser.add_argument("--mosaic", type=float, help="Optional mosaic augmentation probability override.")
    parser.add_argument(
        "--swin-p4-weights",
        "--swin_p4_weights",
        dest="swin_p4_weights",
        type=Path,
        help="Optional checkpoint with pretrained SwinContextBlock for the P4 gated branch.",
    )
    parser.add_argument(
        "--swin-p5-weights",
        "--swin_p5_weights",
        dest="swin_p5_weights",
        type=Path,
        help="Optional checkpoint with pretrained SwinContextBlock for the P5 gated branch.",
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


def _preferred_weight_transfer_strategy(model: Any) -> str | None:
    for layer in model.model.model:
        if isinstance(layer, AdaptiveDetailGatedSwinFusion):
            return "adaptive_gated_shift2"
    return None


def _load_pretrained_weights(model: Any, weights_path: Path) -> None:
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(weights_path))
    pretrained_model = checkpoint.get("ema") or checkpoint["model"]
    source_state_dict = pretrained_model.float().state_dict()
    target_state_dict = model.model.state_dict()

    selected_weights: dict[str, Any] = {}
    selected_counts: Counter[int] = Counter()
    strategy = "exact"
    preferred_strategy = _preferred_weight_transfer_strategy(model)

    for strategy_name, index_remap in _WEIGHT_TRANSFER_STRATEGIES.items():
        candidate_weights, candidate_counts = _collect_compatible_weights(
            source_state_dict,
            target_state_dict,
            index_remap=index_remap,
        )
        is_better = len(candidate_weights) > len(selected_weights)
        is_preferred_tie = (
            preferred_strategy is not None
            and strategy_name == preferred_strategy
            and len(candidate_weights) == len(selected_weights)
            and strategy != preferred_strategy
        )
        if is_better or is_preferred_tie:
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


def _load_checkpoint_state_dict(weights_path: Path) -> dict[str, Any]:
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(weights_path))
    pretrained_model = checkpoint.get("ema") or checkpoint["model"]
    return pretrained_model.float().state_dict()


def _find_fusion_modules(model: Any) -> dict[int, FusionModule]:
    modules: dict[int, FusionModule] = {}
    for layer in model.model.model:
        if isinstance(layer, (GatedSwinFusion, AdaptiveDetailGatedSwinFusion)):
            modules[layer.in_channels] = layer
    return modules


def _candidate_prefixes_for_swin_state(state_dict: dict[str, Any]) -> list[str]:
    candidates = set()
    for key in state_dict:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            candidates.add(".".join(parts[:2]))
    return sorted(candidates)


def _count_swin_matches(state_dict: dict[str, Any], target_module: FusionModule, prefix: str) -> tuple[int, int]:
    own_state = target_module.swin.state_dict()
    matched = 0
    dotted_prefix = prefix.rstrip(".") + "."
    for key, value in state_dict.items():
        if not key.startswith(dotted_prefix):
            continue
        new_key = key[len(dotted_prefix) :]
        own_value = own_state.get(new_key)
        if own_value is not None and own_value.shape == getattr(value, "shape", None):
            matched += 1
    return matched, len(own_state)


def _best_swin_prefix(state_dict: dict[str, Any], target_module: FusionModule) -> tuple[str | None, int, int]:
    best_prefix: str | None = None
    best_matched = 0
    target_total = len(target_module.swin.state_dict())
    for prefix in _candidate_prefixes_for_swin_state(state_dict):
        matched, _ = _count_swin_matches(state_dict, target_module, prefix)
        if matched > best_matched:
            best_prefix = prefix
            best_matched = matched
    if best_prefix is None:
        return None, 0, target_total
    return best_prefix, best_matched, target_total


def load_pretrained_gated_swin_weights(
    model: Any,
    p4_ckpt: Path | None = None,
    p5_ckpt: Path | None = None,
) -> dict[str, Any]:
    fusion_modules = _find_fusion_modules(model)
    p4_gate = fusion_modules.get(128)
    p5_gate = fusion_modules.get(256)

    result: dict[str, Any] = {
        "p4_source": None,
        "p5_source": None,
        "p4_prefix": None,
        "p5_prefix": None,
        "p4_matched": 0,
        "p5_matched": 0,
        "p4_total": len(p4_gate.swin.state_dict()) if p4_gate else 0,
        "p5_total": len(p5_gate.swin.state_dict()) if p5_gate else 0,
        "alpha_mean_p4_before": p4_gate.alpha_mean if p4_gate else None,
        "alpha_mean_p5_before": p5_gate.alpha_mean if p5_gate else None,
        "alpha_mean_p4_after": p4_gate.alpha_mean if p4_gate else None,
        "alpha_mean_p5_after": p5_gate.alpha_mean if p5_gate else None,
        "trainable": bool(
            (p4_gate is None or all(param.requires_grad for param in p4_gate.parameters()))
            and (p5_gate is None or all(param.requires_grad for param in p5_gate.parameters()))
        ),
    }

    if p4_ckpt is not None and p4_gate is not None:
        source_state = _load_checkpoint_state_dict(p4_ckpt)
        result["p4_source"] = str(p4_ckpt)
        prefix, matched, total = _best_swin_prefix(source_state, p4_gate)
        if prefix is not None:
            matched, total = p4_gate.load_swin_weights(source_state, prefix)
            result["p4_prefix"] = prefix
            result["p4_matched"] = matched
            result["p4_total"] = total

    if p5_ckpt is not None and p5_gate is not None:
        source_state = _load_checkpoint_state_dict(p5_ckpt)
        result["p5_source"] = str(p5_ckpt)
        prefix, matched, total = _best_swin_prefix(source_state, p5_gate)
        if prefix is not None:
            matched, total = p5_gate.load_swin_weights(source_state, prefix)
            result["p5_prefix"] = prefix
            result["p5_matched"] = matched
            result["p5_total"] = total

    result["alpha_mean_p4_after"] = p4_gate.alpha_mean if p4_gate else None
    result["alpha_mean_p5_after"] = p5_gate.alpha_mean if p5_gate else None
    result["trainable"] = bool(
        (p4_gate is None or all(param.requires_grad for param in p4_gate.parameters()))
        and (p5_gate is None or all(param.requires_grad for param in p5_gate.parameters()))
    )
    return result


def _log_composite_swin_warmstart(result: dict[str, Any]) -> None:
    print("Composite Swin warm-start:")
    print(f"P4 source: {result['p4_source'] or 'not provided'}")
    if result["p4_source"]:
        print(f"P4 prefix: {result['p4_prefix']}")
        print(f"P4 matched: {result['p4_matched']}/{result['p4_total']} tensors")
    print(f"P5 source: {result['p5_source'] or 'not provided'}")
    if result["p5_source"]:
        print(f"P5 prefix: {result['p5_prefix']}")
        print(f"P5 matched: {result['p5_matched']}/{result['p5_total']} tensors")
    print(
        "raw_alpha before/after: "
        f"p4={result['alpha_mean_p4_before']} -> {result['alpha_mean_p4_after']}, "
        f"p5={result['alpha_mean_p5_before']} -> {result['alpha_mean_p5_after']}"
    )
    print(f"Swin modules remain trainable: {result['trainable']}")


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    model_path = Path(args.model).expanduser()
    weights_path = args.weights.expanduser().resolve() if args.weights else None
    swin_p4_weights = args.swin_p4_weights.expanduser().resolve() if args.swin_p4_weights else None
    swin_p5_weights = args.swin_p5_weights.expanduser().resolve() if args.swin_p5_weights else None

    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2
    if model_path.suffix in {".yaml", ".yml"} and not model_path.resolve().is_file():
        print(f"Model config not found: {model_path.resolve()}", file=sys.stderr)
        return 2
    if weights_path is not None and not weights_path.is_file():
        print(f"Weights checkpoint not found: {weights_path}", file=sys.stderr)
        return 2
    if swin_p4_weights is not None and not swin_p4_weights.is_file():
        print(f"P4 Swin checkpoint not found: {swin_p4_weights}", file=sys.stderr)
        return 2
    if swin_p5_weights is not None and not swin_p5_weights.is_file():
        print(f"P5 Swin checkpoint not found: {swin_p5_weights}", file=sys.stderr)
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
        "swin_p4_weights": str(swin_p4_weights) if swin_p4_weights else None,
        "swin_p5_weights": str(swin_p5_weights) if swin_p5_weights else None,
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
        "lr0": args.lr0,
        "mosaic": args.mosaic,
    }
    _print_run_configuration(run_config)

    try:
        from ultralytics import YOLO

        register_context_modules()
        model = YOLO(str(model_path))
        if weights_path is not None:
            _load_pretrained_weights(model, weights_path)
        if swin_p4_weights is not None or swin_p5_weights is not None:
            composite_result = load_pretrained_gated_swin_weights(
                model,
                p4_ckpt=swin_p4_weights,
                p5_ckpt=swin_p5_weights,
            )
            _log_composite_swin_warmstart(composite_result)

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
        if args.mosaic is not None:
            train_kwargs["mosaic"] = args.mosaic

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
