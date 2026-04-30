from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from custom_models import register_context_modules
from train_swin_context import _load_pretrained_weights
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained context/fusion module parameters against initialized + transferred state."
    )
    parser.add_argument("--model-yaml", required=True, type=Path, help="Model YAML used to create the run.")
    parser.add_argument("--weights", required=True, type=Path, help="Baseline checkpoint loaded before training.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained checkpoint to inspect.")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[11, 12],
        help="Layer indices to compare. Defaults to context/fusion layers 11 and 12.",
    )
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    from ultralytics.nn.tasks import torch_safe_load

    checkpoint, _ = torch_safe_load(str(path))
    model = checkpoint.get("ema") or checkpoint["model"]
    return model.float().state_dict()


def _tensor_delta(init_state: dict[str, torch.Tensor], trained_state: dict[str, torch.Tensor], key: str) -> str:
    init_value = init_state.get(key)
    trained_value = trained_state.get(key)
    if init_value is None or trained_value is None:
        return "missing"
    if init_value.shape != trained_value.shape:
        return f"shape_mismatch init={tuple(init_value.shape)} trained={tuple(trained_value.shape)}"

    diff = (trained_value.detach().float().cpu() - init_value.detach().float().cpu()).abs()
    base = init_value.detach().float().cpu().abs()
    rel = diff.mean() / (base.mean() + 1e-12)
    return (
        f"mean_abs={diff.mean().item():.8f} "
        f"max_abs={diff.max().item():.8f} "
        f"rel_mean={rel.item():.8f}"
    )


def _group_delta(
    init_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
    prefix: str,
) -> tuple[int, float, float, float]:
    total_elements = 0
    sum_abs = 0.0
    max_abs = 0.0
    sum_base_abs = 0.0

    for key, init_value in init_state.items():
        if not key.startswith(prefix):
            continue
        trained_value = trained_state.get(key)
        if trained_value is None or init_value.shape != trained_value.shape:
            continue
        diff = (trained_value.detach().float().cpu() - init_value.detach().float().cpu()).abs()
        base = init_value.detach().float().cpu().abs()
        total_elements += diff.numel()
        sum_abs += float(diff.sum().item())
        max_abs = max(max_abs, float(diff.max().item()))
        sum_base_abs += float(base.sum().item())

    if total_elements == 0:
        return 0, 0.0, 0.0, 0.0
    mean_abs = sum_abs / total_elements
    rel_mean = sum_abs / (sum_base_abs + 1e-12)
    return total_elements, mean_abs, max_abs, rel_mean


def _print_alpha_like_stats(layer: Any, layer_index: int) -> None:
    raw_alpha = getattr(layer, "raw_alpha", None)
    if raw_alpha is not None:
        alpha = torch.sigmoid(raw_alpha.detach().float().cpu())
        print(
            f"model.{layer_index}.alpha: "
            f"mean={alpha.mean().item():.6f} "
            f"min={alpha.min().item():.6f} "
            f"max={alpha.max().item():.6f} "
            f"std={alpha.std().item():.6f}"
        )

    detail_strength = getattr(layer, "detail_strength", None)
    if detail_strength is not None:
        print(f"model.{layer_index}.detail_strength: {float(detail_strength.detach().cpu().item()):.6f}")

    beta = getattr(layer, "beta", None)
    if beta is not None:
        print(f"model.{layer_index}.beta: {float(beta.detach().cpu().item()):.6f}")


def main() -> int:
    args = parse_args()
    model_yaml = args.model_yaml.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    for path, label in [(model_yaml, "Model YAML"), (weights, "Baseline weights"), (checkpoint, "Checkpoint")]:
        if not path.is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    configure_ultralytics(args.yolo_config_dir)
    register_context_modules()

    try:
        from ultralytics import YOLO

        init_model = YOLO(str(model_yaml))
        _load_pretrained_weights(init_model, weights)
        init_state = {key: value.detach().cpu().clone() for key, value in init_model.model.state_dict().items()}

        trained_model = YOLO(str(checkpoint))
        trained_state = _state_dict_from_checkpoint(checkpoint)

        print("Context parameter delta report")
        print("------------------------------")
        print(f"model_yaml:  {model_yaml}")
        print(f"weights:     {weights}")
        print(f"checkpoint:  {checkpoint}")

        for layer_index in args.layers:
            layer = trained_model.model.model[layer_index]
            print(f"\nLayer {layer_index}: {layer.__class__.__name__}")
            _print_alpha_like_stats(layer, layer_index)

            keys = [
                f"model.{layer_index}.raw_alpha",
                f"model.{layer_index}.detail_strength",
                f"model.{layer_index}.beta",
            ]
            for key in keys:
                if key in init_state or key in trained_state:
                    print(f"{key}: {_tensor_delta(init_state, trained_state, key)}")

            prefixes = [
                "swin",
                "wavevit",
                "channel_gate",
                "spatial_gate",
                "delta_proj",
            ]
            for child_prefix in prefixes:
                prefix = f"model.{layer_index}.{child_prefix}."
                count, mean_abs, max_abs, rel_mean = _group_delta(init_state, trained_state, prefix)
                if count:
                    print(
                        f"{prefix}*: elements={count} "
                        f"mean_abs={mean_abs:.8f} max_abs={max_abs:.8f} rel_mean={rel_mean:.8f}"
                    )

            prefix = f"model.{layer_index}."
            count, mean_abs, max_abs, rel_mean = _group_delta(init_state, trained_state, prefix)
            print(
                f"{prefix}ALL: elements={count} "
                f"mean_abs={mean_abs:.8f} max_abs={max_abs:.8f} rel_mean={rel_mean:.8f}"
            )

        return 0
    except Exception as exc:
        print(f"Parameter comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
