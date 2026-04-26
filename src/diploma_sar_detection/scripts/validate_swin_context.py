from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import register_context_modules
from utils import configure_ultralytics
from train_swin_context import _load_pretrained_weights, load_pretrained_gated_swin_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check the YOLO26n Swin-context P5 integration with dummy forward passes."
    )
    parser.add_argument("--baseline-model", default="yolo26n.pt", help="Baseline model name or checkpoint.")
    parser.add_argument(
        "--context-model-yaml",
        type=Path,
        default=Path("models/yolo26n_swin_context_p5.yaml"),
        help="Path to the context-augmented model YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Dummy input image size.")
    parser.add_argument(
        "--variant",
        choices=("auto", "p5", "p4_light", "gated_p4_p5", "adaptive_gated_p4_p5"),
        default="auto",
        help="Expected context variant. Use auto to infer from YAML filename.",
    )
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
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


def _count_parameters(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _capture_layer_output(container: dict[str, Any], key: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        container[key] = output

    return hook


def _resolve_variant(variant: str, model_yaml_path: Path) -> str:
    if variant != "auto":
        return variant

    model_name = model_yaml_path.name.lower()
    if "adaptive_detail_gated_swin_p4_p5" in model_name:
        return "adaptive_gated_p4_p5"
    if "gated_swin_p4_p5" in model_name:
        return "gated_p4_p5"
    if "p4_light" in model_name:
        return "p4_light"
    return "p5"


def _hook_indices(variant: str) -> dict[str, int]:
    if variant in {"gated_p4_p5", "adaptive_gated_p4_p5"}:
        return {
            "p4_backbone": 6,
            "p4_out": 11,
            "p5_backbone": 10,
            "p5_out": 12,
        }
    if variant == "p4_light":
        return {
            "feature_backbone": 6,
            "feature_context": 11,
            "feature_fused": 13,
        }
    if variant == "p5":
        return {
            "feature_backbone": 10,
            "feature_context": 11,
            "feature_fused": 13,
        }
    raise ValueError(f"Unsupported context variant: {variant}")


def _print_layer_table(yolo_model: Any) -> None:
    print("\nLayer table")
    print("-----------")
    for index, layer in enumerate(yolo_model.model.model):
        print(f"{index:2d} | from={str(layer.f):<10} | type={layer.type}")


def _print_gate_stats(name: str, gate: Any, *, detail: bool = False) -> None:
    print(f"alpha_mean_{name}: {gate.alpha_mean:.6f}")
    if hasattr(gate, "alpha_min") and hasattr(gate, "alpha_max"):
        print(f"alpha_min_{name}: {gate.alpha_min:.6f}")
        print(f"alpha_max_{name}: {gate.alpha_max:.6f}")
    if detail:
        print(f"detail_mean_{name}: {gate.detail_mean:.6f}")
        print(f"detail_max_{name}: {gate.detail_max:.6f}")
        print(f"detail_bias_mean_{name}: {gate.detail_bias_mean:.6f}")
        print(f"detail_bias_max_{name}: {gate.detail_bias_max:.6f}")


def main() -> int:
    args = parse_args()
    model_yaml_path = args.context_model_yaml.expanduser().resolve()
    swin_p4_weights = args.swin_p4_weights.expanduser().resolve() if args.swin_p4_weights else None
    swin_p5_weights = args.swin_p5_weights.expanduser().resolve() if args.swin_p5_weights else None
    if not model_yaml_path.is_file():
        print(f"Context model YAML not found: {model_yaml_path}", file=sys.stderr)
        return 2
    if swin_p4_weights is not None and not swin_p4_weights.is_file():
        print(f"P4 Swin checkpoint not found: {swin_p4_weights}", file=sys.stderr)
        return 2
    if swin_p5_weights is not None and not swin_p5_weights.is_file():
        print(f"P5 Swin checkpoint not found: {swin_p5_weights}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)
    variant = _resolve_variant(args.variant, model_yaml_path)

    try:
        from ultralytics import YOLO

        register_context_modules()

        baseline = YOLO(args.baseline_model)
        context = YOLO(str(model_yaml_path))
        if Path(args.baseline_model).expanduser().is_file():
            _load_pretrained_weights(context, Path(args.baseline_model).expanduser().resolve())

        composite_result = None
        if swin_p4_weights is not None or swin_p5_weights is not None:
            composite_result = load_pretrained_gated_swin_weights(
                context,
                p4_ckpt=swin_p4_weights,
                p5_ckpt=swin_p5_weights,
            )

        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        captured: dict[str, Any] = {}
        hook_indices = _hook_indices(variant)
        hooks = [
            context.model.model[layer_index].register_forward_hook(_capture_layer_output(captured, key))
            for key, layer_index in hook_indices.items()
        ]

        with torch.no_grad():
            baseline_output = baseline.model(sample)
            context_output = context.model(sample)

        for hook in hooks:
            hook.remove()

        print("Swin-context sanity checks")
        print("-------------------------")
        print(f"Baseline model: {args.baseline_model}")
        print(f"Context model:  {model_yaml_path}")
        print(f"Variant:        {variant}")
        print(f"Dummy input:    {(1, 3, args.imgsz, args.imgsz)}")
        print(f"Baseline output type: {type(baseline_output).__name__}")
        print(f"Context output type:  {type(context_output).__name__}")
        print(f"Baseline parameters: {_count_parameters(baseline.model):,}")
        print(f"Context parameters:  {_count_parameters(context.model):,}")
        _print_layer_table(context)

        for key in hook_indices:
            value = captured.get(key)
            if value is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Failed to capture tensor for {key}.")
            print(f"{key}: {tuple(value.shape)}")

        if variant in {"gated_p4_p5", "adaptive_gated_p4_p5"}:
            p4_backbone = captured["p4_backbone"]
            p4_out = captured["p4_out"]
            p5_backbone = captured["p5_backbone"]
            p5_out = captured["p5_out"]
            if p4_backbone.shape != p4_out.shape:
                raise RuntimeError(f"Gated P4 shape mismatch. cnn={tuple(p4_backbone.shape)}, out={tuple(p4_out.shape)}.")
            if p5_backbone.shape != p5_out.shape:
                raise RuntimeError(f"Gated P5 shape mismatch. cnn={tuple(p5_backbone.shape)}, out={tuple(p5_out.shape)}.")

            p4_gate = context.model.model[11]
            p5_gate = context.model.model[12]
            print_detail_stats = variant == "adaptive_gated_p4_p5"
            _print_gate_stats("p4", p4_gate, detail=print_detail_stats)
            _print_gate_stats("p5", p5_gate, detail=print_detail_stats)
            if composite_result is not None:
                print(
                    "p4_swin_loaded: "
                    f"{composite_result['p4_matched']}/{composite_result['p4_total']} "
                    f"(prefix={composite_result['p4_prefix']})"
                )
                print(
                    "p5_swin_loaded: "
                    f"{composite_result['p5_matched']}/{composite_result['p5_total']} "
                    f"(prefix={composite_result['p5_prefix']})"
                )
        else:
            feature_backbone = captured["feature_backbone"]
            feature_context = captured["feature_context"]
            feature_fused = captured["feature_fused"]
            if feature_backbone.shape != feature_context.shape:
                raise RuntimeError(
                    "Context branch output shape mismatch. "
                    f"backbone={tuple(feature_backbone.shape)}, context={tuple(feature_context.shape)}."
                )
            if feature_backbone.shape != feature_fused.shape:
                raise RuntimeError(
                    "Fused feature shape mismatch. "
                    f"backbone={tuple(feature_backbone.shape)}, fused={tuple(feature_fused.shape)}."
                )

        print("\nValidation result: OK")
        return 0
    except Exception as exc:
        print(f"Validation result: FAILED - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
