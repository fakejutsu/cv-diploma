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
        choices=("auto", "p5", "p4_light"),
        default="auto",
        help="Expected context variant. Use auto to infer from YAML filename.",
    )
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
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
    if "p4_light" in model_name:
        return "p4_light"
    return "p5"


def _hook_indices(variant: str) -> tuple[int, int, int]:
    if variant == "p4_light":
        return 6, 11, 13
    if variant == "p5":
        return 10, 11, 13
    raise ValueError(f"Unsupported context variant: {variant}")


def main() -> int:
    args = parse_args()
    model_yaml_path = args.context_model_yaml.expanduser().resolve()
    if not model_yaml_path.is_file():
        print(f"Context model YAML not found: {model_yaml_path}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)
    variant = _resolve_variant(args.variant, model_yaml_path)

    try:
        from ultralytics import YOLO

        register_context_modules()

        baseline = YOLO(args.baseline_model)
        context = YOLO(str(model_yaml_path))

        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        captured: dict[str, Any] = {}
        backbone_index, context_index, fused_index = _hook_indices(variant)
        hooks = [
            context.model.model[backbone_index].register_forward_hook(_capture_layer_output(captured, "feature_backbone")),
            context.model.model[context_index].register_forward_hook(_capture_layer_output(captured, "feature_context")),
            context.model.model[fused_index].register_forward_hook(_capture_layer_output(captured, "feature_fused")),
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

        for key in ("feature_backbone", "feature_context", "feature_fused"):
            value = captured.get(key)
            if value is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Failed to capture tensor for {key}.")
            print(f"{key}: {tuple(value.shape)}")

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
