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
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _count_parameters(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _capture_layer_output(container: dict[str, Any], key: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        container[key] = output

    return hook


def main() -> int:
    args = parse_args()
    model_yaml_path = args.context_model_yaml.expanduser().resolve()
    if not model_yaml_path.is_file():
        print(f"Context model YAML not found: {model_yaml_path}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)

    try:
        from ultralytics import YOLO

        register_context_modules()

        baseline = YOLO(args.baseline_model)
        context = YOLO(str(model_yaml_path))

        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        captured: dict[str, Any] = {}
        hooks = [
            context.model.model[10].register_forward_hook(_capture_layer_output(captured, "p5_backbone")),
            context.model.model[11].register_forward_hook(_capture_layer_output(captured, "p5_context")),
            context.model.model[13].register_forward_hook(_capture_layer_output(captured, "p5_fused")),
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
        print(f"Dummy input:    {(1, 3, args.imgsz, args.imgsz)}")
        print(f"Baseline output type: {type(baseline_output).__name__}")
        print(f"Context output type:  {type(context_output).__name__}")
        print(f"Baseline parameters: {_count_parameters(baseline.model):,}")
        print(f"Context parameters:  {_count_parameters(context.model):,}")

        for key in ("p5_backbone", "p5_context", "p5_fused"):
            value = captured.get(key)
            if value is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Failed to capture tensor for {key}.")
            print(f"{key}: {tuple(value.shape)}")

        p5_backbone = captured["p5_backbone"]
        p5_context = captured["p5_context"]
        p5_fused = captured["p5_fused"]
        if p5_backbone.shape != p5_context.shape:
            raise RuntimeError(
                f"Context branch output shape mismatch. P5={tuple(p5_backbone.shape)}, context={tuple(p5_context.shape)}."
            )
        if p5_backbone.shape != p5_fused.shape:
            raise RuntimeError(
                f"Fused P5 shape mismatch. P5={tuple(p5_backbone.shape)}, fused={tuple(p5_fused.shape)}."
            )

        print("\nValidation result: OK")
        return 0
    except Exception as exc:
        print(f"Validation result: FAILED - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
