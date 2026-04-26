from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import ResidualSwinC2PSA, register_context_modules
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO26n P5 Swin C2PSA replacement wiring.")
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=Path("models/yolo26n_p5_swin_c2psa_replacement.yaml"),
        help="Path to model YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Dummy input image size.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _capture_layer_output(container: dict[str, Any], key: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        container[key] = output

    return hook


def _extract_detect_input_channels(detect_layer: Any) -> list[int]:
    channels: list[int] = []
    branches = getattr(detect_layer, "cv2", None)
    if not branches:
        return channels

    for branch in branches:
        first_layer = branch[0]
        conv = getattr(first_layer, "conv", None)
        if conv is None:
            raise RuntimeError("Failed to resolve input channels from Detect head branch.")
        channels.append(int(conv.in_channels))
    return channels


def main() -> int:
    args = parse_args()
    model_yaml = args.model_yaml.expanduser().resolve()
    if not model_yaml.is_file():
        print(f"Model YAML not found: {model_yaml}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)

    try:
        from ultralytics import YOLO

        register_context_modules()
        model = YOLO(str(model_yaml))
        replacement = model.model.model[10]
        if not isinstance(replacement, ResidualSwinC2PSA):
            raise RuntimeError(f"Expected ResidualSwinC2PSA at layer 10, got {type(replacement).__name__}.")

        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        captured: dict[str, Any] = {}
        hooks = [
            model.model.model[9].register_forward_hook(_capture_layer_output(captured, "p5_before")),
            model.model.model[10].register_forward_hook(_capture_layer_output(captured, "p5_after")),
        ]
        with torch.no_grad():
            output = model.model(sample)
        for hook in hooks:
            hook.remove()

        p5_before = captured.get("p5_before")
        p5_after = captured.get("p5_after")
        if not isinstance(p5_before, torch.Tensor) or not isinstance(p5_after, torch.Tensor):
            raise RuntimeError("Failed to capture P5 tensors around replacement layer.")
        if p5_before.shape != p5_after.shape:
            raise RuntimeError(f"P5 shape changed: before={tuple(p5_before.shape)}, after={tuple(p5_after.shape)}")

        detect_layer = model.model.model[-1]
        detect_input_channels = _extract_detect_input_channels(detect_layer)
        strides = [float(value) for value in getattr(detect_layer, "stride", [])]

        print("Swin C2PSA replacement sanity checks")
        print("-----------------------------------")
        print(f"Model: {model_yaml}")
        print(f"Dummy input: {(1, 3, args.imgsz, args.imgsz)}")
        print(f"Output type: {type(output).__name__}")
        print(f"P5 before: {tuple(p5_before.shape)}")
        print(f"P5 after:  {tuple(p5_after.shape)}")
        print(f"alpha_mean: {replacement.alpha_mean:.6f}")
        print(f"alpha_min:  {replacement.alpha_min:.6f}")
        print(f"alpha_max:  {replacement.alpha_max:.6f}")
        print(f"beta:       {float(replacement.beta.detach().cpu().item()):.6f}")
        print(f"delta_abs_mean: {replacement.delta_abs_mean:.6f}")
        print(f"Detect input channels: {detect_input_channels}")
        print(f"Detect strides:        {strides}")

        expected_strides = [8.0, 16.0, 32.0]
        if len(strides) != len(expected_strides) or any(
            abs(actual - expected) > 0.01 for actual, expected in zip(strides, expected_strides)
        ):
            raise RuntimeError(f"Unexpected Detect strides. Expected {expected_strides}, got {strides}.")

        print("\nValidation result: OK")
        return 0
    except Exception as exc:
        print(f"Validation result: FAILED - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
