from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import GatedWaveVitFusion, register_context_modules
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check the YOLO26n gated WaveViT-style P3/P4 integration with a dummy forward pass."
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=Path("models/yolo26n_gated_wavevit_p3_p4.yaml"),
        help="Path to the WaveViT-context model YAML.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Dummy input image size.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _count_parameters(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _capture_layer_output(container: dict[str, Any], key: str):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        container[key] = output

    return hook


def _print_layer_table(yolo_model: Any) -> None:
    print("\nLayer table")
    print("-----------")
    for index, layer in enumerate(yolo_model.model.model):
        print(f"{index:2d} | from={str(layer.f):<10} | type={layer.type}")


def _print_gate_stats(name: str, gate: GatedWaveVitFusion) -> None:
    print(f"alpha_mean_{name}: {gate.alpha_mean:.6f}")
    print(f"alpha_min_{name}: {gate.alpha_min:.6f}")
    print(f"alpha_max_{name}: {gate.alpha_max:.6f}")


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
    model_yaml_path = args.model_yaml.expanduser().resolve()
    if not model_yaml_path.is_file():
        print(f"WaveViT-context model YAML not found: {model_yaml_path}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)

    try:
        from ultralytics import YOLO

        register_context_modules()
        model = YOLO(str(model_yaml_path))

        sample = torch.randn(1, 3, args.imgsz, args.imgsz)
        captured: dict[str, Any] = {}
        hook_indices = {
            "p3_backbone": 4,
            "p4_backbone": 6,
            "p5_backbone": 10,
            "p3_out": 11,
            "p4_out": 12,
        }
        hooks = [
            model.model.model[layer_index].register_forward_hook(_capture_layer_output(captured, key))
            for key, layer_index in hook_indices.items()
        ]

        with torch.no_grad():
            output = model.model(sample)

        for hook in hooks:
            hook.remove()

        print("WaveViT-context sanity checks")
        print("----------------------------")
        print(f"Model:       {model_yaml_path}")
        print(f"Dummy input: {(1, 3, args.imgsz, args.imgsz)}")
        print(f"Output type: {type(output).__name__}")
        print(f"Parameters:  {_count_parameters(model.model):,}")
        _print_layer_table(model)

        for key in hook_indices:
            value = captured.get(key)
            if value is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Failed to capture tensor for {key}.")
            print(f"{key}: {tuple(value.shape)}")

        p3_backbone = captured["p3_backbone"]
        p3_out = captured["p3_out"]
        p4_backbone = captured["p4_backbone"]
        p4_out = captured["p4_out"]
        if p3_backbone.shape != p3_out.shape:
            raise RuntimeError(f"Gated P3 shape mismatch. cnn={tuple(p3_backbone.shape)}, out={tuple(p3_out.shape)}.")
        if p4_backbone.shape != p4_out.shape:
            raise RuntimeError(f"Gated P4 shape mismatch. cnn={tuple(p4_backbone.shape)}, out={tuple(p4_out.shape)}.")

        p3_gate = model.model.model[11]
        p4_gate = model.model.model[12]
        if not isinstance(p3_gate, GatedWaveVitFusion) or not isinstance(p4_gate, GatedWaveVitFusion):
            raise RuntimeError("Expected GatedWaveVitFusion modules at layers 11 and 12.")
        _print_gate_stats("p3", p3_gate)
        _print_gate_stats("p4", p4_gate)

        detect_layer = model.model.model[-1]
        detect_input_channels = _extract_detect_input_channels(detect_layer)
        strides = [float(value) for value in getattr(detect_layer, "stride", [])]
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
