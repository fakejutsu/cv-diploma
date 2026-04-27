from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Type

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import HybridCnnSwinTBackbone, OriginalWaveVitBackbone, SwinTBackbone, WaveVitBackbone, register_backbone
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that YOLO26 transformer backbones are wired correctly in this repository."
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=Path("models/yolo26_cnn_swin_t.yaml"),
        help="Path to model YAML.",
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
        ),
        help="Backbone registration variant. Use auto to infer from --model-yaml path.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to verify loading path.")
    parser.add_argument(
        "--pretrained-backbone",
        type=Path,
        help="Optional local checkpoint path for custom backbone pretrained weights.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size for backbone forward validation.")
    parser.add_argument("--skip-forward", action="store_true", help="Skip dummy forward shape validation.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _extract_expected_index_channels(model_spec: dict[str, Any]) -> list[int]:
    channels: list[int] = []
    for layer in model_spec.get("head", []):
        if not isinstance(layer, list) or len(layer) < 4:
            continue
        module_name = layer[2]
        args = layer[3]
        if module_name == "Index" and isinstance(args, list) and args:
            channels.append(int(args[0]))
    return channels


def _resolve_backbone_variant(arg_variant: str, model_yaml_path: Path) -> str:
    if arg_variant != "auto":
        return arg_variant

    model_name = model_yaml_path.name.lower()
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


def _expected_backbone_class(variant: str) -> Type[Any]:
    if variant == "swin_t":
        return SwinTBackbone
    if variant == "cnn_swin_t":
        return HybridCnnSwinTBackbone
    if variant in {"wavevit_s", "wavevit_b", "wavevit_l"}:
        return WaveVitBackbone
    if variant in {"original_wavevit_s", "original_wavevit_b", "original_wavevit_l"}:
        return OriginalWaveVitBackbone
    raise ValueError(f"Unsupported backbone variant: {variant}")


def _get_backbone_layer(yolo_model: Any) -> Any:
    core_model = getattr(yolo_model, "model", None)
    layers = getattr(core_model, "model", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Failed to resolve model layers from YOLO object.")
    return layers[0]


def _get_detect_layer(yolo_model: Any) -> Any:
    core_model = getattr(yolo_model, "model", None)
    layers = getattr(core_model, "model", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Failed to resolve model layers from YOLO object.")
    return layers[-1]


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
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else None

    if not model_yaml_path.is_file():
        print(f"Model YAML not found: {model_yaml_path}", file=sys.stderr)
        return 2
    if checkpoint_path is not None and not checkpoint_path.is_file():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    backbone_variant = _resolve_backbone_variant(args.backbone_variant, model_yaml_path)
    expected_backbone_cls = _expected_backbone_class(backbone_variant)
    configure_ultralytics(args.yolo_config_dir)

    try:
        model_spec = yaml.safe_load(model_yaml_path.read_text(encoding="utf-8"))
        expected_index_channels = _extract_expected_index_channels(model_spec)
        if not expected_index_channels:
            raise RuntimeError("No expected channels were found from `Index` layers in model YAML.")

        register_backbone(backbone_variant)
        from ultralytics import YOLO

        yaml_model = YOLO(str(model_yaml_path))
        yaml_backbone = _get_backbone_layer(yaml_model)
        if args.pretrained_backbone is not None:
            load_pretrained = getattr(yaml_backbone, "load_pretrained", None)
            if not callable(load_pretrained):
                raise RuntimeError(
                    f"Backbone `{type(yaml_backbone).__name__}` does not support --pretrained-backbone loading."
                )
            load_pretrained(args.pretrained_backbone.expanduser().resolve())
        yaml_detect = _get_detect_layer(yaml_model)

        print("YAML build checks")
        print("-----------------")
        print(f"Backbone variant: {backbone_variant}")
        print(f"Backbone type: {type(yaml_backbone).__name__}")
        print(f"Backbone class: {yaml_backbone.__class__}")

        if not isinstance(yaml_backbone, expected_backbone_cls):
            raise RuntimeError(
                "Backbone type mismatch after registration. "
                f"Expected {expected_backbone_cls.__name__}, got {type(yaml_backbone).__name__}."
            )

        if not args.skip_forward:
            with torch.no_grad():
                sample = torch.randn(1, 3, args.imgsz, args.imgsz)
                features = yaml_backbone(sample)
            actual_channels = [int(feature.shape[1]) for feature in features]
            actual_shapes = [tuple(feature.shape) for feature in features]
            print(f"Expected channels: {expected_index_channels}")
            print(f"Actual channels:   {actual_channels}")
            print(f"Feature shapes:    {actual_shapes}")
            if actual_channels != expected_index_channels:
                raise RuntimeError("Backbone output channels do not match `Index` channel contract.")

            detect_input_channels = _extract_detect_input_channels(yaml_detect)
            print(f"Detect input channels: {detect_input_channels}")
            if detect_input_channels != expected_index_channels:
                raise RuntimeError(
                    "Detect input channels do not match the expected transformer-backbone channel contract "
                    f"{expected_index_channels}. Got {detect_input_channels}."
                )

            strides = [float(value) for value in getattr(yaml_detect, "stride", [])]
            print(f"Detect strides:    {strides}")
            if not strides:
                raise RuntimeError("Detect stride tensor is empty.")
            expected_strides = [8.0, 16.0, 32.0]
            if len(strides) != len(expected_strides) or any(
                abs(actual - expected) > 0.01 for actual, expected in zip(strides, expected_strides)
            ):
                raise RuntimeError(
                    "Unexpected Detect strides for transformer-backbone integration. "
                    f"Expected {expected_strides}, got {strides}."
                )

        if checkpoint_path is not None:
            checkpoint_model = YOLO(str(checkpoint_path))
            checkpoint_backbone = _get_backbone_layer(checkpoint_model)
            print("\nCheckpoint checks")
            print("-----------------")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Backbone type: {type(checkpoint_backbone).__name__}")
            if not isinstance(checkpoint_backbone, expected_backbone_cls):
                raise RuntimeError(
                    "Checkpoint loads with unexpected backbone type. "
                    f"Expected {expected_backbone_cls.__name__}, got {type(checkpoint_backbone).__name__}."
                )

        print("\nValidation result: OK")
        return 0
    except Exception as exc:
        print(f"Validation result: FAILED - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
