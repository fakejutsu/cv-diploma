from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import SwinTBackbone, register_swin_t_backbone
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that YOLO26 Swin-T architecture is wired correctly in this repository."
    )
    parser.add_argument("--model-yaml", type=Path, default=Path("models/yolo26_swin_t.yaml"), help="Path to model YAML.")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to verify loading path.")
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


def _get_backbone_layer(yolo_model: Any) -> Any:
    core_model = getattr(yolo_model, "model", None)
    layers = getattr(core_model, "model", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Failed to resolve model layers from YOLO object.")
    return layers[0]


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

    configure_ultralytics(args.yolo_config_dir)

    try:
        model_spec = yaml.safe_load(model_yaml_path.read_text(encoding="utf-8"))
        expected_channels = _extract_expected_index_channels(model_spec)
        if not expected_channels:
            raise RuntimeError("No expected channels were found from `Index` layers in model YAML.")

        register_swin_t_backbone()
        from ultralytics import YOLO

        yaml_model = YOLO(str(model_yaml_path))
        yaml_backbone = _get_backbone_layer(yaml_model)

        print("YAML build checks")
        print("-----------------")
        print(f"Backbone type: {type(yaml_backbone).__name__}")
        print(f"Backbone class: {yaml_backbone.__class__}")

        if not isinstance(yaml_backbone, SwinTBackbone):
            raise RuntimeError("Backbone is not SwinTBackbone after registration.")

        if not args.skip_forward:
            with torch.no_grad():
                sample = torch.randn(1, 3, args.imgsz, args.imgsz)
                features = yaml_backbone(sample)
            actual_channels = [int(feature.shape[1]) for feature in features]
            actual_shapes = [tuple(feature.shape) for feature in features]
            print(f"Expected channels: {expected_channels}")
            print(f"Actual channels:   {actual_channels}")
            print(f"Feature shapes:    {actual_shapes}")
            if actual_channels != expected_channels:
                raise RuntimeError("Backbone output channels do not match `Index` channel contract.")

        if checkpoint_path is not None:
            checkpoint_model = YOLO(str(checkpoint_path))
            checkpoint_backbone = _get_backbone_layer(checkpoint_model)
            print("\nCheckpoint checks")
            print("-----------------")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Backbone type: {type(checkpoint_backbone).__name__}")
            if not isinstance(checkpoint_backbone, SwinTBackbone):
                raise RuntimeError("Checkpoint loads with a non-Swin backbone.")

        print("\nValidation result: OK")
        return 0
    except Exception as exc:
        print(f"Validation result: FAILED - {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
