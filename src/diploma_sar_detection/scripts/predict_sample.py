from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import configure_ultralytics, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a sample image or directory.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model checkpoint.")
    parser.add_argument("--source", required=True, type=Path, help="Path to an image or directory with images.")
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
        help="Custom backbone registration variant. Use auto to infer order from checkpoint name.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save-dir", required=True, type=Path, help="Directory to save rendered predictions.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def register_custom_backbones_if_available(variant: str) -> None:
    """
    Best-effort registration of local custom backbones.
    Required for checkpoints that depend on repository-local modules (e.g. Swin-T wrapper).
    """

    try:
        from custom_models import register_backbone, register_context_modules

        register_context_modules()
        register_backbone(variant)
    except Exception:
        # Keep baseline inference working even if local custom modules are not available.
        return


def resolve_backbone_variant_candidates(requested_variant: str, model_path: Path) -> tuple[str, ...]:
    if requested_variant != "auto":
        return (requested_variant,)

    model_name = model_path.name.lower()
    if "original_wavevit_l" in model_name:
        return ("original_wavevit_l", "original_wavevit_s", "wavevit_l", "wavevit_s", "cnn_swin_t", "swin_t")
    if "original_wavevit_b" in model_name:
        return ("original_wavevit_b", "original_wavevit_s", "wavevit_b", "wavevit_s", "cnn_swin_t", "swin_t")
    if "original_wavevit" in model_name:
        return ("original_wavevit_s", "original_wavevit_b", "original_wavevit_l", "wavevit_s", "cnn_swin_t", "swin_t")
    if "wavevit_l" in model_name:
        return ("wavevit_l", "wavevit_s", "cnn_swin_t", "swin_t")
    if "wavevit_b" in model_name:
        return ("wavevit_b", "wavevit_s", "cnn_swin_t", "swin_t")
    if "wavevit" in model_name:
        return ("wavevit_s", "wavevit_b", "wavevit_l", "cnn_swin_t", "swin_t")
    if "cnn_swin" in model_name:
        return ("cnn_swin_t", "swin_t")
    if "swin" in model_name:
        return ("swin_t", "cnn_swin_t")
    return ("cnn_swin_t", "swin_t")


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()

    if not model_path.is_file():
        print(f"Model checkpoint not found: {model_path}", file=sys.stderr)
        return 2
    if not source_path.exists():
        print(f"Source path not found: {source_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    config_dir = configure_ultralytics(args.yolo_config_dir)

    try:
        from ultralytics import YOLO

        model = None
        active_backbone_variant = "none"
        candidate_variants = resolve_backbone_variant_candidates(args.backbone_variant, model_path)
        last_model_init_error: Exception | None = None

        for variant in candidate_variants:
            try:
                register_custom_backbones_if_available(variant)
                model = YOLO(str(model_path))
                active_backbone_variant = variant
                break
            except Exception as exc:
                last_model_init_error = exc

        if model is None:
            if last_model_init_error is not None:
                raise last_model_init_error
            raise RuntimeError("Failed to initialize YOLO model.")

        results = model.predict(
            source=str(source_path),
            imgsz=args.imgsz,
            device=device,
            conf=args.conf,
            save=True,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
        )
    except ImportError:
        print("Ultralytics is not installed. Run 'pip install -r requirements.txt' first.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    print("\nInference finished.")
    print(f"Processed items: {len(results)}")
    print(f"YOLO config directory: {config_dir}")
    print(f"Backbone variant: {active_backbone_variant}")
    print(f"Predictions saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
