from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import register_context_modules
from custom_models.distill_swin_p5_model import DistillSwinP5DetectionModel
from utils import configure_ultralytics, resolve_device, resolve_save_dir, set_seed

from train_swin_context import (
    _BASELINE_TO_CONTEXT_LAYER_REMAP,
    _BASELINE_TO_GATED_P4_P5_LAYER_REMAP,
    _collect_compatible_weights,
    _load_checkpoint_state_dict,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer


_WEIGHT_TRANSFER_STRATEGIES = {
    "exact": None,
    "context_shift3": _BASELINE_TO_CONTEXT_LAYER_REMAP,
    "gated_shift2": _BASELINE_TO_GATED_P4_P5_LAYER_REMAP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n + Swin(P5) student with YOLO26m feature distillation.")
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=Path("best_yolo26m.pt"),
        help="Path to the teacher checkpoint (YOLO26m).",
    )
    parser.add_argument(
        "--student-model",
        default="models/yolo26n_swin_context_p5.yaml",
        help="Path to the student model YAML.",
    )
    parser.add_argument(
        "--student-weights",
        type=Path,
        default=Path("yolo26n.pt"),
        help="Optional checkpoint to warm-start the student model before distillation.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--project", default="runs", help="Project directory for Ultralytics outputs.")
    parser.add_argument("--name", default="yolo26m_to_yolo26n_swinp5_distill", help="Run name.")
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
    parser.add_argument("--lr0", type=float, help="Optional initial learning rate override.")
    parser.add_argument("--mosaic", type=float, help="Optional mosaic augmentation probability override.")
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.1,
        help="Weight applied to the P5 feature distillation loss.",
    )
    parser.add_argument(
        "--distill-loss",
        choices=("l1", "mse", "smoothl1"),
        default="smoothl1",
        help="Feature distillation loss type.",
    )
    return parser.parse_args()


def _print_run_configuration(config: dict[str, Any]) -> None:
    print("\nDistillation training configuration")
    print("----------------------------------")
    for key, value in config.items():
        print(f"{key}: {value}")


def _resolve_best_checkpoint(trainer: Any, fallback_save_dir: Path) -> Path:
    best = getattr(trainer, "best", None)
    if best:
        return Path(best)
    return fallback_save_dir / "weights" / "best.pt"


def load_pretrained_student_weights(model: DistillSwinP5DetectionModel, weights_path: Path) -> tuple[str, int, int]:
    source_state_dict = _load_checkpoint_state_dict(weights_path)
    target_state_dict = model.state_dict()

    selected_weights: dict[str, Any] = {}
    selected_counts: Counter[int] = Counter()
    strategy = "exact"

    for strategy_name, index_remap in _WEIGHT_TRANSFER_STRATEGIES.items():
        candidate_weights, candidate_counts = _collect_compatible_weights(
            source_state_dict,
            target_state_dict,
            index_remap=index_remap,
        )
        if len(candidate_weights) > len(selected_weights):
            selected_weights = candidate_weights
            selected_counts = candidate_counts
            strategy = strategy_name

    model.load_state_dict(selected_weights, strict=False)
    detect_layer = model.model[-1]
    detect_matches = selected_counts.get(int(getattr(detect_layer, "i", len(model.model) - 1)), 0)
    print(
        f"Student warm-start: strategy={strategy}, matched={len(selected_weights)}/{len(target_state_dict)}, "
        f"detect_matches={detect_matches}"
    )
    return strategy, len(selected_weights), detect_matches


class DistillDetectionTrainer(DetectionTrainer):
    def __init__(
        self,
        teacher_model_path: Path,
        student_weights_path: Path | None,
        distill_weight: float,
        distill_loss: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.teacher_model_path = teacher_model_path
        self.student_weights_path = student_weights_path
        self.distill_weight = float(distill_weight)
        self.distill_loss = distill_loss
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = DistillSwinP5DetectionModel(
            cfg=cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            distill_weight=self.distill_weight,
            distill_loss=self.distill_loss,
        )
        if self.student_weights_path:
            load_pretrained_student_weights(model, self.student_weights_path)
        elif weights:
            model.load(weights)
        return model

    def _setup_train(self):
        super()._setup_train()
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import unwrap_model

        teacher_yolo = YOLO(str(self.teacher_model_path))
        teacher_model = teacher_yolo.model.to(self.device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False

        student_model = unwrap_model(self.model)
        student_model.set_distillation_teacher(teacher_model, teacher_path=str(self.teacher_model_path))
        self.teacher_model = teacher_model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        return super().get_validator()


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    teacher_model_path = args.teacher_model.expanduser().resolve()
    student_model_path = Path(args.student_model).expanduser()
    student_weights_path = args.student_weights.expanduser().resolve() if args.student_weights else None

    if not data_path.is_file():
        print(f"Dataset config not found: {data_path}", file=sys.stderr)
        return 2
    if not teacher_model_path.is_file():
        print(f"Teacher checkpoint not found: {teacher_model_path}", file=sys.stderr)
        return 2
    if student_model_path.suffix in {".yaml", ".yml"} and not student_model_path.resolve().is_file():
        print(f"Student model config not found: {student_model_path.resolve()}", file=sys.stderr)
        return 2
    if student_weights_path is not None and not student_weights_path.is_file():
        print(f"Student warm-start checkpoint not found: {student_weights_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    set_seed(args.seed)
    config_dir = configure_ultralytics(args.yolo_config_dir)
    project_dir = Path(args.project).expanduser().resolve()
    fallback_save_dir = project_dir / args.name

    run_config = {
        "data": str(data_path),
        "teacher_model": str(teacher_model_path),
        "student_model": str(student_model_path),
        "student_weights": str(student_weights_path) if student_weights_path else None,
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
        "save_period": args.save_period,
        "exist_ok": args.exist_ok,
        "yolo_config_dir": str(config_dir),
        "fraction": args.fraction,
        "lr0": args.lr0,
        "mosaic": args.mosaic,
        "distill_weight": args.distill_weight,
        "distill_loss": args.distill_loss,
    }
    _print_run_configuration(run_config)

    try:
        register_context_modules()
        overrides: dict[str, Any] = {
            "model": str(student_model_path),
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
            "save_period": args.save_period,
            "exist_ok": args.exist_ok,
        }
        if args.cache is not None:
            overrides["cache"] = args.cache
        if args.optimizer is not None:
            overrides["optimizer"] = args.optimizer
        if args.amp is not None:
            overrides["amp"] = args.amp
        if args.lr0 is not None:
            overrides["lr0"] = args.lr0
        if args.mosaic is not None:
            overrides["mosaic"] = args.mosaic

        trainer = DistillDetectionTrainer(
            teacher_model_path=teacher_model_path,
            student_weights_path=student_weights_path,
            distill_weight=args.distill_weight,
            distill_loss=args.distill_loss,
            overrides=overrides,
        )
        trainer.train()
    except Exception as exc:
        print(f"Distillation training failed: {exc}", file=sys.stderr)
        return 1

    save_dir = resolve_save_dir(trainer, fallback_save_dir)
    best_checkpoint = _resolve_best_checkpoint(trainer, save_dir)
    print("\nDistillation training finished.")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
