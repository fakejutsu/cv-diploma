from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom_models import register_context_modules
from custom_models.distill_swin_p5_model import DistillSwinP5DetectionModel
from train_distill import load_pretrained_student_weights
from utils import configure_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check teacher/student P5 distillation wiring.")
    parser.add_argument("--teacher-model", type=Path, default=Path("best_yolo26m.pt"))
    parser.add_argument("--student-model", type=Path, default=Path("models/yolo26n_swin_context_p5.yaml"))
    parser.add_argument("--student-weights", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--distill-loss", choices=("l1", "mse", "smoothl1"), default="smoothl1")
    parser.add_argument("--yolo-config-dir", type=Path)
    return parser.parse_args()


def _count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> int:
    args = parse_args()
    teacher_path = args.teacher_model.expanduser().resolve()
    student_path = args.student_model.expanduser().resolve()
    student_weights = args.student_weights.expanduser().resolve()
    if not teacher_path.is_file():
        print(f"Teacher checkpoint not found: {teacher_path}", file=sys.stderr)
        return 2
    if not student_path.is_file():
        print(f"Student model not found: {student_path}", file=sys.stderr)
        return 2
    if not student_weights.is_file():
        print(f"Student warm-start checkpoint not found: {student_weights}", file=sys.stderr)
        return 2

    configure_ultralytics(args.yolo_config_dir)
    register_context_modules()

    from ultralytics import YOLO

    teacher = YOLO(str(teacher_path)).model
    student = DistillSwinP5DetectionModel(
        cfg=str(student_path),
        nc=5,
        ch=3,
        verbose=False,
        distill_weight=args.distill_weight,
        distill_loss=args.distill_loss,
    )
    load_pretrained_student_weights(student, student_weights)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student.set_distillation_teacher(teacher, teacher_path=str(teacher_path))

    sample = torch.rand(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        baseline_output = teacher(sample)
        student_output = student(sample)
        alignment = student.compute_distillation_alignment(sample)

    print("Distillation sanity checks")
    print("-------------------------")
    print(f"Teacher model: {teacher_path}")
    print(f"Student model: {student_path}")
    print(f"Student weights: {student_weights}")
    print(f"Teacher output type: {type(baseline_output).__name__}")
    print(f"Student output type: {type(student_output).__name__}")
    print(f"Teacher parameters: {_count_parameters(teacher):,}")
    print(f"Student parameters: {_count_parameters(student):,}")
    print(f"teacher_p5: {tuple(alignment['teacher_feature'].shape)}")
    print(f"student_p5: {tuple(alignment['student_feature'].shape)}")
    print(f"adapted_student_p5: {tuple(alignment['adapted_student_feature'].shape)}")
    print(f"distill_loss: {float(alignment['distill_loss'].item()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
