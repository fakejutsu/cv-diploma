from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from custom_models import register_context_modules
from train_swin_context import _load_pretrained_weights
from utils import configure_ultralytics, ensure_directory, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a Swin C2PSA replacement by imitating the original YOLO26n C2PSA layer output."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument("--teacher", required=True, type=Path, help="Teacher YOLO checkpoint with original C2PSA.")
    parser.add_argument(
        "--student-yaml",
        type=Path,
        default=Path("models/yolo26n_p5_swin_c2psa_replacement.yaml"),
        help="Student YAML with ResidualSwinC2PSA replacement.",
    )
    parser.add_argument("--student-weights", required=True, type=Path, help="Baseline checkpoint for student warm-start.")
    parser.add_argument("--teacher-layer", type=int, default=10, help="Teacher layer index to imitate.")
    parser.add_argument("--student-layer", type=int, default=10, help="Student replacement layer index.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of imitation epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default=None, help="Device id like 0 or 'cpu'.")
    parser.add_argument("--project", default="runs", help="Project directory.")
    parser.add_argument("--name", default="p5_swin_c2psa_imitation", help="Run name.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for replacement layer imitation.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Optimizer weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of train split to use.")
    parser.add_argument("--save-period", type=int, default=-1, help="Save periodic checkpoints every N epochs.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--yolo-config-dir", type=Path, help="Directory for Ultralytics settings and cache files.")
    return parser.parse_args()


def _capture_layer_output(container: dict[str, Tensor], key: str):
    def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
        container[key] = output

    return hook


def _build_dataloader(data_yaml: Path, model: Any, imgsz: int, batch: int, workers: int, fraction: float):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    args = get_cfg(
        overrides={
            "task": "detect",
            "mode": "train",
            "data": str(data_yaml),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": int(workers),
            "fraction": float(fraction),
            "rect": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "scale": 0.0,
            "translate": 0.0,
            "fliplr": 0.0,
        }
    )
    stride = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)
    dataset = build_yolo_dataset(args, data["train"], batch, data, mode="train", rect=False, stride=stride)
    return build_dataloader(dataset, batch, workers, shuffle=True, rank=-1), data


def _preprocess_images(batch: dict[str, Any], device: torch.device) -> Tensor:
    images = batch["img"].to(device, non_blocking=True).float()
    return images / 255.0


def _checkpoint_payload(model: Any, args: argparse.Namespace, epoch: int, best_loss: float) -> dict[str, Any]:
    from ultralytics.utils import __version__

    model_copy = copy.deepcopy(model).half().cpu()
    return {
        "epoch": epoch,
        "best_fitness": -best_loss,
        "model": model_copy,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": vars(args),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": __version__,
    }


def _train_only_replacement(student: nn.Module, replacement_layer: nn.Module) -> None:
    """Keep the pretrained detector stable while optimizing only the replacement block."""
    student.eval()
    replacement_layer.train()


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    teacher_path = args.teacher.expanduser().resolve()
    student_yaml = args.student_yaml.expanduser().resolve()
    student_weights = args.student_weights.expanduser().resolve()

    for path, label in [
        (data_path, "Dataset config"),
        (teacher_path, "Teacher checkpoint"),
        (student_yaml, "Student YAML"),
        (student_weights, "Student weights"),
    ]:
        if not path.is_file():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 2

    device_string = resolve_device(args.device)
    device = torch.device("cpu" if device_string == "cpu" else f"cuda:{device_string.split(',')[0]}")
    set_seed(args.seed)
    config_dir = configure_ultralytics(args.yolo_config_dir)
    project_dir = Path(args.project).expanduser().resolve()
    save_dir = project_dir / args.name
    if save_dir.exists() and not args.exist_ok:
        suffix = 2
        while (project_dir / f"{args.name}-{suffix}").exists():
            suffix += 1
        save_dir = project_dir / f"{args.name}-{suffix}"
    weights_dir = save_dir / "weights"
    ensure_directory(weights_dir)

    print("\nC2PSA replacement imitation configuration")
    print("----------------------------------------")
    for key, value in {
        "data": data_path,
        "teacher": teacher_path,
        "student_yaml": student_yaml,
        "student_weights": student_weights,
        "teacher_layer": args.teacher_layer,
        "student_layer": args.student_layer,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": device,
        "save_dir": save_dir,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "fraction": args.fraction,
        "yolo_config_dir": config_dir,
    }.items():
        print(f"{key}: {value}")

    try:
        from ultralytics import YOLO

        register_context_modules()
        teacher = YOLO(str(teacher_path)).model.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        student_yolo = YOLO(str(student_yaml))
        _load_pretrained_weights(student_yolo, student_weights)
        student = student_yolo.model.to(device)

        for parameter in student.parameters():
            parameter.requires_grad = False
        replacement_layer = student.model[args.student_layer]
        for parameter in replacement_layer.parameters():
            parameter.requires_grad = True
        _train_only_replacement(student, replacement_layer)

        trainable = sum(parameter.numel() for parameter in replacement_layer.parameters() if parameter.requires_grad)
        print(f"Trainable replacement parameters: {trainable:,}")

        dataloader, _data = _build_dataloader(data_path, student, args.imgsz, args.batch, args.workers, args.fraction)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in replacement_layer.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.SmoothL1Loss()
        best_loss = float("inf")

        teacher_layer = teacher.model[args.teacher_layer]
        student_layer = student.model[args.student_layer]

        for epoch in range(args.epochs):
            _train_only_replacement(student, replacement_layer)
            running_loss = 0.0
            num_batches = 0
            progress = tqdm(dataloader, total=len(dataloader), desc=f"epoch {epoch + 1}/{args.epochs}")
            for batch in progress:
                images = _preprocess_images(batch, device)
                teacher_capture: dict[str, Tensor] = {}
                student_capture: dict[str, Tensor] = {}
                teacher_hook = teacher_layer.register_forward_hook(_capture_layer_output(teacher_capture, "feature"))
                student_hook = student_layer.register_forward_hook(_capture_layer_output(student_capture, "feature"))
                try:
                    with torch.no_grad():
                        teacher.predict(images)
                    student.predict(images)
                finally:
                    teacher_hook.remove()
                    student_hook.remove()

                teacher_feature = teacher_capture.get("feature")
                student_feature = student_capture.get("feature")
                if teacher_feature is None or student_feature is None:
                    raise RuntimeError("Failed to capture teacher/student layer outputs.")
                if student_feature.shape[-2:] != teacher_feature.shape[-2:]:
                    teacher_feature = torch.nn.functional.interpolate(
                        teacher_feature,
                        size=student_feature.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if student_feature.shape[1] != teacher_feature.shape[1]:
                    raise RuntimeError(
                        "Teacher/student layer channels differ. "
                        f"student={tuple(student_feature.shape)}, teacher={tuple(teacher_feature.shape)}"
                    )

                loss = criterion(student_feature.float(), teacher_feature.detach().float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.detach().item())
                num_batches += 1
                progress.set_postfix(loss=running_loss / max(num_batches, 1))

            epoch_loss = running_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1}/{args.epochs}: imitation_loss={epoch_loss:.6f}")

            last_payload = _checkpoint_payload(student, args, epoch, best_loss)
            torch.save(last_payload, weights_dir / "last.pt")
            if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
                torch.save(last_payload, weights_dir / f"epoch{epoch + 1}.pt")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(_checkpoint_payload(student, args, epoch, best_loss), weights_dir / "best.pt")

        print("\nC2PSA replacement imitation finished.")
        print(f"Run directory: {save_dir}")
        print(f"Best checkpoint: {weights_dir / 'best.pt'}")
        print(f"Last checkpoint: {weights_dir / 'last.pt'}")
        return 0
    except Exception as exc:
        print(f"C2PSA replacement imitation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
