from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_ultralytics(yolo_config_dir: Path | None = None) -> Path:
    config_dir = yolo_config_dir.expanduser().resolve() if yolo_config_dir else project_root() / ".ultralytics"
    ensure_directory(config_dir)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    return config_dir


def find_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*.txt") if path.is_file())


def parse_yolo_label_file(label_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    annotations: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return annotations, [f"{label_path}: failed to read file: {exc}"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path}:{line_number}: expected 5 values "
                f"'class_id x_center y_center width height', got {len(parts)}"
            )
            continue

        class_token, *coord_tokens = parts

        try:
            class_id = int(class_token)
        except ValueError:
            errors.append(f"{label_path}:{line_number}: class_id must be an integer, got '{class_token}'")
            continue

        if class_id < 0:
            errors.append(f"{label_path}:{line_number}: class_id must be >= 0, got {class_id}")
            continue

        try:
            x_center, y_center, width, height = (float(token) for token in coord_tokens)
        except ValueError:
            errors.append(f"{label_path}:{line_number}: bbox values must be numeric")
            continue

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }
        out_of_range = [name for name, value in coordinates.items() if not 0.0 <= value <= 1.0]
        if out_of_range:
            joined = ", ".join(out_of_range)
            errors.append(
                f"{label_path}:{line_number}: values for {joined} must be within [0, 1], "
                f"got {coordinates}"
            )
            continue

        annotations.append({"class_id": class_id, **coordinates})

    return annotations, errors


def render_table(headers: list[str], rows: list[list[Any]]) -> str:
    normalized_headers = [str(header) for header in headers]
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([len(normalized_headers[index]), *(len(row[index]) for row in normalized_rows)])
        for index in range(len(normalized_headers))
    ]

    def render_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render_row(normalized_headers), separator]
    lines.extend(render_row(row) for row in normalized_rows)
    return "\n".join(lines)


def print_section(title: str, lines: Iterable[str]) -> None:
    normalized_lines = [line for line in lines if line]
    print(f"\n{title}")
    print("-" * len(title))
    if normalized_lines:
        for line in normalized_lines:
            print(line)
    else:
        print("None")


def save_json_report(path: Path, report: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False

    return bool(torch.cuda.is_available())


def resolve_device(requested_device: str | None) -> str:
    if requested_device is None:
        return "0" if is_cuda_available() else "cpu"

    normalized = str(requested_device).strip()
    lowered = normalized.lower()

    if lowered == "cpu":
        return "cpu"

    if not is_cuda_available():
        print("CUDA is not available. Falling back to CPU.", file=sys.stderr)
        return "cpu"

    if lowered == "cuda":
        return "0"

    return normalized


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_save_dir(result_object: Any, fallback: Path) -> Path:
    save_dir = getattr(result_object, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return fallback


def extract_detection_metrics(metrics: Any) -> dict[str, float | None]:
    box_metrics = getattr(metrics, "box", None)
    results_dict = getattr(metrics, "results_dict", {}) or {}

    def pick(
        attribute_name: str | None,
        fallback_keys: list[str],
    ) -> float | None:
        if attribute_name and box_metrics is not None:
            value = getattr(box_metrics, attribute_name, None)
            if isinstance(value, (int, float)):
                return float(value)

        for key in fallback_keys:
            value = results_dict.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        return None

    return {
        "precision": pick("mp", ["metrics/precision(B)", "precision"]),
        "recall": pick("mr", ["metrics/recall(B)", "recall"]),
        "mAP50": pick("map50", ["metrics/mAP50(B)", "mAP50"]),
        "mAP50-95": pick("map", ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"]),
    }
