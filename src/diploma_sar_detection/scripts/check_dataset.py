from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from utils import find_image_files, find_label_files, parse_yolo_label_file, print_section, render_table, save_json_report


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and annotations.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Path to dataset root directory.")
    parser.add_argument("--save-report", type=Path, help="Optional path to save a JSON report.")
    return parser.parse_args()


def _build_stem_index(paths: list[Path]) -> tuple[dict[str, Path], list[str]]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in paths:
        stem = path.stem
        if stem in index:
            duplicates.append(stem)
            continue
        index[stem] = path

    return index, sorted(set(duplicates))


def inspect_split(dataset_root: Path, split: str) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    critical_errors: list[str] = []

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    split_report: dict[str, Any] = {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "images": 0,
        "labels": 0,
        "matched_pairs": 0,
        "class_counts": {},
    }

    if not image_dir.is_dir():
        critical_errors.append(f"[{split}] Missing directory: {image_dir}")
    if not label_dir.is_dir() and split != "test":
        critical_errors.append(f"[{split}] Missing directory: {label_dir}")
    elif not label_dir.is_dir() and split == "test":
        warnings.append(f"[{split}] Label directory is missing: {label_dir}. This is acceptable for unlabeled test sets.")
    if critical_errors:
        return split_report, warnings, critical_errors

    image_files = find_image_files(image_dir)
    label_files = find_label_files(label_dir) if label_dir.is_dir() else []
    image_index, duplicate_image_stems = _build_stem_index(image_files)
    label_index, duplicate_label_stems = _build_stem_index(label_files)

    split_report["images"] = len(image_files)
    split_report["labels"] = len(label_files)

    if not image_files:
        warnings.append(f"[{split}] No image files found in {image_dir}")
    if not label_files and split != "test":
        warnings.append(f"[{split}] No label files found in {label_dir}")
    elif not label_files and split == "test":
        warnings.append(f"[{split}] No label files found in {label_dir}. This is acceptable for benchmark test sets.")

    if duplicate_image_stems:
        critical_errors.append(f"[{split}] Duplicate image stems found: {', '.join(duplicate_image_stems)}")
    if duplicate_label_stems:
        critical_errors.append(f"[{split}] Duplicate label stems found: {', '.join(duplicate_label_stems)}")

    image_stems = set(image_index)
    label_stems = set(label_index)

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels and not (split == "test" and not label_files):
        critical_errors.append(
            f"[{split}] Images without matching labels: {', '.join(missing_labels[:20])}"
            + (" ..." if len(missing_labels) > 20 else "")
        )
    if missing_images:
        critical_errors.append(
            f"[{split}] Labels without matching images: {', '.join(missing_images[:20])}"
            + (" ..." if len(missing_images) > 20 else "")
        )

    class_counts: Counter[int] = Counter()
    label_errors: list[str] = []
    for stem in sorted(image_stems & label_stems):
        label_path = label_index[stem]
        annotations, errors = parse_yolo_label_file(label_path)
        label_errors.extend(errors)
        for annotation in annotations:
            class_counts[annotation["class_id"]] += 1

    split_report["matched_pairs"] = len(image_stems & label_stems)
    split_report["class_counts"] = {str(class_id): count for class_id, count in sorted(class_counts.items())}

    if label_errors:
        critical_errors.extend(f"[{split}] {error}" for error in label_errors)

    return split_report, warnings, critical_errors


def build_report(dataset_root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    critical_errors: list[str] = []
    split_reports: dict[str, Any] = {}
    total_class_counts: Counter[int] = Counter()

    for split in SPLITS:
        split_report, split_warnings, split_errors = inspect_split(dataset_root, split)
        split_reports[split] = split_report
        warnings.extend(split_warnings)
        critical_errors.extend(split_errors)

        for class_id, count in split_report["class_counts"].items():
            total_class_counts[int(class_id)] += int(count)

    return {
        "dataset_root": str(dataset_root.resolve()),
        "valid": not critical_errors,
        "splits": split_reports,
        "class_counts_total": {str(class_id): count for class_id, count in sorted(total_class_counts.items())},
        "warnings": warnings,
        "critical_errors": critical_errors,
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nDataset check report")
    print("====================")
    print(f"Dataset root: {report['dataset_root']}")
    print(f"Status: {'OK' if report['valid'] else 'FAILED'}")

    rows: list[list[Any]] = []
    for split in SPLITS:
        split_report = report["splits"][split]
        rows.append(
            [
                split,
                split_report["images"],
                split_report["labels"],
                split_report["matched_pairs"],
            ]
        )

    print("\nSplit summary")
    print("-------------")
    print(render_table(["split", "images", "labels", "matched_pairs"], rows))

    class_rows = [[class_id, count] for class_id, count in report["class_counts_total"].items()]
    print("\nClass distribution")
    print("------------------")
    if class_rows:
        print(render_table(["class_id", "instances"], class_rows))
    else:
        print("No valid annotations found.")

    print_section("Warnings", report["warnings"])
    print_section("Critical errors", report["critical_errors"])


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()

    if not dataset_root.exists():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    report = build_report(dataset_root)
    print_report(report)

    if args.save_report:
        save_json_report(args.save_report.expanduser(), report)
        print(f"\nJSON report saved to: {args.save_report.expanduser()}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
