from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils import ensure_directory, save_json_report


DEFAULT_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO label files.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root directory of the dataset.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to convert. Default: train val",
    )
    parser.add_argument(
        "--annotations-dir",
        default="annotations",
        help="Annotations directory relative to dataset root.",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Images directory relative to dataset root.",
    )
    parser.add_argument(
        "--labels-dir",
        default="labels",
        help="Output labels directory relative to dataset root.",
    )
    parser.add_argument(
        "--create-empty-test-dir",
        action="store_true",
        help="Create an empty labels/test directory for unlabeled benchmark test sets.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        help="Optional path to save a conversion report as JSON.",
    )
    return parser.parse_args()


def build_category_mapping(
    categories: list[dict[str, Any]],
    used_category_ids: set[int],
) -> tuple[dict[int, int], list[str]]:
    sorted_categories = sorted(
        (
            category
            for category in categories
            if int(category["id"]) in used_category_ids and str(category["name"]).lower() != "ignored"
        ),
        key=lambda category: int(category["id"]),
    )
    category_id_to_yolo: dict[int, int] = {}
    names: list[str] = []

    for yolo_id, category in enumerate(sorted_categories):
        category_id = int(category["id"])
        category_id_to_yolo[category_id] = yolo_id
        names.append(str(category["name"]))

    return category_id_to_yolo, names


def coco_bbox_to_yolo(bbox: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2.0) / image_width
    y_center = (y_min + height / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return x_center, y_center, normalized_width, normalized_height


def normalize_name(name: str) -> str:
    return name.replace("\\", "/")


def convert_split(
    dataset_root: Path,
    split: str,
    annotations_dir: str,
    images_dir: str,
    labels_dir: str,
) -> tuple[dict[str, Any], list[str]]:
    split_report: dict[str, Any] = {
        "split": split,
        "annotation_file": "",
        "images_dir": str(dataset_root / images_dir / split),
        "labels_dir": str(dataset_root / labels_dir / split),
        "images": 0,
        "annotations": 0,
        "label_files_written": 0,
        "class_counts": {},
        "names": [],
    }
    errors: list[str] = []

    annotation_path = dataset_root / annotations_dir / f"instances_{split}.json"
    image_dir = dataset_root / images_dir / split
    label_dir = dataset_root / labels_dir / split
    split_report["annotation_file"] = str(annotation_path)

    if not annotation_path.is_file():
        errors.append(f"[{split}] Annotation file not found: {annotation_path}")
        return split_report, errors
    if not image_dir.is_dir():
        errors.append(f"[{split}] Images directory not found: {image_dir}")
        return split_report, errors

    try:
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"[{split}] Failed to read annotation file {annotation_path}: {exc}")
        return split_report, errors

    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if not categories:
        errors.append(f"[{split}] No categories found in {annotation_path}")
        return split_report, errors

    used_category_ids = {
        int(annotation["category_id"])
        for annotation in annotations
        if not annotation.get("iscrowd", 0)
    }
    category_id_to_yolo, names = build_category_mapping(categories, used_category_ids)
    image_id_to_info = {int(image["id"]): image for image in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)
        category_id = int(annotation["category_id"])
        if category_id in category_id_to_yolo:
            class_counts[category_id_to_yolo[category_id]] += 1

    ensure_directory(label_dir)

    label_files_written = 0
    for image_id, image_info in image_id_to_info.items():
        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        file_name = Path(normalize_name(str(image_info["file_name"])))
        image_path = image_dir / file_name.name
        if not image_path.is_file():
            errors.append(f"[{split}] Image referenced in JSON not found: {image_path}")
            continue

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue

            category_id = int(annotation["category_id"])
            if category_id not in category_id_to_yolo:
                errors.append(
                    f"[{split}] Unknown category_id {category_id} in annotation id {annotation.get('id')}"
                )
                continue

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"[{split}] Invalid bbox in annotation id {annotation.get('id')}: {bbox}")
                continue

            x_center, y_center, width, height = coco_bbox_to_yolo(bbox, image_width, image_height)
            if min(x_center, y_center, width, height) < 0:
                errors.append(f"[{split}] Negative normalized bbox in annotation id {annotation.get('id')}")
                continue

            lines.append(
                f"{category_id_to_yolo[category_id]} "
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        label_path = label_dir / f"{file_name.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        label_files_written += 1

    split_report["images"] = len(images)
    split_report["annotations"] = len(annotations)
    split_report["label_files_written"] = label_files_written
    split_report["class_counts"] = {str(class_id): count for class_id, count in sorted(class_counts.items())}
    split_report["names"] = names
    return split_report, errors


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "splits": {},
        "errors": [],
    }

    discovered_names: list[str] | None = None
    for split in args.splits:
        split_report, errors = convert_split(
            dataset_root=dataset_root,
            split=split,
            annotations_dir=args.annotations_dir,
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
        )
        report["splits"][split] = split_report
        report["errors"].extend(errors)
        if split_report["names"]:
            discovered_names = split_report["names"]

    if args.create_empty_test_dir:
        ensure_directory(dataset_root / args.labels_dir / "test")

    if discovered_names is not None:
        report["names"] = discovered_names
        report["nc"] = len(discovered_names)

    print("\nCOCO to YOLO conversion report")
    print("==============================")
    print(f"Dataset root: {report['dataset_root']}")
    for split, split_report in report["splits"].items():
        print(f"\n[{split}]")
        print(f"annotation_file: {split_report['annotation_file']}")
        print(f"images: {split_report['images']}")
        print(f"annotations: {split_report['annotations']}")
        print(f"label_files_written: {split_report['label_files_written']}")

    if "names" in report:
        print(f"\nDetected classes ({report['nc']}): {', '.join(report['names'])}")

    if report["errors"]:
        print("\nErrors")
        print("------")
        for error in report["errors"]:
            print(error)
    else:
        print("\nConversion finished without errors.")

    if args.save_report:
        save_json_report(args.save_report.expanduser(), report)
        print(f"Report saved to: {args.save_report.expanduser()}")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
