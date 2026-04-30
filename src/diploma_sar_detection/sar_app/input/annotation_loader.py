from __future__ import annotations

from pathlib import Path

from sar_app.domain.entities import Annotation, AnnotationObject, BoundingBox, SarImage


class YoloAnnotationLoader:
    """Loads YOLO txt annotations placed next to demo dataset labels."""

    def __init__(self, dataset_root: Path | None, class_names: tuple[str, ...]) -> None:
        self._dataset_root = dataset_root.expanduser().resolve() if dataset_root else None
        self._class_names = class_names

    def load_for_image(self, image: SarImage) -> Annotation | None:
        label_path = self._resolve_label_path(image.path)
        if label_path is None or not label_path.is_file():
            return None

        objects: list[AnnotationObject] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parsed = self._parse_line(raw_line, image)
            if parsed is not None:
                objects.append(parsed)
        return Annotation(image_path=image.path, objects=tuple(objects))

    def _resolve_label_path(self, image_path: Path) -> Path | None:
        if self._dataset_root is None:
            return None

        try:
            relative = image_path.resolve().relative_to(self._dataset_root)
        except ValueError:
            return None

        parts = list(relative.parts)
        if "images" not in parts:
            return None

        image_index = parts.index("images")
        parts[image_index] = "labels"
        label_relative = Path(*parts).with_suffix(".txt")
        return self._dataset_root / label_relative

    def _parse_line(self, raw_line: str, image: SarImage) -> AnnotationObject | None:
        parts = raw_line.strip().split()
        if len(parts) != 5:
            return None

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            return None

        x1 = (x_center - width / 2.0) * image.width
        y1 = (y_center - height / 2.0) * image.height
        x2 = (x_center + width / 2.0) * image.width
        y2 = (y_center + height / 2.0) * image.height
        class_name = self._class_names[class_id] if 0 <= class_id < len(self._class_names) else str(class_id)
        return AnnotationObject(
            class_id=class_id,
            class_name=class_name,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )

