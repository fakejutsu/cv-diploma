from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class AnnotationObject:
    class_id: int
    class_name: str
    bbox: BoundingBox


@dataclass(frozen=True)
class Annotation:
    image_path: Path
    objects: tuple[AnnotationObject, ...]


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    true_positive: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path
    input_sizes: tuple[int, ...]
    class_names: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class SarImage:
    path: Path
    image: Image.Image

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass(frozen=True)
class PreprocessedImage:
    data: np.ndarray
    original_size: tuple[int, int]
    input_size: int
    ratio: float
    pad: tuple[float, float]


@dataclass(frozen=True)
class RenderedImage:
    image: Image.Image


@dataclass(frozen=True)
class DetectionResult:
    source_image: SarImage
    rendered_image: RenderedImage
    detections: tuple[Detection, ...]
    metrics: DetectionMetrics | None
    text_summary: str

