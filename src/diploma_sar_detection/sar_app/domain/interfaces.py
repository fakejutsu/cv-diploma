from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .entities import (
    Annotation,
    Detection,
    DetectionMetrics,
    DetectionResult,
    ModelInfo,
    PreprocessedImage,
    RenderedImage,
    SarImage,
)


class IModelRepository(Protocol):
    def list_models(self) -> list[ModelInfo]:
        ...


class IModelLoader(Protocol):
    def load(self, model: ModelInfo) -> "IDetector":
        ...


class IDetector(Protocol):
    def predict(self, image: SarImage, confidence: float, image_size: int) -> list[Detection]:
        ...


class IImageLoader(Protocol):
    def load(self, path: Path) -> SarImage:
        ...


class IAnnotationLoader(Protocol):
    def load_for_image(self, image: SarImage) -> Annotation | None:
        ...


class IImagePreprocessor(Protocol):
    def prepare(self, image: SarImage, image_size: int) -> PreprocessedImage:
        ...


class IPostprocessor(Protocol):
    def process(
        self,
        raw_output: object,
        preprocessed: PreprocessedImage,
        confidence: float,
        class_names: tuple[str, ...],
    ) -> list[Detection]:
        ...


class IVisualizationService(Protocol):
    def render(self, image: SarImage, detections: list[Detection]) -> RenderedImage:
        ...


class IMetricsService(Protocol):
    def calculate(self, detections: list[Detection], annotation: Annotation) -> DetectionMetrics:
        ...


class IResultBuilder(Protocol):
    def build(
        self,
        image: SarImage,
        detections: list[Detection],
        rendered_image: RenderedImage,
        metrics: DetectionMetrics | None,
    ) -> DetectionResult:
        ...
