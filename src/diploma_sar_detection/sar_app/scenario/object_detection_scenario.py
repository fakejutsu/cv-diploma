from __future__ import annotations

from pathlib import Path

from sar_app.domain.entities import DetectionResult, ModelInfo
from sar_app.domain.interfaces import (
    IAnnotationLoader,
    IImageLoader,
    IMetricsService,
    IModelLoader,
    IResultBuilder,
    IVisualizationService,
)


class ObjectDetectionScenario:
    """Coordinates object detection according to the application architecture."""

    def __init__(
        self,
        model_loader: IModelLoader,
        image_loader: IImageLoader,
        annotation_loader: IAnnotationLoader,
        visualization_service: IVisualizationService,
        metrics_service: IMetricsService,
        result_builder: IResultBuilder,
    ) -> None:
        self._model_loader = model_loader
        self._image_loader = image_loader
        self._annotation_loader = annotation_loader
        self._visualization_service = visualization_service
        self._metrics_service = metrics_service
        self._result_builder = result_builder

    def run(
        self,
        model: ModelInfo,
        image_path: Path,
        image_size: int,
        confidence: float,
    ) -> DetectionResult:
        if image_size not in model.input_sizes:
            allowed = ", ".join(str(size) for size in model.input_sizes)
            raise ValueError(f"Unsupported image_size={image_size}. Allowed values: {allowed}")

        detector = self._model_loader.load(model)
        image = self._image_loader.load(image_path)
        annotation = self._annotation_loader.load_for_image(image)
        detections = detector.predict(image=image, confidence=confidence, image_size=image_size)
        rendered_image = self._visualization_service.render(image=image, detections=detections)
        metrics = self._metrics_service.calculate(detections, annotation) if annotation is not None else None
        return self._result_builder.build(
            image=image,
            detections=detections,
            rendered_image=rendered_image,
            metrics=metrics,
        )
