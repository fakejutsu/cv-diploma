from __future__ import annotations

from sar_app.domain.entities import Detection, DetectionMetrics, DetectionResult, RenderedImage, SarImage


class DetectionResultBuilder:
    def build(
        self,
        image: SarImage,
        detections: list[Detection],
        rendered_image: RenderedImage,
        metrics: DetectionMetrics | None,
    ) -> DetectionResult:
        return DetectionResult(
            source_image=image,
            rendered_image=rendered_image,
            detections=tuple(detections),
            metrics=metrics,
            text_summary=self._build_summary(detections, metrics),
        )

    def _build_summary(self, detections: list[Detection], metrics: DetectionMetrics | None) -> str:
        lines = [f"Обнаружено объектов: {len(detections)}"]
        for index, detection in enumerate(detections, start=1):
            bbox = detection.bbox
            lines.append(
                f"{index}. {detection.class_name}: "
                f"conf={detection.confidence:.3f}, "
                f"bbox=({bbox.x1:.0f}, {bbox.y1:.0f}, {bbox.x2:.0f}, {bbox.y2:.0f})"
            )

        if metrics is not None:
            lines.extend(
                (
                    "",
                    "Метрики обнаружения:",
                    f"precision={metrics.precision:.3f}",
                    f"recall={metrics.recall:.3f}",
                    f"F1={metrics.f1:.3f}",
                    f"mean IoU={metrics.mean_iou:.3f}",
                    f"TP={metrics.true_positive}, FP={metrics.false_positive}, FN={metrics.false_negative}",
                )
            )
        return "\n".join(lines)

