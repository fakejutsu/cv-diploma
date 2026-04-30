from __future__ import annotations

import unittest
from pathlib import Path

from sar_app.domain.entities import Annotation, AnnotationObject, BoundingBox, Detection
from sar_app.metrics.metrics_service import BasicDetectionMetricsService


class BasicDetectionMetricsServiceTest(unittest.TestCase):
    def test_calculates_precision_recall_and_f1(self) -> None:
        annotation = Annotation(
            image_path=Path("image.jpg"),
            objects=(
                AnnotationObject(0, "boat", BoundingBox(0, 0, 100, 100)),
                AnnotationObject(1, "buoy", BoundingBox(200, 200, 260, 260)),
            ),
        )
        detections = [
            Detection(0, "boat", 0.9, BoundingBox(0, 0, 100, 100)),
            Detection(0, "boat", 0.7, BoundingBox(300, 300, 340, 340)),
        ]

        metrics = BasicDetectionMetricsService().calculate(detections, annotation)

        self.assertEqual(metrics.true_positive, 1)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)


if __name__ == "__main__":
    unittest.main()
