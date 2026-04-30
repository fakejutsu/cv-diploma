from __future__ import annotations

from sar_app.domain.entities import Annotation, BoundingBox, Detection, DetectionMetrics


class BasicDetectionMetricsService:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold

    def calculate(self, detections: list[Detection], annotation: Annotation) -> DetectionMetrics:
        matched_annotation_indexes: set[int] = set()
        true_positive = 0
        false_positive = 0
        matched_ious: list[float] = []

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            match_index, match_iou = self._find_best_match(
                detection=detection,
                annotation=annotation,
                used_indexes=matched_annotation_indexes,
            )
            if match_index is None:
                false_positive += 1
                continue

            matched_annotation_indexes.add(match_index)
            true_positive += 1
            matched_ious.append(match_iou)

        false_negative = len(annotation.objects) - len(matched_annotation_indexes)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    def _find_best_match(
        self,
        detection: Detection,
        annotation: Annotation,
        used_indexes: set[int],
    ) -> tuple[int | None, float]:
        best_index: int | None = None
        best_iou = 0.0

        for index, target in enumerate(annotation.objects):
            if index in used_indexes or target.class_id != detection.class_id:
                continue

            iou = self._iou(detection.bbox, target.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is None or best_iou < self._iou_threshold:
            return None, 0.0
        return best_index, best_iou

    def _iou(self, first: BoundingBox, second: BoundingBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = first.area + second.area - intersection
        return intersection / union if union > 0 else 0.0

