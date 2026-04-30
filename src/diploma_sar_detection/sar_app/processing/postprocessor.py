from __future__ import annotations

import numpy as np

from sar_app.domain.entities import BoundingBox, Detection, PreprocessedImage


class YoloOnnxPostprocessor:
    def __init__(self, iou_threshold: float = 0.45) -> None:
        self._iou_threshold = iou_threshold

    def process(
        self,
        raw_output: object,
        preprocessed: PreprocessedImage,
        confidence: float,
        class_names: tuple[str, ...],
    ) -> list[Detection]:
        output = self._primary_output(raw_output)
        predictions = self._normalize_prediction_shape(output)
        candidates = self._to_candidates(predictions, confidence, class_names)
        selected = self._nms(candidates)
        return [self._scale_detection(detection, preprocessed) for detection in selected]

    def _primary_output(self, raw_output: object) -> np.ndarray:
        if isinstance(raw_output, (list, tuple)):
            if not raw_output:
                return np.empty((0, 0), dtype=np.float32)
            return np.asarray(raw_output[0])
        return np.asarray(raw_output)

    def _normalize_prediction_shape(self, output: np.ndarray) -> np.ndarray:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)

        # Common YOLO export shape: [attributes, boxes].
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 128:
            predictions = predictions.T
        return predictions.astype(np.float32, copy=False)

    def _to_candidates(
        self,
        predictions: np.ndarray,
        confidence: float,
        class_names: tuple[str, ...],
    ) -> list[Detection]:
        if predictions.size == 0 or predictions.shape[1] < 6:
            return []

        if predictions.shape[1] == 6:
            return self._parse_nms_output(predictions, confidence, class_names)

        class_count = len(class_names)
        attrs = predictions.shape[1]
        candidates: list[Detection] = []

        if attrs == 4 + class_count:
            boxes = predictions[:, :4]
            class_scores = predictions[:, 4:]
        elif attrs >= 5 + class_count:
            boxes = predictions[:, :4]
            objectness = predictions[:, 4:5]
            class_scores = predictions[:, 5 : 5 + class_count] * objectness
        else:
            return []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = scores >= confidence

        for box, class_id, score in zip(boxes[keep], class_ids[keep], scores[keep]):
            bbox = self._xywh_to_xyxy(box)
            candidates.append(
                Detection(
                    class_id=int(class_id),
                    class_name=self._class_name(int(class_id), class_names),
                    confidence=float(score),
                    bbox=bbox,
                )
            )
        return candidates

    def _parse_nms_output(
        self,
        predictions: np.ndarray,
        confidence: float,
        class_names: tuple[str, ...],
    ) -> list[Detection]:
        candidates: list[Detection] = []
        for row in predictions:
            score = float(row[4])
            if score < confidence:
                continue
            class_id = int(row[5])
            candidates.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_name(class_id, class_names),
                    confidence=score,
                    bbox=BoundingBox(x1=float(row[0]), y1=float(row[1]), x2=float(row[2]), y2=float(row[3])),
                )
            )
        return candidates

    def _xywh_to_xyxy(self, box: np.ndarray) -> BoundingBox:
        x_center, y_center, width, height = (float(value) for value in box[:4])
        return BoundingBox(
            x1=x_center - width / 2.0,
            y1=y_center - height / 2.0,
            x2=x_center + width / 2.0,
            y2=y_center + height / 2.0,
        )

    def _scale_detection(self, detection: Detection, preprocessed: PreprocessedImage) -> Detection:
        pad_x, pad_y = preprocessed.pad
        original_width, original_height = preprocessed.original_size
        bbox = detection.bbox

        x1 = (bbox.x1 - pad_x) / preprocessed.ratio
        y1 = (bbox.y1 - pad_y) / preprocessed.ratio
        x2 = (bbox.x2 - pad_x) / preprocessed.ratio
        y2 = (bbox.y2 - pad_y) / preprocessed.ratio

        scaled = BoundingBox(
            x1=float(np.clip(x1, 0, original_width)),
            y1=float(np.clip(y1, 0, original_height)),
            x2=float(np.clip(x2, 0, original_width)),
            y2=float(np.clip(y2, 0, original_height)),
        )
        return Detection(
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=scaled,
        )

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            overlaps = [
                self._iou(detection.bbox, other.bbox)
                for other in selected
                if other.class_id == detection.class_id
            ]
            if not overlaps or max(overlaps) < self._iou_threshold:
                selected.append(detection)
        return selected

    def _iou(self, first: BoundingBox, second: BoundingBox) -> float:
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = first.area + second.area - intersection
        return intersection / union if union > 0 else 0.0

    def _class_name(self, class_id: int, class_names: tuple[str, ...]) -> str:
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return str(class_id)
