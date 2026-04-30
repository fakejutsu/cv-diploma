from __future__ import annotations

from sar_app.domain.entities import ModelInfo
from sar_app.domain.interfaces import IDetector, IImagePreprocessor, IPostprocessor
from sar_app.model.onnx_detector import OnnxDetector


class OnnxModelLoader:
    """Creates and caches detectors for ONNX models."""

    def __init__(
        self,
        preprocessor: IImagePreprocessor,
        postprocessor: IPostprocessor,
        providers: list[str] | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._providers = providers
        self._active_model_path: str | None = None
        self._active_detector: IDetector | None = None

    def load(self, model: ModelInfo) -> IDetector:
        model_key = str(model.path)
        if self._active_model_path == model_key and self._active_detector is not None:
            return self._active_detector

        detector = OnnxDetector(
            model=model,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            providers=self._providers,
        )
        self._active_model_path = model_key
        self._active_detector = detector
        return detector

