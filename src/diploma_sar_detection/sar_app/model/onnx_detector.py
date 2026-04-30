from __future__ import annotations

from sar_app.domain.entities import Detection, ModelInfo, SarImage
from sar_app.domain.interfaces import IImagePreprocessor, IPostprocessor


class OnnxDetector:
    """ONNX Runtime-backed object detector."""

    def __init__(
        self,
        model: ModelInfo,
        preprocessor: IImagePreprocessor,
        postprocessor: IPostprocessor,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed. Install dependencies first.") from exc

        self._model = model
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

        available = ort.get_available_providers()
        requested = providers or self._default_providers(available)
        active_providers = [provider for provider in requested if provider in available]
        if not active_providers:
            active_providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model.path), providers=active_providers)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, image: SarImage, confidence: float, image_size: int) -> list[Detection]:
        preprocessed = self._preprocessor.prepare(image, image_size)
        raw_output = self._session.run(None, {self._input_name: preprocessed.data})
        return self._postprocessor.process(
            raw_output=raw_output,
            preprocessed=preprocessed,
            confidence=confidence,
            class_names=self._model.class_names,
        )

    def _default_providers(self, available: list[str]) -> list[str]:
        # CoreML can make Tkinter controls visually stick on macOS during session
        # creation. Keep the desktop app on stable providers by default.
        ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return [provider for provider in ordered if provider in available]
