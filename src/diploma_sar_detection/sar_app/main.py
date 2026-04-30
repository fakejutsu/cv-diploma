from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sar_app.config import DEFAULT_MODELS_DIR, read_dataset_info
from sar_app.input.annotation_loader import YoloAnnotationLoader
from sar_app.input.image_loader import PillowImageLoader
from sar_app.metrics.metrics_service import BasicDetectionMetricsService
from sar_app.model.model_repository import FileSystemModelRepository
from sar_app.model.onnx_model_loader import OnnxModelLoader
from sar_app.processing.image_preprocessor import LetterboxImagePreprocessor
from sar_app.processing.postprocessor import YoloOnnxPostprocessor
from sar_app.result.result_builder import DetectionResultBuilder
from sar_app.scenario.object_detection_scenario import ObjectDetectionScenario
from sar_app.ui.main_window import MainWindow
from sar_app.visualization.visualization_service import PillowVisualizationService


def build_app() -> MainWindow:
    dataset_root, dataset_class_names = read_dataset_info()
    preprocessor = LetterboxImagePreprocessor()
    postprocessor = YoloOnnxPostprocessor()
    model_repository = FileSystemModelRepository(DEFAULT_MODELS_DIR)
    model_loader = OnnxModelLoader(preprocessor=preprocessor, postprocessor=postprocessor)
    scenario = ObjectDetectionScenario(
        model_loader=model_loader,
        image_loader=PillowImageLoader(),
        annotation_loader=YoloAnnotationLoader(dataset_root=dataset_root, class_names=dataset_class_names),
        visualization_service=PillowVisualizationService(),
        metrics_service=BasicDetectionMetricsService(),
        result_builder=DetectionResultBuilder(),
    )
    return MainWindow(
        model_repository=model_repository,
        scenario=scenario,
        demo_root=dataset_root,
    )


def main() -> int:
    app = build_app()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

