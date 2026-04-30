from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sar_app.domain.entities import ModelInfo


class FileSystemModelRepository:
    """Reads ONNX model descriptors from a directory."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir

    def list_models(self) -> list[ModelInfo]:
        if not self._models_dir.exists():
            return []

        models: list[ModelInfo] = []
        for metadata_path in sorted(self._models_dir.glob("*.json")):
            model_info = self._read_metadata(metadata_path)
            if model_info is not None:
                models.append(model_info)
        return models

    def _read_metadata(self, metadata_path: Path) -> ModelInfo | None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        model_path = self._resolve_model_path(metadata_path, payload)
        if model_path is None or not model_path.is_file():
            return None

        input_sizes = self._read_input_sizes(payload)
        class_names = tuple(str(name) for name in payload.get("class_names", ()))
        if not input_sizes or not class_names:
            return None

        return ModelInfo(
            name=str(payload.get("name") or model_path.stem),
            path=model_path,
            input_sizes=input_sizes,
            class_names=class_names,
            description=str(payload.get("description") or ""),
        )

    def _resolve_model_path(self, metadata_path: Path, payload: dict[str, Any]) -> Path | None:
        raw_path = payload.get("path")
        if not raw_path:
            return None

        model_path = Path(str(raw_path)).expanduser()
        if not model_path.is_absolute():
            model_path = metadata_path.parent / model_path
        return model_path.resolve()

    def _read_input_sizes(self, payload: dict[str, Any]) -> tuple[int, ...]:
        raw_sizes = payload.get("input_sizes", payload.get("input_size"))
        if isinstance(raw_sizes, int):
            return (raw_sizes,)
        if isinstance(raw_sizes, list):
            sizes = []
            for value in raw_sizes:
                try:
                    sizes.append(int(value))
                except (TypeError, ValueError):
                    continue
            return tuple(sorted(set(size for size in sizes if size > 0)))
        return ()

