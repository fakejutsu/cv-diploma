from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "sar_app" / "models_repository"
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "data" / "dataset.yaml"


def read_dataset_info(dataset_config: Path = DEFAULT_DATASET_CONFIG) -> tuple[Path | None, tuple[str, ...]]:
    if not dataset_config.is_file():
        return None, ()

    try:
        import yaml
    except ImportError:
        return None, ()

    try:
        payload: dict[str, Any] = yaml.safe_load(dataset_config.read_text(encoding="utf-8")) or {}
    except OSError:
        return None, ()

    raw_root = payload.get("path")
    dataset_root = Path(str(raw_root)).expanduser().resolve() if raw_root else None
    names = payload.get("names", ())
    if isinstance(names, dict):
        class_names = tuple(str(names[key]) for key in sorted(names))
    elif isinstance(names, list):
        class_names = tuple(str(name) for name in names)
    else:
        class_names = ()
    return dataset_root, class_names

