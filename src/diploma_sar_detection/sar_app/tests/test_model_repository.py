from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sar_app.model.model_repository import FileSystemModelRepository


class FileSystemModelRepositoryTest(unittest.TestCase):
    def test_reads_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.onnx"
            model_path.write_bytes(b"onnx")
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "name": "Test model",
                        "path": "model.onnx",
                        "input_sizes": [640],
                        "class_names": ["boat"],
                    }
                ),
                encoding="utf-8",
            )

            models = FileSystemModelRepository(root).list_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Test model")
        self.assertEqual(models[0].input_sizes, (640,))
        self.assertEqual(models[0].class_names, ("boat",))


if __name__ == "__main__":
    unittest.main()

