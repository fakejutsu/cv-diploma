from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sar_app.domain.entities import SarImage
from sar_app.input.annotation_loader import YoloAnnotationLoader


class YoloAnnotationLoaderTest(unittest.TestCase):
    def test_loads_matching_label_for_demo_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "images" / "val" / "sample.jpg"
            label_path = root / "labels" / "val" / "sample.txt"
            image_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            Image.new("RGB", (200, 100)).save(image_path)
            label_path.write_text("0 0.5 0.5 0.5 0.4\n", encoding="utf-8")

            image = SarImage(path=image_path, image=Image.new("RGB", (200, 100)))
            annotation = YoloAnnotationLoader(root, ("boat",)).load_for_image(image)

        self.assertIsNotNone(annotation)
        assert annotation is not None
        self.assertEqual(len(annotation.objects), 1)
        bbox = annotation.objects[0].bbox
        self.assertEqual((bbox.x1, bbox.y1, bbox.x2, bbox.y2), (50.0, 30.0, 150.0, 70.0))


if __name__ == "__main__":
    unittest.main()

