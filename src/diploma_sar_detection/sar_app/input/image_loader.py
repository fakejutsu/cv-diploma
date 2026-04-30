from __future__ import annotations

from pathlib import Path

from PIL import Image

from sar_app.domain.entities import SarImage


class PillowImageLoader:
    def load(self, path: Path) -> SarImage:
        image_path = path.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return SarImage(path=image_path, image=rgb_image)

