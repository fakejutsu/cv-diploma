from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from sar_app.domain.entities import PreprocessedImage, SarImage


class LetterboxImagePreprocessor:
    def prepare(self, image: SarImage, image_size: int) -> PreprocessedImage:
        source = image.image.convert("RGB")
        width, height = source.size
        ratio = min(image_size / width, image_size / height)
        new_width = int(round(width * ratio))
        new_height = int(round(height * ratio))

        resized = source.resize((new_width, new_height), Image.Resampling.BILINEAR)
        pad_x = (image_size - new_width) / 2.0
        pad_y = (image_size - new_height) / 2.0
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        right = image_size - new_width - left
        bottom = image_size - new_height - top

        padded = ImageOps.expand(resized, border=(left, top, right, bottom), fill=(114, 114, 114))
        array = np.asarray(padded, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0).astype(np.float32)

        return PreprocessedImage(
            data=array,
            original_size=(width, height),
            input_size=image_size,
            ratio=ratio,
            pad=(left, top),
        )

