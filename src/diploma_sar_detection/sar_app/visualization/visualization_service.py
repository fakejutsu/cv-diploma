from __future__ import annotations

from PIL import ImageDraw, ImageFont

from sar_app.domain.entities import Detection, RenderedImage, SarImage


class PillowVisualizationService:
    def __init__(self) -> None:
        self._palette = (
            "#d62728",
            "#1f77b4",
            "#2ca02c",
            "#9467bd",
            "#ff7f0e",
            "#17becf",
            "#8c564b",
        )

    def render(self, image: SarImage, detections: list[Detection]) -> RenderedImage:
        rendered = image.image.copy()
        draw = ImageDraw.Draw(rendered)
        font = ImageFont.load_default()

        for detection in detections:
            color = self._palette[detection.class_id % len(self._palette)]
            bbox = detection.bbox
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline=color, width=3)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            text_bbox = draw.textbbox((bbox.x1, bbox.y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            label_y = max(0, int(bbox.y1) - text_height - 4)
            draw.rectangle(
                (int(bbox.x1), label_y, int(bbox.x1) + text_width + 6, label_y + text_height + 4),
                fill=color,
            )
            draw.text((int(bbox.x1) + 3, label_y + 2), label, fill="white", font=font)

        return RenderedImage(image=rendered)

