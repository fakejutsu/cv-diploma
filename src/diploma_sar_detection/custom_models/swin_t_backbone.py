from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class SwinTBackbone(nn.Module):
    """Thin timm-based wrapper that exposes Swin-T feature maps to Ultralytics."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        img_size: int = 640,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = True,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for the Swin-T backbone. Install it from requirements.txt.") from exc

        normalized_out_indices = tuple(int(index) for index in out_indices)
        self.model_name = model_name
        self.out_indices = normalized_out_indices
        self.img_size = int(img_size)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
            out_indices=normalized_out_indices,
            strict_img_size=False,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
        )
        self.channels = tuple(int(channel) for channel in self.backbone.feature_info.channels())

    def forward(self, x: Tensor) -> list[Tensor]:
        # Keep original input spatial size to preserve correct Detect stride computation.
        features = self.backbone(x)
        outputs: list[Tensor] = []
        for feature in features:
            if feature.ndim == 4:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            outputs.append(feature)
        return outputs
