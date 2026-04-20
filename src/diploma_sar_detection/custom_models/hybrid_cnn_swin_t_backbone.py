from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn


class HybridCnnSwinTBackbone(nn.Module):
    """
    Hybrid backbone: lightweight CNN stem before timm Swin-T feature extractor.

    The CNN stem keeps spatial resolution and restores 3 channels, so Swin-T
    can be used without patch-embedding surgery while enriching local features.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        stem_channels: Sequence[int] = (32, 64),
        img_size: int = 640,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = True,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for the CNN+Swin-T backbone. Install it from requirements.txt."
            ) from exc

        normalized_out_indices = tuple(int(index) for index in out_indices)
        normalized_stem_channels = tuple(int(channel) for channel in stem_channels)
        if not normalized_stem_channels:
            raise ValueError("stem_channels must contain at least one channel size.")

        self.model_name = model_name
        self.out_indices = normalized_out_indices
        self.stem_channels = normalized_stem_channels
        self.img_size = int(img_size)

        stem_layers: list[nn.Module] = []
        in_channels = 3
        for channels in normalized_stem_channels:
            stem_layers.extend(
                [
                    nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ]
            )
            in_channels = channels
        stem_layers.extend(
            [
                nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.SiLU(inplace=True),
            ]
        )
        self.cnn_stem = nn.Sequential(*stem_layers)

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
        x = self.cnn_stem(x)
        features = self.backbone(x)
        outputs: list[Tensor] = []
        for feature in features:
            if feature.ndim == 4:
                feature = feature.permute(0, 3, 1, 2).contiguous()
            outputs.append(feature)
        return outputs
