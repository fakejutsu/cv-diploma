from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn


class HybridCnnSwinTBackbone(nn.Module):
    """
    Hybrid backbone: YOLO-style stride-preserving stem before timm Swin-T.

    The stem uses Ultralytics YOLO modules (Conv/C3k2) with stride=1 to preserve
    spatial resolution before transformer feature extraction.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
        stem_channels: Sequence[int] = (64, 128),
        stem_depth: int = 2,
        stem_expand_ratio: float = 0.25,
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

        try:
            from ultralytics.nn.modules.block import C3k2
            from ultralytics.nn.modules.conv import Conv
        except ImportError as exc:
            raise ImportError("ultralytics is required for the YOLO-style CNN+Swin-T stem.") from exc

        normalized_out_indices = tuple(int(index) for index in out_indices)
        normalized_stem_channels = tuple(int(channel) for channel in stem_channels)
        if len(normalized_stem_channels) < 2:
            raise ValueError("stem_channels must contain at least two channel sizes, e.g. [64, 128].")

        stem_mid_channels = normalized_stem_channels[0]
        stem_out_channels = normalized_stem_channels[1]
        if stem_mid_channels <= 0 or stem_out_channels <= 0:
            raise ValueError("stem_channels values must be positive.")

        normalized_stem_depth = int(stem_depth)
        if normalized_stem_depth < 1:
            raise ValueError("stem_depth must be >= 1.")

        normalized_stem_expand_ratio = float(stem_expand_ratio)
        if normalized_stem_expand_ratio <= 0:
            raise ValueError("stem_expand_ratio must be > 0.")

        self.model_name = model_name
        self.out_indices = normalized_out_indices
        self.stem_channels = normalized_stem_channels
        self.stem_depth = normalized_stem_depth
        self.stem_expand_ratio = normalized_stem_expand_ratio
        self.img_size = int(img_size)

        self.cnn_stem = nn.Sequential(
            Conv(3, stem_mid_channels, k=3, s=1),
            Conv(stem_mid_channels, stem_out_channels, k=3, s=1),
            C3k2(
                stem_out_channels,
                stem_out_channels,
                n=normalized_stem_depth,
                c3k=False,
                e=normalized_stem_expand_ratio,
            ),
        )

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=stem_out_channels,
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
