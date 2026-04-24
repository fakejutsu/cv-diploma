from __future__ import annotations

from torch import Tensor, nn


class SwinContextBlock(nn.Module):
    """
    Lightweight Swin-style context enhancer for an existing YOLO feature map.

    The block keeps the input/output channel count unchanged so it can be inserted
    into a stock Ultralytics YAML graph without parse_model customization.
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        window_size: int = 7,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        try:
            from torchvision.models.swin_transformer import SwinTransformerBlock
        except ImportError as exc:
            raise ImportError(
                "torchvision with swin_transformer support is required for SwinContextBlock."
            ) from exc

        normalized_channels = int(channels)
        normalized_hidden_dim = int(hidden_dim)
        normalized_num_heads = int(num_heads)
        normalized_window_size = int(window_size)
        normalized_depth = int(depth)
        normalized_mlp_ratio = float(mlp_ratio)
        normalized_dropout = float(dropout)

        if normalized_channels <= 0:
            raise ValueError("channels must be > 0")
        if normalized_hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if normalized_num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if normalized_hidden_dim % normalized_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if normalized_window_size <= 0:
            raise ValueError("window_size must be > 0")
        if normalized_depth <= 0:
            raise ValueError("depth must be > 0")
        if normalized_mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        if normalized_dropout < 0:
            raise ValueError("dropout must be >= 0")

        self.channels = normalized_channels
        self.hidden_dim = normalized_hidden_dim
        self.num_heads = normalized_num_heads
        self.window_size = normalized_window_size
        self.depth = normalized_depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(normalized_channels, normalized_hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(normalized_hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(
                dim=normalized_hidden_dim,
                num_heads=normalized_num_heads,
                window_size=[normalized_window_size, normalized_window_size],
                shift_size=[
                    0 if block_index % 2 == 0 else normalized_window_size // 2,
                    0 if block_index % 2 == 0 else normalized_window_size // 2,
                ],
                mlp_ratio=normalized_mlp_ratio,
                dropout=normalized_dropout,
                attention_dropout=normalized_dropout,
            )
            for block_index in range(normalized_depth)
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(normalized_hidden_dim, normalized_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(normalized_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.output_proj(x)
