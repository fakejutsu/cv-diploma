from __future__ import annotations

import torch
from torch import Tensor, nn

from .swin_context_block import SwinContextBlock


class GatedSwinFusion(nn.Module):
    """Channel-wise gated fusion between a CNN feature map and its Swin-enhanced context."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        depth: int,
        init_alpha: float = 4.0,
    ) -> None:
        super().__init__()

        normalized_in_channels = int(in_channels)
        normalized_init_alpha = float(init_alpha)
        if normalized_in_channels <= 0:
            raise ValueError("in_channels must be > 0")

        self.in_channels = normalized_in_channels
        self.init_alpha = normalized_init_alpha
        self.swin = SwinContextBlock(
            normalized_in_channels,
            int(hidden_dim),
            int(num_heads),
            int(window_size),
            int(depth),
        )
        self.raw_alpha = nn.Parameter(torch.full((1, normalized_in_channels, 1, 1), normalized_init_alpha))

    @property
    def alpha_mean(self) -> float:
        return float(torch.sigmoid(self.raw_alpha.detach()).mean().item())

    def forward(self, x: Tensor) -> Tensor:
        swin = self.swin(x)
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * x + (1.0 - alpha) * swin
