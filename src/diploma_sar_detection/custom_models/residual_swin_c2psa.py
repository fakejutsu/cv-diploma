from __future__ import annotations

import torch
from torch import Tensor, nn

from .swin_context_block import SwinContextBlock


class ResidualSwinC2PSA(nn.Module):
    """Shape-preserving residual Swin replacement for a YOLO P5 context block."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        depth: int,
        init_alpha: float = -1.0,
        init_beta: float = 0.1,
    ) -> None:
        super().__init__()

        normalized_channels = int(channels)
        if normalized_channels <= 0:
            raise ValueError("channels must be > 0")

        self.channels = normalized_channels
        self.init_alpha = float(init_alpha)
        self.init_beta = float(init_beta)
        self.swin = SwinContextBlock(
            normalized_channels,
            int(hidden_dim),
            int(num_heads),
            int(window_size),
            int(depth),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(normalized_channels * 2, normalized_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(normalized_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                normalized_channels,
                normalized_channels,
                kernel_size=3,
                padding=1,
                groups=normalized_channels,
                bias=False,
            ),
            nn.BatchNorm2d(normalized_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(normalized_channels, normalized_channels, kernel_size=1),
        )
        self.raw_alpha = nn.Parameter(torch.full((1, normalized_channels, 1, 1), self.init_alpha))
        self.beta = nn.Parameter(torch.tensor(self.init_beta, dtype=torch.float32))

        self._last_delta_abs_mean = 0.0
        self._init_delta()

    def _init_delta(self) -> None:
        nn.init.normal_(self.delta_proj[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_proj[-1].bias)

    @property
    def alpha_mean(self) -> float:
        return float(torch.sigmoid(self.raw_alpha.detach()).mean().item())

    @property
    def alpha_min(self) -> float:
        return float(torch.sigmoid(self.raw_alpha.detach()).min().item())

    @property
    def alpha_max(self) -> float:
        return float(torch.sigmoid(self.raw_alpha.detach()).max().item())

    @property
    def delta_abs_mean(self) -> float:
        return self._last_delta_abs_mean

    def forward(self, x: Tensor) -> Tensor:
        swin = self.swin(x)
        delta = self.delta_proj(torch.cat((x, swin - x), dim=1))
        alpha = torch.sigmoid(self.raw_alpha)
        with torch.no_grad():
            self._last_delta_abs_mean = float(delta.detach().abs().mean().item())
        return x + self.beta.to(dtype=x.dtype) * alpha * delta
