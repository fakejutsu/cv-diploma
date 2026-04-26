from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .swin_context_block import SwinContextBlock


class AdaptiveDetailGatedSwinFusion(nn.Module):
    """Adaptive CNN/Swin fusion biased toward CNN on locally detailed regions."""

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

        channel_hidden = max(normalized_in_channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(normalized_in_channels * 2, channel_hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel_hidden, normalized_in_channels, kernel_size=1, bias=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
        )
        self.detail_strength = nn.Parameter(torch.tensor(0.25))

        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

        initial_alpha = float(torch.sigmoid(torch.tensor(normalized_init_alpha)).item())
        self._last_alpha_mean = initial_alpha
        self._last_alpha_min = initial_alpha
        self._last_alpha_max = initial_alpha
        self._last_detail_mean = 0.0
        self._last_detail_max = 0.0
        self._last_detail_bias_mean = 0.0
        self._last_detail_bias_max = 0.0

    @property
    def alpha_mean(self) -> float:
        return float(self._last_alpha_mean)

    @property
    def alpha_min(self) -> float:
        return float(self._last_alpha_min)

    @property
    def alpha_max(self) -> float:
        return float(self._last_alpha_max)

    @property
    def detail_mean(self) -> float:
        return float(self._last_detail_mean)

    @property
    def detail_max(self) -> float:
        return float(self._last_detail_max)

    @property
    def detail_bias_mean(self) -> float:
        return float(self._last_detail_bias_mean)

    @property
    def detail_bias_max(self) -> float:
        return float(self._last_detail_bias_max)

    def load_swin_weights(self, state_dict: dict[str, Tensor], prefix: str | None = None) -> tuple[int, int]:
        own_state = self.swin.state_dict()
        matched_state: OrderedDict[str, Tensor] = OrderedDict()

        normalized_prefix = prefix.rstrip(".") if prefix else None
        for key, value in state_dict.items():
            if normalized_prefix:
                dotted_prefix = f"{normalized_prefix}."
                if not key.startswith(dotted_prefix):
                    continue
                new_key = key[len(dotted_prefix) :]
            else:
                new_key = key

            own_value = own_state.get(new_key)
            if own_value is None or own_value.shape != value.shape:
                continue
            matched_state[new_key] = value

        self.swin.load_state_dict(matched_state, strict=False)
        return len(matched_state), len(own_state)

    @staticmethod
    def _detail_maps(x: Tensor) -> tuple[Tensor, Tensor]:
        pooled = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = (x - pooled).abs().mean(dim=1, keepdim=True)
        detail_scale = detail.mean(dim=(2, 3), keepdim=True)
        detail_bias = detail / (detail_scale + 1e-6) - 1.0
        return detail, detail_bias

    def forward(self, x: Tensor) -> Tensor:
        swin = self.swin(x)

        pooled = torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_avg_pool2d(swin, 1)), dim=1)
        channel_logits = self.channel_gate(pooled)

        detail, detail_bias = self._detail_maps(x)
        spatial_input = torch.cat((x.mean(dim=1, keepdim=True), swin.mean(dim=1, keepdim=True), detail_bias), dim=1)
        spatial_logits = self.spatial_gate(spatial_input)

        alpha_logits = self.raw_alpha + channel_logits + spatial_logits + self.detail_strength * detail_bias
        alpha = torch.sigmoid(alpha_logits)

        detached_alpha = alpha.detach()
        detached_detail = detail.detach()
        detached_detail_bias = detail_bias.detach()
        self._last_alpha_mean = float(detached_alpha.mean().item())
        self._last_alpha_min = float(detached_alpha.min().item())
        self._last_alpha_max = float(detached_alpha.max().item())
        self._last_detail_mean = float(detached_detail.mean().item())
        self._last_detail_max = float(detached_detail.max().item())
        self._last_detail_bias_mean = float(detached_detail_bias.mean().item())
        self._last_detail_bias_max = float(detached_detail_bias.max().item())
        return alpha * x + (1.0 - alpha) * swin
