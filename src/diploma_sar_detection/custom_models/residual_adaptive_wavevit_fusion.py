from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .wavevit_context_block import WaveVitContextBlock


class ResidualAdaptiveWaveVitFusion(nn.Module):
    """
    Add a small input-adaptive WaveViT correction to a CNN feature map.

    Unlike convex feature mixing, the baseline feature map remains the identity
    path: `out = x + beta * gate * delta`. This is safer for fine-tuning from an
    already trained YOLO checkpoint because WaveViT cannot directly replace the
    CNN representation at initialization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        kv_stride: int = 1,
        mlp_ratio: float = 2.0,
        init_alpha: float = -2.0,
        init_beta: float = 0.1,
        gate_reduction: int = 4,
    ) -> None:
        super().__init__()

        normalized_in_channels = int(in_channels)
        normalized_hidden_dim = int(hidden_dim)
        normalized_gate_reduction = int(gate_reduction)
        if normalized_in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if normalized_hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if normalized_gate_reduction <= 0:
            raise ValueError("gate_reduction must be > 0")

        gate_hidden = max(1, normalized_in_channels // normalized_gate_reduction)
        spatial_hidden = max(8, min(32, normalized_in_channels // 4))

        self.in_channels = normalized_in_channels
        self.init_alpha = float(init_alpha)
        self.init_beta = float(init_beta)
        self.wavevit = WaveVitContextBlock(
            normalized_in_channels,
            normalized_hidden_dim,
            int(num_heads),
            int(depth),
            int(kv_stride),
            float(mlp_ratio),
        )
        self.raw_alpha = nn.Parameter(torch.full((1, normalized_in_channels, 1, 1), self.init_alpha))
        self.beta = nn.Parameter(torch.tensor(self.init_beta, dtype=torch.float32))

        self.channel_gate = nn.Sequential(
            nn.Conv2d(normalized_in_channels * 2, gate_hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, normalized_in_channels, kernel_size=1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, spatial_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(spatial_hidden, 1, kernel_size=1),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(normalized_in_channels * 3, normalized_in_channels, kernel_size=1),
            nn.BatchNorm2d(normalized_in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                normalized_in_channels,
                normalized_in_channels,
                kernel_size=3,
                padding=1,
                groups=normalized_in_channels,
            ),
            nn.BatchNorm2d(normalized_in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(normalized_in_channels, normalized_in_channels, kernel_size=1),
        )

        self._last_gate_mean = float(torch.sigmoid(self.raw_alpha.detach()).mean().item())
        self._last_gate_min = float(torch.sigmoid(self.raw_alpha.detach()).min().item())
        self._last_gate_max = float(torch.sigmoid(self.raw_alpha.detach()).max().item())
        self._last_delta_abs_mean = 0.0
        self._init_adaptive_layers()

    def _init_adaptive_layers(self) -> None:
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
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
    def gate_mean(self) -> float:
        return self._last_gate_mean

    @property
    def gate_min(self) -> float:
        return self._last_gate_min

    @property
    def gate_max(self) -> float:
        return self._last_gate_max

    @property
    def delta_abs_mean(self) -> float:
        return self._last_delta_abs_mean

    def load_wavevit_weights(self, state_dict: dict[str, Tensor], prefix: str | None = None) -> tuple[int, int]:
        """Load only the inner WaveVitContextBlock weights from a compatible state dict."""

        own_state = self.wavevit.state_dict()
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

        self.wavevit.load_state_dict(matched_state, strict=False)
        return len(matched_state), len(own_state)

    def forward(self, x: Tensor) -> Tensor:
        wave = self.wavevit(x)
        diff = wave - x
        delta = self.delta_proj(torch.cat((x, wave, diff), dim=1))

        channel_context = torch.cat(
            (
                F.adaptive_avg_pool2d(x, 1),
                F.adaptive_avg_pool2d(wave, 1),
            ),
            dim=1,
        )
        channel_gate = self.channel_gate(channel_context)
        spatial_context = torch.cat(
            (
                x.mean(dim=1, keepdim=True),
                wave.mean(dim=1, keepdim=True),
                diff.abs().mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = self.spatial_gate(spatial_context)
        gate = torch.sigmoid(self.raw_alpha + channel_gate + spatial_gate)

        with torch.no_grad():
            detached_gate = gate.detach()
            self._last_gate_mean = float(detached_gate.mean().item())
            self._last_gate_min = float(detached_gate.min().item())
            self._last_gate_max = float(detached_gate.max().item())
            self._last_delta_abs_mean = float(delta.detach().abs().mean().item())

        return x + self.beta.to(dtype=x.dtype) * gate * delta
