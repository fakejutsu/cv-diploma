from __future__ import annotations

import math
from functools import partial
from typing import Sequence

import torch
from torch import Tensor, nn

from .wavevit_context_block import HaarWavelet2D


class WaveVitMlp(nn.Module):
    """PVT-style MLP with a depthwise spatial convolution."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class WaveVitAttention(nn.Module):
    """Wave-ViT attention with Haar wavelet key/value context."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = int(sr_ratio)
        self.wavelet = HaarWavelet2D()

        reduced_dim = max(1, self.dim // 4)
        self.reduce = nn.Sequential(
            nn.Conv2d(self.dim, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = (
            nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            if self.sr_ratio > 1
            else nn.Identity()
        )
        self.q = nn.Linear(self.dim, self.dim)
        self.kv_norm = nn.LayerNorm(self.dim)
        self.kv = nn.Linear(self.dim, self.dim * 2)
        self.proj = nn.Linear(self.dim + reduced_dim, self.dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        spatial = x.transpose(1, 2).reshape(batch, channels, height, width)
        coeffs, original_hw = self.wavelet.dwt(self.reduce(spatial))
        coeffs = self.filter(coeffs)
        local_context = self.wavelet.idwt(coeffs, original_hw)
        local_context = local_context.flatten(2).transpose(1, 2)

        kv_source = self.kv_embed(coeffs).flatten(2).transpose(1, 2)
        kv_source = self.kv_norm(kv_source)
        kv = self.kv(kv_source).reshape(batch, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attended = (q @ key.transpose(-2, -1)) * self.scale
        attended = attended.softmax(dim=-1)
        attended = (attended @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(torch.cat((attended, local_context), dim=-1))


class TokenAttention(nn.Module):
    """Standard token self-attention for later WaveViT stages."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(self.dim, self.dim)
        self.kv = nn.Linear(self.dim, self.dim * 2)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x: Tensor, _height: int, _width: int) -> Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(batch, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]
        attended = (q @ key.transpose(-2, -1)) * self.scale
        attended = attended.softmax(dim=-1)
        attended = (attended @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attended)


class WaveVitBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        sr_ratio: int,
        use_wave_attention: bool,
        norm_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = (
            WaveVitAttention(dim, num_heads, sr_ratio)
            if use_wave_attention
            else TokenAttention(dim, num_heads)
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = WaveVitMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class WaveVitStem(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(self.conv(x))
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class WaveVitDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


_WAVEVIT_CONFIGS = {
    "wavevit_s": {
        "stem_hidden_dim": 32,
        "embed_dims": (64, 128, 320, 448),
        "num_heads": (2, 4, 10, 14),
        "mlp_ratios": (8, 8, 4, 4),
        "depths": (3, 4, 6, 3),
        "sr_ratios": (4, 2, 1, 1),
    },
    "wavevit_b": {
        "stem_hidden_dim": 64,
        "embed_dims": (64, 128, 320, 512),
        "num_heads": (2, 4, 10, 16),
        "mlp_ratios": (8, 8, 4, 4),
        "depths": (3, 4, 12, 3),
        "sr_ratios": (4, 2, 1, 1),
    },
    "wavevit_l": {
        "stem_hidden_dim": 64,
        "embed_dims": (96, 192, 384, 512),
        "num_heads": (3, 6, 12, 16),
        "mlp_ratios": (8, 8, 4, 4),
        "depths": (3, 6, 18, 3),
        "sr_ratios": (4, 2, 1, 1),
    },
}


class WaveVitBackbone(nn.Module):
    """Pure WaveViT backbone wrapper that exposes multi-scale NCHW feature maps."""

    def __init__(
        self,
        model_name: str = "wavevit_s",
        pretrained: bool = False,
        out_indices: Sequence[int] = (1, 2, 3),
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        normalized_name = str(model_name).strip().lower()
        if normalized_name not in _WAVEVIT_CONFIGS:
            supported = ", ".join(sorted(_WAVEVIT_CONFIGS))
            raise ValueError(f"Unsupported WaveViT variant `{model_name}`. Supported: {supported}.")
        if pretrained:
            raise ValueError("WaveVitBackbone does not load external pretrained weights yet.")

        config = _WAVEVIT_CONFIGS[normalized_name]
        self.model_name = normalized_name
        self.out_indices = tuple(int(index) for index in out_indices)
        self.embed_dims = tuple(int(dim) for dim in config["embed_dims"])
        self.channels = tuple(self.embed_dims[index] for index in self.out_indices)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        depths = tuple(int(depth) for depth in config["depths"])
        drop_rates = torch.linspace(0, float(drop_path_rate), sum(depths)).tolist()
        cursor = 0
        stages: list[nn.ModuleDict] = []
        for stage_index, depth in enumerate(depths):
            if stage_index == 0:
                patch_embed = WaveVitStem(
                    int(in_chans),
                    int(config["stem_hidden_dim"]),
                    self.embed_dims[stage_index],
                    norm_layer,
                )
            else:
                patch_embed = WaveVitDownsample(
                    self.embed_dims[stage_index - 1],
                    self.embed_dims[stage_index],
                    norm_layer,
                )

            blocks = nn.ModuleList(
                [
                    WaveVitBlock(
                        dim=self.embed_dims[stage_index],
                        num_heads=int(config["num_heads"][stage_index]),
                        mlp_ratio=float(config["mlp_ratios"][stage_index]),
                        drop_path=float(drop_rates[cursor + block_index]),
                        sr_ratio=int(config["sr_ratios"][stage_index]),
                        use_wave_attention=stage_index < 2,
                        norm_layer=norm_layer,
                    )
                    for block_index in range(depth)
                ]
            )
            stages.append(nn.ModuleDict({"patch_embed": patch_embed, "blocks": blocks}))
            cursor += depth

        self.stages = nn.ModuleList(stages)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x, height, width = stage["patch_embed"](x)
            for block in stage["blocks"]:
                x = block(x, height, width)
            x = x.reshape(x.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
            if stage_index in self.out_indices:
                outputs.append(x)
        return outputs
