from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HaarWavelet2D(nn.Module):
    """Fixed Haar DWT/IDWT implemented with grouped PyTorch convolutions."""

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[-0.5, -0.5], [0.5, 0.5]],
                [[-0.5, 0.5], [-0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("filters", filters[:, None, :, :])

    @staticmethod
    def _pad_to_even(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (height, width)

    def dwt(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        x, original_hw = self._pad_to_even(x)
        channels = int(x.shape[1])
        weight = self.filters.to(dtype=x.dtype).repeat(channels, 1, 1, 1)
        return F.conv2d(x, weight, stride=2, groups=channels), original_hw

    def idwt(self, coefficients: Tensor, original_hw: tuple[int, int]) -> Tensor:
        channels = int(coefficients.shape[1]) // 4
        weight = self.filters.to(dtype=coefficients.dtype).repeat(channels, 1, 1, 1)
        x = F.conv_transpose2d(coefficients, weight, stride=2, groups=channels)
        height, width = original_hw
        return x[..., :height, :width]


class WaveletAttention(nn.Module):
    """WaveViT-style attention that uses wavelet coefficients as compact key/value context."""

    def __init__(self, channels: int, num_heads: int, kv_stride: int = 1) -> None:
        super().__init__()

        normalized_channels = int(channels)
        normalized_heads = int(num_heads)
        normalized_kv_stride = int(kv_stride)
        if normalized_channels <= 0:
            raise ValueError("channels must be > 0")
        if normalized_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if normalized_channels % normalized_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        if normalized_kv_stride <= 0:
            raise ValueError("kv_stride must be > 0")

        self.channels = normalized_channels
        self.num_heads = normalized_heads
        self.head_dim = normalized_channels // normalized_heads
        self.kv_stride = normalized_kv_stride
        self.wavelet = HaarWavelet2D()

        self.q = nn.Conv2d(normalized_channels, normalized_channels, kernel_size=1)
        self.kv = nn.Conv2d(normalized_channels * 4, normalized_channels * 2, kernel_size=1)
        self.wave_filter = nn.Sequential(
            nn.Conv2d(
                normalized_channels * 4,
                normalized_channels * 4,
                kernel_size=3,
                padding=1,
                groups=normalized_channels * 4,
            ),
            nn.GELU(),
            nn.Conv2d(normalized_channels * 4, normalized_channels * 4, kernel_size=1),
        )
        self.proj = nn.Conv2d(normalized_channels, normalized_channels, kernel_size=1)

    def _flatten_heads(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        return x.reshape(batch, self.num_heads, channels // self.num_heads, height * width).transpose(2, 3)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        coefficients, original_hw = self.wavelet.dwt(x)
        wave_context = self.wavelet.idwt(self.wave_filter(coefficients), original_hw)

        kv_source = coefficients
        if self.kv_stride > 1:
            kv_source = F.avg_pool2d(kv_source, kernel_size=self.kv_stride, stride=self.kv_stride)

        q = self._flatten_heads(self.q(x))
        key_value = self.kv(kv_source)
        k, v = key_value.chunk(2, dim=1)
        k = self._flatten_heads(k)
        v = self._flatten_heads(v)

        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)
        return self.proj(attended + wave_context)


class WaveVitBlock(nn.Module):
    """Shape-preserving wavelet-attention block for detector feature maps."""

    def __init__(self, channels: int, num_heads: int, kv_stride: int, mlp_ratio: float) -> None:
        super().__init__()

        normalized_channels = int(channels)
        hidden_channels = max(1, int(math.ceil(normalized_channels * float(mlp_ratio))))

        self.norm1 = nn.BatchNorm2d(normalized_channels)
        self.attn = WaveletAttention(normalized_channels, int(num_heads), int(kv_stride))
        self.norm2 = nn.BatchNorm2d(normalized_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(normalized_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, normalized_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WaveVitContextBlock(nn.Module):
    """WaveViT-style context enhancer with an `NCHW -> NCHW` contract."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_heads: int,
        depth: int,
        kv_stride: int = 1,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()

        normalized_in_channels = int(in_channels)
        normalized_hidden_dim = int(hidden_dim)
        normalized_depth = int(depth)
        if normalized_in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if normalized_hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if normalized_depth <= 0:
            raise ValueError("depth must be > 0")

        self.in_channels = normalized_in_channels
        self.hidden_dim = normalized_hidden_dim
        self.num_heads = int(num_heads)
        self.depth = normalized_depth
        self.kv_stride = int(kv_stride)
        self.mlp_ratio = float(mlp_ratio)

        self.input_proj = nn.Sequential(
            nn.Conv2d(normalized_in_channels, normalized_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(normalized_hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                WaveVitBlock(
                    normalized_hidden_dim,
                    self.num_heads,
                    self.kv_stride,
                    self.mlp_ratio,
                )
                for _ in range(normalized_depth)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(normalized_hidden_dim, normalized_in_channels, kernel_size=1),
            nn.BatchNorm2d(normalized_in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(self.blocks(self.input_proj(x)))
