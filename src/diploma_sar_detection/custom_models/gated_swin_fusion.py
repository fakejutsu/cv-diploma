from __future__ import annotations

from collections import OrderedDict

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

    def load_swin_weights(self, state_dict: dict[str, Tensor], prefix: str | None = None) -> tuple[int, int]:
        """
        Load only the inner SwinContextBlock weights from a checkpoint state dict.

        Parameters are matched by name and shape after removing the optional prefix.
        Non-matching tensors are skipped silently.
        """

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

    def forward(self, x: Tensor) -> Tensor:
        swin = self.swin(x)
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * x + (1.0 - alpha) * swin
