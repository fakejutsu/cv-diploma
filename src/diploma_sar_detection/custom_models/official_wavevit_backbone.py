from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .original_wavevit_backbone import _OFFICIAL_PRETRAINED_ALIASES
from .vendor.wavevit_official import wavevit_b, wavevit_l, wavevit_s


_OFFICIAL_BUILDERS = {
    "official_wavevit_s": wavevit_s,
    "official_wavevit_b": wavevit_b,
    "official_wavevit_l": wavevit_l,
}

_OFFICIAL_CHANNELS = {
    "official_wavevit_s": (64, 128, 320, 448),
    "official_wavevit_b": (64, 128, 320, 512),
    "official_wavevit_l": (96, 192, 384, 512),
}


class OfficialWaveVitBackbone(nn.Module):
    """Thin YOLO wrapper around the vendored official WaveViT implementation."""

    def __init__(
        self,
        model_name: str = "official_wavevit_s",
        pretrained: bool | str | Path = False,
        out_indices: Sequence[int] = (1, 2, 3),
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        normalized_name = str(model_name).strip().lower()
        if normalized_name not in _OFFICIAL_BUILDERS:
            supported = ", ".join(sorted(_OFFICIAL_BUILDERS))
            raise ValueError(f"Unsupported official WaveViT variant `{model_name}`. Supported: {supported}.")

        self.model_name = normalized_name
        self.config_name = normalized_name.removeprefix("official_")
        self.out_indices = tuple(int(index) for index in out_indices)
        embed_dims = _OFFICIAL_CHANNELS[normalized_name]
        self.channels = tuple(embed_dims[index] for index in self.out_indices)
        self.model = _OFFICIAL_BUILDERS[normalized_name](
            pretrained=False,
            in_chans=int(in_chans),
            num_classes=1000,
            token_label=True,
            drop_path_rate=float(drop_path_rate),
        )
        if pretrained:
            if isinstance(pretrained, bool):
                raise ValueError(
                    "OfficialWaveVitBackbone requires an explicit pretrained alias or checkpoint path."
                )
            self.load_pretrained(pretrained)

    def _resolve_pretrained_checkpoint(self, checkpoint: str | Path) -> Path:
        text = str(checkpoint)
        path = Path(text).expanduser()
        if path.is_file():
            return path.resolve()

        normalized_alias = text.strip().lower()
        variant_aliases = _OFFICIAL_PRETRAINED_ALIASES.get(normalized_alias)
        if variant_aliases is None:
            supported = ", ".join(sorted(_OFFICIAL_PRETRAINED_ALIASES))
            raise FileNotFoundError(
                f"Official WaveViT pretrained checkpoint not found: {path}. "
                f"Supported built-in aliases: {supported}."
            )

        entry = variant_aliases[self.config_name]
        cached_path = Path("weights") / "pretrained" / "wavevit" / str(entry["filename"])
        if cached_path.is_file():
            return cached_path.resolve()

        raise FileNotFoundError(
            "Official WaveViT pretrained checkpoint is not present locally.\n"
            f"Alias: {normalized_alias} / {self.config_name}\n"
            f"Expected local path: {cached_path}\n"
            f"Google Drive: {entry['google_drive_url']}\n"
            f"Baidu: {entry['baidu_url']} (access code: nets)\n"
            "Download the checkpoint to the expected path, or pass the downloaded file path explicitly."
        )

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")
        for key in ("state_dict", "model", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint  # type: ignore[return-value]
        raise KeyError("Could not find a state_dict in checkpoint keys: state_dict/model/module.")

    @staticmethod
    def _print_examples(title: str, values: Sequence[str], limit: int) -> None:
        examples = list(values[:limit])
        print(f"{title} ({len(values)} total, first {len(examples)}):")
        for value in examples:
            print(f"  {value}")

    def load_pretrained(self, checkpoint_path: str | Path, *, example_limit: int = 20) -> dict[str, int]:
        path = self._resolve_pretrained_checkpoint(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu")
        source_state = self._extract_state_dict(checkpoint)
        target_state = self.model.state_dict()

        matched: dict[str, Tensor] = {}
        matched_examples: list[str] = []
        ignored_keys: list[str] = []
        unexpected_keys: list[str] = []
        shape_mismatch_keys: list[str] = []
        ignored_prefixes = ("head.", "aux_head.", "post_network.")
        for source_key, value in source_state.items():
            normalized_key = source_key
            for prefix in ("module.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            if normalized_key.startswith(ignored_prefixes):
                ignored_keys.append(source_key)
                continue
            target_value = target_state.get(normalized_key)
            if target_value is None:
                unexpected_keys.append(source_key)
                continue
            if getattr(value, "shape", None) != target_value.shape:
                shape_mismatch_keys.append(
                    f"{source_key}: source={tuple(value.shape)} target={tuple(target_value.shape)}"
                )
                continue
            matched[normalized_key] = value
            if len(matched_examples) < example_limit:
                matched_examples.append(f"{source_key}: shape={tuple(value.shape)}")

        missing_keys = sorted(
            key for key in set(target_state) - set(matched) if not key.startswith(ignored_prefixes)
        )
        self.model.load_state_dict(matched, strict=False)
        report = {
            "loaded": len(matched),
            "missing": len(missing_keys),
            "ignored": len(ignored_keys),
            "unexpected": len(unexpected_keys),
            "shape_mismatch": len(shape_mismatch_keys),
            "total_target": len(target_state),
        }
        print(
            "Official WaveViT pretrained load: "
            f"loaded={report['loaded']}/{report['total_target']} "
            f"missing={report['missing']} ignored={report['ignored']} "
            f"unexpected={report['unexpected']} shape_mismatch={report['shape_mismatch']}"
        )
        self._print_examples("Official WaveViT source key examples", list(source_state.keys()), example_limit)
        self._print_examples("Official WaveViT loaded examples", matched_examples, example_limit)
        self._print_examples("Official WaveViT missing examples", missing_keys, example_limit)
        self._print_examples("Official WaveViT ignored examples", ignored_keys, example_limit)
        self._print_examples("Official WaveViT unexpected examples", unexpected_keys, example_limit)
        self._print_examples("Official WaveViT shape mismatch examples", shape_mismatch_keys, example_limit)
        return report

    def forward(self, x: Tensor) -> list[Tensor]:
        batch = x.shape[0]
        outputs: list[Tensor] = []
        for stage_index in range(self.model.num_stages):
            patch_embed = getattr(self.model, f"patch_embed{stage_index + 1}")
            block = getattr(self.model, f"block{stage_index + 1}")
            x, height, width = patch_embed(x)
            for layer in block:
                x = layer(x, height, width)
            norm = getattr(self.model, f"norm{stage_index + 1}")
            x = norm(x)
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if stage_index in self.out_indices:
                outputs.append(x)
        return outputs
