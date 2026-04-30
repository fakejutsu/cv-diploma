from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .wavevit_backbone import _WAVEVIT_CONFIGS, WaveVitBlock, WaveVitDownsample, WaveVitStem


_OFFICIAL_PRETRAINED_ALIASES = {
    "imagenet_1k_224": {
        "wavevit_s": {
            "filename": "wavevit_s_224.pth",
            "google_drive_id": "1DLdqLRPiARBXAc9DQQtamVZnFtMWBurd",
            "google_drive_url": "https://drive.google.com/file/d/1DLdqLRPiARBXAc9DQQtamVZnFtMWBurd/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/1JfNgmBE5ieAGsBermjpSzQ",
        },
        "wavevit_b": {
            "filename": "wavevit_b_224.pth",
            "google_drive_id": "1g-_eP_ty1JmEsiRqmErwFgf4XTo0ML61",
            "google_drive_url": "https://drive.google.com/file/d/1g-_eP_ty1JmEsiRqmErwFgf4XTo0ML61/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/16DLjX7dKaZuULn7OrxVqyw",
        },
        "wavevit_l": {
            "filename": "wavevit_l_224.pth",
            "google_drive_id": "11MitPmWgg3GB02ndLT4rFO3xc6n3Jmyh",
            "google_drive_url": "https://drive.google.com/file/d/11MitPmWgg3GB02ndLT4rFO3xc6n3Jmyh/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/1L-ZQ1eiv4sdefvdi7Z4-bw",
        },
    },
    "imagenet_1k_384": {
        "wavevit_s": {
            "filename": "wavevit_s_384.pth",
            "google_drive_id": "1aR_5zZ-iVDbUyEEMkvYNV7Nj78C7dJ6G",
            "google_drive_url": "https://drive.google.com/file/d/1aR_5zZ-iVDbUyEEMkvYNV7Nj78C7dJ6G/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/1RLxSDwVG1wQs29s0x3_6QA",
        },
        "wavevit_b": {
            "filename": "wavevit_b_384.pth",
            "google_drive_id": "1mrtGVPIJ8-F9SDofuIwBccXa-xA_E7s1",
            "google_drive_url": "https://drive.google.com/file/d/1mrtGVPIJ8-F9SDofuIwBccXa-xA_E7s1/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/1KYINKOLO4Bf9kabDMjN-TA",
        },
        "wavevit_l": {
            "filename": "wavevit_l_384.pth",
            "google_drive_id": "1Rda4zS2JG0MlcQqrtxKzSWh2149_e5wV",
            "google_drive_url": "https://drive.google.com/file/d/1Rda4zS2JG0MlcQqrtxKzSWh2149_e5wV/view?usp=sharing",
            "baidu_url": "https://pan.baidu.com/s/1BdQBbdUeeAo8CojNifURGw",
        },
    },
}


class OriginalWaveVitBackbone(nn.Module):
    """
    WaveViT backbone variant that preserves the original stage architecture.

    This wrapper intentionally omits the ImageNet classification/token-label
    heads and exposes stage outputs as NCHW feature maps for YOLO/FPN usage.
    It uses the repository-local Haar wavelet backend to avoid a hard
    dependency on `torch_wavelets`.
    """

    def __init__(
        self,
        model_name: str = "original_wavevit_s",
        pretrained: bool | str | Path = False,
        out_indices: Sequence[int] = (1, 2, 3),
        drop_path_rate: float = 0.0,
        in_chans: int = 3,
    ) -> None:
        super().__init__()

        normalized_name = str(model_name).strip().lower()
        config_name = normalized_name.removeprefix("original_")
        if config_name not in _WAVEVIT_CONFIGS:
            supported = ", ".join(f"original_{name}" for name in sorted(_WAVEVIT_CONFIGS))
            raise ValueError(f"Unsupported original WaveViT variant `{model_name}`. Supported: {supported}.")
        config = _WAVEVIT_CONFIGS[config_name]
        self.model_name = normalized_name
        self.config_name = config_name
        self.out_indices = tuple(int(index) for index in out_indices)
        self.embed_dims = tuple(int(dim) for dim in config["embed_dims"])
        self.channels = tuple(self.embed_dims[index] for index in self.out_indices)
        self.num_stages = len(self.embed_dims)

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
            stages.append(
                nn.ModuleDict(
                    {
                        "patch_embed": patch_embed,
                        "blocks": blocks,
                        "norm": norm_layer(self.embed_dims[stage_index]),
                    }
                )
            )
            cursor += depth

        self.stages = nn.ModuleList(stages)
        self.apply(self._init_weights)
        if pretrained:
            if isinstance(pretrained, bool):
                raise ValueError(
                    "OriginalWaveVitBackbone requires an explicit checkpoint path for pretrained=True. "
                    "Use scripts/train_swin.py --pretrained-backbone <wavevit_checkpoint.pth>."
                )
            self.load_pretrained(pretrained)

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
            module.weight.data.normal_(0, (2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs: list[Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            x, height, width = stage["patch_embed"](x)
            for block in stage["blocks"]:
                x = block(x, height, width)
            x = stage["norm"](x)
            x = x.reshape(x.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
            if stage_index in self.out_indices:
                outputs.append(x)
        return outputs

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
    def _remap_official_key(key: str) -> str | None:
        normalized = key
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]

        ignored_prefixes = ("head.", "aux_head.", "post_network.")
        if normalized.startswith(ignored_prefixes):
            return None

        for stage_index in range(4):
            official_stage = stage_index + 1
            patch_prefix = f"patch_embed{official_stage}."
            if normalized.startswith(patch_prefix):
                return f"stages.{stage_index}.patch_embed.{normalized[len(patch_prefix):]}"

            block_prefix = f"block{official_stage}."
            if normalized.startswith(block_prefix):
                tail = normalized[len(block_prefix) :]
                tail = tail.replace(".attn.kv.0.", ".attn.kv_norm.")
                tail = tail.replace(".attn.kv.1.", ".attn.kv.")
                return f"stages.{stage_index}.blocks.{tail}"

            norm_prefix = f"norm{official_stage}."
            if normalized.startswith(norm_prefix):
                return f"stages.{stage_index}.norm.{normalized[len(norm_prefix):]}"

        return normalized

    @staticmethod
    def _print_examples(title: str, values: Sequence[str], limit: int) -> None:
        examples = list(values[:limit])
        print(f"{title} ({len(values)} total, first {len(examples)}):")
        for value in examples:
            print(f"  {value}")

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
                f"Original WaveViT pretrained checkpoint not found: {path}. "
                f"Supported built-in aliases: {supported}."
            )

        entry = variant_aliases[self.config_name]
        cache_dir = Path("weights") / "pretrained" / "wavevit"
        cached_path = cache_dir / str(entry["filename"])
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

    def load_pretrained(self, checkpoint_path: str | Path, *, example_limit: int = 20) -> dict[str, int]:
        path = self._resolve_pretrained_checkpoint(checkpoint_path)

        checkpoint = torch.load(path, map_location="cpu")
        source_state = self._extract_state_dict(checkpoint)
        target_state = self.state_dict()

        matched: dict[str, Tensor] = {}
        matched_examples: list[str] = []
        ignored_keys: list[str] = []
        unexpected_keys: list[str] = []
        shape_mismatch_keys: list[str] = []
        for source_key, value in source_state.items():
            target_key = self._remap_official_key(source_key)
            if target_key is None:
                ignored_keys.append(source_key)
                continue
            target_value = target_state.get(target_key)
            if target_value is None:
                unexpected_keys.append(f"{source_key} -> {target_key}")
                continue
            if getattr(value, "shape", None) != target_value.shape:
                shape_mismatch_keys.append(
                    f"{source_key} -> {target_key}: source={tuple(value.shape)} target={tuple(target_value.shape)}"
                )
                continue
            matched[target_key] = value
            if len(matched_examples) < example_limit:
                matched_examples.append(f"{source_key} -> {target_key}: shape={tuple(value.shape)}")

        missing_keys = sorted(set(target_state) - set(matched))
        self.load_state_dict(matched, strict=False)
        report = {
            "loaded": len(matched),
            "missing": len(missing_keys),
            "ignored": len(ignored_keys),
            "unexpected": len(unexpected_keys),
            "shape_mismatch": len(shape_mismatch_keys),
            "total_target": len(target_state),
        }
        print(
            "Original WaveViT pretrained load: "
            f"loaded={report['loaded']}/{report['total_target']} "
            f"missing={report['missing']} ignored={report['ignored']} "
            f"unexpected={report['unexpected']} shape_mismatch={report['shape_mismatch']}"
        )
        self._print_examples("Original WaveViT source key examples", list(source_state.keys()), example_limit)
        self._print_examples("Original WaveViT loaded examples", matched_examples, example_limit)
        self._print_examples("Original WaveViT missing examples", missing_keys, example_limit)
        self._print_examples("Original WaveViT ignored examples", ignored_keys, example_limit)
        self._print_examples("Original WaveViT unexpected examples", unexpected_keys, example_limit)
        self._print_examples("Original WaveViT shape mismatch examples", shape_mismatch_keys, example_limit)
        return report
