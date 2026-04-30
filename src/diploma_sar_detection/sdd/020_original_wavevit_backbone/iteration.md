# 020_original_wavevit_backbone

## Goal
Добавить side-by-side вариант backbone, максимально близкий к оригинальному `WaveViT` из `YehLi/ImageNetModel/classification/wavevit.py`, но с `features_only` forward для YOLO detection pipeline.

Граница итерации:
- не менять существующий `WaveVitBackbone` / `models/yolo26_wavevit_s.yaml`;
- не добавлять обязательную dependency `torch_wavelets`, чтобы не ломать рабочий pipeline;
- добавить отдельные variants `original_wavevit_s`, `original_wavevit_b`, `original_wavevit_l`;
- сохранить выходной контракт YOLO: реальные stage outputs в `BCHW`, strides `[8, 16, 32]`.

## Inputs
- External reference: `YehLi/ImageNetModel/classification/wavevit.py`.
- Existing registration path: `custom_models/register.py`, `scripts/train_swin.py`.
- Existing validation guard: `scripts/validate_swin_backbone.py`.

## Findings
- Confirmed by external code:
  - official `wavevit_s` uses `Stem -> DownSamples` stages;
  - stages 0/1 use `WaveAttention`, stages 2/3 use standard attention;
  - stage configs:
    - `wavevit_s`: embed dims `[64, 128, 320, 448]`, depths `[3, 4, 6, 3]`;
    - `wavevit_b`: embed dims `[64, 128, 320, 512]`, depths `[3, 4, 12, 3]`;
    - `wavevit_l`: embed dims `[96, 192, 384, 512]`, depths `[3, 6, 18, 3]`.
- Confirmed by repo:
  - `torch_wavelets` is not installed locally;
  - current YOLO custom-backbone registration can safely support additional variants side-by-side.

## Decisions
1. Add `OriginalWaveVitBackbone` as a separate class.
2. Preserve original stage architecture and attention split, but expose stage outputs instead of classification/token-label heads.
3. Use local Haar DWT/IDWT backend to keep runtime dependency-free.
4. Add `models/yolo26_original_wavevit_s.yaml`.
5. Extend train/validate/predict backbone variant choices without changing defaults.
6. Add explicit local checkpoint pretrained loading for `OriginalWaveVitBackbone`.
7. Print diagnostic examples during pretrained loading: source keys, loaded mappings, missing keys, ignored heads, unexpected keys and shape mismatches.
8. Add built-in official ImageNet-1K aliases and a pretrained YAML variant.

## Spec updates
- Adds original-architecture WaveViT backbone variants for detection.
- `original_wavevit_s` YAML consumes stage outputs `[1, 2, 3]` as `P3/P4/P5`.
- `OriginalWaveVitBackbone.load_pretrained()` remaps official classification checkpoint keys from `patch_embed*/block*/norm*` to local `stages.*` keys and ignores classification/token-label heads.
- Pretrained loading report includes example keys, so partial loads are diagnosable instead of relying only on counts.
- `scripts/train_swin.py --pretrained-backbone <path>` loads custom backbone pretrained weights before Ultralytics training.
- `models/yolo26_original_wavevit_s_imagenet.yaml` uses the `imagenet_1k_224` alias and expects `weights/pretrained/wavevit/wavevit_s_224.pth`.
- No WaveViT ImageNet-21K or SAR-specific pretrained checkpoint has been established; current built-in source is official ImageNet-1K.

## Open questions
- Whether P2/stride-4 should be added for small-object recall after the first smoke run.
