# 021_vendored_official_wavevit

## Goal
Перейти от manual-port WaveViT к vendored official WaveViT implementation, сохранив YOLO detection pipeline рабочим.

Граница итерации:
- не удалять предыдущие экспериментальные WaveViT-style файлы, чтобы не ломать историю и старые runs;
- добавить новый variant `official_wavevit_s`;
- использовать official module/stage classes как источник истины;
- wrapper должен только отдавать `P3/P4/P5` feature maps в `BCHW`;
- сохранить stride contract `[8, 16, 32]`.

## Inputs
- Official source: `YehLi/ImageNetModel/classification/wavevit.py`.
- Official pretrained checkpoints from `YehLi/ImageNetModel/classification/README.md`.
- Current YOLO custom backbone registration.

## Decisions
1. Vendor official WaveViT code into `custom_models/vendor/wavevit_official.py`.
2. Provide local `torch_wavelets` fallback with the same `DWT_2D`/`IDWT_2D` symbols before importing/using official code.
3. Add `OfficialWaveVitBackbone` wrapper that builds official `WaveViT`, strips classification use from forward, and returns stage outputs.
4. Add `models/yolo26_official_wavevit_s.yaml`.
5. Add `official_wavevit_s` registration and CLI support.

## Spec updates
- `official_wavevit_s` becomes the preferred route for original WaveViT experiments.
- Prior `original_wavevit_s` manual-port remains available but is no longer the recommended original-checkpoint path.
- Added separate YAMLs for non-pretrained and ImageNet-1K pretrained official WaveViT-S.
- Official checkpoint loading uses exact official keys without remapping, ignoring only classification/token-label heads.

## Open questions
- Whether official checkpoint includes buffers that require exact `torch_wavelets`; fallback should be compatible but must be verified with loaded/missing/unexpected report.
- Whether adding P2/stride-4 head is needed after first smoke validation.
