# 019_wavevit_pure_backbone

## Goal
Добавить чистый `WaveViT` backbone variant для YOLO26 detection pipeline.

Граница итерации:
- не менять существующие Swin/WaveViT context варианты;
- не добавлять external dependency `torch_wavelets`;
- сохранить YOLO `Detect` stride contract `[8, 16, 32]`;
- начать с `wavevit_s` как минимального официального варианта.

## Inputs
- External reference: `YehLi/ImageNetModel/classification/wavevit.py`.
- Existing Swin backbone registration path: `custom_models/swin_t_backbone.py`, `custom_models/register.py`, `scripts/train_swin.py`.
- Existing local Haar DWT implementation: `custom_models/wavevit_context_block.py`.

## Findings
- Confirmed by external code:
  - official WaveViT has `wavevit_s`, `wavevit_b`, `wavevit_l`;
  - backbone-only parameter counts are approximately:
    - `wavevit_s`: `21.79M`;
    - `wavevit_b`: `32.49M`;
    - `wavevit_l`: `56.45M`.
- Confirmed by repo:
  - existing WaveViT code is a context block, not a pure multi-scale backbone;
  - `torch_wavelets` is not installed locally.

## Decisions
1. Add local `WaveVitBackbone` with stage-wise multi-scale outputs.
2. Use parameter-free Haar DWT/IDWT from local code instead of `torch_wavelets`.
3. Expose variants through `register_backbone("wavevit_s"|"wavevit_b"|"wavevit_l")`.
4. Add `models/yolo26_wavevit_s.yaml` with `Index -> FPN/PAN -> Detect`.
5. Add a WaveViT-specific validator for dummy forward, output channels and strides.

## Spec updates
- Adds a pure WaveViT backbone path parallel to existing pure Swin-T backbone path.
- First supported YAML targets `wavevit_s` channels `[128, 320, 448]` for `P3/P4/P5`.

## Open questions
- Whether training from scratch can converge on the current SAR dataset.
- Whether ImageNet/pretrained official weights should be ported later.
- Whether full attention in the `P4` stage is too expensive at `imgsz=640`.
