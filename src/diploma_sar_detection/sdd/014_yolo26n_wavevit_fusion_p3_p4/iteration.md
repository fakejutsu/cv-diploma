# 014_yolo26n_wavevit_fusion_p3_p4

## Goal
Добавить отдельный вариант `YOLO26n + GatedWaveVitFusion(P3, P4)`, где штатный CNN backbone/neck/head сохраняются, а WaveViT-style wavelet attention используется только как context/fusion side branch для small-object уровней.

Граница итерации:
- не заменять штатный YOLO26n backbone;
- не менять формат датасета, loss, optimizer schedule и augmentations;
- не менять существующие Swin/Swin-gated YAML;
- реализовать минимальный проверяемый WaveViT-style context path на `P3/P4`.

## Inputs
- [`custom_models/gated_swin_fusion.py`](../../custom_models/gated_swin_fusion.py)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- [`models/yolo26n_gated_swin_p4_p5.yaml`](../../models/yolo26n_gated_swin_p4_p5.yaml)
- [`models/yolo26n_swin_context_p4_light.yaml`](../../models/yolo26n_swin_context_p4_light.yaml)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)
- External reference: `YehLi/ImageNetModel/classification/wavevit.py`

## Findings
- Confirmed by repo:
  - Текущие context/fusion варианты уже сохраняют штатный YOLO26n backbone и добавляют локальные YAML-модули через `register_context_modules()`.
  - Для текущего YOLO26n YAML фактические каналы feature layers `P3/P4/P5`: `128/128/256`.
  - Existing gated path использует два fusion-модуля перед neck и сохраняет стандартный `Detect(P3, P4, P5)` контракт.
- Inferred:
  - Для small objects более полезно начать с `P3/P4`, а не `P4/P5`, потому что `P3/8` сохраняет больше spatial detail.
  - Полный WaveViT backbone replacement несёт больший риск по каналам, checkpoint compatibility и feature stride contract.
  - Для первой версии достаточно WaveViT-style wavelet attention block с shape-preserving контрактом `NCHW -> NCHW`; exact ImageNet classification head не нужен.

## Decisions
1. Добавить `WaveVitContextBlock`, который сохраняет shape входного feature map.
2. Использовать фиксированное Haar DWT/IDWT через PyTorch conv/conv_transpose, без новой runtime-зависимости на `PyWavelets`.
3. Добавить `GatedWaveVitFusion`, совместимый по смыслу с `GatedSwinFusion`: `out = alpha * x + (1 - alpha) * wave`.
4. Создать отдельный YAML `models/yolo26n_gated_wavevit_p3_p4.yaml`.
5. В YAML применять fusion к:
   - `P3` layer `4`, channels `128`;
   - `P4` layer `6`, channels `128`;
   - `P5` оставить штатным.
6. Добавить отдельный validator `scripts/validate_wavevit_context.py`, который проверяет dummy forward, shape contract, Detect strides и gate statistics.
7. Расширить baseline warm-start remap в `scripts/train_swin_context.py` отдельной strategy `wavevit_p3_p4_shift2`, совпадающей по layer shift с old gated layout, но отличимой в логах.

## Spec updates
- В систему добавляется новый context-enhancement variant:
  - `YOLO26n + GatedWaveVitFusion(P3, P4)`.
- Новый variant не заменяет backbone и не меняет Detect head.
- `WaveVitContextBlock` использует wavelet attention над `NCHW` feature map и сохраняет входной shape.
- `GatedWaveVitFusion` содержит обучаемый channel-wise `raw_alpha` shape `[1, C, 1, 1]`.
- `P3_out` и `P4_out` обязаны сохранять shape исходных `P3/P4`.
- `Detect` обязан сохранить strides `[8, 16, 32]`.

## Open questions
- Окажется ли `P3` WaveViT-style attention слишком дорогим по памяти на `imgsz=640`.
- Нужен ли следующий light вариант `P4 only`, если `P3/P4` будет медленным.
- Нужен ли later warm-start из внешних WaveViT weights, если random context branch покажет стабильный smoke train.

## Next iteration
- Провести smoke train и сравнить с текущими Swin-gated вариантами по:
  - runtime/memory;
  - `alpha_mean` на `P3/P4`;
  - small-object class metrics.
