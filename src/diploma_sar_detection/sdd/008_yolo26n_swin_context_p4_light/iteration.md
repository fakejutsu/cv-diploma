# Iteration 008 — YOLO26n Swin Context P4 Light

## Goal
Добавить новый экспериментальный вариант `YOLO26n + light SwinContextBlock(P4)` без изменения штатного CNN-backbone, loss и `Detect` head.

Целевой сценарий:
- baseline `YOLO26n backbone`
- взять штатный `P4`
- `P4 -> SwinContextBlock(light) -> P4_context`
- `Concat(P4, P4_context) -> Conv1x1 -> fused P4`
- использовать `fused P4` в neck вместо исходного `P4`

Граница итерации:
- только `P4`;
- только light-конфигурация блока;
- только `concat` fusion;
- не менять существующий `P5`-вариант;
- acceptance: `model builds -> forward works -> remap still valid -> train command ready`.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`sdd/007_yolo26n_swin_context_p5/iteration.md`](../007_yolo26n_swin_context_p5/iteration.md)
- [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- [`/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml`](/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml)

## Findings
- Confirmed by repo:
  - `007` уже реализовал отдельный path `YOLO26n + SwinContextBlock(P5)` через [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml).
  - `train_swin_context.py` уже умеет грузить baseline `yolo26n.pt` через remap для графа, в котором перед head вставлены три новых слоя.
  - В штатном `YOLO26n` слое `6` находится `P4`-feature map, а слой `10` соответствует `P5` перед началом head.
- Inferred:
  - Новый `P4-light` вариант можно сделать с тем же downstream index shift `+3` для head, что и в `P5`-варианте, если вставить `SwinContextBlock + Concat + Conv` в начале head.
  - Это позволит сохранить действующую remap-логику warm start без отдельной индексной схемы.

## Decisions
1. Добавить новый YAML `models/yolo26n_swin_context_p4_light.yaml`.
2. Оставить backbone `YOLO26n` без изменений.
3. Вставить `SwinContextBlock` не в backbone, а в начале head, используя исходный `P4` из слоя `6`.
4. Использовать light-параметры:
   - `in_channels=128`
   - `hidden_dim=64`
   - `num_heads=2`
   - `window_size=7`
   - `depth=1`
5. Считать корректным shape-контракт:
   - `P4_backbone: [B, 128, 40, 40]`
   - `P4_context: [B, 128, 40, 40]`
   - `Concat: [B, 256, 40, 40]`
   - `P4_fused: [B, 128, 40, 40]`
6. Использовать `P4_fused` только в neck, не переподключая им дальнейшее формирование `P5`.
7. Расширить `validate_swin_context.py`, чтобы он валидировал и `P5`, и `P4-light` варианты.

## Spec updates
- В систему добавляется ещё один context-enhancement variant:
  - `YOLO26n + SwinContextBlock(P4-light)`.
- `P4-light` path предназначен для честного A/B against:
  - baseline `YOLO26n`;
  - `YOLO26n + SwinContextBlock(P5)`.

## Open questions
- Достаточно ли `P4-light` как отдельного ablation, или следующий шаг должен быть сразу `P4+P5`?
- Требуется ли отдельный более явный лог remap для разных context YAML в `train_swin_context.py`?

## Next iteration
Сравнить baseline vs `P5` vs `P4-light` по одному smoke-протоколу и решить, нужен ли объединённый `P4+P5` context path.
