# Iteration 009 — YOLO26n Gated Swin P4 P5

## Goal
Добавить новый вариант `YOLO26n + channel-wise gated Swin fusion` на уровнях `P4` и `P5`, не заменяя штатный CNN-backbone и не меняя `Detect` head.

Целевой сценарий:
- baseline `YOLO26n backbone`
- `P4 -> GatedSwinFusion -> P4_out`
- `P5 -> GatedSwinFusion -> P5_out`
- далее штатный neck/head, но с `P4_out` и `P5_out` вместо исходных `P4/P5`

Граница итерации:
- gate только channel-wise, не scalar;
- отдельный gate для `P4` и `P5`;
- `Detect`, `nc`, `scales`, dataset contract и train script не ломать;
- acceptance: `model builds -> forward works -> alpha visible -> pretrained transfer valid`.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`sdd/008_yolo26n_swin_context_p4_light/iteration.md`](../008_yolo26n_swin_context_p4_light/iteration.md)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- [`custom_models/register.py`](../../custom_models/register.py)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)
- [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml)
- [`models/yolo26n_swin_context_p4_light.yaml`](../../models/yolo26n_swin_context_p4_light.yaml)

## Findings
- Confirmed by repo:
  - Уже есть отдельные context-variants:
    - `YOLO26n + SwinContextBlock(P5)`;
    - `YOLO26n + light SwinContextBlock(P4)`.
  - `train_swin_context.py` уже умеет выбирать стратегию warm start по числу matched weights.
  - `validate_swin_context.py` уже умеет проверять `P5` и `P4-light` variants через dummy input.
- Inferred:
  - Для gated variant безопаснее вставить только два новых модуля перед head:
    - `GatedSwinFusion(P4)`;
    - `GatedSwinFusion(P5)`.
  - Это даёт стабильный index shift `+2` относительно baseline `YOLO26n`, который можно добавить как отдельную remap-стратегию warm start.

## Decisions
1. Реализовать новый модуль `custom_models/gated_swin_fusion.py`.
2. Реализовать `GatedSwinFusion` как:
   - `SwinContextBlock` внутри;
   - `raw_alpha` shape `[1, C, 1, 1]`;
   - `alpha = sigmoid(raw_alpha)`;
   - `out = alpha * x + (1 - alpha) * swin(x)`.
3. Инициализировать `raw_alpha` значением `4.0`, чтобы старт был близок к чистому CNN.
4. Создать отдельный YAML `models/yolo26n_gated_swin_p4_p5.yaml`.
5. В этом YAML:
   - `P4` брать из слоя `6`;
   - `P5` брать из слоя `10`;
   - в neck использовать gated `P4/P5` вместо исходных.
6. Не менять `Detect` head по смыслу и не менять `nc`.
7. Расширить `validate_swin_context.py` вариантом `gated_p4_p5` с выводом:
   - таблицы слоёв;
   - shape-контракта;
   - `alpha_mean` для `P4` и `P5`.
8. Расширить warm-start remap в `train_swin_context.py` отдельной стратегией для gated variant.

## Spec updates
- В систему добавляется ещё один context-enhancement variant:
  - `YOLO26n + GatedSwinFusion(P4, P5)`.
- Новый variant использует channel-wise gating и стартует в режиме “почти CNN”, а Swin-сигнал подмешивается постепенно в процессе обучения.

## Open questions
- Нужно ли в следующей итерации логировать `alpha_mean` прямо в train loop по эпохам?
- Нужен ли позже asymmetric gate-init для `P4` и `P5`, или текущий `init_alpha=4.0` достаточен для обеих ветвей?

## Next iteration
Сравнить baseline vs `P5` vs `P4-light` vs `gated_p4_p5` на едином протоколе и проверить динамику `alpha_mean` во времени.
