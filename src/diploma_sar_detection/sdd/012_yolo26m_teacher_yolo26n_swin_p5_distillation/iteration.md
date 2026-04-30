# 012_yolo26m_teacher_yolo26n_swin_p5_distillation

## Goal
Добавить distillation-путь, где teacher `YOLO26m` передаёт high-level семантику student `YOLO26n + SwinContextBlock(P5)` через feature distillation только на уровне `P5`.

## Inputs
- [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml)
- [`best_yolo26m.pt`](../../best_yolo26m.pt)
- [`yolo26n.pt`](../../yolo26n.pt)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)

## Findings
- `YOLO26m` даёт лучший overall результат и может выступать teacher.
- Pure `Swin-T` может быть сильнее baseline на отдельных классах, но overall проигрывает и слишком тяжёлый.
- `YOLO26n + SwinContextBlock(P5)` уже реализован как компактный student-кандидат.

## Decisions
- Teacher фиксируется как `YOLO26m`.
- Student фиксируется как `YOLO26n + SwinContextBlock(P5)`.
- Distillation делается только на fused `P5` student (`layer 13`) против semantic `P5` teacher (`layer 10`).
- Для выравнивания каналов используется student-side `1x1` adapter `256 -> 512`.
- Первая версия использует только feature distillation (`smoothl1`) без `P4`, без bbox/logit distill.

## Extension
- `train_distill.py` и `validate_distill_setup.py` обобщены для альтернативных student-архитектур через CLI-параметры:
  - `--student-backbone-variant`
  - `--student-distill-layer`
  - `--student-distill-channels`
  - `--teacher-distill-layer`
  - `--teacher-distill-channels`
- Для pure `Swin-T + neck` student подтверждён рабочий distillation contract:
  - student `P5` = `layer 15`
  - student `P5 channels` = `768`
  - teacher `P5` остаётся `layer 10`, `512 channels`.

## Spec updates
- Добавлен новый custom model class для train-only distillation:
  [`custom_models/distill_swin_p5_model.py`](../../custom_models/distill_swin_p5_model.py)
- Добавлен новый entrypoint:
  [`scripts/train_distill.py`](../../scripts/train_distill.py)
- Добавлен sanity-check:
  [`scripts/validate_distill_setup.py`](../../scripts/validate_distill_setup.py)

## Open questions
- Даст ли `P5-only distillation` measurable gain относительно обычного `YOLO26n + Swin(P5)`.
- Нужен ли второй этап с `P4-light`.
- Имеет ли смысл потом добавлять logit distillation поверх `P5` feature distill.

## Next iteration
- Если `P5-only distillation` даст прирост, расширить на `P4-light` и/или добавить classification distillation.
