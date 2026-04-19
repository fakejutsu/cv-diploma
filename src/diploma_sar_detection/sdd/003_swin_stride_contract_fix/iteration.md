# Iteration 003 — Swin Stride Contract Fix

## Goal
Устранить деградацию обучения Swin-T (нулевой mAP в ранних эпохах), вызванную нарушением stride-контракта между backbone и Detect head.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`custom_models/swin_t_backbone.py`](../../custom_models/swin_t_backbone.py)
- [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py)
- Runtime-логи обучения Swin-T с `mAP50=0` на первых эпохах.

## Findings
- Confirmed by repo/runtime:
  - `SwinTBackbone.forward()` принудительно ресайзил вход к фиксированному `img_size=640`.
  - Из-за этого при сборке модели `Detect.stride` вычислялся неверно (`[3.2, 6.4, 12.8]` вместо `[8, 16, 32]`).
  - Нарушенный stride-контракт приводит к некорректной геометрии детекций и деградации метрик в training/val.

## Decisions
1. Убрать принудительный resize в `SwinTBackbone.forward()` и сохранить исходный spatial size входа.
2. Расширить guard-валидацию `scripts/validate_swin_backbone.py`:
   - проверять не только channel contract (`192/384/768`), но и `Detect.stride == [8, 16, 32]`.

## Spec updates
- Зафиксировано требование: Swin backbone не должен менять spatial размер входа внутри `forward()`.
- Зафиксировано требование: guard-валидация обязана падать при несоответствии stride-контракта.

## Open questions
- Нужен ли отдельный smoke-ран (`fraction < 1`) как обязательный этап после архитектурных правок backbone?

## Next iteration
Добавить документированный smoke-протокол запуска Swin (короткий train + validate) и минимальные acceptance-критерии по метрикам/динамике loss.
