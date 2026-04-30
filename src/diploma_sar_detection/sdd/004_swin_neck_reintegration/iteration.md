# Iteration 004 — Swin Neck Reintegration

## Goal
Вернуть neck (FPN/PAN) в `YOLO26 + Swin-T` архитектуру между `SwinTBackbone` и `Detect`, чтобы уйти от минимального `Index -> Detect` и повысить устойчивость feature fusion.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)
- [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py)
- Runtime-наблюдения по train/val метрикам Swin-конфига без neck.

## Findings
- Confirmed by repo (до изменений):
  - архитектура использовала прямой путь `Swin (P3/P4/P5) -> Detect` без FPN/PAN neck.
  - guard-скрипт проверял тип backbone, channel contract backbone и stride контракты Detect.
- Inferred:
  - отсутствие neck ограничивает multi-scale fusion и может быть субоптимальным для малых/сложных объектов.

## Decisions
1. Реинтегрировать neck в [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml):
   - top-down FPN: `Upsample + Concat + C2f`;
   - bottom-up PAN: `Conv(s=2) + Concat + C2f`;
   - Detect входы оставить в канальном контракте `192/384/768`.
2. Усилить guard в [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py):
   - дополнительно проверять фактические входные каналы `Detect` (через ветви head), а не только backbone channel contract.

## Spec updates
- Для Swin-T scaffold теперь зафиксирован путь `Swin -> Index -> FPN/PAN neck -> Detect`.
- Guard-проверка расширена: `Detect` обязан получать каналы `[192, 384, 768]` и stride `[8, 16, 32]`.

## Open questions
- Нужен ли отдельный облегчённый neck-вариант (меньше повторов `C2f`) для более быстрого обучения на ограниченном времени?
- Какие целевые пороги сравнения с baseline считать acceptance-критерием для нового neck на фиксированном протоколе?

## Next iteration
Добавить регламент A/B-сравнения `Swin(no-neck)` vs `Swin(neck)` на фиксированном протоколе (epochs/imgsz/batch/seed), включая сводный отчет по `mAP50-95`, времени эпохи и VRAM.
