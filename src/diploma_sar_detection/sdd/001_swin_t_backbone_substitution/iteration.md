# Iteration 001 — Swin-T Backbone Substitution State Sync

## Goal
Актуализировать SDD-состояние фичи подмены `swin_t` backbone для пайплайна обучения и зафиксировать, что уже реализовано в коде, а что ещё не подтверждено.

## Inputs
- [`sdd/README.md`](../README.md)
- [`AGENTS.md`](../../AGENTS.md)
- [`scripts/train_swin.py`](../../scripts/train_swin.py)
- [`custom_models/register.py`](../../custom_models/register.py)
- [`custom_models/swin_t_backbone.py`](../../custom_models/swin_t_backbone.py)
- [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)
- [`requirements.txt`](../../requirements.txt)

## Findings
- Confirmed by repo:
  - Swin-путь изолирован в отдельном скрипте обучения и не затрагивает baseline-путь.
  - Подмена `TorchVision` выполняется явно перед `YOLO(model_yaml)`.
  - Реализована timm-обёртка `SwinTBackbone` с выдачей multi-scale feature maps.
  - YAML для Swin уже согласован с тремя уровнями признаков (`192/384/768`) и `Detect`.
- Inferred:
  - Текущий статус фичи — рабочий scaffold для экспериментов, но не production-ready замена backbone.
- Not established in repo:
  - Нет закреплённого в SDD/артефактах факта успешного smoke-train/val именно для Swin-пути.

## Decisions
1. Зафиксировать текущее состояние как `implemented scaffold`.
2. Не декларировать фичу как полностью завершённую до появления подтверждённых обучающих артефактов и метрик.
3. Использовать `sdd/CURRENT_SPEC.md` как источник истины по состоянию подмены backbone на текущий момент.

## Spec updates
- Добавлен `sdd/CURRENT_SPEC.md` с явным разделением:
  - `Confirmed by repo`
  - `Inferred`
  - `Not established in repo`
- Состояние фичи подмены `swin_t` формализовано без изменения runtime-поведения кода.

## Open questions
- Какие минимальные smoke-параметры обучения считаются достаточными для перевода статуса из scaffold в validated?
- На каком наборе данных и по каким метрикам фиксируем критерий приемки для Swin-T варианта?
- Нужно ли формализовать fallback-стратегию, если monkey-patch `TorchVision` перестанет быть совместимым с будущей версией `ultralytics`?

## Next iteration
Провести и зафиксировать короткий воспроизводимый прогон `train_swin.py` + `validate.py` (с параметрами smoke-run), затем обновить `CURRENT_SPEC.md` раздел `Not established in repo`.
