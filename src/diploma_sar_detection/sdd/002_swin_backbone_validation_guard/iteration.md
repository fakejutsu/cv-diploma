# Iteration 002 — Swin Backbone Validation Guard

## Goal
Проверить, что новая архитектура с `Swin-T` backbone корректно интегрирована в пайплайн, и устранить разрыв между train и standalone validate/predict для кастомных чекпоинтов.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`scripts/train_swin.py`](../../scripts/train_swin.py)
- [`scripts/validate.py`](../../scripts/validate.py)
- [`scripts/predict_sample.py`](../../scripts/predict_sample.py)
- [`custom_models/register.py`](../../custom_models/register.py)
- [`custom_models/swin_t_backbone.py`](../../custom_models/swin_t_backbone.py)
- [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)

## Findings
- Confirmed by repo:
  - Архитектура из YAML собирается с `SwinTBackbone` в качестве первого слоя после вызова `register_swin_t_backbone()`.
  - Контракт каналов согласован: backbone выдаёт `192/384/768`, что соответствует `Index`-слоям в head.
- Confirmed by runtime:
  - Для standalone `validate.py`/`predict_sample.py` требовался `PYTHONPATH=.` при загрузке кастомного чекпоинта, иначе `ultralytics` пытался установить несуществующий pip-пакет `custom_models`.

## Decisions
1. Добавить устойчивый путь загрузки локальных кастомных backbone в `validate.py` и `predict_sample.py` через:
   - добавление root проекта в `sys.path`;
   - best-effort вызов `register_swin_t_backbone()`.
2. Добавить отдельный технический скрипт `scripts/validate_swin_backbone.py` для быстрой проверки интеграции архитектуры:
   - сборка модели из YAML;
   - проверка типа backbone;
   - проверка channel contract через dummy-forward;
   - опциональная проверка загрузки checkpoint.

## Spec updates
- Зафиксировано, что standalone validate/predict поддерживают локальные кастомные backbone без внешней установки `custom_models`.
- Зафиксировано наличие отдельного guard-скрипта для проверки корректности Swin-интеграции.

## Open questions
- Нужна ли автоматизация этого guard-скрипта в CI при появлении CI-контура в проекте?
- Нужны ли дополнительные проверки spatial strides/feature sizes в guard-скрипте под разные `imgsz`?

## Next iteration
Добавить smoke-команду в README/SDD регламент (единая команда проверки `validate_swin_backbone.py` + короткий train/val), чтобы ускорить регрессионную проверку при изменении YAML/backbone.
