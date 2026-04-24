# CURRENT_SPEC

Updated: 2026-04-24

## Scope
Текущее состояние экспериментальных Swin-based интеграций в training pipeline YOLO26:
- legacy pure `Swin-T` backbone replacement;
- hybrid `CNN -> Swin-T` backbone replacement;
- `YOLO26n + SwinContextBlock(P5)` как context-enhancement путь без замены штатного CNN-backbone.

## System State

### Confirmed by repo
- В проекте есть отдельный entrypoint для Swin-based экспериментов: [`scripts/train_swin.py`](../scripts/train_swin.py).
- Для context-enhancement пути добавлен отдельный entrypoint: [`scripts/train_swin_context.py`](../scripts/train_swin_context.py).
- Перед созданием модели вызывается `register_backbone(variant)`, который monkey-patch'ит `ultralytics.nn.tasks.TorchVision` на нужный класс:
  - `swin_t -> SwinTBackbone`;
  - `cnn_swin_t -> HybridCnnSwinTBackbone`.
  Реализация: [`custom_models/register.py`](../custom_models/register.py).
- Для context-enhancement пути используется отдельная регистрация локальных YAML-модулей через `register_context_modules()`, которая экспортирует `SwinContextBlock` в `ultralytics.nn.tasks`: [`custom_models/register.py`](../custom_models/register.py).
- `SwinTBackbone` реализован как timm-обёртка (`features_only=True`) с выдачей multi-scale features и приведением к формату `NCHW`: [`custom_models/swin_t_backbone.py`](../custom_models/swin_t_backbone.py).
- `HybridCnnSwinTBackbone` реализует `YOLO-style stride-preserving stem (Conv/C3k2) -> timm Swin-T(features_only)` и выдаёт multi-scale features в `NCHW`: [`custom_models/hybrid_cnn_swin_t_backbone.py`](../custom_models/hybrid_cnn_swin_t_backbone.py).
- `SwinContextBlock` реализован как лёгкий Swin-style context enhancer поверх уже вычисленного YOLO feature map `P5`, использует `torchvision.models.swin_transformer.SwinTransformerBlock` и сохраняет `NCHW`-контракт входа/выхода: [`custom_models/swin_context_block.py`](../custom_models/swin_context_block.py).
- Оба backbone не выполняют принудительный внутренний resize входа; spatial размер задаётся внешним training pipeline (`imgsz`) и сохраняет корректный stride-контракт для `Detect`.
- В гибридном пути stem не выполняет downsample (stride=1), поэтому `Detect` сохраняет ожидаемые strides `[8,16,32]`.
- Поддерживаются три YAML-шаблона архитектуры:
  - [`models/yolo26_cnn_swin_t.yaml`](../models/yolo26_cnn_swin_t.yaml) — дефолтный гибридный путь;
  - [`models/yolo26_swin_t.yaml`](../models/yolo26_swin_t.yaml) — legacy pure Swin-T путь;
  - [`models/yolo26n_swin_context_p5.yaml`](../models/yolo26n_swin_context_p5.yaml) — `YOLO26n + SwinContextBlock(P5)` без замены backbone.
- В `models/yolo26n_swin_context_p5.yaml`:
  - штатный `YOLO26n` backbone сохранён;
  - после `P5` добавлен `SwinContextBlock`;
  - выполняется `Concat(P5, context)` и `Conv1x1`-проекция обратно в fused `P5`;
  - затем используется штатный `YOLO26n` neck/head.
- Путь baseline обучения сохранён отдельно и не заменён автоматически: [`scripts/train_baseline.py`](../scripts/train_baseline.py).
- Зависимости для Swin-пути объявлены в `requirements.txt` (`torch`, `torchvision`, `timm`, `ultralytics`).
- Entry points [`scripts/validate.py`](../scripts/validate.py) и [`scripts/predict_sample.py`](../scripts/predict_sample.py):
  - автоматически добавляют root проекта в `sys.path`;
  - поддерживают `--backbone-variant (auto|swin_t|cnn_swin_t)`;
  - вызывают `register_context_modules()` best-effort;
  - в `auto` определяют приоритет варианта по имени чекпоинта и используют fallback на второй вариант.
- Есть отдельная проверка архитектурной интеграции: [`scripts/validate_swin_backbone.py`](../scripts/validate_swin_backbone.py).
- Guard-скрипт `validate_swin_backbone.py` поддерживает `--backbone-variant (auto|swin_t|cnn_swin_t)` и проверяет:
  - channel contract backbone (`192/384/768`);
  - входные каналы `Detect` (`192/384/768`);
  - ожидаемые `Detect` strides `[8,16,32]`.
- Для context-пути добавлен отдельный sanity-check: [`scripts/validate_swin_context.py`](../scripts/validate_swin_context.py).
- `validate_swin_context.py` подтверждает для dummy input:
  - сборку baseline и context-модели;
  - корректный `forward`;
  - shape-контракт `P5_backbone -> P5_context -> P5_fused`;
  - различие по числу параметров baseline vs modified.

### Inferred
- Все Swin-based пути остаются экспериментальными scaffold-режимами, а не полностью верифицированной заменой базового пайплайна для всех сценариев.
- Для стабильного обучения критична согласованность `out_indices`/каналов между Swin-based backbone, neck и `Detect` слоями в YAML.
- `models/yolo26n_swin_context_p5.yaml` на текущем шаге привязан к `YOLO26n` scale (`n`) и не является универсальным шаблоном для `s/m/l/x`.

### Not established in repo
- Нет зафиксированного сравнения метрик baseline vs pure `Swin-T` vs hybrid `CNN+Swin-T` vs `YOLO26n + SwinContextBlock(P5)` на одном и том же датасете.
- Нет CI/авто-проверки, которая бы гарантировала, что подмена `TorchVision` не ломается при обновлениях `ultralytics`.
- Для context-path пока не зафиксирована отдельная стратегия экспорт/ONNX-совместимости.

## Contracts
- CLI-контракт `scripts/train_swin.py` должен оставаться совместимым с текущими флагами.
- CLI-контракт `scripts/train_swin_context.py` должен оставаться совместимым с текущими флагами.
- Формат датасета и `data/dataset.yaml` не меняется из-за Swin-based интеграции.
- Для Swin-based путей обязателен контракт выходов в neck/head: `P3/P4/P5` каналы `192/384/768`, `Detect` strides `[8,16,32]`.
- Для `YOLO26n + SwinContextBlock(P5)` обязателен контракт:
  - `P5_backbone`, `P5_context` и `P5_fused` имеют одинаковые spatial размеры;
  - `P5_fused` возвращается к каналам штатного `YOLO26n P5`;
  - baseline `YOLO26n` backbone/neck/head не заменяются.
- Артефакты обучения сохраняются в стандартной структуре `runs/<name>/weights/{best,last}.pt`.
