# CURRENT_SPEC

Updated: 2026-04-20

## Scope
Текущее состояние Swin-based кастомных backbone в training pipeline YOLO26, где дефолтный scaffold — гибрид `CNN -> Swin-T`, а legacy pure `Swin-T` сохранён для сравнения.

## System State

### Confirmed by repo
- В проекте есть отдельный entrypoint для Swin-based экспериментов: [`scripts/train_swin.py`](../scripts/train_swin.py).
- Перед созданием модели вызывается `register_backbone(variant)`, который monkey-patch'ит `ultralytics.nn.tasks.TorchVision` на нужный класс:
  - `swin_t -> SwinTBackbone`;
  - `cnn_swin_t -> HybridCnnSwinTBackbone`.
  Реализация: [`custom_models/register.py`](../custom_models/register.py).
- `SwinTBackbone` реализован как timm-обёртка (`features_only=True`) с выдачей multi-scale features и приведением к формату `NCHW`: [`custom_models/swin_t_backbone.py`](../custom_models/swin_t_backbone.py).
- `HybridCnnSwinTBackbone` реализует `CNN stem -> timm Swin-T(features_only)` и выдаёт multi-scale features в `NCHW`: [`custom_models/hybrid_cnn_swin_t_backbone.py`](../custom_models/hybrid_cnn_swin_t_backbone.py).
- Оба backbone не выполняют принудительный внутренний resize входа; spatial размер задаётся внешним training pipeline (`imgsz`) и сохраняет корректный stride-контракт для `Detect`.
- Поддерживаются два YAML-шаблона архитектуры:
  - [`models/yolo26_cnn_swin_t.yaml`](../models/yolo26_cnn_swin_t.yaml) — дефолтный гибридный путь;
  - [`models/yolo26_swin_t.yaml`](../models/yolo26_swin_t.yaml) — legacy pure Swin-T путь.
  В обоих случаях head использует `Index` с каналами `192/384/768`, затем FPN/PAN neck (`Upsample/Concat/C2f` и `Conv/Concat/C2f`) и `Detect`.
- Путь baseline обучения сохранён отдельно и не заменён автоматически: [`scripts/train_baseline.py`](../scripts/train_baseline.py).
- Зависимости для Swin-пути объявлены в `requirements.txt` (`torch`, `torchvision`, `timm`, `ultralytics`).
- Entry points [`scripts/validate.py`](../scripts/validate.py) и [`scripts/predict_sample.py`](../scripts/predict_sample.py):
  - автоматически добавляют root проекта в `sys.path`;
  - поддерживают `--backbone-variant (auto|swin_t|cnn_swin_t)`;
  - в `auto` определяют приоритет варианта по имени чекпоинта и используют fallback на второй вариант.
- Есть отдельная проверка архитектурной интеграции: [`scripts/validate_swin_backbone.py`](../scripts/validate_swin_backbone.py).
- Guard-скрипт `validate_swin_backbone.py` поддерживает `--backbone-variant (auto|swin_t|cnn_swin_t)` и проверяет:
  - channel contract backbone (`192/384/768`);
  - входные каналы `Detect` (`192/384/768`);
  - ожидаемые `Detect` strides `[8,16,32]`.

### Inferred
- Фича нацелена на экспериментальный режим интеграции (`scaffold`), а не на полностью верифицированную замену backbone для всех сценариев.
- Для стабильного обучения критична согласованность `out_indices`/каналов между Swin-based backbone, neck и `Detect` слоями в YAML.

### Not established in repo
- Нет зафиксированного сравнения метрик baseline vs pure `Swin-T` vs hybrid `CNN+Swin-T` на одном и том же датасете.
- Нет CI/авто-проверки, которая бы гарантировала, что подмена `TorchVision` не ломается при обновлениях `ultralytics`.

## Contracts
- CLI-контракт `scripts/train_swin.py` должен оставаться совместимым с текущими флагами.
- Формат датасета и `data/dataset.yaml` не меняется из-за Swin-based интеграции.
- Для Swin-based путей обязателен контракт выходов в neck/head: `P3/P4/P5` каналы `192/384/768`, `Detect` strides `[8,16,32]`.
- Артефакты обучения сохраняются в стандартной структуре `runs/<name>/weights/{best,last}.pt`.
