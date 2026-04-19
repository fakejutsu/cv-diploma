# CURRENT_SPEC

Updated: 2026-04-19

## Scope
Текущее состояние фичи подмены стандартного YOLO-backbone на `Swin-T` в training pipeline.

## System State

### Confirmed by repo
- В проекте есть отдельный entrypoint для экспериментов со Swin-T: [`scripts/train_swin.py`](../scripts/train_swin.py).
- Перед созданием модели вызывается `register_swin_t_backbone()`, который monkey-patch'ит `ultralytics.nn.tasks.TorchVision` на `SwinTBackbone`: [`custom_models/register.py`](../custom_models/register.py).
- `SwinTBackbone` реализован как timm-обёртка (`features_only=True`) с выдачей multi-scale features и приведением к формату `NCHW`: [`custom_models/swin_t_backbone.py`](../custom_models/swin_t_backbone.py).
- Кастомная архитектура описана в [`models/yolo26_swin_t.yaml`](../models/yolo26_swin_t.yaml):
  - backbone использует `TorchVision` с `swin_tiny_patch4_window7_224` и `out_indices=[1,2,3]`;
  - head выбирает карты признаков через `Index` с каналами `192/384/768` и передаёт их в `Detect`.
- Путь baseline обучения сохранён отдельно и не заменён автоматически: [`scripts/train_baseline.py`](../scripts/train_baseline.py).
- Зависимости для Swin-пути объявлены в `requirements.txt` (`torch`, `torchvision`, `timm`, `ultralytics`).

### Inferred
- Фича нацелена на экспериментальный режим интеграции (`scaffold`), а не на полностью верифицированную замену backbone для всех сценариев.
- Для стабильного обучения критична согласованность `out_indices`/каналов между Swin backbone и `Index`/`Detect` слоями в YAML.

### Not established in repo
- Нет подтверждённого в репозитории результата короткого/полного обучения `train_swin.py` с сохранённым валидным чекпоинтом.
- Нет зафиксированного сравнения метрик baseline vs Swin-T на одном и том же датасете.
- Нет CI/авто-проверки, которая бы гарантировала, что подмена `TorchVision` не ломается при обновлениях `ultralytics`.

## Contracts
- CLI-контракт `scripts/train_swin.py` должен оставаться совместимым с текущими флагами.
- Формат датасета и `data/dataset.yaml` не меняется из-за интеграции Swin-T.
- Артефакты обучения сохраняются в стандартной структуре `runs/<name>/weights/{best,last}.pt`.
