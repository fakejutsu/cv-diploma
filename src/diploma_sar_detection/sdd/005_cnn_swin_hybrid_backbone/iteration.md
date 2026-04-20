# Iteration 005 — CNN+Swin Hybrid Backbone

## Goal
Заменить дефолтный чистый `Swin-T` backbone на гибридный `CNN -> Swin` путь (по аналогии с практикой для SeaDronesSee-подобных задач), чтобы сначала извлекать локальные признаки через CNN-стем, затем дообогащать их трансформером.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`sdd/004_swin_neck_reintegration/iteration.md`](../004_swin_neck_reintegration/iteration.md)
- [`custom_models/swin_t_backbone.py`](../../custom_models/swin_t_backbone.py)
- [`custom_models/register.py`](../../custom_models/register.py)
- [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)
- [`scripts/train_swin.py`](../../scripts/train_swin.py)
- [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py)

## Findings
- Confirmed by repo (после реализации):
  - Добавлен `HybridCnnSwinTBackbone` с пайплайном `CNN stem -> timm Swin-T(features_only)` и выходом в `NCHW`: [`custom_models/hybrid_cnn_swin_t_backbone.py`](../../custom_models/hybrid_cnn_swin_t_backbone.py).
  - В реестре backbone добавлен явный выбор варианта `register_backbone(variant)` + алиасы совместимости:
    - `swin_t -> SwinTBackbone`;
    - `cnn_swin_t -> HybridCnnSwinTBackbone`.
    Реализация: [`custom_models/register.py`](../../custom_models/register.py), [`custom_models/__init__.py`](../../custom_models/__init__.py).
  - Добавлен отдельный YAML для гибрида: [`models/yolo26_cnn_swin_t.yaml`](../../models/yolo26_cnn_swin_t.yaml), при этом legacy YAML [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml) сохранён.
  - `scripts/train_swin.py`:
    - дефолтный `--model` переключён на гибридный YAML;
    - добавлен `--backbone-variant (auto|swin_t|cnn_swin_t)` с auto-резолвом по имени модели.
  - `scripts/validate.py` и `scripts/predict_sample.py`:
    - добавлен `--backbone-variant`;
    - в `auto` реализован fallback с приоритизацией варианта по имени чекпоинта.
  - Guard-скрипт [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py):
    - поддерживает оба варианта через `--backbone-variant`;
    - проверяет type-contract, каналы `192/384/768` и strides `[8,16,32]`.

## Decisions
1. Ввести новый backbone-класс `HybridCnnSwinTBackbone` в `custom_models/` с пайплайном:
   - `CNN stem` (несколько `Conv+BN+SiLU` блоков, без хардкода CUDA),
   - затем `timm Swin-T` как extractor multi-scale features (`features_only=True`, `out_indices=[1,2,3]`),
   - выход в `NCHW`, совместимый с текущим `Index/FPN/PAN/Detect`.
2. Не ломать baseline- и legacy-путь:
   - сохранить существующий `SwinTBackbone`,
   - добавить отдельный YAML для гибрида (`models/yolo26_cnn_swin_t.yaml`) и переключение через CLI.
3. Расширить регистрацию backbone:
   - добавить явный выбор варианта регистрации (`swin_t` vs `cnn_swin_t`) вместо неявной замены одного класса на все случаи.
4. Обновить архитектурный guard:
   - `scripts/validate_swin_backbone.py` должен валидировать гибридный вариант по тем же критичным контрактам (каналы `192/384/768`, stride `[8,16,32]`),
   - при необходимости переименовать скрипт в более общий (`validate_hybrid_backbone.py`) или оставить алиас для обратной совместимости.
5. Ввести минимальный smoke-протокол для гибрида:
   - короткий train (`--fraction < 1`) + validate + predict,
   - отчёт по `mAP50-95`, времени эпохи и VRAM для сравнения с текущим `Swin-T` конфигом.

## Spec updates
- Основной экспериментальный Swin-путь переключён на гибрид `CNN -> Swin`.
- Контракты выходных уровней (`P3/P4/P5`, `192/384/768`, stride `[8,16,32]`) сохранены неизменными для совместимости с neck/head.
- Legacy pure `Swin-T` путь сохранён и доступен через явный выбор `--backbone-variant swin_t`.

## Open questions
- Какой профиль CNN-стема выбрать первым:
  - лёгкий (минимум параметров) или усиленный (лучше локальные текстуры, но выше VRAM/latency)?
- Нужна ли частичная заморозка Swin-блоков на первых эпохах для стабилизации обучения гибрида?
- Нужна ли отдельная калибровка параметров аугментаций/learning-rate для гибрида до полного A/B?

## Next iteration
Провести фиксированный smoke A/B (`Swin-T` vs `CNN+Swin`) на одном протоколе (seed/imgsz/batch/epochs/fraction), зафиксировать `mAP50-95`, время эпохи и VRAM, затем обновить acceptance-критерии в `CURRENT_SPEC`.
