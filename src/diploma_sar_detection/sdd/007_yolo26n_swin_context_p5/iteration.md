# Iteration 007 — YOLO26n Swin Context P5

## Goal
Добавить минимально рабочее контекстное усиление `YOLO26n` через `SwinContextBlock` над штатным `P5`, не заменяя существующий CNN-backbone и не вводя вторую полноценную backbone-ветвь.

Целевой сценарий итерации:
- `YOLO26n backbone -> P5`
- `P5 -> SwinContextBlock -> S5_context`
- `Concat(P5, S5_context) -> projection -> fused P5`
- далее штатный `YOLO26n neck/head`

Граница итерации:
- только `P5`;
- только `concat` fusion;
- без raw-image Swin branch;
- без `freeze_swin_epochs`;
- без `swin_lr_mult`;
- acceptance: `model builds -> forward works -> train starts`.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`sdd/006_yolo_stem_stride_preserving/iteration.md`](../006_yolo_stem_stride_preserving/iteration.md)
- [`AGENTS.md`](../../AGENTS.md)
- [`scripts/train_baseline.py`](../../scripts/train_baseline.py)
- [`scripts/train_swin.py`](../../scripts/train_swin.py)
- [`scripts/validate.py`](../../scripts/validate.py)
- [`scripts/predict_sample.py`](../../scripts/predict_sample.py)
- [`custom_models/register.py`](../../custom_models/register.py)
- [`models/yolo26_cnn_swin_t.yaml`](../../models/yolo26_cnn_swin_t.yaml)
- [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)
- [`/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml`](/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml)

## Findings
- Confirmed by repo:
  - Baseline-путь обучения остаётся отдельным и использует штатный `YOLO(args.model)`: [`scripts/train_baseline.py`](../../scripts/train_baseline.py).
  - Текущий Swin-based путь в репозитории реализован как подмена `ultralytics.nn.tasks.TorchVision` через `register_backbone(variant)`: [`custom_models/register.py`](../../custom_models/register.py).
  - Уже существуют два варианта backbone replacement:
    - pure `Swin-T`;
    - hybrid `CNN -> Swin-T`.
  - Оба текущих YAML для Swin-пути описывают замену backbone, а не auxiliary/context branch:
    - [`models/yolo26_swin_t.yaml`](../../models/yolo26_swin_t.yaml)
    - [`models/yolo26_cnn_swin_t.yaml`](../../models/yolo26_cnn_swin_t.yaml)
  - Штатный граф `YOLO26n` собирается из upstream YAML и использует стандартный backbone с выходами `P3/P4/P5`, после чего идёт neck/head с `Detect`: [`/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml`](/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/cfg/models/26/yolo26.yaml).
  - В проекте уже есть зависимости `torch`, `torchvision`, `timm`, `ultralytics`: [`requirements.txt`](../../requirements.txt).
  - Реализован новый локальный модуль [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py), который использует `torchvision.models.swin_transformer.SwinTransformerBlock` для контекстного усиления feature map `P5`.
  - Реализована регистрация `SwinContextBlock` в `ultralytics.nn.tasks` через `register_context_modules()`: [`custom_models/register.py`](../../custom_models/register.py).
  - Добавлен новый модельный конфиг [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml), в котором:
    - штатный `YOLO26n` backbone сохранён;
    - после `P5` вставлен `SwinContextBlock`;
    - после `Concat(P5, context)` выполняется `Conv1x1`-проекция обратно в fused `P5`;
    - далее используется штатный `YOLO26n neck/head`.
  - Добавлен отдельный entrypoint [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py) для обучения context-модели с опциональной загрузкой baseline-весов через `--weights`.
  - Добавлен отдельный sanity-check [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py), который подтверждает:
    - сборку baseline и context-модели;
    - прохождение `forward`;
    - shape-контракт `P5_backbone`, `P5_context`, `P5_fused`;
    - число параметров baseline и modified модели.
  - Локально подтверждён shape-контракт на dummy input `640x640`:
    - `P5_backbone: (1, 256, 20, 20)`;
    - `P5_context: (1, 256, 20, 20)`;
    - `P5_fused: (1, 256, 20, 20)`.
  - Локально подтверждён старт smoke train через [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py) с загрузкой `yolo26n.pt`, построением модели, переносом pretrained-весов и запуском train/val dataloader.
- Inferred:
  - Текущий YAML [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml) привязан к `YOLO26n` scale (`n`) и требует отдельного обобщения для `s/m/l/x`.
  - Параметры обучения вида `freeze_swin_epochs` и `swin_lr_mult` требуют отдельной train-time логики и остаются за пределами итерации `007`.

## Decisions
1. Реализовать первую версию как `YOLO26n + SwinContextBlock(P5)`, где `YOLO26n` остаётся основным детектором, а трансформер работает только как модуль глобального контекста над `P5`.
2. Не использовать в этой итерации отдельную raw-image Swin branch и не вводить второй backbone.
3. Не использовать в этой итерации `timm.create_model(..., features_only=True)` для второй параллельной ветви; вместо этого использовать локальный `SwinContextBlock`, принимающий feature map `P5`.
4. Ограничить первую реализацию одной точкой fusion:
   - вход: `P5`;
   - context output: `S5_context`;
   - fusion: `Concat(P5, S5_context)` с последующей проекцией обратно к каналам `P5`.
5. На первой итерации использовать только `concat` fusion. Варианты `add` и `gated` перенести в следующую итерацию.
6. Реализовать новый путь отдельным YAML и отдельным entrypoint, не ломая baseline и уже существующие Swin-backbone scaffolds.
7. Сохранить baseline без изменений по поведению:
   - существующие модули `YOLO26n` не удалять;
   - baseline CLI не ломать.
8. Для первой итерации считать достаточными следующие проверки:
   - модель собирается;
   - `forward` на dummy input проходит;
   - короткий smoke train стартует;
   - shapes и количество параметров baseline/modified выводятся отдельной sanity-проверкой.
9. Параметры `freeze_swin_epochs`, `swin_lr_mult`, multi-level fusion (`P4+P5`) и полноценную Swin branch вынести в следующую итерацию.

## Spec updates
- Экспериментальный путь `007` реализован как context enhancement поверх штатного `YOLO26n`, а не как backbone replacement.
- Итоговый scaffold итерации:
  - baseline `YOLO26n backbone`;
  - новый `SwinContextBlock` над `P5`;
  - `concat` fusion только на `P5`;
  - неизменённый `YOLO26n neck/head`.
- Для `007` добавлены:
  - отдельный YAML `models/yolo26n_swin_context_p5.yaml`;
  - отдельный train entrypoint `scripts/train_swin_context.py`;
  - отдельный sanity-check `scripts/validate_swin_context.py`.
- Для первой итерации training pipeline не расширяется кастомной логикой optimizer groups или временной заморозки.
- Baseline и существующие Swin-backbone scaffolds сохраняются как отдельные пути для сравнения и истории.

## Open questions
- Нужно ли обобщать `models/yolo26n_swin_context_p5.yaml` на масштабы `s/m/l/x`, или для сравнения в дипломе достаточно зафиксировать путь только для `YOLO26n`?
- Нужен ли в следующей итерации отдельный `FusionBlock` как локальный модуль, или достаточно продолжить собирать fusion из стандартных YAML-блоков `Concat + Conv`?
- Стоит ли делать `P4+P5` через два независимых context-блока, или нужен единый multi-level context path?
- Требуется ли фиксировать экспортный контракт (`export/onnx`) после стабилизации training/val/predict-пути?

## Next iteration
`008_yolo26n_swin_context_branch_expansion`

Фокус следующей итерации:
- расширить context path до `P4+P5`;
- добавить `FusionBlock` с `concat/add/gated`;
- рассмотреть полноценную Swin branch от input или early feature map;
- добавить `freeze_swin_epochs`;
- добавить `swin_lr_mult`;
- определить более зрелый train-time контракт для optimizer parameter groups.
