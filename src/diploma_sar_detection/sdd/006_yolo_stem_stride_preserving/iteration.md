# Iteration 006 — YOLO Stem Stride-Preserving

## Goal
Перевести гибридный `CNN -> Swin-T` backbone на YOLO26n-style stem (на базе `Conv`/`C3k2` блоков Ultralytics) в stride-preserving режиме, чтобы сохранить текущий `Detect` stride-контракт `[8,16,32]`.

## Inputs
- [`sdd/CURRENT_SPEC.md`](../CURRENT_SPEC.md)
- [`sdd/005_cnn_swin_hybrid_backbone/iteration.md`](../005_cnn_swin_hybrid_backbone/iteration.md)
- [`custom_models/hybrid_cnn_swin_t_backbone.py`](../../custom_models/hybrid_cnn_swin_t_backbone.py)
- [`models/yolo26_cnn_swin_t.yaml`](../../models/yolo26_cnn_swin_t.yaml)
- [`scripts/validate_swin_backbone.py`](../../scripts/validate_swin_backbone.py)

## Findings
- Confirmed by repo (после реализации):
  - `HybridCnnSwinTBackbone` переведён на YOLO-style stem:
    - `Conv(3 -> stem_mid, k=3, s=1)`,
    - `Conv(stem_mid -> stem_out, k=3, s=1)`,
    - `C3k2(stem_out -> stem_out, n=stem_depth, e=stem_expand_ratio)`.
  - Вход в Swin теперь подаётся без возврата к 3 каналам, через `timm.create_model(..., in_chans=stem_out)`.
  - Гибридный YAML обновлён на stem-каналы `[64,128]`.
  - Guard-проверка для `cnn_swin_t` и `swin_t` подтверждает сохранение контрактов:
    - channels `192/384/768`,
    - `Detect` strides `[8,16,32]`.

## Decisions
1. Пересобрать `HybridCnnSwinTBackbone` так, чтобы stem состоял из YOLO-блоков:
   - `Conv(3 -> stem_mid, k=3, s=1)`,
   - `Conv(stem_mid -> stem_out, k=3, s=1)`,
   - `C3k2(stem_out -> stem_out, n=stem_depth, e=stem_expand_ratio)`.
2. Убрать обязательный возврат к 3 каналам перед Swin и использовать `timm.create_model(..., in_chans=stem_out)`.
3. Сохранить stride-preserving контракт (без внутренних resize/downsample в stem), чтобы head оставался совместим с текущим YAML.
4. Обновить дефолт параметров гибридного YAML под YOLO-style stem (`[64,128]`) и сохранить остальную архитектуру без изменений.

## Spec updates
- Зафиксировано, что гибридный путь использует YOLO26n-style stride-preserving stem на `Conv/C3k2`.
- Контракт выходов в neck/head (`192/384/768`, `[8,16,32]`) сохранён неизменным.

## Open questions
- Нужна ли отдельная конфигурация stem для “lite” режима (например, `[32,64]`) как быстрый A/B against `[64,128]`?
- Стоит ли фиксировать `stem_depth=2` как дефолт или вынести в экспериментальный CLI-параметр позже?

## Next iteration
Провести A/B smoke между `YOLO-style stride-preserving stem ([64,128])` и lite-конфигом (`[32,64]`) с одинаковым протоколом (seed/imgsz/batch/fraction) и обновить acceptance-критерии.
