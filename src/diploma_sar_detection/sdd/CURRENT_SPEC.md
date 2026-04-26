# CURRENT_SPEC

Updated: 2026-04-26

## Scope
Текущее состояние экспериментальных transformer/context интеграций в training pipeline YOLO26:
- legacy pure `Swin-T` backbone replacement;
- hybrid `CNN -> Swin-T` backbone replacement;
- `YOLO26n + SwinContextBlock(P5)` как context-enhancement путь без замены штатного CNN-backbone;
- `YOLO26n + light SwinContextBlock(P4)` как отдельный light context variant;
- `YOLO26n + GatedSwinFusion(P4, P5)` как channel-wise gated context variant.
- `YOLO26n + GatedSwinFusion(P4, P5)` с softer gate-start и composite pretrained Swin warm-start.
- `YOLO26n + GatedSwinFusion(P4, P5)` с conservative gate-start (`init_alpha=6.0`) для random-Swin runs от dataset-pretrained baseline.
- `YOLO26n + AdaptiveDetailGatedSwinFusion(P4, P5)` как detail-aware adaptive fusion variant без замены штатного backbone.
- `YOLO26n + GatedWaveVitFusion(P3, P4)` как WaveViT-style wavelet-attention context variant без замены штатного backbone.
- `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)` как residual/adaptive WaveViT-style correction variant без замены штатного backbone.
- `YOLO26m teacher -> lightweight Swin-based student` через `P5` feature distillation.

## System State

### Confirmed by repo
- В проекте есть отдельный entrypoint для Swin-based экспериментов: [`scripts/train_swin.py`](../scripts/train_swin.py).
- Для context-enhancement пути добавлен отдельный entrypoint: [`scripts/train_swin_context.py`](../scripts/train_swin_context.py).
- Перед созданием модели вызывается `register_backbone(variant)`, который monkey-patch'ит `ultralytics.nn.tasks.TorchVision` на нужный класс:
  - `swin_t -> SwinTBackbone`;
  - `cnn_swin_t -> HybridCnnSwinTBackbone`.
  Реализация: [`custom_models/register.py`](../custom_models/register.py).
- Для context-enhancement пути используется отдельная регистрация локальных YAML-модулей через `register_context_modules()`, которая экспортирует local context/fusion modules в `ultralytics.nn.tasks`: [`custom_models/register.py`](../custom_models/register.py).
- `SwinTBackbone` реализован как timm-обёртка (`features_only=True`) с выдачей multi-scale features и приведением к формату `NCHW`: [`custom_models/swin_t_backbone.py`](../custom_models/swin_t_backbone.py).
- `HybridCnnSwinTBackbone` реализует `YOLO-style stride-preserving stem (Conv/C3k2) -> timm Swin-T(features_only)` и выдаёт multi-scale features в `NCHW`: [`custom_models/hybrid_cnn_swin_t_backbone.py`](../custom_models/hybrid_cnn_swin_t_backbone.py).
- `SwinContextBlock` реализован как лёгкий Swin-style context enhancer поверх уже вычисленного YOLO feature map `P5`, использует `torchvision.models.swin_transformer.SwinTransformerBlock` и сохраняет `NCHW`-контракт входа/выхода: [`custom_models/swin_context_block.py`](../custom_models/swin_context_block.py).
- `GatedSwinFusion` реализован как channel-wise gated fusion между исходным CNN feature map и его Swin-enhanced контекстом, использует обучаемый `raw_alpha` shape `[1, C, 1, 1]`: [`custom_models/gated_swin_fusion.py`](../custom_models/gated_swin_fusion.py).
- `AdaptiveDetailGatedSwinFusion` реализован как вход-зависимый fusion между исходным CNN feature map и его Swin-enhanced контекстом, использует `raw_alpha` prior, channel gate, spatial gate и detail-aware bias: [`custom_models/adaptive_detail_gated_swin_fusion.py`](../custom_models/adaptive_detail_gated_swin_fusion.py).
- `WaveVitContextBlock` реализован как shape-preserving WaveViT-style context enhancer поверх YOLO feature map, использует фиксированный Haar DWT/IDWT и attention с wavelet key/value context: [`custom_models/wavevit_context_block.py`](../custom_models/wavevit_context_block.py).
- `GatedWaveVitFusion` реализован как channel-wise gated fusion между исходным CNN feature map и его WaveViT-style контекстом, использует обучаемый `raw_alpha` shape `[1, C, 1, 1]`: [`custom_models/gated_wavevit_fusion.py`](../custom_models/gated_wavevit_fusion.py).
- `ResidualAdaptiveWaveVitFusion` реализован как residual correction поверх исходного CNN feature map: `out = x + beta * gate * delta`, где `gate` зависит от channel context и spatial/detail signals: [`custom_models/residual_adaptive_wavevit_fusion.py`](../custom_models/residual_adaptive_wavevit_fusion.py).
- Оба backbone не выполняют принудительный внутренний resize входа; spatial размер задаётся внешним training pipeline (`imgsz`) и сохраняет корректный stride-контракт для `Detect`.
- В гибридном пути stem не выполняет downsample (stride=1), поэтому `Detect` сохраняет ожидаемые strides `[8,16,32]`.
- Поддерживаются десять YAML-шаблонов архитектуры:
  - [`models/yolo26_cnn_swin_t.yaml`](../models/yolo26_cnn_swin_t.yaml) — дефолтный гибридный путь;
  - [`models/yolo26_swin_t.yaml`](../models/yolo26_swin_t.yaml) — legacy pure Swin-T путь;
  - [`models/yolo26n_swin_context_p5.yaml`](../models/yolo26n_swin_context_p5.yaml) — `YOLO26n + SwinContextBlock(P5)` без замены backbone;
  - [`models/yolo26n_swin_context_p4_light.yaml`](../models/yolo26n_swin_context_p4_light.yaml) — `YOLO26n + light SwinContextBlock(P4)` без замены backbone;
  - [`models/yolo26n_gated_swin_p4_p5.yaml`](../models/yolo26n_gated_swin_p4_p5.yaml) — `YOLO26n + GatedSwinFusion(P4, P5)` без замены backbone;
  - [`models/yolo26n_gated_swin_p4_p5_pretrained.yaml`](../models/yolo26n_gated_swin_p4_p5_pretrained.yaml) — `YOLO26n + GatedSwinFusion(P4, P5)` с `init_alpha=2.0` и optional pretrained Swin subweight warm-start;
  - [`models/yolo26n_gated_swin_p4_p5_alpha6.yaml`](../models/yolo26n_gated_swin_p4_p5_alpha6.yaml) — `YOLO26n + GatedSwinFusion(P4, P5)` с `init_alpha=6.0` для random-Swin запуска от dataset-pretrained baseline.
  - [`models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml`](../models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml) — `YOLO26n + AdaptiveDetailGatedSwinFusion(P4, P5)` с input-adaptive channel/spatial/detail gate.
  - [`models/yolo26n_gated_wavevit_p3_p4.yaml`](../models/yolo26n_gated_wavevit_p3_p4.yaml) — `YOLO26n + GatedWaveVitFusion(P3, P4)` с wavelet-attention context на small-object уровнях.
  - [`models/yolo26n_residual_adaptive_wavevit_p3_p4.yaml`](../models/yolo26n_residual_adaptive_wavevit_p3_p4.yaml) — `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)` с малой adaptive residual WaveViT-поправкой.
- Для distillation-пути teacher задаётся отдельным checkpoint `YOLO26m`, а student может быть:
  - существующий [`models/yolo26n_swin_context_p5.yaml`](../models/yolo26n_swin_context_p5.yaml);
  - pure Swin student на [`models/yolo26_swin_t.yaml`](../models/yolo26_swin_t.yaml) при явной регистрации `--student-backbone-variant swin_t`.
- В `models/yolo26n_swin_context_p5.yaml`:
  - штатный `YOLO26n` backbone сохранён;
  - после `P5` добавлен `SwinContextBlock`;
  - выполняется `Concat(P5, context)` и `Conv1x1`-проекция обратно в fused `P5`;
  - затем используется штатный `YOLO26n` neck/head.
- В `models/yolo26n_swin_context_p4_light.yaml`:
  - штатный `YOLO26n` backbone сохранён;
  - в начале head над `P4` из слоя `6` добавлен light `SwinContextBlock`;
  - выполняется `Concat(P4, context)` и `Conv1x1`-проекция обратно в fused `P4`;
  - далее в neck используется `fused P4`, а `P5` остаётся штатным.
- В `models/yolo26n_gated_swin_p4_p5.yaml`:
  - штатный `YOLO26n` backbone сохранён;
  - над `P4` и `P5` добавлены отдельные `GatedSwinFusion`;
  - для `P4` используется gate shape `[1, 128, 1, 1]`;
  - для `P5` используется gate shape `[1, 256, 1, 1]`;
  - в neck используются gated `P4/P5`, а `Detect` head остаётся штатным по смыслу.
- В `models/yolo26n_gated_swin_p4_p5_pretrained.yaml`:
  - wiring совпадает с `models/yolo26n_gated_swin_p4_p5.yaml`;
  - `init_alpha=2.0` задаётся только в YAML, без изменения default внутри `GatedSwinFusion`;
  - `scripts/train_swin_context.py` и `scripts/validate_swin_context.py` поддерживают optional `--swin-p4-weights/--swin-p5-weights`;
  - при их передаче загружаются только inner `swin`-параметры соответствующих `GatedSwinFusion`, без перезаписи `raw_alpha`.
- В `models/yolo26n_gated_swin_p4_p5_alpha6.yaml`:
  - wiring совпадает с `models/yolo26n_gated_swin_p4_p5.yaml`;
  - `init_alpha=6.0` задаётся только в YAML;
  - сценарий предназначен для запуска от dataset-pretrained baseline checkpoint без pretrained Swin subweights.
- В `models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml`:
  - wiring совпадает с `models/yolo26n_gated_swin_p4_p5.yaml`;
  - `P4` и `P5` проходят через отдельные `AdaptiveDetailGatedSwinFusion`;
  - gate остаётся совместимым с `raw_alpha` prior и inner `SwinContextBlock` warm-start;
  - `alpha` зависит от текущих CNN/Swin признаков и local detail energy, а не только от статического параметра.
- В `models/yolo26n_gated_wavevit_p3_p4.yaml`:
  - штатный `YOLO26n` backbone сохранён;
  - `P3` из слоя `4` и `P4` из слоя `6` проходят через отдельные `GatedWaveVitFusion`;
  - для `P3` используется gate shape `[1, 128, 1, 1]`;
  - для `P4` используется gate shape `[1, 128, 1, 1]`;
  - `P5` остаётся штатным;
  - в neck используются gated `P3/P4`, а `Detect` head остаётся штатным по смыслу.
- В `models/yolo26n_residual_adaptive_wavevit_p3_p4.yaml`:
  - wiring совпадает с `models/yolo26n_gated_wavevit_p3_p4.yaml`;
  - `P3` и `P4` проходят через отдельные `ResidualAdaptiveWaveVitFusion`;
  - baseline CNN feature остаётся identity path;
  - WaveViT branch добавляет residual delta через channel/spatial adaptive gate;
  - `P5` остаётся штатным;
  - `Detect` head остаётся штатным по смыслу.
- Для distillation-пути:
  - teacher `YOLO26m` используется только во время train и не участвует в inference student;
  - student warm-start задаётся отдельным checkpoint через `--student-weights`;
  - дополнительный loss считается только на `P5`;
  - конкретный student `P5 layer/channels` задаются CLI-параметрами и не захардкожены только под один student;
  - channel alignment выполняется student-side `1x1` adapter.
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
  - shape-контракт `feature_backbone -> feature_context -> feature_fused` для `P5` и `P4-light` вариантов;
  - shape-контракт `P4_cnn -> P4_out` и `P5_cnn -> P5_out` для `gated_p4_p5`;
  - `alpha_mean` для `P4/P5` gated modules;
  - `alpha_mean/min/max`, `detail_mean/max` и `detail_bias_mean/max` для adaptive gated modules;
  - optional matched counts для pretrained `P4/P5` Swin subweights;
  - различие по числу параметров baseline vs modified.
- Добавлен отдельный sanity-check WaveViT-context setup:
  [`scripts/validate_wavevit_context.py`](../scripts/validate_wavevit_context.py), который подтверждает для dummy input:
  - сборку `YOLO26n + GatedWaveVitFusion(P3, P4)` и `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)`;
  - shape-контракт `P3_cnn -> P3_out` и `P4_cnn -> P4_out`;
  - `alpha_mean/min/max` для `P3/P4` gated modules;
  - для residual/adaptive variant дополнительно `gate_mean/min/max`, `delta_abs_mean` и `beta`;
  - `Detect` strides `[8, 16, 32]` и входные каналы `[64, 128, 256]`.
- Добавлен отдельный sanity-check distillation setup:
  [`scripts/validate_distill_setup.py`](../scripts/validate_distill_setup.py), который подтверждает сборку teacher/student и `P5` shape alignment.

### Inferred
- Все Swin-based пути остаются экспериментальными scaffold-режимами, а не полностью верифицированной заменой базового пайплайна для всех сценариев.
- Для стабильного обучения критична согласованность `out_indices`/каналов между Swin-based backbone, neck и `Detect` слоями в YAML.
- `models/yolo26n_swin_context_p5.yaml` на текущем шаге привязан к `YOLO26n` scale (`n`) и не является универсальным шаблоном для `s/m/l/x`.
- `models/yolo26n_swin_context_p4_light.yaml` так же привязан к `YOLO26n` scale (`n`) и не является универсальным шаблоном для `s/m/l/x`.
- `models/yolo26n_gated_swin_p4_p5.yaml` так же привязан к `YOLO26n` scale (`n`) и не является универсальным шаблоном для `s/m/l/x`.
- `models/yolo26n_gated_swin_p4_p5_pretrained.yaml` так же привязан к `YOLO26n` scale (`n`) и не является универсальным шаблоном для `s/m/l/x`.
- `models/yolo26n_gated_wavevit_p3_p4.yaml` так же привязан к текущему `YOLO26n` layer/channel layout и не является универсальным шаблоном для `s/m/l/x`.
- `models/yolo26n_residual_adaptive_wavevit_p3_p4.yaml` так же привязан к текущему `YOLO26n` layer/channel layout и не является универсальным шаблоном для `s/m/l/x`.

### Not established in repo
- Нет зафиксированного сравнения метрик baseline vs pure `Swin-T` vs hybrid `CNN+Swin-T` vs `YOLO26n + SwinContextBlock(P5)` vs `YOLO26n + light SwinContextBlock(P4)` vs `YOLO26n + GatedSwinFusion(P4, P5)` vs `YOLO26n + GatedWaveVitFusion(P3, P4)` vs `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)` на одном и том же датасете.
- Нет CI/авто-проверки, которая бы гарантировала, что подмена `TorchVision` не ломается при обновлениях `ultralytics`.
- Для context-path пока не зафиксирована отдельная стратегия экспорт/ONNX-совместимости.

## Contracts
- CLI-контракт `scripts/train_swin.py` должен оставаться совместимым с текущими флагами.
- CLI-контракт `scripts/train_swin_context.py` должен оставаться совместимым с текущими флагами.
- `scripts/train_swin_context.py` дополнительно поддерживает optional проброс `lr0` и `mosaic` в `ultralytics` train kwargs.
- Добавлен отдельный entrypoint [`scripts/train_distill.py`](../scripts/train_distill.py) для teacher/student distillation.
- Формат датасета и `data/dataset.yaml` не меняется из-за Swin-based интеграции.
- Для Swin-based путей обязателен контракт выходов в neck/head: `P3/P4/P5` каналы `192/384/768`, `Detect` strides `[8,16,32]`.
- Для `YOLO26n + SwinContextBlock(P5)` обязателен контракт:
  - `P5_backbone`, `P5_context` и `P5_fused` имеют одинаковые spatial размеры;
  - `P5_fused` возвращается к каналам штатного `YOLO26n P5`;
  - baseline `YOLO26n` backbone/neck/head не заменяются.
- Для `YOLO26n + light SwinContextBlock(P4)` обязателен контракт:
  - `P4_backbone`, `P4_context` и `P4_fused` имеют одинаковые spatial размеры;
  - `P4_fused` возвращается к каналам штатного `YOLO26n P4`;
  - в neck вместо исходного `P4` используется `P4_fused`;
  - baseline `YOLO26n` backbone/neck/head не заменяются.
- Для `YOLO26n + GatedSwinFusion(P4, P5)` обязателен контракт:
  - `P4_out` и `P5_out` сохраняют shape исходных `P4/P5`;
  - `alpha4` shape `[1, 128, 1, 1]`, `alpha5` shape `[1, 256, 1, 1]`;
  - gate является channel-wise и обучаемым;
  - `Detect` head остаётся штатным по смыслу.
- Для `YOLO26n + AdaptiveDetailGatedSwinFusion(P4, P5)` обязателен контракт:
  - `P4_out` и `P5_out` сохраняют shape исходных `P4/P5`;
  - `raw_alpha` остаётся базовым prior shape `[1, C, 1, 1]`;
  - `alpha` зависит от входных признаков через channel gate, spatial gate и detail-aware bias;
  - inner `swin` поддерживает тот же `load_swin_weights(...)` контракт, что и `GatedSwinFusion`;
  - `Detect` head остаётся штатным по смыслу.
- Для `YOLO26n + GatedWaveVitFusion(P3, P4)` обязателен контракт:
  - `P3_out` и `P4_out` сохраняют shape исходных `P3/P4`;
  - `alpha3` shape `[1, 128, 1, 1]`, `alpha4` shape `[1, 128, 1, 1]`;
  - gate является channel-wise и обучаемым;
  - `P5` остаётся штатным;
  - `Detect` head остаётся штатным по смыслу со strides `[8, 16, 32]`.
- Для `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)` обязателен контракт:
  - `P3_out` и `P4_out` сохраняют shape исходных `P3/P4`;
  - residual branch имеет форму `out = x + beta * gate * delta`;
  - `gate` зависит от global channel context и local spatial/detail context;
  - `P5` остаётся штатным;
  - `Detect` head остаётся штатным по смыслу со strides `[8, 16, 32]`.
- Артефакты обучения сохраняются в стандартной структуре `runs/<name>/weights/{best,last}.pt`.
