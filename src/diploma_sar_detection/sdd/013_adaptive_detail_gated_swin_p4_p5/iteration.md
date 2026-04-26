# 013_adaptive_detail_gated_swin_p4_p5

## Goal
Добавить отдельный вариант `YOLO26n + AdaptiveDetailGatedSwinFusion(P4, P5)`, где fusion gate остаётся совместимым с текущим gated-пайплайном, но становится вход-зависимым:
- локально детальные области сильнее сохраняют CNN-признаки;
- менее детальные и более контекстные области могут сильнее использовать `SwinContextBlock`;
- `raw_alpha` остаётся базовым prior, а не единственным gate-сигналом.

Граница итерации:
- не менять старый `GatedSwinFusion`;
- не менять baseline/context/gated YAML;
- не добавлять distillation, новый head, новый optimizer schedule или P5-only redesign;
- реализовать только более умный P4/P5 fusion module поверх существующего pipeline.

## Inputs
- [`sdd/009_yolo26n_gated_swin_p4_p5/iteration.md`](../009_yolo26n_gated_swin_p4_p5/iteration.md)
- [`sdd/010_yolo26n_gated_swin_p4_p5_pretrained_softgate/iteration.md`](../010_yolo26n_gated_swin_p4_p5_pretrained_softgate/iteration.md)
- [`custom_models/gated_swin_fusion.py`](../../custom_models/gated_swin_fusion.py)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)
- [`models/yolo26n_gated_swin_p4_p5.yaml`](../../models/yolo26n_gated_swin_p4_p5.yaml)

## Findings
- Confirmed by repo:
  - `GatedSwinFusion` использует только обучаемый channel-wise `raw_alpha`, не зависящий от текущего входа.
  - Старый gated-вариант уже используется как отдельный A/B-кандидат и должен остаться воспроизводимым.
  - `train_swin_context.py` уже умеет remap baseline `YOLO26n` весов на gated P4/P5 layout через shift `+2`.
  - Composite Swin warm-start уже загружает только inner `swin`-веса и не должен перезаписывать gate-параметры.
- Inferred:
  - Почти равномерный learned `alpha` по каналам делает старый gate слишком грубым для различия мелких деталей и контекста.
  - Для SAR small-object классов полезно явно смещать gate в пользу CNN на областях с высокой local detail energy.

## Decisions
1. Оставить `GatedSwinFusion` без изменений.
2. Добавить новый модуль `AdaptiveDetailGatedSwinFusion`.
3. Сохранить внутри нового модуля `SwinContextBlock` и метод `load_swin_weights(...)`, чтобы composite Swin warm-start остался совместимым.
4. Использовать формулу:
   - `swin = self.swin(x)`;
   - `channel_gate = MLP(GAP(x), GAP(swin))`;
   - `spatial_gate = Conv(mean(x), mean(swin), detail_bias)`;
   - `detail_bias = abs(x - avg_pool_3x3(x))` с spatial-нормализацией;
   - `alpha = sigmoid(raw_alpha + channel_gate + spatial_gate + detail_strength * detail_bias)`;
   - `out = alpha * x + (1 - alpha) * swin`.
5. Инициализировать последние слои channel/spatial gate нулями, чтобы старт был контролируемым и не ломал warm-start поведение.
6. Инициализировать `detail_strength` осторожно, чтобы первая версия не превращала gate в нестабильный attention-блок.
7. Создать отдельный YAML `models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml`, wiring которого совпадает со старым gated P4/P5 layout.
8. Добавить отдельную warm-start strategy `adaptive_gated_shift2`, даже если layer remap совпадает с `gated_shift2`, чтобы логи явно различали old gated и adaptive gated.
9. Расширить validator отдельным вариантом `adaptive_gated_p4_p5` с печатью:
   - P4/P5 shape contract;
   - `alpha_mean/min/max`;
   - `detail_mean/max`;
   - `detail_bias_mean/max`.

## Spec updates
- В систему добавляется новый context-enhancement variant:
  - `YOLO26n + AdaptiveDetailGatedSwinFusion(P4, P5)`.
- Новый variant не заменяет backbone и не меняет Detect head.
- Новый gate является вход-зависимым и содержит:
  - базовый `raw_alpha` prior;
  - channel-wise adaptive gate;
  - spatial adaptive gate;
  - detail-aware bias в пользу CNN на локально детальных областях.
- Старый `YOLO26n + GatedSwinFusion(P4, P5)` остаётся отдельным воспроизводимым baseline для A/B.

## Open questions
- Даст ли detail-aware bias различимую динамику `alpha` между `P4` и `P5` на реальных SAR данных.
- Нужно ли в следующей итерации вводить разные коэффициенты detail/context для `P4` и `P5`.
- Не окажется ли adaptive gate слишком закрытым при `init_alpha=4.0`; если да, нужен отдельный soft-gate YAML.

## Next iteration
- Провести smoke/full train и сравнить не только mAP, но и gate-поведение:
  - `alpha_mean` P4 vs P5;
  - ширину `alpha_min/max`;
  - small-object классы `swimmer`, `life_saving_appliances`, `buoy`.
