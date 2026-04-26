# 015_residual_adaptive_wavevit_fusion_p3_p4

## Goal
Заменить статическое convex mixing поведение `GatedWaveVitFusion(P3, P4)` отдельным residual/adaptive fusion вариантом:
- baseline CNN feature map остаётся identity path;
- WaveViT-style branch добавляет только малую обучаемую поправку;
- gate зависит от channel context и local spatial/detail signals.

Граница итерации:
- не менять существующий `GatedWaveVitFusion`;
- не менять baseline/Swin/static WaveViT YAML;
- не менять датасет, loss, optimizer schedule и augmentations;
- добавить отдельный A/B-кандидат `ResidualAdaptiveWaveVitFusion(P3, P4)`.

## Inputs
- [`sdd/014_yolo26n_wavevit_fusion_p3_p4/iteration.md`](../014_yolo26n_wavevit_fusion_p3_p4/iteration.md)
- [`custom_models/gated_wavevit_fusion.py`](../../custom_models/gated_wavevit_fusion.py)
- [`custom_models/wavevit_context_block.py`](../../custom_models/wavevit_context_block.py)
- [`models/yolo26n_gated_wavevit_p3_p4.yaml`](../../models/yolo26n_gated_wavevit_p3_p4.yaml)
- Training observation: static WaveViT gate stayed near its initialization and did not outperform `best_yolo26n.pt`.

## Findings
- Confirmed by run logs:
  - Static `GatedWaveVitFusion` technically trains after fixing `train_swin_context.py` weight transfer.
  - `alpha=0.982` and `alpha=0.881` stayed almost unchanged during training.
  - Static `30% WaveViT` did not beat baseline on overall or per-class validation metrics.
- Inferred:
  - Direct convex mixing shifts detector feature distribution and is too coarse for small-object SAR signals.
  - A residual correction is safer for fine-tuning from a strong YOLO checkpoint.
  - A gate should combine global channel context and spatial/detail context instead of using only static channel scalars.

## Decisions
1. Add `ResidualAdaptiveWaveVitFusion`.
2. Use formula:
   - `wave = WaveVitContextBlock(x)`;
   - `delta = delta_proj(cat(x, wave, wave - x))`;
   - `channel_gate = MLP(GAP(x), GAP(wave))`;
   - `spatial_gate = Conv(mean(x), mean(wave), mean(abs(wave - x)))`;
   - `gate = sigmoid(raw_alpha + channel_gate + spatial_gate)`;
   - `out = x + beta * gate * delta`.
3. Initialize `raw_alpha=-2.0` (`gate ~= 0.119`) and `beta=0.1`.
4. Initialize channel/spatial adaptive gate heads to zero so the initial gate is controlled by `raw_alpha`.
5. Initialize final `delta_proj` conv with small weights instead of exact zeros, so the residual branch starts near identity but still allows gate gradients.
6. Create `models/yolo26n_residual_adaptive_wavevit_p3_p4.yaml`.
7. Reuse `scripts/validate_wavevit_context.py` for both static and residual/adaptive WaveViT variants.

## Spec updates
- Adds a new context-enhancement variant:
  - `YOLO26n + ResidualAdaptiveWaveVitFusion(P3, P4)`.
- New variant does not replace backbone and does not change Detect head.
- `P3_out` and `P4_out` preserve source shapes.
- WaveViT influence is residual, not direct feature replacement.
- Validator reports prior alpha, effective gate statistics, `delta_abs_mean`, and `beta`.

## Open questions
- Will residual/adaptive WaveViT correction produce measurable gain over `best_yolo26n.pt`.
- Whether `init_alpha=-2.0` and `beta=0.1` are too conservative for fast adaptation.
- Whether later variants should freeze baseline layers for a short gate/delta warm-up.

## Next iteration
- Run pre-transfer validation and short fine-tune.
- Compare against:
  - `best_yolo26n.pt`;
  - static `GatedWaveVitFusion(P3, P4)` alpha2/alpha30;
  - Swin adaptive gated P4/P5 after the weight-transfer fix.
