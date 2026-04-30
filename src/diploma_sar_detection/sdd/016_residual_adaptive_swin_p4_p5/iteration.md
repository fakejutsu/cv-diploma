# 016_residual_adaptive_swin_p4_p5

## Goal
Добавить отдельный вариант `YOLO26n + ResidualAdaptiveSwinFusion(P4, P5)`, чтобы заменить прямое convex mixing поведение adaptive Swin на безопасную residual correction:
- baseline CNN feature остаётся identity path;
- SwinContextBlock добавляет малую адаптивную поправку;
- gate зависит от global channel context и local spatial/detail signals.

Граница итерации:
- не менять существующие gated/adaptive Swin YAML;
- не менять WaveViT варианты;
- не менять датасет, loss, optimizer schedule и augmentations;
- добавить отдельный A/B-кандидат.

## Inputs
- [`custom_models/adaptive_detail_gated_swin_fusion.py`](../../custom_models/adaptive_detail_gated_swin_fusion.py)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- [`models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml`](../../models/yolo26n_adaptive_detail_gated_swin_p4_p5.yaml)
- Observation: adaptive Swin modules learn their Swin/gate weights but convex mixing did not beat baseline.

## Findings
- Confirmed by run logs:
  - Adaptive Swin pre-transfer validation can match or slightly exceed `best_yolo26n.pt`.
  - `raw_alpha` and `detail_strength` barely move, while `swin.*`, `channel_gate.*`, and `spatial_gate.*` change substantially.
- Inferred:
  - Swin branch is trainable, unlike WaveViT branch in tested setups.
  - Direct convex mixing may still perturb feature distribution enough to prevent `mAP50-95` improvement.
  - Residual correction can preserve baseline feature geometry while allowing Swin to add targeted adjustments.

## Decisions
1. Add `ResidualAdaptiveSwinFusion`.
2. Use formula:
   - `swin = SwinContextBlock(x)`;
   - `delta = delta_proj(cat(x, swin, swin - x))`;
   - `channel_gate = MLP(GAP(x), GAP(swin))`;
   - `spatial_gate = Conv(mean(x), mean(swin), detail_map(x))`;
   - `gate = sigmoid(raw_alpha + channel_gate + spatial_gate)`;
   - `out = x + beta * gate * delta`.
3. Initialize `raw_alpha=-1.0` and `beta=0.5`, giving stronger potential residual influence than the first WaveViT residual variant.
4. Initialize adaptive gate heads to zero and final `delta_proj` conv with small weights.
5. Create `models/yolo26n_residual_adaptive_swin_p4_p5.yaml`.
6. Extend `validate_swin_context.py` with `residual_adaptive_p4_p5`.

## Spec updates
- Adds `YOLO26n + ResidualAdaptiveSwinFusion(P4, P5)`.
- New variant preserves backbone and Detect head.
- `P4_out` and `P5_out` preserve source shapes.
- Swin influence is residual, not direct feature replacement.
- Validator reports prior alpha, effective gate statistics, detail statistics, `delta_abs_mean`, and `beta`.

## Open questions
- Whether residual Swin correction can beat `best_yolo26n.pt` where convex adaptive Swin did not.
- Whether P4-only residual Swin would outperform P4/P5 if P5 context hurts localization.

## Next iteration
- Run pre-transfer validation.
- Train with `--freeze-layers 0-10`.
- Compare against `best_yolo26n.pt`, adaptive detail gated Swin, and WaveViT residual variants.
