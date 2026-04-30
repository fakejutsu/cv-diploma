# 022_staged_context_freeze_unfreeze

## Goal
Добавить staged training режим для context/fusion экспериментов:
- первые `N` эпох обучать только context/fusion blocks;
- затем разморозить остальные слои и продолжить end-to-end training.

## Inputs
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- Existing context modules:
  - `GatedSwinFusion`
  - `AdaptiveDetailGatedSwinFusion`
  - `GatedWaveVitFusion`
  - `ResidualAdaptiveSwinFusion`
  - `ResidualAdaptiveWaveVitFusion`
  - `ResidualSwinC2PSA`

## Decisions
1. Add CLI flags:
   - `--context-only-epochs N`
   - `--context-layer-indices ...`
2. If `--context-only-epochs > 0`, freeze all layers except context layers before `model.train()`.
3. Register an Ultralytics callback on `on_train_epoch_start`; when current epoch reaches `N`, set all model parameters back to trainable.
4. Keep existing `--freeze-layers` behavior for full-run freezing; staged context freeze is separate and should not require manually listing all non-context layers.

## Spec updates
- `train_swin_context.py` can now run a two-phase context warm-up.
- Default context layer indices are inferred from known context/fusion module types.

## Open questions
- Whether unfreezing after optimizer creation updates all parameter groups in every Ultralytics version; this is expected to work because parameters remain in the model/optimizer, but should be checked in smoke logs.
