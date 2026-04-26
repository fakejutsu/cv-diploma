# 017_teacher_guided_residual_swin_distillation

## Goal
Добавить teacher-guided distillation для residual/adaptive Swin student, чтобы transformer-вставки получали прямой dense supervision signal, а не только слабый градиент от detection loss.

Граница итерации:
- не менять формат датасета;
- не менять baseline train entrypoint;
- не удалять существующий single-layer P5 distillation scaffold;
- добавить multi-feature distillation как отдельный режим `scripts/train_distill.py --distill-mode multi`.

## Inputs
- [`scripts/train_distill.py`](../../scripts/train_distill.py)
- [`custom_models/distill_swin_p5_model.py`](../../custom_models/distill_swin_p5_model.py)
- [`custom_models/residual_adaptive_swin_fusion.py`](../../custom_models/residual_adaptive_swin_fusion.py)
- [`models/yolo26n_residual_adaptive_swin_p4_p5.yaml`](../../models/yolo26n_residual_adaptive_swin_p4_p5.yaml)

## Findings
- Confirmed by runs:
  - Static/insert-only transformer branches often do not move enough under detection loss alone.
  - Adaptive Swin branch updates its internal Swin/gate weights, but convex/residual variants did not yet beat baseline by mAP50-95.
- Inferred:
  - Transformer residual branch needs an explicit teacher feature target to learn useful corrections.
  - P4/P5 feature distillation should provide denser gradients than detector loss alone.

## Decisions
1. Add `DistillMultiFeatureDetectionModel`.
2. Support multiple student/teacher feature pairs with separate `1x1` adapters.
3. Extend `scripts/train_distill.py` with:
   - `--distill-mode single|multi`;
   - `--student-distill-layers`;
   - `--student-distill-channel-list`;
   - `--teacher-distill-layers`;
   - `--teacher-distill-channel-list`;
   - `--freeze-layers`.
4. Keep existing single-layer P5 distillation mode backward-compatible.
5. Use feature loss mean across all configured feature pairs.

## Spec updates
- `scripts/train_distill.py` supports teacher-guided multi-feature distillation.
- Intended first student:
  - `models/yolo26n_residual_adaptive_swin_p4_p5.yaml`.
- Intended first feature pairs:
  - student `P4/P5`: layers `11,12`, channels `128,256`;
  - teacher `P4/P5`: layers `6,10`, channels configured by the teacher checkpoint, initially `512,512` for YOLO26m-like teacher.
- Distillation loss is added to detection loss:
  - `L = L_det + distill_weight * mean(feature_losses) * batch_size`.

## Open questions
- Whether YOLO26m teacher P4 channels are exactly `512` for the current checkpoint.
- Whether best teacher should be YOLO26m or the best available YOLO26n/m run.
- Whether feature-only warm-up should freeze neck/head (`0-10 13-25`) before freeze-backbone training.

## Next iteration
- Validate multi-feature setup with a dummy batch.
- Run short distillation warm-up:
  - freeze `0-10 13-25`;
  - train only residual Swin inserts and distill adapters.
- Then run freeze-backbone distillation:
  - freeze `0-10`;
  - train residual Swin inserts plus neck/head.
