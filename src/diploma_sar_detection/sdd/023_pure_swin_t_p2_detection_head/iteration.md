# 023_pure_swin_t_p2_detection_head

## Goal
Добавить отдельный pure Swin-T вариант с P2/stride-4 feature level для small-object detection при `imgsz=1024`.

Граница итерации:
- не менять существующий `models/yolo26_swin_t.yaml`;
- не менять patch size `4`, чтобы сохранить timm pretrained Swin-T compatibility;
- добавить отдельный YAML с `out_indices=[0, 1, 2, 3]`;
- Detect должен использовать 4 уровня: `P2/P3/P4/P5` со strides `[4, 8, 16, 32]`.

## Inputs
- Existing pure Swin-T wrapper: `custom_models/swin_t_backbone.py`.
- Existing pure Swin-T YAML: `models/yolo26_swin_t.yaml`.
- Validation guard: `scripts/validate_swin_backbone.py`.

## Decisions
1. Add `models/yolo26_swin_t_p2.yaml`.
2. Use Swin-T stage outputs:
   - stage 0: `P2/4`, `96` channels;
   - stage 1: `P3/8`, `192` channels;
   - stage 2: `P4/16`, `384` channels;
   - stage 3: `P5/32`, `768` channels.
3. Build 4-scale FPN/PAN with final Detect inputs `[96, 192, 384, 768]`.
4. Update validation guard to accept expected strides `[4, 8, 16, 32]` when 4 `Index` outputs are present.

## Spec updates
- Adds a small-object-oriented pure Swin-T backbone variant without changing the existing 3-scale pure Swin baseline.

## Open questions
- Whether P2 increases swimmer/buoy/life_saving_appliances recall enough to justify extra memory at `imgsz=1024`.
- Whether P2 head needs lower batch size than current pure Swin-T due to higher-resolution feature maps.
