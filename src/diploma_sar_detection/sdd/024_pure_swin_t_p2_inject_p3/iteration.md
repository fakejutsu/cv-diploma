# 024_pure_swin_t_p2_inject_p3

## Goal
Добавить pure Swin-T вариант, который использует P2/stride-4 detail features без изменения YOLO Detect head.

Граница итерации:
- не менять существующий `models/yolo26_swin_t.yaml`;
- не удалять 4-scale эксперимент `models/yolo26_swin_t_p2.yaml`;
- сохранить Detect как 3-scale `Detect(P3, P4, P5)`;
- использовать Swin stage0/P2 только как detail injection в P3.

## Inputs
- `models/yolo26_swin_t.yaml`
- `models/yolo26_swin_t_p2.yaml`
- `scripts/validate_swin_backbone.py`

## Decisions
1. Add `models/yolo26_swin_t_p2_inject_p3.yaml`.
2. Use Swin outputs `[0, 1, 2, 3]`:
   - `P2/4`: 96 channels;
   - `P3/8`: 192 channels;
   - `P4/16`: 384 channels;
   - `P5/32`: 768 channels.
3. Downsample P2 to stride 8 and fuse it with P3:
   - `P2_down = Conv(P2, 96, k=3, s=2)`;
   - `P3_enhanced = C2f(concat(P2_down, P3), 192)`.
4. Keep the old 3-scale FPN/PAN style after `P3_enhanced`.
5. Keep Detect inputs `[P3, P4, P5]` and strides `[8, 16, 32]`.
6. Update validation guard so extra backbone/index features are allowed when Detect uses only the last 3 pyramid levels.

## Spec updates
- Adds a small-object detail injection path without adding a fourth Detect branch.
- This variant is less disruptive than `023_pure_swin_t_p2_detection_head`.

## Open questions
- Whether P2 injection improves small-object recall without increasing false positives.
- Whether residual/gated injection is needed if direct C2f fusion is too disruptive.
