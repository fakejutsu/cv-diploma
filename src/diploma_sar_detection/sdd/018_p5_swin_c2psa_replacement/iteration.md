# 018_p5_swin_c2psa_replacement

## Goal
Заменить штатный `C2PSA(P5)` module на shape-preserving residual Swin context block.

Граница итерации:
- не менять P3/P4 пути;
- не добавлять side-branch fusion;
- не менять Detect head;
- заменить именно context/attention-like module `layer 10`.

## Inputs
- [`models/yolo26n_gated_swin_p4_p5.yaml`](../../models/yolo26n_gated_swin_p4_p5.yaml)
- [`custom_models/swin_context_block.py`](../../custom_models/swin_context_block.py)
- Prior results: side-branch transformer fusion/distillation did not beat `best_yolo26n.pt`.

## Findings
- Confirmed by repo:
  - `layer 10` in YOLO26n-style local YAML is `C2PSA(P5)`.
  - Downstream neck/head expects P5 channels `256`, Detect input channels `[64, 128, 256]`, and strides `[8, 16, 32]`.
- Inferred:
  - Replacing an existing context block is cleaner than adding random concat/projection branches.
  - A residual near-identity start is required to preserve warm-start behavior.

## Decisions
1. Add `ResidualSwinC2PSA`.
2. Use formula:
   - `swin = SwinContextBlock(x)`;
   - `delta = delta_proj(cat(x, swin - x))`;
   - `out = x + beta * sigmoid(raw_alpha) * delta`.
3. Initialize `raw_alpha=-1.0`, `beta=0.1`, and final `delta_proj` conv with `std=1e-3`.
4. Create `models/yolo26n_p5_swin_c2psa_replacement.yaml` with unchanged layer indices after `10`.
5. Add `scripts/validate_swin_c2psa_replacement.py`.
6. Add explicit preferred warm-start strategy `p5_swin_c2psa_exact`.
7. Add `scripts/train_c2psa_replacement_imitation.py` to pretrain only layer `10` by imitating the original YOLO26n C2PSA output.

## Spec updates
- Adds `YOLO26n + ResidualSwinC2PSA(P5)` as a direct context-module replacement.
- Replacement preserves `P5` shape and channels.
- Detect head remains unchanged.
- Adds an imitation warm-up script for replacing C2PSA without starting from a randomly behaving main-path module.

## Open questions
- Whether replacing `C2PSA` loses useful pretrained behavior from layer 10 despite near-identity residual init.
- Whether `beta=0.1` is too conservative.

## Next iteration
- Run pre-transfer validation.
- Train with `--freeze-layers 0-9`.
- Compare against `best_yolo26n.pt`.
