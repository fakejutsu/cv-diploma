# Custom model templates

This directory stores architecture templates for experiments beyond the baseline Ultralytics checkpoints.

Current templates:

- `yolo26_cnn_swin_t.yaml` — hybrid `YOLO-style stride-preserving stem (Conv/C3k2) -> Swin-T` with YOLO26 neck/head wiring (current default scaffold).
- `yolo26_swin_t.yaml` — legacy pure `Swin-T` backbone scaffold kept for A/B comparisons.
- `yolo26n_swin_context_p5.yaml` — `YOLO26n` with an additional `SwinContextBlock` over the backbone `P5`, followed by `Concat + Conv1x1` fusion back into the stock neck/head.
- `yolo26n_swin_context_p4_light.yaml` — `YOLO26n` with a light `SwinContextBlock` over backbone `P4`, followed by `Concat + Conv1x1` fusion back into the stock neck/head.

Important:

- The baseline project is fully working with stock `yolo26*.pt` checkpoints.
- Swin-based paths are scaffolds for custom-backbone experimentation.
- `yolo26n_swin_context_p5.yaml` is currently aligned to the `YOLO26n` scale (`n`) and is not yet a generic template for `s/m/l/x`.
- `yolo26n_swin_context_p4_light.yaml` is also aligned to the `YOLO26n` scale (`n`) and is not yet a generic template for `s/m/l/x`.
- Before long training runs, validate tensor shapes and neck/head wiring with a short smoke test.
