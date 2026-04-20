# Custom model templates

This directory stores architecture templates for experiments beyond the baseline Ultralytics checkpoints.

Current templates:

- `yolo26_cnn_swin_t.yaml` — hybrid `CNN stem -> Swin-T` backbone with YOLO26 neck/head wiring (current default scaffold).
- `yolo26_swin_t.yaml` — legacy pure `Swin-T` backbone scaffold kept for A/B comparisons.

Important:

- The baseline project is fully working with stock `yolo26*.pt` checkpoints.
- Swin-based paths are scaffolds for custom-backbone experimentation.
- Before long training runs, validate tensor shapes and neck/head wiring with a short smoke test.
