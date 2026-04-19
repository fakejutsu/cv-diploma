# Custom model templates

This directory stores architecture templates for experiments beyond the baseline Ultralytics checkpoints.

Current template:

- `yolo26_swin_t.yaml` — starting point for a `YOLO26 + Swin-T` experiment.

Important:

- The baseline project is fully working with stock `yolo26*.pt` checkpoints.
- The Swin-T path is a scaffold for custom-backbone experimentation.
- Before long training runs, validate tensor shapes and neck/head wiring with a short smoke test.
