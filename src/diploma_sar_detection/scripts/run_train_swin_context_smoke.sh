#!/usr/bin/env bash
set -euo pipefail

python3 scripts/train_swin_context.py \
  --data data/dataset.yaml \
  --model models/yolo26n_swin_context_p5.yaml \
  --weights yolo26n.pt \
  --project runs_cmp \
  --name yolo26n_swin_context_p5_smoke \
  --epochs 10 \
  --imgsz 640 \
  --batch 24 \
  --device 0 \
  --workers 4 \
  --patience 10 \
  --seed 42 \
  --no-amp \
  --optimizer SGD \
  --fraction 0.1
