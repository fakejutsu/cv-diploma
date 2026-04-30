# Iteration 007 Runbook

Этот runbook фиксирует практический порядок запуска для итерации `007_yolo26n_swin_context_p5`.

## Цель

Сравнить три экспериментальных пути на одном датасете:

1. baseline `YOLO26n`
2. legacy pure `Swin-T` backbone replacement
3. `YOLO26n + SwinContextBlock(P5)`

## Быстрая проверка архитектуры

Для context-пути сначала проверить сборку модели и shape-контракт:

```bash
python3 scripts/validate_swin_context.py --imgsz 640
```

Ожидаемые shape'ы для `YOLO26n` при `imgsz=640`:

- `P5_backbone: (1, 256, 20, 20)`
- `P5_context: (1, 256, 20, 20)`
- `P5_fused: (1, 256, 20, 20)`

## Команды запуска

### 1. Baseline YOLO26n

```bash
python3 scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model yolo26n.pt \
  --project runs_cmp \
  --name baseline_yolo26n_smoke \
  --epochs 10 \
  --imgsz 640 \
  --batch 24 \
  --device 0 \
  --workers 4 \
  --patience 10 \
  --seed 42 \
  --no-amp \
  --fraction 0.1
```

### 2. Legacy pure Swin-T backbone

```bash
python3 scripts/train_swin.py \
  --data data/dataset.yaml \
  --model models/yolo26_swin_t.yaml \
  --project runs_cmp \
  --name swint_neck_smoke \
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
```

### 3. YOLO26n + SwinContextBlock(P5)

```bash
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
```

## Валидация и инференс

### Валидация context-модели

```bash
python3 scripts/validate.py \
  --model runs_cmp/yolo26n_swin_context_p5_smoke/weights/best.pt \
  --data data/dataset.yaml \
  --imgsz 640 \
  --batch 24 \
  --device 0 \
  --project runs_cmp
```

### Инференс context-модели

```bash
python3 scripts/predict_sample.py \
  --model runs_cmp/yolo26n_swin_context_p5_smoke/weights/best.pt \
  --source /path/to/sample_or_dir \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --save-dir runs_cmp/predict_yolo26n_swin_context_p5
```

## Ограничения текущей итерации

- Текущий context YAML рассчитан на `YOLO26n` scale (`n`).
- В `007` нет `P4+P5` fusion.
- В `007` нет `freeze_swin_epochs`.
- В `007` нет `swin_lr_mult`.
- В `007` нет `gated/add` fusion.
