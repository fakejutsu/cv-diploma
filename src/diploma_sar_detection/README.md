# diploma_sar_detection

Baseline project for training and evaluating an Ultralytics YOLO26 object detection model on the SeaDronesSee Object Detection v2 dataset.

The current scope is the first baseline stage:

- COCO to YOLO label export when needed
- dataset structure check
- dataset config preparation
- baseline YOLO26 training without backbone replacement
- standalone validation of the best checkpoint
- sample inference on images

The project is intentionally small and readable so it can be extended later for:

- freeze backbone experiments
- full fine-tuning after Swin integration

## Requirements

- Python 3.11+
- Linux or Windows
- CUDA-capable GPU is recommended, but CPU mode is supported

## Install dependencies

Linux:

```bash
cd diploma_sar_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd diploma_sar_detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, install it explicitly after `requirements.txt` using the official PyTorch selector for your image and driver.

## Project structure

```text
diploma_sar_detection/
|-- data/
|   |-- dataset.yaml
|   `-- README_data.md
|-- scripts/
|   |-- check_dataset.py
|   |-- convert_coco_to_yolo.py
|   |-- predict_sample.py
|   |-- train_baseline.py
|   |-- utils.py
|   `-- validate.py
|-- runs/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Expected dataset structure

```text
DATASET_ROOT/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

If the exported dataset differs from this layout, run the dataset checker first and fix the structure before training.

## Example `dataset.yaml`

Edit [data/dataset.yaml](./data/dataset.yaml):

```yaml
path: /absolute/path/to/SeaDronesSee_yolo
train: images/train
val: images/val
test: images/test
nc: 5
names:
  - swimmer
  - boat
  - jetski
  - life_saving_appliances
  - buoy
```

## Commands

### 1. Check dataset

```bash
python scripts/check_dataset.py --dataset-root /path/to/dataset --save-report report.json
```

The checker validates:

- required directories
- image and label counts per split
- image to label matching by stem
- YOLO label line format
- normalized coordinate range
- class statistics

It prints a readable summary and returns a non-zero exit code if critical problems are found.

### Optional: convert COCO annotations to YOLO labels

For datasets like SeaDronesSee Object Detection v2 that ship with `annotations/instances_train.json` and `annotations/instances_val.json`, first export YOLO labels:

```bash
python scripts/convert_coco_to_yolo.py \
  --dataset-root /path/to/dataset \
  --splits train val \
  --create-empty-test-dir \
  --save-report runs/convert_report.json
```

This creates:

- `labels/train/*.txt`
- `labels/val/*.txt`
- optionally an empty `labels/test/`

### 2. Train baseline

```bash
python scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model yolo26n.pt \
  --epochs 10 \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --cache disk \
  --amp \
  --project runs \
  --name baseline_yolo26
```

Defaults:

- `epochs=10`
- `imgsz=640`
- `batch=4`
- `workers=4`
- `patience=20`
- `seed=42`

If CUDA is unavailable, the script falls back to CPU and prints a clear message.
Ultralytics settings and cache are stored inside `.ultralytics/` in the project by default to make local and cloud runs more predictable.

Useful training flags:

- `--cache disk` to reduce repeated image decoding overhead
- `--amp` to use mixed precision on GPU
- `--resume` together with `--model path/to/last.pt` to continue an interrupted run
- `--save-period N` to keep intermediate checkpoints every `N` epochs
- `--exist-ok` if you intentionally want to reuse a run directory

Quick smoke test before renting a GPU:

```bash
python scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 2 \
  --device cpu \
  --workers 0 \
  --fraction 0.01 \
  --project runs \
  --name smoke_yolo26
```

This checks that:

- dataset config is valid
- labels load correctly
- Ultralytics can build the dataloader
- the model starts training without committing to a full local run

### 3. Validate best model

```bash
python scripts/validate.py \
  --model runs/baseline_yolo26/weights/best.pt \
  --data data/dataset.yaml \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --project runs
```

The script runs `model.val(...)`, prints:

- `mAP50`
- `mAP50-95`
- `precision`
- `recall`

Validation artifacts are saved into a dedicated folder inside `runs/`.

### 4. Run inference on samples

```bash
python scripts/predict_sample.py \
  --model runs/baseline_yolo26/weights/best.pt \
  --source /path/to/sample_images \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --save-dir runs/predict_baseline
```

`--source` supports:

- a single image
- a directory with images

Rendered predictions with bounding boxes are saved to the requested directory.

## Where results are stored

- training: the resolved `project/name` directory that Ultralytics prints at the end of training
- validation: `runs/val_<model>_<split>_<timestamp>/` unless overridden
- inference: the directory passed through `--save-dir`
- dataset check report: the path passed through `--save-report`

Typical training artifacts from Ultralytics include:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- plots and training summaries

## Cloud Preparation Checklist

### Do locally before upload

- Make sure `labels/train` and `labels/val` already exist.
- Run `python scripts/check_dataset.py --dataset-root /path/to/dataset`.
- Keep `dataset.yaml` in sync with the final class list.
- Save the project together with `requirements.txt` and the scripts folder.

### Prepare the cloud machine

- Create and activate a fresh virtual environment.
- Install project requirements.
- Install a CUDA-enabled PyTorch build that matches the machine image.
- Verify GPU access:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Upload and configure

- Upload the project folder.
- Upload the dataset in YOLO format.
- Edit `data/dataset.yaml` so `path` points to the dataset location on the cloud machine.
- Prefer Linux-style absolute paths such as `/workspace/data/SeaDronesSee`.

### Run a smoke test in cloud

```bash
python scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --fraction 0.02 \
  --cache disk \
  --amp \
  --project runs \
  --name smoke_cloud
```

### Start the full cloud baseline

```bash
python scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --cache disk \
  --amp \
  --save-period 10 \
  --project runs \
  --name baseline_yolo26_50ep
```

If the instance stops or disconnects, resume from the last checkpoint:

```bash
python scripts/train_baseline.py \
  --data data/dataset.yaml \
  --model runs/baseline_yolo26_50ep/weights/last.pt \
  --resume \
  --device 0 \
  --project runs \
  --name baseline_yolo26_50ep
```

## Notes for further extension

This baseline keeps the default Ultralytics architecture flow and does not introduce:

- custom backbone YAML
- frozen layers
- Swin integration

That separation is intentional so the baseline remains a clean reference point for later diploma experiments.
