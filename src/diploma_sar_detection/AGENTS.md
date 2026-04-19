# AGENTS.md

## Scope
Operational entrypoint for coding agents in this repository.
Use this file for safe execution rules; use README/docs for detailed explanations.

## Evidence Policy
- `Confirmed by repo`: behavior visible in code/config/docs in this repo.
- `Inferred`: safe assumption from current structure; verify before major edits.
- `Not established in repo`: do not assume it exists.

## Project Snapshot (Confirmed by repo)
- CV object detection pipeline built on `ultralytics` YOLO.
- Primary flow: dataset prep/check -> train -> validate -> sample inference.
- Two training paths: baseline YOLO checkpoint and Swin-T scaffold.

## Key Paths (Confirmed by repo)
- `scripts/`: executable entrypoints (`train_baseline.py`, `train_swin.py`, `validate.py`, `predict_sample.py`, `check_dataset.py`, `convert_coco_to_yolo.py`).
- `scripts/utils.py`: shared runtime helpers (seed, device, YOLO config dir, metrics extraction).
- `data/dataset.yaml`: Ultralytics dataset config.
- `custom_models/`: Swin-T wrapper + registration hook.
- `models/yolo26_swin_t.yaml`: custom architecture scaffold.
- `README.md`, `data/README_data.md`, `models/README_models.md`: operational context.

## Setup / Environment (Confirmed by repo)
- Run `python3 -m venv .venv` and activate it.
- Run `pip install -r requirements.txt`.
- Use Python 3.11+ (documented in README).
- If GPU is needed, install matching CUDA-enabled PyTorch build explicitly.

## Canonical Commands (Confirmed by repo)
- Dataset check: `python scripts/check_dataset.py --dataset-root <DATASET_ROOT> [--save-report <report.json>]`
- COCO->YOLO conversion: `python scripts/convert_coco_to_yolo.py --dataset-root <DATASET_ROOT> --splits train val [--create-empty-test-dir] [--save-report <report.json>]`
- Baseline train: `python scripts/train_baseline.py --data data/dataset.yaml --model yolo26n.pt --project runs --name <run_name> [flags]`
- Swin scaffold train: `python scripts/train_swin.py --data data/dataset.yaml --model models/yolo26_swin_t.yaml --project runs --name <run_name> [flags]`
- Validate checkpoint: `python scripts/validate.py --model <best.pt> --data data/dataset.yaml --project runs [--split val|test]`
- Inference samples: `python scripts/predict_sample.py --model <best.pt> --source <image_or_dir> --save-dir <out_dir>`

## Dataset & Config Conventions (Confirmed by repo)
- Keep `dataset.yaml` keys compatible: `path`, `train`, `val`, `test`, `nc`, `names`.
- Keep class order in `names` aligned with label class IDs.
- Expected layout under dataset root: `images/{train,val,test}` and `labels/{train,val,test}`.
- Run `check_dataset.py` before changing training/data logic.
- Do not change label format: each line must stay `class_id x_center y_center width height` normalized to `[0,1]`.

## Model / Architecture Conventions
- Confirmed by repo:
- Baseline path uses Ultralytics `YOLO(args.model)` directly.
- Swin path requires `register_swin_t_backbone()` before model build.
- `models/yolo26_swin_t.yaml` depends on the monkey-patched `TorchVision` symbol.
- Inferred:
- When changing Swin feature outputs/channels, also update `Index`/`Detect` wiring in YAML.

## Artifacts, Runs, Checkpoints (Confirmed by repo)
- Training artifacts are under `--project/--name` (default `runs/<name>`).
- Expect `weights/best.pt` and `weights/last.pt` from Ultralytics runs.
- Validation artifacts go to `runs/val_<model>_<split>_<timestamp>` unless `--name` overrides.
- Inference artifacts are written to `--save-dir`.
- `YOLO_CONFIG_DIR` is set via `configure_ultralytics()` (default `<repo>/.ultralytics`).

## Safe-Change Boundaries
- Confirmed by repo:
- Read entrypoint args and `scripts/utils.py` before editing behavior.
- Preserve CLI compatibility for existing script flags unless task explicitly requires breaking change.
- Keep default training knobs (`epochs`, `imgsz`, `batch`, `workers`, `patience`, `seed`) stable unless requested.
- Inferred:
- Do not change checkpoint file naming/location expectations without updating all consuming scripts/docs.
- Do not remove run artifacts or infra directories without understanding downstream usage.

## Required Agent Workflow
1. Read `data/dataset.yaml` + target entrypoint script before any edit.
2. Trace data contracts first (dataset checker + label parser) before changing training/validation logic.
3. Do not change dataset format, config keys, or checkpoint compatibility without explicit reason.
4. Validate edits with a minimal run first (use small `--fraction`, small `--epochs`, `--device cpu` if needed).
5. Only then run full command variants.

## CV/ML Editing Rules
- Enforce shape safety: verify tensor/channel assumptions when touching `custom_models/` or model YAML.
- Preserve explicit device handling via `resolve_device()`; do not hardcode CUDA-only paths.
- Preserve reproducibility path via `set_seed()` unless task requires different determinism behavior.
- Keep metric consistency: maintain `extract_detection_metrics()` keys (`precision`, `recall`, `mAP50`, `mAP50-95`).
- Treat augmentation as Ultralytics-default (Not customized in repo); avoid silent augmentation changes.
- Maintain backward compatibility for `best.pt`/`last.pt` workflows (`train` -> `validate`/`predict`).

## PR / Validation Checklist
- Run only relevant scripts for changed area (at least one minimal smoke command).
- For data-pipeline edits, run `check_dataset.py` on a representative dataset root.
- For train-path edits, run a 1-epoch or small-fraction smoke train and ensure checkpoint path resolves.
- For validation/inference edits, run `validate.py` or `predict_sample.py` once with a known checkpoint.
- Update `README.md` only when CLI behavior or required workflow changed.

## Not Established in Repo
- No configured lint/format/test framework files (`pyproject.toml`, `pytest`, `ruff`, `Makefile`, CI configs) were found.
- No external experiment tracker integration (e.g., W&B/MLflow) is defined in repo code.

## Additional References
- See `README.md` for end-to-end command examples.
- See `data/README_data.md` for dataset config details.
- See `models/README_models.md` for custom model template notes.
