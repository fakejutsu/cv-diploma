# Dataset configuration

`dataset.yaml` is the Ultralytics dataset config used by training and validation scripts.

## Expected dataset layout

```text
DATASET_ROOT/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

## How to fill `dataset.yaml`

Update these fields before training:

- `path`: absolute path to the dataset root.
- `train`: path to the training images relative to `path`.
- `val`: path to the validation images relative to `path`.
- `test`: path to the test images relative to `path`.
- `nc`: number of classes.
- `names`: class names in the same order as class ids in label files.

Example:

```yaml
path: /data/SeaDronesSee_OD_v2_yolo
train: images/train
val: images/val
test: images/test
nc: 6
names:
  - swimmer
  - boat
  - buoy
  - life_saving_appliance
  - jetski
  - unknown
```

If your dataset ships with COCO JSON annotations instead of YOLO labels, first run `scripts/convert_coco_to_yolo.py` to generate `labels/train` and `labels/val`.
