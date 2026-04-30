# 011_yolo26n_gated_alpha6_lowlr_nomosaic

## Goal
Добавить отдельный gated-вариант `YOLO26n` для запуска от dataset-pretrained baseline checkpoint, где:
- `Swin` ветви стартуют случайно;
- gate стартует ещё ближе к чистому CNN (`init_alpha=6.0`);
- обучение можно запускать с более мягким `AdamW` learning rate;
- `mosaic` можно отключать через `train_swin_context.py`.

## Inputs
- [`models/yolo26n_gated_swin_p4_p5.yaml`](../../models/yolo26n_gated_swin_p4_p5.yaml)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)
- dataset-pretrained baseline checkpoint [`baseline_yolo26_100ep/weights/best.pt`](../../baseline_yolo26_100ep/weights/best.pt)

## Findings
- На коротком горизонте `pretrained softgate` не дал быстрого выигрыша над random gated.
- Для следующей проверки нужен более консервативный старт: ближе к baseline CNN, ниже `lr0`, без `mosaic`.

## Decisions
- Создать новый YAML [`models/yolo26n_gated_swin_p4_p5_alpha6.yaml`](../../models/yolo26n_gated_swin_p4_p5_alpha6.yaml) вместо изменения существующих gated-моделей.
- Оставить `Swin` ветви случайно инициализированными, без `--swin-p4-weights/--swin-p5-weights`.
- В `scripts/train_swin_context.py` добавить optional проброс `--lr0` и `--mosaic`.
- Для `AdamW` выбрать рекомендуемый старт `lr0=0.0005`.

## Spec updates
- Добавлен отдельный `alpha=6.0` gated YAML для `YOLO26n`.
- `train_swin_context.py` может передавать `lr0` и `mosaic` в `ultralytics` train kwargs.

## Open questions
- Даст ли более консервативный gate-start лучший early-training stability на горизонте `10-15` эпох.
- Нужен ли аналогичный режим с `close_mosaic=0`, если `mosaic=0.0` уже полностью отключает аугментацию.

## Next iteration
- Сравнить `alpha=4.0` vs `alpha=6.0` при одном и том же baseline checkpoint, `AdamW lr0=5e-4`, `mosaic=0.0`, `30-50` эпох.
