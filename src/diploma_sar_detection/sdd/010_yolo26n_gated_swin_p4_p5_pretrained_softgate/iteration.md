# 010_yolo26n_gated_swin_p4_p5_pretrained_softgate

## Goal
Добавить отдельный gated-вариант `YOLO26n`, который:
- сохраняет текущий YOLO warm-start из `yolo26n.pt`;
- использует более мягкий gate-start (`init_alpha=2.0`);
- догружает pretrained `SwinContextBlock` веса для `P4` и `P5` из предыдущих экспериментов;
- не изменяет старую gated-модель с `init_alpha=4.0`.

## Inputs
- [`models/yolo26n_gated_swin_p4_p5.yaml`](../../models/yolo26n_gated_swin_p4_p5.yaml)
- [`models/yolo26n_swin_context_p4_light.yaml`](../../models/yolo26n_swin_context_p4_light.yaml)
- [`models/yolo26n_swin_context_p5.yaml`](../../models/yolo26n_swin_context_p5.yaml)
- [`custom_models/gated_swin_fusion.py`](../../custom_models/gated_swin_fusion.py)
- [`scripts/train_swin_context.py`](../../scripts/train_swin_context.py)
- [`scripts/validate_swin_context.py`](../../scripts/validate_swin_context.py)

## Findings
- Старые `P4-light` и `P5` checkpoint содержат веса `SwinContextBlock` как `model.11.*`, а не `model.11.swin.*`.
- Поэтому composite warm-start нельзя жёстко привязывать к одному prefix; нужен matching по `state_dict` и `shape`.
- Текущий `gated_shift2` remap для YOLO-части уже корректно матчится с gated-моделью и не должен изменяться.

## Decisions
- Создать новый YAML [`models/yolo26n_gated_swin_p4_p5_pretrained.yaml`](../../models/yolo26n_gated_swin_p4_p5_pretrained.yaml) вместо изменения существующего gated-конфига.
- Оставить default `init_alpha=4.0` внутри `GatedSwinFusion`; значение `2.0` задаётся только через новый YAML.
- Добавить optional CLI args `--swin-p4-weights/--swin_p4_weights` и `--swin-p5-weights/--swin_p5_weights`.
- После обычного YOLO warm-start выполнять выбор лучшего source prefix для inner `swin` по числу совпавших `name + shape` параметров.
- `raw_alpha` не загружать из старых checkpoint и оставить обучаемым.

## Spec updates
- Добавлен новый YAML pretrained-soft-gate варианта.
- `train_swin_context.py` и `validate_swin_context.py` поддерживают optional composite Swin warm-start.
- Validator печатает:
  - layer table;
  - shape-контракт;
  - `alpha_mean`;
  - matched counts для `P4/P5` Swin subweights.

## Open questions
- Нужна ли в следующей итерации асимметричная инициализация gate для `P4` и `P5`.
- Нужно ли в будущем выделять отдельный helper-модуль под warm-start, если composite загрузка станет сложнее.

## Next iteration
- Проверить метрики `baseline` vs `P5` vs `P4-light` vs `gated-pretrained-softgate` на одном датасете и одинаковом train protocol.
