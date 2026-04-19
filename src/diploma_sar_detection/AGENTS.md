# AGENTS.md

## Область применения
Операционная точка входа для coding-агентов в этом репозитории.  
Используй этот файл для правил безопасного выполнения; подробности смотри в README/docs.

- При анализе поведения системы предпочитай SDD-артефакты (`sdd/`) вместо README.

## Политика достоверности
- `Confirmed by repo`: поведение явно подтверждено кодом/конфигами/документацией.
- `Inferred`: безопасное предположение на основе структуры; проверяй перед серьёзными изменениями.
- `Not established in repo`: не считать существующим.

## Обзор проекта (Confirmed by repo)
- CV-пайплайн для детекции объектов на базе `ultralytics` YOLO.
- Основной поток: подготовка/проверка датасета → обучение → валидация → инференс.
- Два пути обучения: базовый YOLO-чекпоинт и Swin-T scaffold.

## Ключевые пути (Confirmed by repo)
- `scripts/`: entrypoint-скрипты (`train_baseline.py`, `train_swin.py`, `validate.py`, `predict_sample.py`, `check_dataset.py`, `convert_coco_to_yolo.py`).
- `scripts/utils.py`: общие runtime-хелперы (seed, device, YOLO config dir, метрики).
- `data/dataset.yaml`: конфигурация датасета Ultralytics.
- `custom_models/`: обёртка Swin-T + регистрация.
- `models/yolo26_swin_t.yaml`: кастомная архитектура.
- `README.md`, `data/README_data.md`, `models/README_models.md`: контекст.

## Настройка / окружение (Confirmed by repo)
- Выполни `python3 -m venv .venv` и активируй окружение.
- Выполни `pip install -r requirements.txt`.
- Используй Python 3.11+.
- Для GPU установи соответствующую CUDA-версию PyTorch.

## Основные команды (Confirmed by repo)
- Проверка датасета:  
  `python scripts/check_dataset.py --dataset-root <DATASET_ROOT> [--save-report <report.json>]`
- Конвертация COCO→YOLO:  
  `python scripts/convert_coco_to_yolo.py --dataset-root <DATASET_ROOT> --splits train val [--create-empty-test-dir] [--save-report <report.json>]`
- Обучение baseline:  
  `python scripts/train_baseline.py --data data/dataset.yaml --model yolo26n.pt --project runs --name <run_name> [flags]`
- Обучение Swin:  
  `python scripts/train_swin.py --data data/dataset.yaml --model models/yolo26_swin_t.yaml --project runs --name <run_name> [flags]`
- Валидация:  
  `python scripts/validate.py --model <best.pt> --data data/dataset.yaml --project runs [--split val|test]`
- Инференс:  
  `python scripts/predict_sample.py --model <best.pt> --source <image_or_dir> --save-dir <out_dir>`

## Датасет и конфиги (Confirmed by repo)
- Соблюдай ключи `dataset.yaml`: `path`, `train`, `val`, `test`, `nc`, `names`.
- Порядок классов в `names` должен совпадать с ID.
- Структура: `images/{train,val,test}` и `labels/{train,val,test}`.
- Перед изменениями запускай `check_dataset.py`.
- Не меняй формат разметки: `class_id x_center y_center width height` (нормализовано).

## Модели и архитектура
- Confirmed by repo:
    - Baseline использует `YOLO(args.model)`.
    - Swin требует `register_swin_t_backbone()`.
    - YAML зависит от monkey-patch `TorchVision`.
- Inferred:
    - При изменении каналов обновляй `Index`/`Detect`.

## Артефакты и чекпоинты (Confirmed by repo)
- Артефакты в `runs/<name>`.
- Чекпоинты: `weights/best.pt`, `weights/last.pt`.
- Валидация: `runs/val_<model>_<split>_<timestamp>`.
- Инференс: `--save-dir`.
- `YOLO_CONFIG_DIR` задаётся через `configure_ultralytics()`.

## Границы безопасных изменений
- Confirmed by repo:
    - Читай entrypoints и `scripts/utils.py` перед изменениями.
    - Сохраняй CLI-совместимость.
    - Не трогай дефолты обучения без причины.
- Inferred:
    - Не меняй пути/имена чекпоинтов без обновления зависимых мест.
    - Не удаляй артефакты без понимания.

## Рабочий процесс агента
1. Прочитай `dataset.yaml` и entrypoint.
2. Разбери контракт данных.
3. Не меняй формат датасета/чекпоинтов без причины.
4. Сначала минимальный запуск.
5. Потом полный.

## SDD Iteration Rules
- Все SDD-артефакты хранятся в `sdd/`.
- Итерация: `<номер>_<имя>` (пример: `001_swin_backbone_train`).
- Используй только последнюю итерацию.
- Старые — только история.

Каждая итерация должна содержать:
- цель
- решения
- изменения спеки
- открытые вопросы

Запрещено:
- менять прошлые итерации
- дробить итерацию

## Правила разработки через SDD
- Все нетривиальные изменения выполняются через SDD.

Перед реализацией:
- Найди/создай текущую итерацию.
- Зафиксируй цель.

Во время:
- Синхронизируй код с SDD.
- Не добавляй поведение вне SDD.

После:
- Обнови текущее состояние системы.
- Проверь соответствие итерации и кода.

Предпочтительно:
- маленькие итерации
- одна задача = одна итерация

Запрещено:
- большие изменения без SDD
- смешивать задачи
- использовать SDD как декор

Если код и SDD расходятся:
- SDD считается устаревшим
- обнови SDD или остановись

См. формат:
- `sdd/README.md`

## CV/ML правила
- Проверяй shape.
- Не хардкодь CUDA.
- Сохраняй seed.
- Не ломай метрики.
- Не меняй аугментации молча.
- Сохраняй совместимость чекпоинтов.

## Проверка изменений
- Запусти минимальный сценарий.
- Для датасета — `check_dataset.py`.
- Для обучения — короткий train.
- Для инференса — `predict_sample.py`.

## Not Established in Repo
- Нет линтеров/тестов/CI.
- Нет W&B/MLflow.

## Ссылки
- `README.md`
- `data/README_data.md`
- `models/README_models.md`