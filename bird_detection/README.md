# Описание проекта

## BirdWatcher: умное коллекционирование птиц из вашей кормушки

Сервис превращает наблюдение за дикой природой в игру: пользователь собирает «коллекцию» птиц, узнаёт их виды и следит за частотой визитов. Белки тоже считаются гостями — их детекция вызывает другое уведомление, но не классифицируется по видам (так как в России живёт всего 2 вида белок). Как показало исследование релевантных обсуждений в интернете, пользователям было бы интересно получать информацию о наличии белки в кормушке (и реагировать на них в реальном времени).

Идея проекта состоит в создании системы компьютерного зрения. Пользователь с помощью компактной камеры (например, Raspberry Pi Camera), установленной в кормушке, или собственного смартфона загружает изображение в систему, где нейросетевая модель определяет, кто прилетел к кормушке: птица или белка. 

На получаемых снимках модель детектирует объекты, определяет вид птицы (из 51 категории) и выделяет белок отдельным классом. Мы решаем задачу детекции, так как в кормушке могут находиться сразу несколько животных разных видов, или никого (если снимки с камеры отправляются автоматически).

Пользователь получает уведомление с фото и разметкой (bounding box и класс объекта), а в приложении накапливается коллекция видов птиц, которых он «поймал» камерой. Последнее вносит элемент геймификации, чтобы привлечь пользователей, а также получаемые данные могут быть использованы исследователями для наблюдения за миграцией различных видов птиц.

Целью проекта является создание надёжной ML-систему, которая:
1. Определяет, присутствует ли на снимке птица или белка.  
2. Классифицирует птицу по виду (из 51 категорий) или белку.  
3. Сохраняет статистику визитов и новых видов.  
4. Отправляет пользователю уведомления с изображением и bounding boxes.

В случае полной реализации, для взаимодействия с пользователем планируется использовать чат-бот в Telegram. Это также позволит собирать обратную связь о качестве разметки для сбора дополнительных данных для обучения.


## Целевые метрики

| Категория                    | Метрика и цель / порог |  Обоснование  |
| ---------------------------- | ---------------------------------- | ------------------------- |
| **Детекция объектов**        | mAP@0.50 ≥ **0.65**<br>mAP@[.50:.95] ≥ **0.40**                                    | Приемлемое качество локализации объектов  |
| **Белка**                    | Accuracy ≥ **0.90** | Критичный класс: важно не путать с птицами |
| **Классификация видов птиц** | Top-1 Precision ≥ **0.70**<br>Top-1 Recall ≥ **0.70**<br>Top-3 Accuracy ≥ **0.85** | Баланс между точностью и полнотой; UX допускает неопределённость в top-3             |
| **Сервис (SLO)**             | p95 latency ≤ **1.5 сек**                                                            | Чтобы пользователь успевал увидеть гостя «вживую», если фото приходят с камеры автоматически  |
| **Ошибки API**               | ≤ **1 %** неуспешных запросов   | Стабильность сервиса   |


## Набор данных
### Птицы
- **Источник:** [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/).  
- **Оригинальный объём:** 200 видов, 11 788 изображений (5 994 train / 5 794 test).  
- **Аннотации:** bounding box, 15 keypoints, 312 атрибутов.  
- **В проекте:** отобран **51 вид**, который более вероятно увидеть в кормушке (~3 000 изображений).  
- **Особенность:** обычно по одной птице на кадре, удобно для детекции «одного класса на изображение».
### Белки
- **Источник:** [OpenImageV7 subset](https://www.kaggle.com/datasets/olgreyfox/openimagev7-raccoonsquirrelskunkmouserabbit?select=OID_YOLOv8_Dataset) c заранее выгруженными YOLOv8 split’ами (`train/`, `val/`).  
- **Оригинальный состав:** включает 5 классов (raccoon, squirrel, skunk, mouse, rabbit) с нормализованными YOLO-метками.  
- **В проекте:** используем только класс `squirrel` (id `84` в исходных .txt файлах), собирая изображения из обоих сплитов в единую выборку (~1 800 изображений).  
- **Особенность:** кадры часто содержат несколько животных или частичные попадания; наш скрипт автоматически вычисляет пиксельные боксы и исключает остальные классы.

<!-- ### Пустые кадры
- **Класс:** `none`.  
- Используются для обучения модели распознавать отсутствие объектов.  
- Разметка: один label без bbox.  
- Оценка влияния пустых кадров — отдельный эксперимент.
 -->


## План экспериментов

В качестве базовой модели будем использовать SSD из torchvision с предобученными весами, которую мы дообучим на собранных данных. Для повышения устойчивости мы пошагово проверим группы аугментаций из albumentations: сначала геометрические (flip, rotate) и цветовые (RandomBrightnessContrast, ColorJitter), после чего погодные и доменно-специфичные эффекты вроде MotionBlur, RandomFog, Rain, RandomGamma и Blur. Отдельно оценим эффект использования синтетических сцен, созданных с помощью Mosaic и CutMix, с несколькими объектами, а также добавление “пустых” кадров, чтобы снизить ложные срабатывания. В финале сравним дообученные SSD и Faster R-CNN (обе модели предобучены на COCOv1) и выберем конфигурацию, которая даёт лучший компромисс между mAP и задержкой.



# Запуск модели


## Quickstart

1) **Export Kaggle credentials** (same as your `curl` would use):

```bash
export KAGGLE_USERNAME=your_user
export KAGGLE_KEY=your_key
```

2) **Download & extract**:

```bash
python -m scripts.fetch_datasets
# or
 python -m mlops_data.downloader wenewone/cub2002011 olgreyfox/openimagev7-raccoonsquirrelskunkmouserabbit \
  --raw-dir data/raw --extracted-dir data/extracted
```

# TODO: изменить директорию назначения для согласования

## MLOps-пайплайн обучения

### Установка зависимостей

```bash
python -m pip install -r requirements.txt
```

Файл `requirements.txt` содержит PyTorch/Torchvision, Albumentations, Hydra/OmegaConf, TensorBoard и утилиты вроде tqdm.

### Конфигурация

Hydra хранит конфиги в `bird_detection/conf`:

- `config.yaml` — дефолты, подключающие секции.
- `data/` — пути, разбиение train/val/test, mean/std для нормализации входов.
- `model/` — параметры архитектур (есть `fasterrcnn.yaml` и `ssd.yaml`, можно добавить новые).
- `training/` — оптимизатор, scheduler, AMP, пути сохранения чекпоинтов/логов.
- `augmentations/` — списки Albumentations трансформов и настройки Mosaic/CutMix.

Любой параметр перекрывается из CLI, например:

```bash
python -m bird_detection.detection.train \
  model=ssd \
  training.batch_size=2 \
  training.epochs=1 \
  augmentations.train.advanced.mosaic.enabled=true \
  augmentations.train.advanced.mosaic.prob=0.3
```

### Запуск обучения

```bash
python -m bird_detection.detection.train
```

Скрипт выполняет:

1. Считывает конфиг, фиксирует `seed`, подготавливает логи (Python logging + TensorBoard).
2. Собирает пайплайны аугментаций/нормализации (mean/std берутся из конфигурации данных).
3. Загружает изображения птиц/белок, преобразует bbox’ы, делает процентный сплит и строит `DataLoader`’ы.
4. Инстанцирует выбранную модель (`FasterRCNN MobileNetV3 FPN` или `SSD300 VGG16`), оптимизатор и LR-scheduler.
5. Запускает train loop в стиле [официального туториала torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html): суммирует loss-ы из выхода модели, логирует в консоль/TensorBoard, при необходимости использует AMP и clip grad.
6. Сохраняет лучший чекпоинт (`training.checkpoint_dir`), прогоняет тестовый сплит и экспортирует веса в формате, совместимом с Hugging Face (`pytorch_model.bin`, `config.json`, `label_map.json`) в `training.model_dir`.

Запуски Hydra складываются в `./outputs/<timestamp>`; там же логи (`tensorboard/`), `.hydra/` и итоговая модель (`trained_model/`). Переопределяйте пути в конфиге или из CLI, если требуется другая структура каталогов.

### End-to-end пайплайн

1. **Установить окружение** (однократно):
   ```bash
   poetry install
   ```
2. **Скачать сырьё с Kaggle** (там лежат CUB и OpenImage):
   ```bash
   export KAGGLE_USERNAME=...
   export KAGGLE_KEY=...
   python -m bird_detection.data_scripts.fetch_datasets
   ```
3. **Подготовить выборки**:
   ```bash
   poetry run python bird_detection/data_scripts/extract_cub_birds.py \
     --data-root "data/raw/cub birds/CUB_200_2011/CUB_200_2011" \
     --dest-root bird_detection/data/selected/birds

   poetry run python bird_detection/data_scripts/extract_squirrels.py \
     --source-root data/raw/yolo_rodents/OID_YOLOv8_Dataset \
     --dest-root bird_detection/data/selected/squirrels
   ```
4. **Запустить обучение** (пример для SSD):
   ```bash
   poetry run python -m bird_detection.detection.train \
     model=ssd \
     training.epochs=5 \
     training.batch_size=4
   ```
   После завершения готовый чекпоинт и HF-совместимая модель окажутся в `outputs/<run>/trained_model/`.
5. **Продолжить обучение** (если нужно):
   ```bash
   poetry run python -m bird_detection.detection.train \
     training.resume_from="outputs/<run>/checkpoints/last_checkpoint.pt" \
     training.epochs=10
   ```
6. **Инференс**: можно загрузить сохранённый пакет и сделать предсказания:
   ```bash
   poetry run python - <<'PY'
   import torch
   from bird_detection.detection.inference import load_detector
   from bird_detection.detection.predictions import format_prediction_for_api

   model, metadata, _ = load_detector("outputs/<run>/trained_model", device="cpu")

   # image_tensor: torch.Tensor shape [3,H,W] в диапазоне [0,1]
   image_tensor = torch.rand(3, 512, 512)

   with torch.no_grad():
       outputs = model([image_tensor])
   formatted = format_prediction_for_api(outputs[0], metadata)
   print(formatted)
   PY
   ```

### Продолжение обучения

Каждая эпоха сохраняет состояние тренировки в `<run>/checkpoints/last_checkpoint.pt`, а лучший валидционный результат — в `best_checkpoint.pt` (плюс веса `best_model.pt`). Чтобы продолжить обучение, укажите путь к чекпоинту:

```bash
poetry run python -m bird_detection.detection.train \
  training.resume_from="outputs/2025-11-08/19-55-19/checkpoints/last_checkpoint.pt" \
  training.epochs=5
```

Скрипт восстановит веса модели, оптимизатор, LR-scheduler и (если включён AMP) `GradScaler`, затем продолжит с эпохи `epoch + 1`.

### Конфиги Hydra

- Основной файл `conf/config.yaml` перечисляет default-группы: `data`, `model`, `training`, `logging`, `augmentations`.
- Каждая группа лежит в собственной папке (`conf/data/dataset.yaml`, `conf/model/fasterrcnn.yaml`, и т.д.) и отвечает за конкретный аспект.
- Любой параметр можно переопределить прямо из CLI:  
  ```bash
  poetry run python -m bird_detection.detection.train \
    model=ssd \
    data.train_split=0.85 \
    training.batch_size=8 \
    augmentations.resize_if_needed.enabled=false
  ```
- Hydra автоматически создаёт подпапку в `outputs/` с копией конфигов (`.hydra/config.yaml`), так что каждый прогон воспроизводим.

3) **Отбор данных из загруженных датасетов**

   - **Белки (OpenImage V7 subset)**:

     ```bash
    python bird_detection/data_scripts/extract_squirrels.py \
      --source-root data/raw/yolo_rodents/OID_YOLOv8_Dataset \
      --dest-root data/selected/squirrels \
      --class-id 84 \
      --max-dim 500 \
      --clear-dest
    ```

    Скрипт копирует изображения и создает `labels_pixel/` с боксами формата
    `class_id x_min y_min width height` (в пикселях). Параметр `--max-dim`
    ограничивает максимальную сторону кадра (по умолчанию 500 px); передайте `0`,
    если требуется сохранить исходное разрешение. Метку 84 имеет класс белок.

   - **Птицы (CUB-200-2011)**:

     ```bash
    python bird_detection/data_scripts/extract_cub_birds.py \
      --data-root "data/raw/cub birds/CUB_200_2011/CUB_200_2011" \
      --dest-root data/selected/birds \
      --include 9 10 11 12 14 15 16 17 19 21 26 27 28 29 35 47 48 49 54 55 56 57 \
                67 68 69 73 74 75 76 91 94 95 96 97 116 118 120 129 130 132 133 \
                148 171 175 189 191 192 193 195 196 200 \
      --max-dim 500 \
      --clear-dest
    ```

    Можно передавать значения `--include` как числовые ID, части названий
    (`--include Cardinal Towhee`) или комбинировать; если аргумент опущен,
    выгружаются все 200 классов. Параметры `--source-root` и `--dest-root`
    можно менять, если хотите складывать выборки в другие каталоги. Параметр
    `--max-dim` синхронизирует масштаб с беличьим датасетом (порог 500 px),
    передайте `0`, чтобы оставить оригинальное разрешение CUB.

    В выборке сохранены виды птиц, для которых характерно посещение кормушек — преимущественно воробьинообразные, питающиеся семенами, салом, фруктами или нектаром. Исключены виды, не использующие кормушки, в том числе морские, водоплавающие и насекомоядные птицы.



## Training CLI reminders

* Запуск с Hydra: `python -m bird_detection.detection.train training.epochs=1 model=ssd augmentations.train.advanced.mosaic.enabled=true`.
* TensorBoard логи живут в `${hydra:run.dir}/tensorboard`, output-модель — `${hydra:run.dir}/trained_model`.

## Testing reminders

* Pytests для всего проекта: `pytest`. Полезные подсuites:
  * `pytest tests/test_augmentations.py` — проверяет `DetectionTransformPipeline` (0-1 тензоры, нормализация, ресайз, интеграция с Albumentations, поведение при удалении боксов).
  * `pytest tests/test_data_scripts.py` — валидирует экспорт белок и CUB: структура файлов, форматы `labels_pixel`, ресайз, обрезка, CSV с отображением классов.
  * `pytest tests/test_dataloaders.py` — sanity-check датасет/колайт и `DatasetMetadata` (ограничение диапазонов, область боксов, уникальные ID птиц/белок).
  * `pytest tests/test_predictions.py` — форматирование выводов модели в API (`bbox`, фильтрация по порогу, fallback `not_found`).
* Визуальная проверка денормализации: `python -m bird_detection.data_scripts.visualize_bboxes --images-dir bird_detection/data/selected/birds/<class_dir>/images --labels-dir bird_detection/data/selected/birds/<class_dir>/labels_pixel --output-dir outputs/bbox_viz/<class_dir> --limit 5 --shuffle`.
