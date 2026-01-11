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
- **Оригинальный состав:** 200 видов птиц, 11 788 изображений (5 994 train / 5 794 test).  
- **Аннотации:** bounding box, 15 keypoints, 312 атрибутов.  
- **В проекте:** отобран **51 вид**, который более вероятно увидеть в кормушке (~3 000 изображений).  
- **Особенность:** обычно по одной птице на кадре, удобно для детекции «одного класса на изображение».
### Белки
- **Источник:** [OpenImageV7 subset](https://www.kaggle.com/datasets/olgreyfox/openimagev7-raccoonsquirrelskunkmouserabbit?select=OID_YOLOv8_Dataset) c заранее выгруженными YOLOv8 split’ами (`train/`, `val/`).  
- **Оригинальный состав:** включает 5 классов (raccoon, squirrel, skunk, mouse, rabbit) с нормализованными YOLO-метками.  
- **В проекте:** используем только класс `squirrel` (id `84` в исходных .txt файлах), собирая изображения из обоих сплитов в единую выборку (~1 800 изображений).  
- **Особенность:** кадры часто содержат несколько животных или частичные попадания; наш скрипт автоматически вычисляет пиксельные боксы и исключает остальные классы. Также разметка довольно шумная, отчего интересно далее будет увидеть как мы получим подвыборку с плохим качеством предсказаний (тем не менее, на всех картинках находятся грызуны, так что животные скорее будут отнесены к белкам, чем к птицам).

<!-- ### Пустые кадры
- **Класс:** `none`.  
- Используются для обучения модели распознавать отсутствие объектов.  
- Разметка: один label без bbox.  
- Оценка влияния пустых кадров — отдельный эксперимент.
 -->


## План экспериментов

В качестве базовой модели будем использовать SSD из torchvision с предобученными весами, которую мы дообучим на собранных данных. Для повышения устойчивости мы пошагово проверим группы аугментаций из albumentations: сначала геометрические (flip, rotate) и цветовые (RandomBrightnessContrast, ColorJitter), после чего погодные и доменно-специфичные эффекты вроде MotionBlur, RandomFog, Rain, RandomGamma и Blur. Отдельно оценим эффект использования синтетических сцен, созданных с помощью Mosaic и CutMix, с несколькими объектами, а также добавление "пустых" кадров, чтобы снизить ложные срабатывания. В финале сравним дообученные SSD и Faster R-CNN (обе модели предобучены на COCOv1) и выберем конфигурацию, которая даёт лучший компромисс между mAP и задержкой.

### Этапы экспериментов

| Эксперимент | Модель | Аугментации | Цель | Метрика успеха |
|------------|--------|-------------|------|----------------|
| **Baseline** | SSD300 | Нет | Базовая производительность | mAP@0.5 ≥ 0.50 |
| **Exp-1** | SSD300 | Геометрические (flip, rotate) + цветовые (brightness, contrast) | Улучшение устойчивости | +5% mAP |
| **Exp-2** | SSD300 | + Погодные эффекты (fog, rain, blur) | Робастность к условиям съёмки | +3% mAP |
| **Exp-3** | SSD300 | + Mosaic + CutMix | Детекция нескольких объектов | +5% mAP, улучшение recall |
| **Exp-4** | SSD300 | + Пустые кадры | Снижение false positives | FP rate < 5% |
| **Exp-5** | Faster R-CNN | Лучшая конфигурация из Exp 1-4 | Сравнение архитектур | mAP@0.5 ≥ 0.65 |
| **Final** | SSD/Faster R-CNN | Оптимальная комбинация | Балансировка mAP и latency | mAP@0.5 ≥ 0.65, p95 ≤ 1.5s |


## Управление данными и моделями с DVC

Проект использует **DVC (Data Version Control)** для версионирования данных и ML-пайплайна. Данные и модели хранятся в удалённом S3-совместимом хранилище (Yandex Cloud Object Storage).

### Необходимые переменные окружения

Для работы с проектом нужны ключи доступа:

#### Yandex Object Storage (для DVC remote):
```bash
export AWS_ACCESS_KEY_ID="your_yandex_key_id"
export AWS_SECRET_ACCESS_KEY="your_yandex_secret_key"
```

#### Kaggle API (для загрузки сырых данных):
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

### 🚀 Быстрый старт

```bash
# Клонировать репозиторий
git clone <repository-url>
cd bird-detection

# Установить зависимости
poetry install

# Настроить переменные окружения (см. выше)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."

# Загрузить данные из DVC remote
poetry run dvc pull

# Воспроизвести pipeline
poetry run dvc repro
```

### 📁 Структура данных и моделей

**Физическое расположение:**

```
bird-detection/
├── data/
│   ├── raw/                          # ZIP архивы (~2.8 GB)
│   │   ├── cub2002011.zip
│   │   └── openimagev7-....zip
│   ├── extracted/                    # Распакованные данные (~8.4 GB, генерируется), не версионируются
│   └── selected/                     # Генерируется pipeline
├── bird_detection/data/selected/     # Подготовленные данные (~152 MB)
│   ├── birds/                        # 51 вид, ~3000 изображений
│   └── squirrels/                    # ~1800 изображений
└── outputs/                          # Результаты обучения
    ├── trained_model/                # Обученная модель
    ├── tensorboard/                  # Логи обучения
    └── evaluation/                   # Метрики и графики
```

**Что хранится в удалённом хранилище:**
- ✅ `data/raw/` — исходные ZIP архивы (2.8 GB)
- ✅ `bird_detection/data/selected/` — подготовленные данные (152 MB)
- ✅ `outputs/trained_model/` — обученная модель (117 MB)
- ✅ `outputs/evaluation/` — метрики оценки

### 🔄 DVC Pipeline

Pipeline состоит из 4 автоматических стадий:

```
data/raw (DVC) → extract → prepare → train → evaluate
```

#### 1. **extract** — Распаковка архивов
Распаковывает ZIP файлы из `data/raw/` в `data/extracted/`

```bash
poetry run dvc repro extract
```

#### 2. **prepare** — Подготовка данных
Извлекает 51 вид птиц из CUB-200-2011 и белок из OpenImage V7, приводит к единому формату.

```bash
poetry run dvc repro prepare
```

**Результат:** `bird_detection/data/selected/` с готовыми данными для обучения

#### 3. **train** — Обучение модели
Обучает SSD300 с предобученными весами на подготовленных данных.

```bash
poetry run dvc repro train
```

**Результат:** `outputs/trained_model/` с весами модели

**Параметры** настраиваются в `bird_detection/conf/`. По умолчанию:
- Модель: SSD300 VGG16 (torchvision)
- Epochs: 10 (по умолчанию)
- Batch size: 4
- Optimizer: Adam, lr=0.0001

#### 4. **evaluate** — Оценка модели
Вычисляет mAP@0.5, строит PR-кривые и confusion matrix на тестовой выборке.

```bash
poetry run dvc repro evaluate
```

**Результат:** `outputs/evaluation/` с метриками и визуализациями

### 🎯 Запуск полного pipeline

```bash
# Запустить все стадии последовательно
poetry run dvc repro

# Или отдельные стадии
poetry run dvc repro train     # Только обучение
poetry run dvc repro evaluate  # Только оценка
```

### 💾 Версионирование и синхронизация

```bash
# Отправить данные и модели в remote
poetry run dvc push

# Получить данные и модели из remote
poetry run dvc pull

# Посмотреть метрики
poetry run dvc metrics show

# Визуализировать pipeline
poetry run dvc dag
```

### 🔬 Работа с экспериментами

```bash
# Изменить параметры и переобучить
poetry run dvc repro train -f

# Зафиксировать эксперимент
git add dvc.lock
git commit -m "Experiment 1: baseline SSD300"
git tag exp-1

# Сравнить метрики между экспериментами
poetry run dvc metrics diff exp-1 exp-2

# Вернуться к предыдущему эксперименту
git checkout exp-1
poetry run dvc repro
```

### MLflow: трекинг экспериментов

MLflow автоматически отслеживает все запуски обучения. Каждый `python -m bird_detection.detection.train` создаёт отдельный run с фиксацией параметров, метрик и артефактов.

#### Запуск UI

```bash
cd bird-detection
poetry run mlflow ui --backend-store-uri mlruns
```

#### Что отслеживается

- **Параметры**: все гиперпараметры из конфигурации (lr, batch_size, model, optimizer...)
- **Метрики**: train_loss, val_loss на каждой эпохе + графики
- **Артефакты**: модель, чекпоинты, конфигурация, dvc.lock
- **Теги**: DVC хеши датасетов для отслеживания версии данных

#### Структура хранения

```
mlruns/
├── <experiment_id>/              # ID эксперимента (по имени из конфигурации, по умолчанию "bird_detection")
│   ├── meta.yaml                 # Имя и время создания эксперимента
│   │
│   ├── <run_id_1>/               # Первый запуск
│   │   ├── params/               # Все параметры (lr, epochs, model...)
│   │   ├── metrics/              # История метрик (train_loss, val_loss...)
│   │   ├── tags/                 # DVC хеши, device, user
│   │   └── artifacts/            # Модель, чекпоинты, dvc.lock
│   │
│   └── <run_id_2>/               # Второй запуск (полностью независим)
│       └── ...
│
```

- **Experiment** = группа run'ов с одним именем (например, `bird_detection`)
- **Run** = один запуск обучения с уникальным ID
- Имя эксперимента задаётся в конфигурации через `mlflow.experiment_name` (можно менять через CLI)

```bash
# Создать отдельный эксперимент для тюнинга lr
python -m bird_detection.detection.train mlflow.experiment_name="lr_tuning"
```

####  MLflow → DVC

**MLflow** — для быстрых экспериментов с гиперпараметрами:
```bash
# Запуск множества экспериментов подряд (без git/dvc между ними)
python -m bird_detection.detection.train training.lr=0.0001 training.epochs=5
python -m bird_detection.detection.train training.lr=0.0005 training.epochs=5
python -m bird_detection.detection.train training.lr=0.001 training.epochs=10

# Отбор лучших после сравнения результатов в MLFlow
```

**DVC** — для фиксации лучшей модели в pipeline:
```bash
# После выбора лучших параметров обновляем dvc.yaml (подставив нужный файл конфигурации) и запускаем:
poetry run dvc repro
git add dvc.lock dvc.yaml
git commit -m "Best model at Experiment 2: lr=0.0005, epochs=5"
poetry run dvc push
```

**Важно**: `mlruns/` хранится локально и не версионируется.

###  Метрики качества

После обучения метрики доступны в:
- `outputs/evaluation/metrics.json` — числовые метрики (mAP, AP по классам)
- `outputs/evaluation/pr_curves.json` — данные PR-кривых
- `outputs/evaluation/confusion_matrix.png` — визуализация ошибок
- `outputs/tensorboard/` — логи обучения (TensorBoard) (не версионируются)

```bash
# Просмотр TensorBoard
tensorboard --logdir outputs/tensorboard
```

### Удалённое хранилище

- **Провайдер:** Yandex Cloud Object Storage
- **Bucket:** mlops-homework-17
- **Регион:** ru-central1
- **Конфигурация:** `.dvc/config`

**Настройка доступа:**
1. Создайте сервисный аккаунт в Yandex Cloud
2. Получите ключи доступа (Access Key ID и Secret Key)
3. Экспортируйте переменные окружения (см. выше)


# Использование кода

## Быстрый старт (рекомендуется)

Самый простой способ начать работу с проектом:

```bash
# 1. Клонировать репозиторий
git clone <repository-url>
cd bird-detection

# 2. Установить зависимости
poetry install

# 3. Настроить доступ к Yandex Object Storage
export AWS_ACCESS_KEY_ID="your_key_id"
export AWS_SECRET_ACCESS_KEY="your_secret_key"

# 4. Загрузить данные и воспроизвести pipeline
poetry run dvc pull    # Скачает данные из remote (~3 GB)
poetry run dvc repro   # Запустит весь pipeline
```

После выполнения у вас будет:
- Подготовленные данные в `bird_detection/data/selected/`
- Обученная модель в `outputs/trained_model/`
- Метрики оценки в `outputs/evaluation/`

---


## Описание других модулей

Для удобства, отобранные и подготовленные куски выбранных датасетов сохранены в bird_detection/data/selected.


### Запуск обучения

```bash
python -m bird_detection.detection.train
```

Скрипт выполняет:

1. Считывает конфиг, фиксирует `seed`, подготавливает логи (Python logging + TensorBoard).
2. Собирает пайплайны аугментаций/нормализации (mean/std берутся из конфигурации данных).
3. Загружает изображения птиц/белок, преобразует bbox’ы, делает процентный сплит и строит `DataLoader`’ы.
   * Сплит стратифицирован по классам (каждый вид птиц и белка сохраняют долю в train/val/test),
     а сам train-loader использует `WeightedRandomSampler`, чтобы в батчах не доминировал частый класс (белки).
4. Инстанцирует выбранную модель (`FasterRCNN MobileNetV3 FPN` или `SSD300 VGG16`), оптимизатор и LR-scheduler.
5. Запускает train loop в стиле [официального туториала torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html): суммирует loss-ы из выхода модели, логирует в консоль/TensorBoard, при необходимости использует AMP и clip grad.
6. Сохраняет лучший чекпоинт (`training.checkpoint_dir`), прогоняет тестовый сплит и экспортирует веса в формате, совместимом с Hugging Face (`pytorch_model.bin`, `config.json`, `label_map.json`) в `training.model_dir`.

Запуски складываются в `./outputs/<timestamp>`; там же логи (`tensorboard/`), `.hydra/` и итоговая модель (`trained_model/`). Переопределяйте пути в конфиге или из CLI, если требуется другая структура каталогов.

### End-to-end пайплайн

> **Примечание:** Если вы клонировали репозиторий и хотите начать с готовых данных, используйте DVC:
> ```bash
> git clone <repository-url>
> cd bird-detection
> poetry install
> dvc pull          # Загрузить подготовленные данные
> dvc repro         # Воспроизвести обучение и оценку
> ```

Если вы хотите подготовить данные с нуля:

1. **Установить окружение** (однократно):
   ```bash
   poetry install
   ```
2. **Скачать сырые наборы данных с Kaggle** (там лежат CUB и OpenImage):
   ```bash
   export KAGGLE_USERNAME=...
   export KAGGLE_KEY=...
   python -m bird_detection.data_scripts.fetch_datasets
   ```
3. **Подготовить выборки**:
   ```bash
   poetry run python bird_detection/data_scripts/extract_cub_birds.py \
     --data-root "data/extracted/cub birds/cub2002011" \
     --dest-root bird_detection/data/selected/birds

   poetry run python bird_detection/data_scripts/extract_squirrels.py \
     --source-root data/extracted/openimagev7-raccoonsquirrelskunkmouserabbit \
     --dest-root bird_detection/data/selected/squirrels
   ```
4. **Запустить обучение** (пример для SSD):
   ```bash
   poetry run python -m bird_detection.detection.train \
     model=ssd \
     training.epochs=5 \
     training.batch_size=8
   ```
   После завершения готовый чекпоинт и совместимая с Hugging Face модель окажутся в `outputs/<run>/trained_model/`.
5. **Продолжить обучение** (если нужно):
   ```bash
   poetry run python -m bird_detection.detection.train \
     training.resume_from="outputs/<run>/checkpoints/last_checkpoint.pt" \
     training.epochs=10
   ```
6. **Инференс**: можно загрузить сохранённый пакет и сделать предсказания. За
   подготовку модели к инференсу отвечает модуль
   `bird_detection/detection/inference.py`, в котором живёт хелпер
   `load_detector`:
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
  
  Каждый элемент `ApiPrediction` содержит список `alternatives` в котором до трёх (по
  умолчанию) альтернативных классов. Число кандидатов
  настраивается параметром `model.prediction_topk`.

### Продолжение обучения

Каждая эпоха сохраняет состояние тренировки в `<run>/checkpoints/last_checkpoint.pt`, а лучший валидционный результат — в `best_checkpoint.pt` (плюс веса `best_model.pt`). Чтобы продолжить обучение, укажите путь к чекпоинту:

```bash
poetry run python -m bird_detection.detection.train \
  training.resume_from="outputs/2025-11-08/19-55-19/checkpoints/last_checkpoint.pt" \
  training.epochs=5
```

Скрипт восстановит веса модели, оптимизатор, LR-scheduler и (если включён AMP) `GradScaler`, затем продолжит с эпохи `epoch + 1`.

### Конфигурация

Hydra хранит конфиги в `bird_detection/conf`.
Основной файл `conf/config.yaml` перечисляет default-группы: `data`, `model`, `training`, `logging`, `augmentations`.

- `config.yaml` — значения, использующиеся по умолчанию.
- `data/` — пути, разбиение train/val/test, mean/std для нормализации входов.
- `model/` — параметры архитектур (есть `fasterrcnn.yaml` и `ssd.yaml`, можно добавить новые).
- `training/` — оптимизатор, scheduler, AMP, пути сохранения чекпоинтов/логов.
- `augmentations/` — списки Albumentations трансформов и настройки Mosaic/CutMix.

Любой параметр можно переопределить с помощью CLI, например:

```bash
poetry run python -m bird_detection.detection.train \
  model=ssd \
  data.train_split=0.85 \
  training.batch_size=8 \
  training.epochs=1 \
  augmentations.train.advanced.mosaic.enabled=true \
  augmentations.train.advanced.mosaic.prob=0.3
```
Hydra автоматически создаёт подпапку в `outputs/` с копией конфигов (`.hydra/config.yaml`), так что каждый прогон воспроизводим.


### Отбор данных из загруженных датасетов

   - **Белки (OpenImage V7 subset)**:

  ``` bash
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

  ``` bash
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

### Тесты

* Pytests для всего проекта: `pytest`. Полезные sub-suites:
  * `pytest tests/test_augmentations.py` — проверяет `DetectionTransformPipeline` (0-1 тензоры, нормализация, ресайз, интеграция с Albumentations, поведение при удалении боксов).
  * `pytest tests/test_data_scripts.py` — валидирует экспорт белок и CUB: структура файлов, форматы `labels_pixel`, ресайз, обрезка, CSV с отображением классов.
  * `pytest tests/test_dataloaders.py` — sanity-check датасет/колайт и `DatasetMetadata` (ограничение диапазонов, область боксов, уникальные ID птиц/белок).
  * `pytest tests/test_predictions.py` — форматирование выводов модели в API (`bbox`, фильтрация по порогу, fallback `not_found`).
* Визуальная проверка денормализации: `python -m bird_detection.data_scripts.visualize_bboxes --images-dir bird_detection/data/selected/birds/<class_dir>/images --labels-dir bird_detection/data/selected/birds/<class_dir>/labels_pixel --output-dir outputs/bbox_viz/<class_dir> --limit 5 --shuffle`.


### Заметки
* TensorBoard логи живут в `${hydra:run.dir}/tensorboard`, output-модель — `${hydra:run.dir}/trained_model`.
