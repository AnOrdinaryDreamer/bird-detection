# =============================================================================
# Dockerfile для офлайн-инференса модели детекции птиц
# (Оптимизированная версия с PyTorch CPU-only)
# =============================================================================
#
# Сборка:
#   docker build -t ml-app:v1 .
#
# Запуск:
#   docker run -v ./images:/data/input -v ./output:/data/output \
#       ml-app:v1 --input_path /data/input --output_path /data/output/preds.csv
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Загрузка модели через DVC (этот слой будет выброшен)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS model-downloader

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем DVC с S3 поддержкой (boto3)
RUN pip install --no-cache-dir 'dvc[s3]==3.66.1' boto3

WORKDIR /download

# Инициализируем git (DVC требует)
RUN git init

# Копируем DVC конфигурацию
COPY dvc.yaml dvc.lock ./
COPY .dvc/ ./.dvc/

# Загружаем модель
RUN dvc pull outputs/trained_model -v

# -----------------------------------------------------------------------------
# Stage 2: Финальный образ для инференса
# -----------------------------------------------------------------------------
FROM python:3.11-slim

LABEL description="Bird Detection Model - Offline Inference"
LABEL version="1.0"

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_INPUT=1

WORKDIR /app

# Установка системных зависимостей для OpenCV и Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry без создания виртуального окружения
RUN pip install --no-cache-dir poetry==1.8.2

# ВАЖНО: не создаём venv, ставим в системный Python
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Устанавливаем PyTorch CPU-only
RUN pip install --no-cache-dir \
    torch==2.2.1+cpu \
    torchvision==0.17.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные зависимости через Poetry (без torch, torchvision, dvc, nvidia)
# Экспортируем зависимости и фильтруем ненужные
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --only main && \
    grep -v "^torch==" requirements.txt | \
    grep -v "^torchvision==" | \
    grep -v "^dvc" | \
    grep -v "^nvidia-" > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt && \
    rm requirements.txt requirements-filtered.txt

# Копируем модель из stage 1
COPY --from=model-downloader /download/outputs/trained_model /app/model

# Копируем только код для инференса
COPY bird_detection/__init__.py ./bird_detection/
COPY bird_detection/predict.py ./bird_detection/
COPY bird_detection/detection/ ./bird_detection/detection/
COPY bird_detection/conf/ ./bird_detection/conf/

# Удаляем ключи из образа (опционально, для безопасности)
RUN rm -rf .dvc/ outputs/

VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["python", "-m", "bird_detection.predict"]
CMD ["--help"]
