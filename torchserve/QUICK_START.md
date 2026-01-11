# TorchServe

## Запуск (одна команда)

```bash
docker run -d \
    --name bird-detection-torchserve \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 8082:8082 \
    bird-detection-serve:v1
```


## Проверка работоспособности

```bash
# Health check
curl http://localhost:8080/ping
# Ожидаемый ответ: {"status": "Healthy"}

# Статус модели
curl http://localhost:8081/models/bird_detection | python3 -m json.tool
```

## Inference примеры

### Детекция на изображении

```bash
# С локальным файлом
curl -X POST http://localhost:8080/predictions/bird_detection \
    -T /path/to/image.jpg \
    -H "Content-Type: application/octet-stream" | python3 -m json.tool

# С изображением из проекта (птица)
curl -X POST http://localhost:8080/predictions/bird_detection \
    -T bird_detection/data/selected/birds/026_074_Florida_Jay/images/Florida_Jay_0108_64694.jpg \
    -H "Content-Type: application/octet-stream" | python3 -m json.tool

# С изображением из проекта (белка)
curl -X POST http://localhost:8080/predictions/bird_detection \
    -T bird_detection/data/selected/squirrels/images/db2cc1f44a8ca429.jpg \
    -H "Content-Type: application/octet-stream" | python3 -m json.tool
```

### Python клиент

```python
import requests

# Отправка изображения
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/predictions/bird_detection',
        data=f,
        headers={'Content-Type': 'application/octet-stream'}
    )

predictions = response.json()
print(f"Найдено детекций: {predictions['num_detections']}")

for detection in predictions['detections']:
    print(f"Класс: {detection['label']}")
    print(f"Уверенность: {detection['score']:.2%}")
    print(f"BBox: {detection['bbox']}")
    print()
```

## Формат ответа

```json
{
    "detections": [
        {
            "label": "squirrel",
            "label_id": 52,
            "score": 0.5208,
            "bbox": {
                "x_min": 40.37,
                "y_min": 52.84,
                "x_max": 422.25,
                "y_max": 295.19
            },
            "alternatives": [
                {
                    "label": "069.Rufous_Hummingbird",
                    "label_id": 25,
                    "score": 0.19
                }
            ]
        }
    ],
    "num_detections": 1
}
```

## Управление контейнером

```bash
# Логи
docker logs -f bird-detection-torchserve

# Остановка
docker stop bird-detection-torchserve

# Запуск снова
docker start bird-detection-torchserve

# Удаление
docker rm -f bird-detection-torchserve

# Статистика
docker stats bird-detection-torchserve
```

## Эндпоинты

- **8080** - Inference API (предсказания)
- **8081** - Management API (управление моделями)
- **8082** - Metrics API (метрики Prometheus)

## Масштабирование workers

```bash
# Увеличить до 4 workers
curl -X PUT "http://localhost:8081/models/bird_detection?min_worker=2&max_worker=4"

# Проверить
curl http://localhost:8081/models/bird_detection | python3 -m json.tool
```

## Метрики

```bash
# Все метрики
curl http://localhost:8082/metrics

# Только для bird_detection
curl "http://localhost:8082/metrics?name=bird_detection"
```


## Полная пересборка

Если нужно пересобрать с нуля:

```bash
# 1. Остановить и удалить контейнер
docker stop bird-detection-torchserve
docker rm bird-detection-torchserve

# 2. Удалить образ
docker rmi bird-detection-serve:v1

# 3. Пересобрать
cd bird-detection
docker build -f torchserve/Dockerfile -t bird-detection-serve:v1 .

# 4. Запустить
docker run -d --name bird-detection-torchserve \
    -p 8080:8080 -p 8081:8081 -p 8082:8082 \
    bird-detection-serve:v1
```

---
