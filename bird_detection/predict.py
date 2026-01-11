"""
Скрипт офлайн-инференса для bird detection модели.

Использование:
    python -m bird_detection.predict --input_path /path/to/images --output_path /path/to/preds.csv

Входные данные:
    - Путь к одному изображению (jpg, jpeg, png)
    - Путь к папке с изображениями
    
Выходные данные:
    CSV файл с колонками:
    - image_path: путь к изображению
    - detection_id: порядковый номер детекции на изображении  
    - label: предсказанный класс (вид птицы или 'squirrel')
    - score: уверенность модели (0-1)
    - x_min, y_min, x_max, y_max: координаты bounding box в пикселях
    - alternatives: топ-3 альтернативных класса с оценками (JSON)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from bird_detection.detection.inference import load_detector
from bird_detection.detection.predictions import format_prediction_for_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Поддерживаемые форматы изображений
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Путь к модели по умолчанию (внутри контейнера)
DEFAULT_MODEL_DIR = Path("/app/model")


def get_image_paths(input_path: Path) -> List[Path]:
    """
    Получить список путей к изображениям из входного пути.
    
    Args:
        input_path: Путь к файлу или директории
        
    Returns:
        Список путей к изображениям
    """
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {input_path.suffix}. "
                f"Поддерживаются: {SUPPORTED_EXTENSIONS}"
            )
    elif input_path.is_dir():
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        if not images:
            raise ValueError(f"Изображения не найдены в директории: {input_path}")
        return sorted(images)
    else:
        raise FileNotFoundError(f"Путь не существует: {input_path}")


def load_image_as_tensor(image_path: Path) -> torch.Tensor:
    """
    Загрузить изображение и преобразовать в тензор.
    
    Модель ожидает тензор [C, H, W] в диапазоне [0, 1] в RGB формате.
    
    Args:
        image_path: Путь к изображению
        
    Returns:
        Тензор [3, H, W] в диапазоне [0, 1]
    """
    image = Image.open(image_path).convert("RGB")
    # Преобразуем в тензор [0, 1]
    tensor = transforms.ToTensor()(image)
    return tensor


def run_inference(
    model: torch.nn.Module,
    image_paths: List[Path],
    metadata,
    device: torch.device,
    score_threshold: float = 0.25,
    batch_size: int = 1,
) -> List[dict]:
    """
    Запустить инференс на списке изображений.
    
    Args:
        model: Загруженная модель
        image_paths: Пути к изображениям
        metadata: Метаданные датасета с маппингом классов
        device: Устройство для вычислений
        score_threshold: Порог уверенности для фильтрации детекций
        batch_size: Размер батча (пока только 1)
        
    Returns:
        Список словарей с результатами детекций
    """
    results = []
    model.eval()
    
    with torch.no_grad():
        for i, image_path in enumerate(image_paths):
            logger.info(f"Обработка [{i+1}/{len(image_paths)}]: {image_path.name}")
            
            try:
                image_tensor = load_image_as_tensor(image_path)
                image_tensor = image_tensor.to(device)
                
                # Модель ожидает список изображений
                outputs = model([image_tensor])
                
                # Форматируем предсказания
                predictions = format_prediction_for_api(
                    raw_prediction=outputs[0],
                    metadata=metadata,
                    score_threshold=score_threshold,
                    max_detections=100,  # Разумный лимит
                    include_background=False,
                )
                
                # Добавляем результаты
                for det_id, pred in enumerate(predictions):
                    result = {
                        "image_path": str(image_path),
                        "detection_id": det_id,
                        "label": pred.label,
                        "score": round(pred.score, 4),
                    }
                    
                    # Добавляем bbox если есть
                    if pred.bbox:
                        result.update({
                            "x_min": round(pred.bbox["x_min"], 2),
                            "y_min": round(pred.bbox["y_min"], 2),
                            "x_max": round(pred.bbox["x_max"], 2),
                            "y_max": round(pred.bbox["y_max"], 2),
                        })
                    else:
                        result.update({
                            "x_min": None,
                            "y_min": None,
                            "x_max": None,
                            "y_max": None,
                        })
                    
                    # Добавляем альтернативы как JSON
                    if pred.alternatives:
                        alternatives_dict = [
                            {"label": alt.label, "score": round(alt.score, 4)}
                            for alt in pred.alternatives
                        ]
                        result["alternatives"] = json.dumps(alternatives_dict)
                    else:
                        result["alternatives"] = None
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "detection_id": 0,
                    "label": "error",
                    "score": 0.0,
                    "x_min": None,
                    "y_min": None,
                    "x_max": None,
                    "y_max": None,
                    "alternatives": json.dumps({"error": str(e)}),
                })
    
    return results


def save_results_to_csv(results: List[dict], output_path: Path) -> None:
    """
    Сохранить результаты в CSV файл.
    
    Args:
        results: Список словарей с результатами
        output_path: Путь для сохранения CSV
    """
    if not results:
        logger.warning("Нет результатов для сохранения")
        return
    
    # Создаём директорию если нужно
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "image_path", "detection_id", "label", "score",
        "x_min", "y_min", "x_max", "y_max", "alternatives"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Результаты сохранены в: {output_path}")


def find_model_dir() -> Path:
    """
    Найти директорию с моделью.
    
    Порядок поиска:
    1. /app/model (внутри контейнера)
    2. outputs/trained_model (локально)
    
    Returns:
        Путь к директории с моделью
    """
    candidates = [
        DEFAULT_MODEL_DIR,
        Path("outputs/trained_model"),
        Path("/app/outputs/trained_model"),
    ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "pytorch_model.bin").exists():
            return candidate
    
    raise FileNotFoundError(
        f"Модель не найдена. Проверены пути: {[str(c) for c in candidates]}. "
        "Укажите путь явно через --model_dir или выполните dvc pull."
    )


def main(args: Optional[List[str]] = None) -> int:
    """
    Главная функция скрипта.
    
    Args:
        args: Аргументы командной строки (для тестирования)
        
    Returns:
        Код возврата (0 - успех, 1 - ошибка)
    """
    parser = argparse.ArgumentParser(
        description="Офлайн-инференс модели детекции птиц",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработка одного изображения
  python -m bird_detection.predict --input_path photo.jpg --output_path results.csv
  
  # Обработка папки с изображениями
  python -m bird_detection.predict --input_path /data/images --output_path /output/preds.csv
  
  # С указанием порога уверенности
  python -m bird_detection.predict --input_path images/ --output_path preds.csv --threshold 0.5
        """,
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Путь к изображению или директории с изображениями",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Путь для сохранения результатов (CSV файл)",
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Путь к директории с моделью (по умолчанию: автоопределение)",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Порог уверенности для фильтрации детекций (по умолчанию: 0.25)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для вычислений: 'cpu' или 'cuda' (по умолчанию: автоопределение)",
    )
    
    parsed_args = parser.parse_args(args)
    
    try:
        # Определяем устройство
        if parsed_args.device:
            device = torch.device(parsed_args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {device}")
        
        # Находим модель
        if parsed_args.model_dir:
            model_dir = Path(parsed_args.model_dir)
        else:
            model_dir = find_model_dir()
        logger.info(f"Загрузка модели из: {model_dir}")
        
        # Загружаем модель
        model, metadata, config = load_detector(model_dir, device=device)
        logger.info(f"Модель загружена. Классов: {metadata.num_classes}")
        
        # Получаем список изображений
        input_path = Path(parsed_args.input_path)
        image_paths = get_image_paths(input_path)
        logger.info(f"Найдено изображений: {len(image_paths)}")
        
        # Запускаем инференс
        results = run_inference(
            model=model,
            image_paths=image_paths,
            metadata=metadata,
            device=device,
            score_threshold=parsed_args.threshold,
        )
        
        # Сохраняем результаты
        output_path = Path(parsed_args.output_path)
        save_results_to_csv(results, output_path)
        
        # Выводим статистику
        total_detections = len([r for r in results if r["label"] not in ("not_found", "error")])
        logger.info(f"Обработка завершена. Всего детекций: {total_detections}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
