"""Detection subpackage exposing training utilities."""

from .augmentations import build_transform_pipeline
from .dataloaders import build_dataloaders
from .models import build_model
from .trainer import DetectionTrainer
from .inference import load_detector

__all__ = [
    "build_transform_pipeline",
    "build_dataloaders",
    "build_model",
    "DetectionTrainer",
    "load_detector",
]
