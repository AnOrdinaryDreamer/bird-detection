"""
Utilities for preparing bird and squirrel object detection datasets.

This package currently exposes helpers to parse dataset archives downloaded
from Kaggle and build PyTorch DataLoader objects.
"""

from .data_loading import (  # noqa: F401
    CUB200DetectionDataset,
    OpenImageSubsetDataset,
)
from .dataloaders import build_dataloaders  # noqa: F401

__all__ = [
    "CUB200DetectionDataset",
    "OpenImageSubsetDataset",
    "build_dataloaders",
]
