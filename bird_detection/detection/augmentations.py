"""Utilities to build augmentation pipelines for detection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
import cv2
from torchvision.transforms import functional as F


AugConfig = Dict[str, Any]


def _ensure_sequence(items: Optional[Iterable[Any]]) -> List[AugConfig]:
    if not items:
        return []
    normalized: List[AugConfig] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            normalized.append(item)
        else:
            raise ValueError(f"Unsupported augmentation config: {item}")
    return normalized


def _build_single_aug(
    name: str, params: Optional[Dict[str, Any]] = None
) -> A.BasicTransform:
    params = params or {}
    key = name.lower()
    if key == "flip":
        return A.HorizontalFlip(**params)
    if key == "rotate":
        # Use constant fill value when rotating beyond image bounds.
        return A.Rotate(border_mode=0, **params)
    if key in {"randombrightnesscontrast", "random_brightness_contrast"}:
        return A.RandomBrightnessContrast(**params)
    if key in {"colorjitter", "color_jitter"}:
        return A.ColorJitter(**params)
    if key in {"motionblur", "motion_blur"}:
        return A.MotionBlur(**params)
    if key in {"randomfog", "random_fog"}:
        return A.RandomFog(**params)
    if key in {"rain", "randomrain", "random_rain"}:
        return A.RandomRain(**params)
    if key in {"randomgamma", "random_gamma"}:
        return A.RandomGamma(**params)
    if key == "blur":
        return A.Blur(**params)
    if key == "resize":
        return A.Resize(**params)
    raise KeyError(f"Unsupported augmentation: {name}")


def build_albumentations(
    items: Optional[Sequence[AugConfig]],
    bbox_params: Optional[Dict[str, Any]] = None,
) -> Optional[A.Compose]:
    configs = _ensure_sequence(items)
    if not configs:
        return None
    transforms = [_build_single_aug(cfg["name"], cfg.get("params")) for cfg in configs]
    bbox_settings = bbox_params or {
        "format": "pascal_voc",
        "label_fields": ["labels"],
        "min_visibility": 0.01,
    }
    return A.Compose(transforms, bbox_params=A.BboxParams(**bbox_settings))


@dataclass
class DetectionTransformPipeline:
    """Apply optional resize, Albumentations transforms, and convert to tensors."""

    aug: Optional[A.Compose]
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    resize_cfg: Optional[Dict[str, Any]] = None

    def __call__(
        self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bboxes = boxes.astype(np.float32)
        cls = labels.astype(np.int64)

        if self.resize_cfg and self.resize_cfg.get("enabled", False):
            target = float(self.resize_cfg.get("target_longest_side", 0) or 0)
            tolerance = float(self.resize_cfg.get("tolerance", 0.1))
            if target > 0:
                height, width = image.shape[:2]
                longest = max(height, width)
                diff_ratio = abs(longest - target) / target
                if diff_ratio > tolerance:
                    scale = target / longest
                    new_h = max(1, int(round(height * scale)))
                    new_w = max(1, int(round(width * scale)))
                    image = cv2.resize(
                        image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )
                    if bboxes.size > 0:
                        bboxes *= scale

        if self.aug is not None:
            result = self.aug(
                image=image,
                bboxes=bboxes.tolist(),
                labels=cls.tolist(),
            )
            image = result["image"]
            bboxes = np.array(result["bboxes"], dtype=np.float32)
            cls = np.array(result["labels"], dtype=np.int64)

        tensor_image = F.to_tensor(image)
        if self.mean is not None and self.std is not None:
            tensor_image = F.normalize(tensor_image, mean=self.mean, std=self.std)

        boxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(cls, dtype=torch.int64)
        return tensor_image, boxes_tensor, labels_tensor


def build_transform_pipeline(
    items: Optional[Sequence[AugConfig]],
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    resize_cfg: Optional[Dict[str, Any]] = None,
) -> DetectionTransformPipeline:
    aug = build_albumentations(items)
    mean_tuple = tuple(mean) if mean else None
    std_tuple = tuple(std) if std else None
    return DetectionTransformPipeline(
        aug=aug,
        mean=mean_tuple,
        std=std_tuple,
        resize_cfg=resize_cfg,
    )
