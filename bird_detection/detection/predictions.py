"""Helpers to convert raw model outputs into API-friendly structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .dataloaders import DatasetMetadata


@dataclass
class ApiPrediction:
    label: str
    score: float
    bbox: Optional[Dict[str, float]]


def _ensure_tensor(value: Optional[torch.Tensor], shape: Sequence[int]) -> torch.Tensor:
    if value is None:
        return torch.empty(*shape)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(value)}")
    return value.detach().cpu()


def _box_to_dict(box: torch.Tensor) -> Dict[str, float]:
    x_min, y_min, x_max, y_max = [float(coord) for coord in box.tolist()]
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def format_prediction_for_api(
    raw_prediction: Dict[str, torch.Tensor],
    metadata: DatasetMetadata,
    score_threshold: float = 0.25,
    max_detections: Optional[int] = None,
    include_background: bool = False,
    not_found_label: str = "not_found",
) -> List[ApiPrediction]:
    """
    Convert torchvision-style detections into API-ready structures.

    Args:
        raw_prediction: model output with ``boxes``, ``labels`` and ``scores`` tensors.
        metadata: dataset metadata with label mappings.
        score_threshold: discard detections below this score.
        max_detections: Optional limit on number of returned detections.
        include_background: keep background detections (id 0) if True.
        not_found_label: label to emit when no predictions survive filtering.

    Returns:
        List of ApiPrediction entries ordered by the incoming scores.
    """

    boxes = _ensure_tensor(raw_prediction.get("boxes"), (0, 4))
    labels = _ensure_tensor(raw_prediction.get("labels"), (0,))
    scores = _ensure_tensor(raw_prediction.get("scores"), (0,))
    count = min(boxes.shape[0], labels.shape[0], scores.shape[0])

    predictions: List[ApiPrediction] = []
    for idx in range(count):
        score = float(scores[idx].item())
        if score < score_threshold:
            continue
        label_id = int(labels[idx].item())
        if label_id == 0 and not include_background:
            continue
        label_name = metadata.id2label.get(label_id, f"class_{label_id}")
        bbox = _box_to_dict(boxes[idx])
        predictions.append(
            ApiPrediction(
                label=label_name,
                score=score,
                bbox=bbox,
            )
        )
        if max_detections and len(predictions) >= max_detections:
            break

    if not predictions:
        predictions.append(ApiPrediction(label=not_found_label, score=1.0, bbox=None))
    return predictions
