"""Helpers to convert raw model outputs into API-friendly structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from .dataloaders import DatasetMetadata


@dataclass
class ClassAlternative:
    label: str
    score: float


@dataclass
class ApiPrediction:
    label: str
    score: float
    bbox: Optional[Dict[str, float]]
    alternatives: Optional[List[ClassAlternative]] = None


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
    labels = _ensure_tensor(raw_prediction.get("labels"), (0,)).to(torch.int64)
    scores = _ensure_tensor(raw_prediction.get("scores"), (0,))
    topk_labels = _ensure_tensor(raw_prediction.get("topk_labels"), (0, 0)).to(
        torch.int64
    )
    topk_scores = _ensure_tensor(raw_prediction.get("topk_scores"), (0, 0))
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
        alternatives = _build_alternatives(
            idx,
            topk_labels,
            topk_scores,
            metadata,
            include_background=include_background,
        )
        predictions.append(
            ApiPrediction(
                label=label_name,
                score=score,
                bbox=bbox,
                alternatives=alternatives,
            )
        )
        if max_detections and len(predictions) >= max_detections:
            break

    if not predictions:
        predictions.append(ApiPrediction(label=not_found_label, score=1.0, bbox=None))
    return predictions


def _build_alternatives(
    det_index: int,
    topk_labels: torch.Tensor,
    topk_scores: torch.Tensor,
    metadata: DatasetMetadata,
    include_background: bool,
) -> Optional[List[ClassAlternative]]:
    if topk_labels.ndim != 2 or topk_scores.ndim != 2:
        return None
    if det_index >= topk_labels.shape[0]:
        return None
    alt_count = min(topk_labels.shape[1], topk_scores.shape[1])
    if alt_count == 0:
        return None
    alternatives: List[ClassAlternative] = []
    for alt_idx in range(alt_count):
        label_id = int(topk_labels[det_index, alt_idx].item())
        if label_id == 0 and not include_background:
            continue
        score = float(topk_scores[det_index, alt_idx].item())
        label_name = metadata.id2label.get(label_id, f"class_{label_id}")
        alternatives.append(ClassAlternative(label=label_name, score=score))
    return alternatives or None
