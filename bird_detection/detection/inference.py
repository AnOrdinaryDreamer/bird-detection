"""Utilities for loading trained detectors for inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .dataloaders import DatasetMetadata
from .models import build_model
from .topk import enable_topk_predictions


def _load_metadata(cfg: Dict[str, Any]) -> DatasetMetadata:
    id2label_raw = cfg.get("id2label") or {}
    label2id_raw = cfg.get("label2id") or {}

    id2label = {int(k): v for k, v in id2label_raw.items()}
    label2id = {str(k): int(v) for k, v in label2id_raw.items()}
    if not id2label:
        # ensure background exists even if config is older
        id2label[0] = "__background__"
        label2id["__background__"] = 0
    return DatasetMetadata(id2label=id2label, label2id=label2id)


def load_detector(
    model_dir: Path | str,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, DatasetMetadata, Dict[str, Any]]:
    """
    Load a trained detector (weights + metadata) from ``save_pretrained`` format.

    Args:
        model_dir: Directory containing ``pytorch_model.bin`` and ``config.json``.
        device: Torch device to place the model on.

    Returns:
        Tuple of (model, metadata, full_config).
    """

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing pytorch_model.bin in {model_dir}")

    with config_path.open("r", encoding="utf-8") as handle:
        full_cfg: Dict[str, Any] = json.load(handle)

    metadata = _load_metadata(full_cfg)
    model_cfg = full_cfg.get("model")
    if not model_cfg:
        raise KeyError("Model configuration not found within config.json")

    model = build_model(model_cfg, metadata.num_classes)
    topk = model_cfg.get("prediction_topk")
    if topk is None:
        topk = 3
    enable_topk_predictions(model, int(topk))
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata, full_cfg
