"""Utility helpers for training scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .dataloaders import DatasetMetadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_pretrained(
    model: torch.nn.Module,
    save_directory: Path,
    metadata: DatasetMetadata,
    full_config: Dict[str, Any],
) -> None:
    """Persist model weights and metadata following Hugging Face conventions."""
    save_directory.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_directory / "pytorch_model.bin")

    config = dict(full_config)
    config.setdefault("id2label", metadata.id2label)
    config.setdefault("label2id", metadata.label2id)
    with (save_directory / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    with (save_directory / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"id2label": metadata.id2label, "label2id": metadata.label2id},
            handle,
            indent=2,
            ensure_ascii=False,
        )
