import json
from pathlib import Path

import torch

from bird_detection.detection.dataloaders import DatasetMetadata
from bird_detection.detection.inference import load_detector
from bird_detection.detection.models import build_model


def test_load_detector_roundtrip(tmp_path):
    metadata = DatasetMetadata(
        id2label={0: "__background__", 1: "bird"},
        label2id={"__background__": 0, "bird": 1},
    )
    model_cfg = {
        "name": "ssd300_vgg16",
        "pretrained": False,
        "pretrained_backbone": False,
        "freeze_backbone": False,
        "weights": None,
        "weights_backbone": None,
    }
    config = {
        "model": model_cfg,
        "id2label": metadata.id2label,
        "label2id": metadata.label2id,
    }
    model_dir = Path(tmp_path)
    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle)

    reference_model = build_model(model_cfg, metadata.num_classes)
    torch.save(reference_model.state_dict(), model_dir / "pytorch_model.bin")

    loaded_model, loaded_metadata, loaded_cfg = load_detector(model_dir)

    assert loaded_metadata.id2label == metadata.id2label
    assert loaded_metadata.label2id == metadata.label2id
    assert loaded_cfg["model"]["name"] == "ssd300_vgg16"
    assert set(loaded_model.state_dict().keys()) == set(
        reference_model.state_dict().keys()
    )
