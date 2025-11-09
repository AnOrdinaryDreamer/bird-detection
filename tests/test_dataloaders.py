from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pytest

from bird_detection.detection.augmentations import DetectionTransformPipeline
from bird_detection.detection.dataloaders import (
    BirdSquirrelDetectionDataset,
    SampleRecord,
    detection_collate,
    _build_class_mappings,
)


def _make_image(path: Path, size=(32, 18), color=(50, 60, 70)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def _make_record(tmp_path, name: str, boxes, labels, image_size=(32, 18), image_id=0):
    image_path = _make_image(tmp_path / f"{name}.jpg", size=image_size)
    if boxes:
        box_array = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    else:
        box_array = np.zeros((0, 4), dtype=np.float32)
    return SampleRecord(
        image_path=image_path,
        boxes=box_array,
        labels=np.array(labels, dtype=np.int64),
        width=image_size[0],
        height=image_size[1],
        image_id=image_id,
    )


def _pipeline():
    return DetectionTransformPipeline(aug=None, mean=None, std=None)


def test_dataset_returns_expected_structure(tmp_path):
    record = _make_record(
        tmp_path,
        "sample",
        boxes=[[1.0, 2.0, 10.0, 12.0]],
        labels=[3],
        image_size=(40, 20),
        image_id=42,
    )
    dataset = BirdSquirrelDetectionDataset([record], transform=_pipeline())

    image_tensor, target = dataset[0]

    assert image_tensor.shape == (3, 20, 40)
    assert image_tensor.dtype == torch.float32
    assert 0.0 <= image_tensor.min().item() <= 1.0
    assert 0.0 <= image_tensor.max().item() <= 1.0

    assert set(target.keys()) == {"boxes", "labels", "image_id", "area", "iscrowd"}
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["image_id"].item() == 42
    assert target["boxes"].shape == (1, 4)
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert 0 <= x1 < x2 <= 40
    assert 0 <= y1 < y2 <= 20
    assert target["area"][0].item() == pytest.approx((y2 - y1) * (x2 - x1))


def test_detection_collate_preserves_feature_lists(tmp_path):
    record_a = _make_record(tmp_path, "a", boxes=[[0, 0, 5, 5]], labels=[1], image_id=1)
    record_b = _make_record(tmp_path, "b", boxes=[], labels=[], image_id=2)
    dataset = BirdSquirrelDetectionDataset([record_a, record_b], transform=_pipeline())

    batch = [dataset[0], dataset[1]]
    images, targets = detection_collate(batch)

    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2
    assert targets[0]["labels"].tolist() == [1]
    assert targets[1]["labels"].numel() == 0


def test_build_class_mappings_assigns_unique_ids(tmp_path):
    mapping_csv = tmp_path / "mapping.csv"
    mapping_csv.write_text(
        "class_index,class_name\n1,Bird One\n2,Bird Two\n",
        encoding="utf-8",
    )

    bird_map, squirrel_map, metadata = _build_class_mappings(
        bird_mapping_csv=mapping_csv,
        include_squirrels=True,
        squirrel_source_label=84,
    )

    assert bird_map == {1: 1, 2: 2}
    assert squirrel_map == {84: 3}
    assert metadata.id2label[0] == "__background__"
    assert metadata.id2label[3] == "squirrel"
    assert len(set(metadata.id2label.values())) == len(metadata.id2label)
