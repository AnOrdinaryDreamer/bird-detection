"""Dataset and dataloader utilities for bird / squirrel detection."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentations import DetectionTransformPipeline

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 10
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass
class SampleRecord:
    image_path: Path
    boxes: np.ndarray  # (N, 4) in xyxy format
    labels: np.ndarray  # (N,)
    width: int
    height: int
    image_id: int


@dataclass
class DatasetMetadata:
    id2label: Dict[int, str]
    label2id: Dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.id2label)


def _read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return width, height


def _load_boxes(
    label_path: Path,
    width: int,
    height: int,
    class_mapping: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    boxes: List[List[float]] = []
    labels: List[int] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            class_idx = int(float(tokens[0]))
            if class_idx not in class_mapping:
                continue
            x_min = float(tokens[1])
            y_min = float(tokens[2])
            w = float(tokens[3])
            h = float(tokens[4])
            if w <= 1 or h <= 1:
                continue
            x_max = min(x_min + w, width)
            y_max = min(y_min + h, height)
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_mapping[class_idx])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _load_bird_mapping(csv_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[int(row["class_index"])] = row["class_name"]
    return mapping


def _gather_bird_records(
    root: Path,
    class_mapping: Dict[int, int],
    image_id_start: int,
) -> Tuple[List[SampleRecord], int]:
    records: List[SampleRecord] = []
    image_id = image_id_start
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        image_dir = class_dir / "images"
        label_dir = class_dir / "labels_pixel"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            width, height = _read_image_size(image_path)
            boxes, labels = _load_boxes(label_path, width, height, class_mapping)
            if boxes.shape[0] == 0:
                continue
            records.append(
                SampleRecord(
                    image_path=image_path,
                    boxes=boxes,
                    labels=labels,
                    width=width,
                    height=height,
                    image_id=image_id,
                )
            )
            image_id += 1
    return records, image_id


def _gather_squirrel_records(
    root: Path,
    class_mapping: Dict[int, int],
    image_id_start: int,
) -> Tuple[List[SampleRecord], int]:
    records: List[SampleRecord] = []
    image_id = image_id_start
    image_dir = root / "images"
    label_dir = root / "labels_pixel"
    if not image_dir.exists() or not label_dir.exists():
        # try legacy folder name
        label_dir = root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return records, image_id
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            alternatives = list(image_dir.glob(f"{label_path.stem}.*"))
            if not alternatives:
                continue
            image_path = alternatives[0]
        width, height = _read_image_size(image_path)
        boxes, labels = _load_boxes(label_path, width, height, class_mapping)
        if boxes.shape[0] == 0:
            continue
        records.append(
            SampleRecord(
                image_path=image_path,
                boxes=boxes,
                labels=labels,
                width=width,
                height=height,
                image_id=image_id,
            )
        )
        image_id += 1
    return records, image_id


def _build_class_mappings(
    bird_mapping_csv: Path,
    include_squirrels: bool,
    squirrel_source_label: int,
) -> Tuple[Dict[int, int], Dict[int, int], DatasetMetadata]:
    bird_mapping = _load_bird_mapping(bird_mapping_csv)
    id2label: Dict[int, str] = {0: "__background__"}
    label2id: Dict[str, int] = {"__background__": 0}

    bird_label_map: Dict[int, int] = {}
    for class_index in sorted(bird_mapping.keys()):
        next_id = len(id2label)
        id2label[next_id] = bird_mapping[class_index]
        label2id[bird_mapping[class_index]] = next_id
        bird_label_map[class_index] = next_id

    squirrel_label_map: Dict[int, int] = {}
    if include_squirrels:
        squirrel_id = len(id2label)
        id2label[squirrel_id] = "squirrel"
        label2id["squirrel"] = squirrel_id
        squirrel_label_map[squirrel_source_label] = squirrel_id

    metadata = DatasetMetadata(id2label=id2label, label2id=label2id)
    return bird_label_map, squirrel_label_map, metadata


def _stratified_split_records(
    records: Sequence[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    test_ratio: Optional[float] = None,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    total = train_ratio + val_ratio + test_ratio
    if total > 1.0 + 1e-6:
        raise ValueError("Train + val + test split ratio cannot exceed 1.0")

    by_label: Dict[int, List[SampleRecord]] = {}
    for sample in records:
        if sample.labels.size == 0:
            label = 0
        else:
            label = int(sample.labels[0])
        by_label.setdefault(label, []).append(sample)

    rng = random.Random(seed)
    train_split: List[SampleRecord] = []
    val_split: List[SampleRecord] = []
    test_split: List[SampleRecord] = []

    for samples in by_label.values():
        rng.shuffle(samples)
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        test_end = val_end + int(n * test_ratio)
        train_split.extend(samples[:train_end])
        val_split.extend(samples[train_end:val_end])
        test_split.extend(samples[val_end:test_end])
        test_split.extend(samples[test_end:])  # остаток из-за округления тоже в test

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    rng.shuffle(test_split)
    return train_split, val_split, test_split


def _sample_weights(samples: Sequence[SampleRecord]) -> List[float]:
    label_counts: Dict[int, int] = {}
    for sample in samples:
        if sample.labels.size == 0:
            label = 0
        else:
            label = int(sample.labels[0])
        label_counts[label] = label_counts.get(label, 0) + 1

    weights: List[float] = []
    for sample in samples:
        if sample.labels.size == 0:
            label = 0
        else:
            label = int(sample.labels[0])
        weights.append(1.0 / label_counts[label])
    return weights


class BirdSquirrelDetectionDataset(Dataset):
    """Dataset that optionally applies Mosaic and CutMix augmentations."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        transform: DetectionTransformPipeline,
        mosaic_cfg: Optional[Dict[str, float]] = None,
        cutmix_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.mosaic_cfg = mosaic_cfg or {}
        self.cutmix_cfg = cutmix_cfg or {}
        self.mosaic_enabled = bool(self.mosaic_cfg.get("enabled", False))
        self.cutmix_enabled = bool(self.cutmix_cfg.get("enabled", False))
        self.mosaic_prob = (
            float(self.mosaic_cfg.get("prob", 0.0)) if self.mosaic_enabled else 0.0
        )
        self.mosaic_size = int(self.mosaic_cfg.get("output_size", 1024))
        self.cutmix_prob = (
            float(self.cutmix_cfg.get("prob", 0.0)) if self.cutmix_enabled else 0.0
        )
        self.cutmix_alpha = float(self.cutmix_cfg.get("alpha", 1.0))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, record: SampleRecord) -> np.ndarray:
        with Image.open(record.image_path) as img:
            return np.array(img.convert("RGB"))

    def _resize_image_and_boxes(
        self, image: np.ndarray, boxes: np.ndarray, new_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_w, target_h = new_size
        img = Image.fromarray(image).resize((target_w, target_h), RESAMPLE_BILINEAR)
        scale_x = target_w / image.shape[1]
        scale_y = target_h / image.shape[0]
        scaled = boxes.copy()
        if scaled.shape[0] > 0:
            scaled[:, [0, 2]] *= scale_x
            scaled[:, [1, 3]] *= scale_y
        return np.array(img), scaled

    def _sample_indices(self, count: int, exclude: int) -> List[int]:
        choices = list(range(len(self.samples)))
        choices.remove(exclude)
        return random.sample(choices, min(count, len(choices)))

    def _apply_mosaic(self, base_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_indices = [base_idx] + self._sample_indices(3, base_idx)
        tile_size = self.mosaic_size // 2
        canvas = np.zeros((self.mosaic_size, self.mosaic_size, 3), dtype=np.uint8)
        offsets = [
            (0, 0),
            (0, tile_size),
            (tile_size, 0),
            (tile_size, tile_size),
        ]
        combined_boxes: List[np.ndarray] = []
        combined_labels: List[np.ndarray] = []
        for idx, (top, left) in zip(candidate_indices, offsets):
            record = self.samples[idx]
            image = self._load_image(record)
            image, boxes = self._resize_image_and_boxes(
                image, record.boxes, (tile_size, tile_size)
            )
            canvas[top : top + tile_size, left : left + tile_size] = image
            if boxes.shape[0] > 0:
                shifted = boxes.copy()
                shifted[:, [0, 2]] += left
                shifted[:, [1, 3]] += top
                combined_boxes.append(shifted)
                combined_labels.append(record.labels.copy())
        if combined_boxes:
            boxes = np.concatenate(combined_boxes, axis=0)
            labels = np.concatenate(combined_labels, axis=0)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        return canvas, boxes, labels

    def _apply_cutmix(self, base_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        other_indices = self._sample_indices(1, base_idx)
        if not other_indices:
            record = self.samples[base_idx]
            image = self._load_image(record)
            return image, record.boxes.copy(), record.labels.copy()
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        record_a = self.samples[base_idx]
        record_b = self.samples[other_indices[0]]
        image_a = self._load_image(record_a)
        image_b = self._load_image(record_b)
        target_size = (
            max(image_a.shape[1], image_b.shape[1]),
            max(image_a.shape[0], image_b.shape[0]),
        )
        image_a, boxes_a = self._resize_image_and_boxes(
            image_a, record_a.boxes, target_size
        )
        image_b, boxes_b = self._resize_image_and_boxes(
            image_b, record_b.boxes, target_size
        )
        blended = (lam * image_a + (1 - lam) * image_b).astype(np.uint8)
        boxes = np.concatenate([boxes_a, boxes_b], axis=0)
        labels = np.concatenate([record_a.labels, record_b.labels], axis=0)
        return blended, boxes, labels

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        image = self._load_image(record)
        boxes = record.boxes.copy()
        labels = record.labels.copy()

        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            image, boxes, labels = self._apply_mosaic(idx)
        elif self.cutmix_prob > 0 and random.random() < self.cutmix_prob:
            image, boxes, labels = self._apply_cutmix(idx)

        tensor_image, tensor_boxes, tensor_labels = self.transform(
            image=image,
            boxes=boxes,
            labels=labels,
        )

        areas = (tensor_boxes[:, 3] - tensor_boxes[:, 1]) * (
            tensor_boxes[:, 2] - tensor_boxes[:, 0]
        )
        target = {
            "boxes": tensor_boxes,
            "labels": tensor_labels,
            "image_id": torch.tensor([record.image_id]),
            "area": areas,
            "iscrowd": torch.zeros((tensor_labels.shape[0],), dtype=torch.int64),
        }

        return tensor_image, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(
    data_cfg: Dict[str, Any],
    train_transform: DetectionTransformPipeline,
    eval_transform: DetectionTransformPipeline,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    advanced_train_aug: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    """Construct dataloaders for train/val/test splits."""
    birds_dir = Path(data_cfg["birds_dir"])
    mapping_csv = Path(data_cfg["birds_mapping_csv"])
    include_squirrels = bool(data_cfg.get("include_squirrels", False))
    squirrel_dir = data_cfg.get("squirrels_dir")
    squirrel_label_id = int(data_cfg.get("squirrel_label_id", 84))

    (
        bird_label_map,
        squirrel_label_map,
        metadata,
    ) = _build_class_mappings(
        bird_mapping_csv=mapping_csv,
        include_squirrels=include_squirrels,
        squirrel_source_label=squirrel_label_id,
    )

    records: List[SampleRecord] = []
    birds, next_id = _gather_bird_records(birds_dir, bird_label_map, image_id_start=0)
    records.extend(birds)

    if include_squirrels and squirrel_dir:
        squirrel_path = Path(squirrel_dir)
        squirrels, next_id = _gather_squirrel_records(
            squirrel_path, squirrel_label_map, next_id
        )
        records.extend(squirrels)

    if not records:
        raise RuntimeError("No samples found for training. Check data paths.")

    train_ratio = float(data_cfg.get("train_split", 0.8))
    val_ratio = float(data_cfg.get("val_split", 0.1))
    test_ratio = data_cfg.get("test_split")
    if test_ratio is not None:
        test_ratio = float(test_ratio)
    train_samples, val_samples, test_samples = _stratified_split_records(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        test_ratio=test_ratio,
    )

    mosaic_cfg = (advanced_train_aug or {}).get("mosaic")
    cutmix_cfg = (advanced_train_aug or {}).get("cutmix")

    train_dataset = BirdSquirrelDetectionDataset(
        train_samples,
        transform=train_transform,
        mosaic_cfg=mosaic_cfg,
        cutmix_cfg=cutmix_cfg,
    )
    val_dataset = BirdSquirrelDetectionDataset(
        val_samples,
        transform=eval_transform,
    )
    test_dataset = BirdSquirrelDetectionDataset(
        test_samples,
        transform=eval_transform,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate,
    )

    train_weights = _sample_weights(train_samples)
    train_sampler = WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True
    )
    train_loader = DataLoader(
        train_dataset, shuffle=False, sampler=train_sampler, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, metadata
