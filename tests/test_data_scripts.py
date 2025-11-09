from pathlib import Path

import pytest
from PIL import Image

from bird_detection.data_scripts.extract_cub_birds import (
    cub_bbox_to_normalized,
    export_cub_dataset,
)
from bird_detection.data_scripts.extract_squirrels import (
    clip,
    conditional_resize,
    extract_from_source,
    find_image,
)


def _setup_source(tmp_path, image_size=(80, 40), label_lines=None, stem="sample"):
    source_dir = tmp_path / "source"
    label_dir = source_dir / "labels"
    image_dir = source_dir / "images"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}.jpg"
    Image.new("RGB", image_size, color="white").save(image_path)
    lines = label_lines or ["84 0.5 0.5 0.25 0.5"]
    with (label_dir / f"{stem}.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return source_dir, image_path


def _setup_dest(tmp_path):
    dest_root = tmp_path / "dest"
    dest_image_dir = dest_root / "images"
    dest_label_dir = dest_root / "labels_pixel"
    dest_image_dir.mkdir(parents=True, exist_ok=True)
    dest_label_dir.mkdir(parents=True, exist_ok=True)
    return dest_image_dir, dest_label_dir


def _read_label(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    assert text
    tokens = text.split()
    return tokens[0], list(map(int, tokens[1:]))


def test_extract_from_source_exports_pixel_labels(tmp_path):
    source_dir, _ = _setup_source(
        tmp_path,
        image_size=(80, 40),
        label_lines=["84 0.5 0.5 0.25 0.5"],
    )
    dest_image_dir, dest_label_dir = _setup_dest(tmp_path)

    kept = extract_from_source(
        source_dir,
        dest_image_dir,
        dest_label_dir,
        class_id="84",
        max_dim=None,
    )

    assert kept == 1
    exported_image = dest_image_dir / "sample.jpg"
    exported_label = dest_label_dir / "sample.txt"
    assert exported_image.exists()
    assert exported_label.exists()
    with Image.open(exported_image) as img:
        assert img.size == (80, 40)

    cls, coords = _read_label(exported_label)
    assert cls == "84"
    assert coords == [30, 10, 20, 20]


def test_extract_from_source_resizes_consistently(tmp_path):
    source_dir, _ = _setup_source(
        tmp_path,
        image_size=(80, 40),
        label_lines=["84 0.25 0.5 0.5 0.25"],
    )
    dest_image_dir, dest_label_dir = _setup_dest(tmp_path)

    kept = extract_from_source(
        source_dir,
        dest_image_dir,
        dest_label_dir,
        class_id="84",
        max_dim=40,
    )

    assert kept == 1
    exported_image = dest_image_dir / "sample.jpg"
    with Image.open(exported_image) as img:
        assert img.size == (40, 20)

    _, coords = _read_label(dest_label_dir / "sample.txt")
    x_min, y_min, width, height = coords
    assert (x_min, width) == (0, 20)
    # Height accounts for ceil/floor operations; expect 6px tall.
    assert height == 6
    assert y_min >= 0


def test_extract_from_source_clips_boxes_to_image_bounds(tmp_path):
    label_lines = [
        "84 0.05 0.5 0.6 0.4",  # extends left
        "84 0.95 0.5 0.4 0.4",  # extends right
        "84 0.5 0.05 0.4 0.6",  # extends top/bottom
    ]
    source_dir, _ = _setup_source(
        tmp_path, image_size=(100, 50), label_lines=label_lines
    )
    dest_image_dir, dest_label_dir = _setup_dest(tmp_path)

    extract_from_source(
        source_dir, dest_image_dir, dest_label_dir, class_id="84", max_dim=None
    )

    exported_image = dest_image_dir / "sample.jpg"
    with Image.open(exported_image) as img:
        width, height = img.size

    with (dest_label_dir / "sample.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            _, x, y, w, h = line.split()
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            assert 0 <= x < width
            assert 0 <= y < height
            assert 1 <= w <= width
            assert 1 <= h <= height
            assert x + w <= width
            assert y + h <= height


def test_extract_from_source_label_structure_and_ranges(tmp_path):
    label_lines = [
        "84 0.5 0.5 0.4 0.3",
    ]
    source_dir, _ = _setup_source(
        tmp_path, image_size=(60, 30), label_lines=label_lines
    )
    dest_image_dir, dest_label_dir = _setup_dest(tmp_path)

    extract_from_source(
        source_dir, dest_image_dir, dest_label_dir, class_id="84", max_dim=None
    )

    label_path = dest_label_dir / "sample.txt"
    content = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert content, "label file should not be empty"
    for line in content:
        tokens = line.split()
        assert len(tokens) == 5
        assert tokens[0] == "84"
        numeric = list(map(int, tokens[1:]))
        x, y, w, h = numeric
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0


def test_find_image_prefers_jpg_and_falls_back(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    fallback = image_dir / "item.png"
    Image.new("RGB", (10, 10), color="red").save(fallback)
    chosen = find_image(image_dir, "item")
    assert chosen == fallback
    preferred = image_dir / "item.jpg"
    Image.new("RGB", (10, 10), color="blue").save(preferred)
    assert find_image(image_dir, "item") == preferred


def test_conditional_resize_flags_changes():
    image = Image.new("RGB", (200, 100), color="white")
    unchanged, resized = conditional_resize(image, max_dim=300)
    assert not resized
    assert unchanged.size == (200, 100)
    downsized, resized = conditional_resize(image, max_dim=50)
    assert resized
    assert max(downsized.size) == 50


@pytest.mark.parametrize(
    "value,lower,upper,expected",
    [
        (-1, 0, 10, 0),
        (5, 0, 10, 5),
        (42, 0, 10, 10),
    ],
)
def test_clip_bounds(value, lower, upper, expected):
    assert clip(value, lower, upper) == expected


def _write_lines(path: Path, lines):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _create_cub_dataset(tmp_path, entries):
    data_root = tmp_path / "cub"
    image_dir = data_root / "images"
    class_map = {}
    image_lines = []
    label_lines = []
    box_lines = []

    for entry in entries:
        class_map[entry["class_id"]] = entry["class_name"]
        image_id = entry["image_id"]
        image_rel = entry["relative_path"]
        image_lines.append(f"{image_id} {image_rel}")
        label_lines.append(f"{image_id} {entry['class_id']}")
        x, y, w, h = entry["bbox"]
        box_lines.append(f"{image_id} {x} {y} {w} {h}")
        full_image_path = image_dir / image_rel
        full_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", entry.get("image_size", (100, 50)), color="white").save(
            full_image_path
        )

    data_root.mkdir(parents=True, exist_ok=True)
    class_lines = [f"{cid} {name}" for cid, name in sorted(class_map.items())]

    _write_lines(data_root / "classes.txt", class_lines)
    _write_lines(data_root / "images.txt", image_lines)
    _write_lines(data_root / "image_class_labels.txt", label_lines)
    _write_lines(data_root / "bounding_boxes.txt", box_lines)
    return data_root


def test_cub_bbox_to_normalized_clips_to_image_bounds():
    result = cub_bbox_to_normalized(
        0.5, -5.0, 120.0, 100.0, image_width=100, image_height=50
    )
    assert result == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_export_cub_dataset_creates_per_class_pixel_labels(tmp_path):
    entries = [
        {
            "image_id": 1,
            "relative_path": "001.Class_One/img1.jpg",
            "class_id": 1,
            "class_name": "Class One",
            "bbox": (10.0, 20.0, 30.0, 10.0),
            "image_size": (100, 50),
        }
    ]
    data_root = _create_cub_dataset(tmp_path, entries)
    dest_root = tmp_path / "export"

    export_cub_dataset(
        data_root=data_root,
        dest_root=dest_root,
        include_tokens=[],
        clear_dest=True,
        max_dim=None,
    )

    class_dir = dest_root / "000_Class_One"
    exported_image = class_dir / "images" / "img1.jpg"
    exported_label = class_dir / "labels_pixel" / "img1.txt"
    assert exported_image.exists()
    assert exported_label.exists()
    label_text = exported_label.read_text(encoding="utf-8").strip()
    assert label_text == "0 9 19 30 10"


def test_export_cub_dataset_respects_include_filters(tmp_path):
    entries = [
        {
            "image_id": 1,
            "relative_path": "001.Class_One/img1.jpg",
            "class_id": 1,
            "class_name": "Class One",
            "bbox": (10.0, 10.0, 20.0, 20.0),
            "image_size": (80, 80),
        },
        {
            "image_id": 2,
            "relative_path": "002.Class_Two/img2.jpg",
            "class_id": 2,
            "class_name": "Class Two",
            "bbox": (5.0, 5.0, 10.0, 10.0),
            "image_size": (80, 80),
        },
    ]
    data_root = _create_cub_dataset(tmp_path, entries)
    dest_root = tmp_path / "export"

    export_cub_dataset(
        data_root=data_root,
        dest_root=dest_root,
        include_tokens=["Class Two"],
        clear_dest=True,
        max_dim=None,
    )

    class_one_dir = dest_root / "000_Class_One"
    class_two_dir = dest_root / "000_Class_Two"
    assert not class_one_dir.exists()
    assert class_two_dir.exists()
    labels = list((class_two_dir / "labels_pixel").glob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8").strip().startswith("0 ")

    mapping_csv = (dest_root / "class_mapping.csv").read_text(encoding="utf-8")
    assert "Class Two" in mapping_csv
    assert "Class One" not in mapping_csv


def test_export_cub_dataset_writes_mapping_with_counts(tmp_path):
    entries = [
        {
            "image_id": 1,
            "relative_path": "001.First/img1.jpg",
            "class_id": 3,
            "class_name": "First Bird",
            "bbox": (5.0, 10.0, 20.0, 10.0),
            "image_size": (80, 60),
        },
        {
            "image_id": 2,
            "relative_path": "001.First/img2.jpg",
            "class_id": 3,
            "class_name": "First Bird",
            "bbox": (10.0, 10.0, 5.0, 5.0),
            "image_size": (80, 60),
        },
    ]
    data_root = _create_cub_dataset(tmp_path, entries)
    dest_root = tmp_path / "export"

    export_cub_dataset(
        data_root=data_root,
        dest_root=dest_root,
        include_tokens=[],
        clear_dest=True,
        max_dim=None,
    )

    mapping_path = dest_root / "class_mapping.csv"
    assert mapping_path.exists()
    rows = mapping_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "class_index,class_id,class_name,sample_count"
    assert rows[1].endswith(",First Bird,2"), "sample count should equal exported files"
