import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from PIL import Image


def load_mapping(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            idx_str, value = line.strip().split(" ", maxsplit=1)
            mapping[int(idx_str)] = value
    return mapping


def load_float_mapping(path: Path) -> Dict[int, List[float]]:
    mapping: Dict[int, List[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            idx = int(tokens[0])
            values = [float(x) for x in tokens[1:]]
            mapping[idx] = values
    return mapping


def sanitize_class_name(name: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "class"


def resolve_exclusions(classes: Dict[int, str], exclude_tokens: Iterable[str]) -> Set[int]:
    normalized = {cid: name.lower() for cid, name in classes.items()}
    sanitized = {cid: sanitize_class_name(name).lower() for cid, name in classes.items()}

    excluded: Set[int] = set()
    for token in exclude_tokens:
        key = token.strip().lower()
        if not key:
            continue
        if key.isdigit():
            excluded.add(int(key))
            continue
        matches = [
            cid
            for cid, name in normalized.items()
            if key == name or key in name
        ]
        if not matches:
            matches = [
                cid
                for cid, value in sanitized.items()
                if key == value or key in value
            ]
        excluded.update(matches)
    return excluded


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def cub_bbox_formats(
    x: float, y: float, width: float, height: float, image_width: int, image_height: int
) -> List[float]:
    # CUB bounding boxes use 1-based inclusive coordinates, adjust to 0-based.
    x1 = x - 1.0
    y1 = y - 1.0
    x2 = x1 + width
    y2 = y1 + height

    x1 = clip(x1, 0.0, float(image_width))
    y1 = clip(y1, 0.0, float(image_height))
    x2 = clip(x2, 0.0, float(image_width))
    y2 = clip(y2, 0.0, float(image_height))

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width == 0 or box_height == 0:
        raise ValueError("Degenerate bounding box after clipping.")

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return [
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
        x1,
        y1,
        box_width,
        box_height,
    ]


def export_cub_dataset(
    data_root: Path,
    dest_root: Path,
    exclude_tokens: Iterable[str],
    clear_dest: bool,
) -> None:
    metadata_dir = data_root
    image_dir = data_root / "images"

    classes = load_mapping(metadata_dir / "classes.txt")
    images = load_mapping(metadata_dir / "images.txt")
    labels = load_mapping(metadata_dir / "image_class_labels.txt")
    boxes = load_float_mapping(metadata_dir / "bounding_boxes.txt")

    excluded_ids = resolve_exclusions(classes, exclude_tokens)

    kept_class_ids = sorted(
        {int(cid) for cid in labels.values() if int(cid) not in excluded_ids}
    )
    if not kept_class_ids:
        raise RuntimeError("No classes remaining after applying exclusions.")

    class_id_to_index = {cid: idx for idx, cid in enumerate(kept_class_ids)}

    if clear_dest and dest_root.exists():
        shutil.rmtree(dest_root)
    ensure_dir(dest_root)

    class_counts = defaultdict(int)

    for image_id_str, relative_path in images.items():
        image_id = int(image_id_str)
        class_id = int(labels[image_id])
        if class_id in excluded_ids:
            continue

        bbox_values = boxes.get(image_id)
        if not bbox_values:
            continue

        class_index = class_id_to_index[class_id]
        class_name = classes[class_id]
        class_dir = dest_root / f"{class_index:03d}_{sanitize_class_name(class_name)}"
        image_dest_dir = class_dir / "images"
        label_dest_dir = class_dir / "labels"
        label_pixel_dir = class_dir / "labels_pixel"
        ensure_dir(image_dest_dir)
        ensure_dir(label_dest_dir)
        ensure_dir(label_pixel_dir)

        source_image_path = image_dir / relative_path
        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing image file: {source_image_path}")

        with Image.open(source_image_path) as img:
            width, height = img.size

        try:
            (
                center_x,
                center_y,
                box_width,
                box_height,
                x_min,
                y_min,
                width_px,
                height_px,
            ) = cub_bbox_formats(
                *bbox_values, width, height
            )
        except ValueError:
            continue

        dest_image_path = image_dest_dir / source_image_path.name
        shutil.copy2(source_image_path, dest_image_path)

        label_path = label_dest_dir / f"{source_image_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as label_file:
            label_file.write(
                f"{class_index} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            )

        pixel_label_path = label_pixel_dir / f"{source_image_path.stem}.txt"
        with pixel_label_path.open("w", encoding="utf-8") as pixel_label_file:
            pixel_label_file.write(
                f"{class_index} {x_min:.2f} {y_min:.2f} {width_px:.2f} {height_px:.2f}\n"
            )

        class_counts[class_id] += 1

    mapping_path = dest_root / "class_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class_index", "class_id", "class_name", "sample_count"])
        for class_id in kept_class_ids:
            writer.writerow(
                [
                    class_id_to_index[class_id],
                    class_id,
                    classes[class_id],
                    class_counts[class_id],
                ]
            )

    print(f"Export complete. {sum(class_counts.values())} images processed.")
    print(f"Class mapping saved to {mapping_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CUB-200-2011 bounding boxes in YOLO and pixel formats with per-class folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/cub birds/CUB_200_2011/CUB_200_2011"),
        help="Directory containing CUB metadata and images folders.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("bird_detection/data/selected/birds"),
        help="Destination directory for per-class data (images/, labels/, labels_pixel/).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Class ids or substrings of class names to exclude.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Remove destination directory before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_cub_dataset(
        data_root=args.data_root,
        dest_root=args.dest_root,
        exclude_tokens=args.exclude,
        clear_dest=args.clear_dest,
    )


if __name__ == "__main__":
    main()
