import argparse
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def conditional_resize(
    image: Image.Image, max_dim: Optional[int]
) -> Tuple[Image.Image, bool]:
    """Downscale image if its longer side exceeds max_dim."""
    if max_dim is None:
        return image, False
    longer_side = max(image.size)
    if longer_side <= max_dim:
        return image, False
    scale = max_dim / float(longer_side)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(new_size, Image.LANCZOS), True


def find_image(image_dir: Path, stem: str) -> Path:
    """Return the first image whose stem matches the label stem."""
    default_candidate = image_dir / f"{stem}.jpg"
    if default_candidate.exists():
        return default_candidate

    matches = list(image_dir.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate image for label '{stem}' in {image_dir}"
        )
    return matches[0]


def extract_from_source(
    source_dir: Path,
    dest_image_dir: Path,
    dest_pixel_label_dir: Path,
    class_id: str,
    max_dim: Optional[int],
) -> int:
    label_dir = source_dir / "labels"
    image_dir = source_dir / "images"
    if not label_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(
            f"Expected YOLO directory structure (images/labels) under {source_dir}"
        )

    kept = 0
    for label_file in label_dir.glob("*.txt"):
        with label_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        selected = [line for line in lines if line.split()[0] == class_id]
        if not selected:
            continue

        image_path = find_image(image_dir, label_file.stem)
        with Image.open(image_path) as image:
            processed_image, resized = conditional_resize(image, max_dim)
            width, height = processed_image.size

            dest_image_path = dest_image_dir / image_path.name
            if resized:
                processed_image.save(dest_image_path)
            else:
                shutil.copy2(image_path, dest_image_path)

        pixel_lines = []
        for line in selected:
            parts = line.split()
            _, cx, cy, bw, bh = parts[:5]
            center_x = float(cx) * width
            center_y = float(cy) * height
            box_width = float(bw) * width
            box_height = float(bh) * height

            x1 = clip(center_x - box_width / 2.0, 0.0, width)
            y1 = clip(center_y - box_height / 2.0, 0.0, height)
            x2 = clip(center_x + box_width / 2.0, 0.0, width)
            y2 = clip(center_y + box_height / 2.0, 0.0, height)

            x_min = int(math.floor(x1))
            y_min = int(math.floor(y1))
            x_max = int(math.ceil(x2))
            y_max = int(math.ceil(y2))

            width_px = max(1, x_max - x_min)
            height_px = max(1, y_max - y_min)
            pixel_lines.append(f"{parts[0]} {x_min} {y_min} {width_px} {height_px}")

        with (dest_pixel_label_dir / label_file.name).open(
            "w", encoding="utf-8"
        ) as pixel_out:
            pixel_out.write("\n".join(pixel_lines) + "\n")

        kept += 1

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract YOLO labels for a target class and export pixel-space boxes."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/yolo_rodents/OID_YOLOv8_Dataset"),
        help="Root directory of the YOLO dataset containing split folders.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("bird_detection/data/selected/squirrels"),
        help="Destination directory for the filtered subset (images/ and labels_pixel/).",
    )
    parser.add_argument(
        "--class-id",
        type=str,
        default="84",
        help="YOLO class identifier to extract.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=("train", "val"),
        help="Relative directories under the source root that contain images/ and labels/ folders.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Clear destination directory before extraction.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=500,
        help="If > 0, downscale images so the longer side <= max_dim before saving.",
    )
    args = parser.parse_args()

    if args.clear_dest and args.dest_root.exists():
        shutil.rmtree(args.dest_root)

    dest_image_dir = args.dest_root / "images"
    dest_pixel_label_dir = args.dest_root / "labels_pixel"
    dest_image_dir.mkdir(parents=True, exist_ok=True)
    dest_pixel_label_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for source in args.sources:
        source_dir = args.source_root / source
        kept = extract_from_source(
            source_dir,
            dest_image_dir,
            dest_pixel_label_dir,
            args.class_id,
            args.max_dim if args.max_dim and args.max_dim > 0 else None,
        )
        print(f"{source}: kept {kept} label files for class {args.class_id}")
        total += kept

    print(f"Total files kept: {total}")


if __name__ == "__main__":
    main()
