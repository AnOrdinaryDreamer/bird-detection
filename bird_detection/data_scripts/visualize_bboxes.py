"""Render bounding boxes to quickly inspect rescaled/denormalized labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .extract_squirrels import clip, find_image

Bbox = Tuple[float, float, float, float]


def _color_for_label(label: str) -> Tuple[int, int, int]:
    seed = hash(label) & 0xFFFFFFFF
    rng = random.Random(seed)
    return tuple(rng.randint(80, 255) for _ in range(3))


def _text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except AttributeError:  # pragma: no cover - Pillow<9 fallback
        return font.getsize(text)


def _parse_line(
    line: str,
    label_format: str,
    image_size: Tuple[int, int],
) -> Optional[Tuple[str, Bbox]]:
    tokens = line.strip().split()
    if len(tokens) < 5:
        return None
    label = tokens[0]
    width, height = image_size
    if label_format == "pixel":
        x_min = clip(float(tokens[1]), 0.0, float(width))
        y_min = clip(float(tokens[2]), 0.0, float(height))
        x_max = clip(float(tokens[1]) + float(tokens[3]), 0.0, float(width))
        y_max = clip(float(tokens[2]) + float(tokens[4]), 0.0, float(height))
    else:  # YOLO normalized center/width/height
        cx = float(tokens[1]) * width
        cy = float(tokens[2]) * height
        box_w = float(tokens[3]) * width
        box_h = float(tokens[4]) * height
        x_min = clip(cx - box_w / 2.0, 0.0, float(width))
        y_min = clip(cy - box_h / 2.0, 0.0, float(height))
        x_max = clip(cx + box_w / 2.0, 0.0, float(width))
        y_max = clip(cy + box_h / 2.0, 0.0, float(height))
    if x_max <= x_min or y_max <= y_min:
        return None
    return label, (x_min, y_min, x_max, y_max)


def _load_boxes(
    label_path: Path,
    label_format: str,
    image_size: Tuple[int, int],
) -> List[Tuple[str, Bbox]]:
    boxes: List[Tuple[str, Bbox]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_line(line, label_format, image_size)
            if parsed:
                boxes.append(parsed)
    return boxes


def draw_boxes(image: Image.Image, boxes: Sequence[Tuple[str, Bbox]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for label, (x_min, y_min, x_max, y_max) in boxes:
        color = _color_for_label(label)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=2)
        text_w, text_h = _text_size(font, label)
        text_x = max(0, x_min)
        text_y = max(0, y_min - text_h - 2)
        draw.rectangle(
            [(text_x, text_y), (text_x + text_w + 4, text_y + text_h + 2)],
            fill=color,
        )
        draw.text((text_x + 2, text_y + 1), label, fill=(0, 0, 0), font=font)
    return annotated


def visualize_file(
    label_path: Path,
    images_dir: Path,
    output_dir: Path,
    label_format: str,
) -> Optional[Path]:
    image_path = find_image(images_dir, label_path.stem)
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    boxes = _load_boxes(label_path, label_format, image.size)
    if not boxes:
        return None
    annotated = draw_boxes(image, boxes)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label_path.stem}.jpg"
    annotated.save(output_path)
    return output_path


def visualize_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    label_format: str = "pixel",
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[Path]:
    label_files = sorted(labels_dir.glob("*.txt"))
    if shuffle:
        random.shuffle(label_files)
    selected = label_files[:limit] if limit else label_files
    results: List[Path] = []
    for label_path in selected:
        output_path = visualize_file(label_path, images_dir, output_dir, label_format)
        if output_path:
            results.append(output_path)
    return results


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay bounding boxes onto images to inspect denormalization/rescaling.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory with source images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory with label files (*.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bbox_viz"),
        help="Directory to store annotated previews.",
    )
    parser.add_argument(
        "--label-format",
        choices=("pixel", "yolo"),
        default="pixel",
        help="Label encoding: pixel xywh or YOLO normalized center/wh.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render at most N files (default: all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle labels before selecting the limit.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if not args.labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    outputs = visualize_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        label_format=args.label_format,
        limit=args.limit,
        shuffle=args.shuffle,
    )
    if not outputs:
        print("No labels produced drawable boxes.")
    else:
        print(f"Rendered {len(outputs)} previews to {args.output_dir}")


if __name__ == "__main__":
    main()
