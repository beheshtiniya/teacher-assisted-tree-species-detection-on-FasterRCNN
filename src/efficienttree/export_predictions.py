from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = [
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"
]

CLASS_COLORS = {
    1: (255, 80, 80),
    2: (60, 200, 90),
    3: (70, 140, 255),
    4: (255, 180, 40),
}


def find_image(source_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = source_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
        candidate_upper = source_dir / f"{stem}{suffix.upper()}"
        if candidate_upper.is_file():
            return candidate_upper
    return None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]

    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass

    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO prediction TXT files to pixel XYXY CSV and "
            "draw class 1-4 boxes on the original images."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Optional second confidence filter."
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Source folder not found: {args.source_dir}")

    if not args.labels_dir.is_dir():
        raise NotADirectoryError(f"Prediction labels not found: {args.labels_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    image_count = 0
    box_count = 0
    missing_images: list[str] = []

    font = load_font(16)

    for label_file in sorted(args.labels_dir.glob("*.txt")):
        image_path = find_image(args.source_dir, label_file.stem)

        if image_path is None:
            missing_images.append(label_file.stem)
            continue

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        width, height = image.size
        draw = ImageDraw.Draw(image)

        for line_number, raw_line in enumerate(
            label_file.read_text(encoding="utf-8-sig").splitlines(),
            start=1
        ):
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) not in {5, 6}:
                raise ValueError(
                    f"Invalid prediction at {label_file}:{line_number}: {line}"
                )

            yolo_class = int(float(fields[0]))
            x_center = float(fields[1])
            y_center = float(fields[2])
            box_width = float(fields[3])
            box_height = float(fields[4])
            confidence = float(fields[5]) if len(fields) == 6 else 1.0

            if confidence < args.min_conf:
                continue

            # EfficientTree uses classes 0-3; original CSV uses 1-4.
            class_id = yolo_class + 1

            xmin = (x_center - box_width / 2.0) * width
            ymin = (y_center - box_height / 2.0) * height
            xmax = (x_center + box_width / 2.0) * width
            ymax = (y_center + box_height / 2.0) * height

            xmin = clamp(xmin, 0.0, float(width))
            ymin = clamp(ymin, 0.0, float(height))
            xmax = clamp(xmax, 0.0, float(width))
            ymax = clamp(ymax, 0.0, float(height))

            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            draw.rectangle(
                [xmin, ymin, xmax, ymax],
                outline=color,
                width=3
            )

            label = f"Class {class_id}  {confidence:.2f}"
            text_bbox = draw.textbbox((xmin, ymin), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_y = max(0.0, ymin - text_h - 6)
            draw.rectangle(
                [
                    xmin,
                    label_y,
                    xmin + text_w + 8,
                    label_y + text_h + 6
                ],
                fill=color
            )
            draw.text(
                (xmin + 4, label_y + 3),
                label,
                fill=(0, 0, 0),
                font=font
            )

            rows.append(
                {
                    "filename": image_path.name,
                    "class": class_id,
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2),
                    "confidence": round(confidence, 6),
                }
            )
            box_count += 1

        # PNG is easier to inspect than TIFF and does not alter original images.
        output_image = args.output_dir / f"{image_path.stem}.png"
        image.save(output_image, format="PNG")
        image_count += 1

    fieldnames = [
        "filename",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "confidence",
    ]

    with args.csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("PREDICTION EXPORT COMPLETE")
    print("=" * 72)
    print(f"Annotated images: {image_count}")
    print(f"Saved boxes: {box_count}")
    print(f"Annotated folder: {args.output_dir}")
    print(f"CSV: {args.csv}")

    if missing_images:
        print(f"Missing source images: {len(missing_images)}")
        for name in missing_images[:10]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
