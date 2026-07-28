from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "filename",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "confidence",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert EfficientTree pixel predictions from source coordinates "
            "to the GT coordinate system."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-width", type=float, default=640.0)
    parser.add_argument("--source-height", type=float, default=640.0)
    parser.add_argument("--target-width", type=float, default=256.0)
    parser.add_argument("--target-height", type=float, default=256.0)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {args.input}")

    dimensions = (
        args.source_width,
        args.source_height,
        args.target_width,
        args.target_height,
    )
    if min(dimensions) <= 0:
        raise ValueError("All image dimensions must be positive.")

    scale_x = args.target_width / args.source_width
    scale_y = args.target_height / args.source_height

    rows: list[dict[str, object]] = []

    with args.input.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"Unexpected columns in {args.input}: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                filename = Path(
                    str(row["filename"]).strip().strip('"')
                ).name
                class_id = int(float(row["class"]))
                xmin = float(row["xmin"]) * scale_x
                ymin = float(row["ymin"]) * scale_y
                xmax = float(row["xmax"]) * scale_x
                ymax = float(row["ymax"]) * scale_y
                confidence = float(row["confidence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid row {row_number} in {args.input}"
                ) from exc

            xmin = clamp(xmin, 0.0, args.target_width)
            ymin = clamp(ymin, 0.0, args.target_height)
            xmax = clamp(xmax, 0.0, args.target_width)
            ymax = clamp(ymax, 0.0, args.target_height)

            if xmax <= xmin or ymax <= ymin:
                continue

            rows.append(
                {
                    "filename": filename,
                    "class": class_id,
                    "xmin": round(xmin, 2),
                    "ymin": round(ymin, 2),
                    "xmax": round(xmax, 2),
                    "ymax": round(ymax, 2),
                    "confidence": round(confidence, 6),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("PREDICTIONS CONVERTED TO GT COORDINATES")
    print("=" * 72)
    print(
        f"Source: {args.source_width:g}x{args.source_height:g}"
    )
    print(
        f"Target: {args.target_width:g}x{args.target_height:g}"
    )
    print(f"Scale X: {scale_x:.6f}")
    print(f"Scale Y: {scale_y:.6f}")
    print(f"Rows written: {len(rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
