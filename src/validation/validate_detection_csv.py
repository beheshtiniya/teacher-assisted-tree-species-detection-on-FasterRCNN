from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_COLUMNS = [
    "filename",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "confidence",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate final combined GT + prediction CSV."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--width", type=float, default=256.0)
    parser.add_argument("--height", type=float, default=256.0)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)

    total = 0
    gt_rows = 0
    prediction_rows = 0

    with args.csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            total += 1

            class_id = int(float(row["class"]))
            confidence = float(row["confidence"])
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            if class_id not in {1, 2, 3, 4}:
                raise ValueError(
                    f"Invalid class at row {row_number}"
                )

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"Invalid confidence at row {row_number}"
                )

            if not (
                0.0 <= xmin < xmax <= args.width
                and 0.0 <= ymin < ymax <= args.height
            ):
                raise ValueError(
                    f"Box outside {args.width:g}x{args.height:g} "
                    f"at row {row_number}: "
                    f"{xmin}, {ymin}, {xmax}, {ymax}"
                )

            if abs(confidence - 1.0) < 1e-9:
                gt_rows += 1
            else:
                prediction_rows += 1

    print(
        f"{args.split}: PASS | total={total} | "
        f"GT={gt_rows} | predictions={prediction_rows} | "
        f"coordinates within {args.width:g}x{args.height:g}"
    )


if __name__ == "__main__":
    main()
