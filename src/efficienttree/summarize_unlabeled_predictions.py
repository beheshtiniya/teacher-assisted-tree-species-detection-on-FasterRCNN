from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create image-level statistics and a no-detection list for "
            "unlabeled EfficientTree pseudo-labels."
        )
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--no-detection-list", type=Path, required=True)
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)

    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)

    image_names = sorted(
        path.name
        for path in args.images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    counts: Counter[str] = Counter()

    with args.predictions.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(
                f"Prediction CSV has no filename column: {args.predictions}"
            )

        for row in reader:
            filename = Path(
                str(row["filename"]).strip().strip('"')
            ).name.lower()
            counts[filename] += 1

    summary_rows = []
    no_detection = []

    for filename in image_names:
        box_count = counts[filename.lower()]

        summary_rows.append(
            {
                "filename": filename,
                "pseudo_label_count": box_count,
                "has_prediction": int(box_count > 0),
            }
        )

        if box_count == 0:
            no_detection.append(filename)

    args.summary.parent.mkdir(parents=True, exist_ok=True)

    with args.summary.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "pseudo_label_count",
                "has_prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    args.no_detection_list.write_text(
        "\n".join(no_detection),
        encoding="utf-8-sig",
    )

    print("=" * 72)
    print("UNLABELED PSEUDO-LABEL SUMMARY")
    print("=" * 72)
    print(f"Unlabeled images: {len(image_names)}")
    print(f"Images with pseudo-labels: {len(image_names) - len(no_detection)}")
    print(f"Images without pseudo-labels: {len(no_detection)}")
    print(f"Pseudo-label boxes: {sum(counts.values())}")
    print(f"Summary CSV: {args.summary}")
    print(f"No-detection list: {args.no_detection_list}")


if __name__ == "__main__":
    main()
