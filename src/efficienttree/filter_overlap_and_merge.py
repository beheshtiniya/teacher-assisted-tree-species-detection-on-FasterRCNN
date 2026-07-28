from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GT_COLUMNS = [
    "filename",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]

FINAL_COLUMNS = GT_COLUMNS + ["confidence"]


def image_key(value: str) -> str:
    return Path(str(value).strip().strip('"')).name.lower()


def read_rows(
    path: Path,
    require_confidence: bool,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    required = set(GT_COLUMNS)
    if require_confidence:
        required.add("confidence")

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        column_map = {
            str(name).strip().lower(): name
            for name in reader.fieldnames
        }

        missing = required - set(column_map)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {sorted(missing)}"
            )

        for row_number, raw in enumerate(reader, start=2):
            try:
                row: dict[str, Any] = {
                    "filename": Path(
                        str(
                            raw[column_map["filename"]]
                        ).strip().strip('"')
                    ).name,
                    "class": int(
                        float(raw[column_map["class"]])
                    ),
                    "xmin": float(raw[column_map["xmin"]]),
                    "ymin": float(raw[column_map["ymin"]]),
                    "xmax": float(raw[column_map["xmax"]]),
                    "ymax": float(raw[column_map["ymax"]]),
                }

                if require_confidence:
                    row["confidence"] = float(
                        raw[column_map["confidence"]]
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid row {row_number} in {path}"
                ) from exc

            if not row["filename"]:
                continue

            if row["class"] not in {1, 2, 3, 4}:
                raise ValueError(
                    f"Class must be 1-4 at {path}:{row_number}"
                )

            if (
                row["xmax"] <= row["xmin"]
                or row["ymax"] <= row["ymin"]
            ):
                raise ValueError(
                    f"Invalid XYXY box at {path}:{row_number}"
                )

            rows.append(row)

    return rows


def iou(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    inter_xmin = max(first["xmin"], second["xmin"])
    inter_ymin = max(first["ymin"], second["ymin"])
    inter_xmax = min(first["xmax"], second["xmax"])
    inter_ymax = min(first["ymax"], second["ymax"])

    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height

    first_area = (
        (first["xmax"] - first["xmin"])
        * (first["ymax"] - first["ymin"])
    )
    second_area = (
        (second["xmax"] - second["xmin"])
        * (second["ymax"] - second["ymin"])
    )

    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def final_row(
    row: dict[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "filename": row["filename"],
        "class": int(row["class"]),
        "xmin": round(float(row["xmin"]), 2),
        "ymin": round(float(row["ymin"]), 2),
        "xmax": round(float(row["xmax"]), 2),
        "ymax": round(float(row["ymax"]), 2),
        "confidence": round(float(confidence), 6),
    }


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove every prediction that overlaps any human GT by more "
            "than the IoU threshold, regardless of class, then merge GT "
            "with the remaining missing-object candidates."
        )
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--split", default="split")
    args = parser.parse_args()

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1.")

    gt_rows = read_rows(
        args.ground_truth,
        require_confidence=False,
    )
    prediction_rows = read_rows(
        args.predictions,
        require_confidence=True,
    )

    gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gt in gt_rows:
        gt_by_image[image_key(gt["filename"])].append(gt)

    kept_predictions: list[dict[str, Any]] = []
    removed_predictions: list[dict[str, Any]] = []

    for prediction in prediction_rows:
        image_gt = gt_by_image.get(
            image_key(prediction["filename"]),
            [],
        )

        best_iou = 0.0
        best_gt: dict[str, Any] | None = None

        for gt in image_gt:
            overlap = iou(prediction, gt)
            if overlap > best_iou:
                best_iou = overlap
                best_gt = gt

        # Strictly greater than 0.50, exactly as requested.
        # Class is intentionally ignored.
        if best_iou > args.iou:
            removed_row = final_row(
                prediction,
                prediction["confidence"],
            )
            removed_row.update(
                {
                    "matched_gt_class": (
                        int(best_gt["class"])
                        if best_gt is not None
                        else ""
                    ),
                    "max_iou_with_gt": round(best_iou, 6),
                    "same_class": (
                        int(prediction["class"])
                        == int(best_gt["class"])
                        if best_gt is not None
                        else False
                    ),
                }
            )
            removed_predictions.append(removed_row)
        else:
            kept_predictions.append(
                final_row(
                    prediction,
                    prediction["confidence"],
                )
            )

    combined_rows = [
        final_row(gt, 1.0)
        for gt in gt_rows
    ] + kept_predictions

    kept_predictions.sort(
        key=lambda row: (
            image_key(row["filename"]),
            -float(row["confidence"]),
            int(row["class"]),
        )
    )
    removed_predictions.sort(
        key=lambda row: (
            image_key(row["filename"]),
            -float(row["confidence"]),
        )
    )

    write_csv(
        args.combined,
        FINAL_COLUMNS,
        combined_rows,
    )
    write_csv(
        args.candidates,
        FINAL_COLUMNS,
        kept_predictions,
    )
    write_csv(
        args.removed,
        FINAL_COLUMNS
        + [
            "matched_gt_class",
            "max_iou_with_gt",
            "same_class",
        ],
        removed_predictions,
    )

    gt_counts = Counter(int(row["class"]) for row in gt_rows)
    kept_counts = Counter(
        int(row["class"]) for row in kept_predictions
    )
    removed_counts = Counter(
        int(row["class"]) for row in removed_predictions
    )

    print("=" * 72)
    print(f"CLASS-AGNOSTIC GT FILTER COMPLETE: {args.split}")
    print("=" * 72)
    print(f"Human GT boxes: {len(gt_rows)}")
    print(f"Predictions in 256 coordinates: {len(prediction_rows)}")
    print(
        "Predictions removed because IoU with ANY GT "
        f"> {args.iou:.2f}: {len(removed_predictions)}"
    )
    print(
        f"Missing-object candidates kept: {len(kept_predictions)}"
    )
    print(f"Combined rows: {len(combined_rows)}")
    print("-" * 72)

    for class_id in range(1, 5):
        print(
            f"Class {class_id}: "
            f"GT={gt_counts[class_id]} | "
            f"kept={kept_counts[class_id]} | "
            f"removed={removed_counts[class_id]}"
        )

    print("-" * 72)
    print(f"Combined CSV: {args.combined}")
    print(f"Candidates only: {args.candidates}")
    print(f"Removed overlap audit: {args.removed}")


if __name__ == "__main__":
    main()
