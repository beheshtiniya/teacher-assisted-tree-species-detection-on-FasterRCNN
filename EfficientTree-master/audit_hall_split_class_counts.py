from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit YOLO class distributions in EfficientTree Hall split list files."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing train/val/test split txt files.",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "train_10_percent.txt",
            "val.txt",
            "test.txt",
            "target_remaining_train_10_percent.txt",
            "target_pavilion.txt",
        ],
    )
    return parser.parse_args()


def parse_pair_line(line: str) -> tuple[Path, Path | None]:
    text = line.strip()
    if not text:
        raise ValueError("empty line")

    parts = text.rsplit(maxsplit=1)
    if len(parts) == 2 and Path(parts[1]).suffix.lower() == ".txt":
        return Path(parts[0]), Path(parts[1])

    # Image-only list, such as Pavilion target.
    return Path(text), None


def audit_list(path: Path, num_classes: int) -> None:
    print("=" * 88)
    print(f"FILE: {path}")
    print("=" * 88)

    if not path.is_file():
        print("STATUS: MISSING SPLIT FILE")
        return

    rows = 0
    image_missing = 0
    label_missing = 0
    empty_labels = 0
    malformed_labels = 0
    class_counts: Counter[int] = Counter()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        rows += 1

        try:
            image_path, label_path = parse_pair_line(raw)
        except Exception as exc:
            print(f"BAD SPLIT LINE {line_number}: {exc}: {raw!r}")
            continue

        if not image_path.is_file():
            image_missing += 1

        if label_path is None:
            continue

        if not label_path.is_file():
            label_missing += 1
            continue

        label_lines = [
            x.strip()
            for x in label_path.read_text(encoding="utf-8-sig").splitlines()
            if x.strip()
        ]
        if not label_lines:
            empty_labels += 1
            continue

        for label_line_number, label_line in enumerate(label_lines, 1):
            fields = label_line.split()
            try:
                class_id = int(float(fields[0]))
            except Exception:
                malformed_labels += 1
                print(
                    f"BAD LABEL: {label_path}:{label_line_number}: "
                    f"{label_line!r}"
                )
                continue
            class_counts[class_id] += 1

    expected = set(range(num_classes))
    found = set(class_counts)
    missing_classes = sorted(expected - found)
    unexpected_classes = sorted(found - expected)

    print(f"rows                 = {rows}")
    print(f"missing images       = {image_missing}")
    print(f"missing labels       = {label_missing}")
    print(f"empty label files    = {empty_labels}")
    print(f"malformed labels     = {malformed_labels}")
    print(f"total boxes          = {sum(class_counts.values())}")
    print(f"class counts         = {dict(sorted(class_counts.items()))}")
    print(f"missing class IDs    = {missing_classes}")
    print(f"unexpected class IDs = {unexpected_classes}")


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()

    print(f"SPLIT DIRECTORY: {split_dir}")
    print(f"EXPECTED CLASS IDS: 0..{args.num_classes - 1}")

    for name in args.files:
        audit_list(split_dir / name, args.num_classes)


if __name__ == "__main__":
    main()
