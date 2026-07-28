from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

REQUIRED_COLUMNS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
VALID_CLASSES = {1, 2, 3, 4}
IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the user's natural (unbalanced) tree dataset for EfficientTree: "
            "resize all images to 640x640, convert CSV xyxy boxes to YOLO labels, "
            "and create full train/val/test/unlabeled list files."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root containing images and labels folders.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Folder containing all source images, recursively. Default: <root>\\images",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Folder containing CSV files. Default: <root>\\labels",
    )
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--unlabeled-list", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <root>\\efficienttree_640_full_unbalanced",
    )
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before rebuilding it.",
    )
    parser.add_argument(
        "--copy-without-resize",
        action="store_true",
        help="Debug option only: copy instead of resizing.",
    )
    return parser.parse_args()


def normalize_filename(value: object) -> str:
    return str(value).strip().replace("\\", "/")


def read_csv_checked(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{split} CSV not found: {path}")

    df = pd.read_csv(path).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{split}: missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["filename"] = df["filename"].map(normalize_filename)
    df["class"] = pd.to_numeric(df["class"], errors="raise").astype(int)
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.isna().any().any():
        raise ValueError(f"{split}: NaN detected in required columns")

    found = set(df["class"].unique().tolist())
    invalid_classes = sorted(found - VALID_CLASSES)
    if invalid_classes:
        raise ValueError(
            f"{split}: invalid classes {invalid_classes}; expected only 1,2,3,4"
        )

    bad = (df["xmax"] <= df["xmin"]) | (df["ymax"] <= df["ymin"])
    if bad.any():
        examples = df.loc[bad].head(10).to_dict("records")
        raise ValueError(f"{split}: invalid bounding boxes. Examples: {examples}")

    return df


def read_unlabeled(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Unlabeled list not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        names = [normalize_filename(line) for line in f if line.strip()]
    if not names:
        raise ValueError("Unlabeled list is empty")
    duplicates = len(names) - len(set(names))
    if duplicates:
        raise ValueError(f"Unlabeled list contains {duplicates} duplicate names")
    return names


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    by_rel: Dict[str, Path] = {}
    by_name: Dict[str, List[Path]] = defaultdict(list)
    for path in images_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(images_dir).as_posix()
        by_rel[rel.lower()] = path
        by_name[path.name.lower()].append(path)

    if not by_rel:
        raise RuntimeError(f"No supported images found under {images_dir}")

    index: Dict[str, Path] = dict(by_rel)
    duplicate_basenames = {k: v for k, v in by_name.items() if len(v) > 1}
    for basename, paths in by_name.items():
        if len(paths) == 1:
            index[basename] = paths[0]

    if duplicate_basenames:
        print(
            f"Warning: {len(duplicate_basenames)} duplicate basenames found. "
            "Relative paths remain usable, but ambiguous bare filenames will fail."
        )
    return index


def resolve_source(name: str, image_index: Mapping[str, Path]) -> Path:
    key = normalize_filename(name).lower()
    if key in image_index:
        return image_index[key]
    basename = Path(key).name
    if basename in image_index:
        return image_index[basename]
    raise FileNotFoundError(f"Source image not found for entry: {name}")


def assert_no_name_leakage(named_sets: Mapping[str, set[str]]) -> Dict[str, int]:
    names = list(named_sets)
    report: Dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = named_sets[left] & named_sets[right]
            key = f"{left}_vs_{right}"
            report[key] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Filename leakage detected: {key}={len(overlap)}; "
                    f"examples={sorted(overlap)[:20]}"
                )
    return report


def save_resized_rgb(source: Path, destination: Path, size: int, copy_only: bool) -> Tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(source, destination)
        with Image.open(source) as im:
            return im.size

    with Image.open(source) as im:
        original_size = im.size
        im = im.convert("RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        if im.size != (size, size):
            im = im.resize((size, size), resampling)

        suffix = destination.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            im.save(destination, compression="tiff_deflate")
        elif suffix in {".jpg", ".jpeg"}:
            im.save(destination, quality=95, subsampling=0, optimize=True)
        elif suffix == ".png":
            im.save(destination, optimize=True)
        else:
            im.save(destination)
        return original_size


def xyxy_to_yolo(
    cls_source: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> Tuple[int, float, float, float, float]:
    xmin = min(max(float(xmin), 0.0), float(width))
    xmax = min(max(float(xmax), 0.0), float(width))
    ymin = min(max(float(ymin), 0.0), float(height))
    ymax = min(max(float(ymax), 0.0), float(height))

    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Degenerate box after clipping: {(cls_source, xmin, ymin, xmax, ymax)}"
        )

    cls_yolo = int(cls_source) - 1
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height

    values = (x_center, y_center, box_w, box_h)
    if not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"Normalized box outside [0,1]: {values}")
    return cls_yolo, x_center, y_center, box_w, box_h


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(str(line).rstrip() + "\n")


def process_labeled_split(
    split: str,
    df: pd.DataFrame,
    image_index: Mapping[str, Path],
    output_dir: Path,
    target_size: int,
    copy_only: bool,
) -> Dict[str, object]:
    output_images = output_dir / "images" / split
    output_labels = output_dir / "labels" / split
    split_file = output_dir / "splits" / f"{split}.txt"

    grouped = df.groupby("filename", sort=False)
    output_image_paths: List[str] = []
    class_counts: Counter[int] = Counter()
    source_sizes: Counter[str] = Counter()

    total = len(grouped)
    for idx, (filename, rows) in enumerate(grouped, start=1):
        source = resolve_source(filename, image_index)
        relative = Path(normalize_filename(filename))
        destination = output_images / relative
        original_w, original_h = save_resized_rgb(
            source, destination, target_size, copy_only
        )
        source_sizes[f"{original_w}x{original_h}"] += 1

        label_path = output_labels / relative.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        yolo_lines: List[str] = []
        for row in rows.itertuples(index=False):
            converted = xyxy_to_yolo(
                int(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                original_w,
                original_h,
            )
            cls_yolo, xc, yc, bw, bh = converted
            class_counts[cls_yolo] += 1
            yolo_lines.append(
                f"{cls_yolo} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
            )
        write_lines(label_path, yolo_lines)
        output_image_paths.append(str(destination.resolve()))

        if idx % 250 == 0 or idx == total:
            print(f"[{split}] {idx}/{total} images")

    write_lines(split_file, output_image_paths)
    return {
        "images": len(output_image_paths),
        "boxes": int(sum(class_counts.values())),
        "class_counts_yolo_0_to_3": dict(sorted(class_counts.items())),
        "source_image_sizes": dict(source_sizes),
        "split_file": str(split_file.resolve()),
    }


def process_unlabeled(
    names: Sequence[str],
    image_index: Mapping[str, Path],
    output_dir: Path,
    target_size: int,
    copy_only: bool,
) -> Dict[str, object]:
    output_images = output_dir / "images" / "unlabeled"
    output_labels = output_dir / "labels" / "unlabeled"
    split_file = output_dir / "splits" / "target_unlabeled.txt"
    output_paths: List[str] = []
    source_sizes: Counter[str] = Counter()

    for idx, filename in enumerate(names, start=1):
        source = resolve_source(filename, image_index)
        relative = Path(normalize_filename(filename))
        destination = output_images / relative
        original_w, original_h = save_resized_rgb(
            source, destination, target_size, copy_only
        )
        source_sizes[f"{original_w}x{original_h}"] += 1

        # Empty placeholders avoid missing-label scan errors. They are not used as GT in SSOD.
        empty_label = output_labels / relative.with_suffix(".txt")
        empty_label.parent.mkdir(parents=True, exist_ok=True)
        empty_label.write_text("", encoding="utf-8")

        output_paths.append(str(destination.resolve()))
        if idx % 500 == 0 or idx == len(names):
            print(f"[unlabeled] {idx}/{len(names)} images")

    write_lines(split_file, output_paths)
    return {
        "images": len(output_paths),
        "source_image_sizes": dict(source_sizes),
        "split_file": str(split_file.resolve()),
        "empty_label_placeholders": len(output_paths),
    }


def make_smoke_lists(output_dir: Path) -> Dict[str, str]:
    split_dir = output_dir / "splits"
    limits = {
        "train": 64,
        "val": 64,
        "test": 64,
        "target_unlabeled": 128,
    }
    smoke_paths: Dict[str, str] = {}
    for name, limit in limits.items():
        source = split_dir / f"{name}.txt"
        lines = source.read_text(encoding="utf-8").splitlines()
        destination = split_dir / f"smoke_{name}.txt"
        write_lines(destination, lines[:limit])
        smoke_paths[name] = str(destination.resolve())
    return smoke_paths


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    images_dir = Path(args.images_dir) if args.images_dir else root / "images"
    labels_dir = Path(args.labels_dir) if args.labels_dir else root / "labels"
    train_csv = Path(args.train_csv) if args.train_csv else labels_dir / "train_labels.csv"
    val_csv = Path(args.val_csv) if args.val_csv else labels_dir / "val_labels.csv"
    test_csv = Path(args.test_csv) if args.test_csv else labels_dir / "test_labels.csv"
    unlabeled_list = (
        Path(args.unlabeled_list)
        if args.unlabeled_list
        else labels_dir / "unlabeled_images.txt"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "efficienttree_640_full_unbalanced"
    )

    if args.size <= 0 or args.size % 32 != 0:
        raise ValueError("--size must be a positive multiple of 32")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}\n"
                "Run with --overwrite only when you intend to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading annotations...")
    frames = {
        "train": read_csv_checked(train_csv, "train"),
        "val": read_csv_checked(val_csv, "val"),
        "test": read_csv_checked(test_csv, "test"),
    }
    unlabeled_names = read_unlabeled(unlabeled_list)

    split_sets = {
        key: set(frame["filename"].tolist()) for key, frame in frames.items()
    }
    split_sets["unlabeled"] = set(unlabeled_names)
    leakage_report = assert_no_name_leakage(split_sets)

    print("Indexing source images recursively...")
    image_index = build_image_index(images_dir)
    print(f"Indexed {len(image_index):,} path/name keys")

    report: Dict[str, object] = {
        "mode": "full_natural_unbalanced",
        "target_size": [args.size, args.size],
        "source": {
            "root": str(root.resolve()),
            "images_dir": str(images_dir.resolve()),
            "train_csv": str(train_csv.resolve()),
            "val_csv": str(val_csv.resolve()),
            "test_csv": str(test_csv.resolve()),
            "unlabeled_list": str(unlabeled_list.resolve()),
        },
        "class_mapping": {"1": 0, "2": 1, "3": 2, "4": 3},
        "balancing_or_resampling": False,
        "preprocessing_augmentation": False,
        "filename_leakage": leakage_report,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        report["splits"][split] = process_labeled_split(
            split,
            frames[split],
            image_index,
            output_dir,
            args.size,
            args.copy_without_resize,
        )

    report["splits"]["unlabeled"] = process_unlabeled(
        unlabeled_names,
        image_index,
        output_dir,
        args.size,
        args.copy_without_resize,
    )
    report["smoke_lists"] = make_smoke_lists(output_dir)

    report_path = output_dir / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    counts_path = output_dir / "class_counts.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_yolo", "boxes"])
        for split in ["train", "val", "test"]:
            counts = report["splits"][split]["class_counts_yolo_0_to_3"]
            for cls in range(4):
                writer.writerow([split, cls, counts.get(cls, counts.get(str(cls), 0))])

    print("\nPreparation completed successfully.")
    print(f"Output: {output_dir.resolve()}")
    print(f"Audit:  {report_path.resolve()}")
    print("No balancing or resampling was performed.")
    print("Full train.txt was created; train_10_percent.txt is not used.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
