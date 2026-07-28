from __future__ import annotations

import argparse

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# Values below are overwritten by the pipeline controller.
ORIGINAL_TRAIN_CSV = ""
ORIGINAL_VAL_CSV = ""
ORIGINAL_TEST_CSV = ""
ORIGINAL_UNLABELED_IMAGES_TXT = ""
FUSED_TRAIN_PREDICTIONS_CSV = ""
FUSED_UNLABELED_PREDICTIONS_CSV = ""
OUTPUT_DIR = ""
MIN_PSEUDO_CONFIDENCE = 0.50
INCLUDE_TRAIN_MISSING_BOXES = True
ALLOWED_CLASSES = [1, 2, 3, 4]

REQUIRED_COLUMNS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]


def _key(value: str) -> str:
    return os.path.basename(str(value).strip()).lower()


def _read_labels(path: str, require_confidence: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["filename"] = df["filename"].astype(str).map(
        lambda x: os.path.basename(x.strip())
    )
    df["_key"] = df["filename"].map(_key)
    df["class"] = pd.to_numeric(df["class"], errors="raise").astype(int)

    for column in ["xmin", "ymin", "xmax", "ymax"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    valid_box = (df["xmax"] > df["xmin"]) & (df["ymax"] > df["ymin"])
    df = df.loc[valid_box & df["class"].isin(ALLOWED_CLASSES)].copy()

    if require_confidence:
        if "confidence" not in df.columns:
            raise ValueError(f"{path}: confidence column is required")
        df["confidence"] = pd.to_numeric(df["confidence"], errors="raise")
        valid_conf = df["confidence"].between(0.0, 1.0, inclusive="both")
        if not valid_conf.all():
            raise ValueError(f"{path}: confidence values must be inside [0, 1]")
        df = df.loc[df["confidence"] >= MIN_PSEUDO_CONFIDENCE].copy()

    return df.reset_index(drop=True)


def _read_unlabeled_names(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    names: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            value = line.split(",")[0].strip().strip('"').strip("'")
            if value.lower() in {"filename", "file_name", "image", "image_name"}:
                continue
            name = os.path.basename(value)
            key = _key(name)
            if key and key not in seen:
                seen.add(key)
                names.append(name)
    return names


def _split_allowed_pseudo(
    train_pseudo: pd.DataFrame,
    unlabeled_pseudo: pd.DataFrame,
    train_gt: pd.DataFrame,
    val_gt: pd.DataFrame,
    test_gt: pd.DataFrame,
    original_unlabeled: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_keys = set(train_gt["_key"])
    blocked_keys = set(val_gt["_key"]).union(test_gt["_key"])
    unlabeled_keys = {_key(name) for name in original_unlabeled}

    accepted_parts = []
    rejected_parts = []

    if INCLUDE_TRAIN_MISSING_BOXES:
        accepted_train = train_pseudo.loc[
            train_pseudo["_key"].isin(train_keys)
            & ~train_pseudo["_key"].isin(blocked_keys)
        ].copy()
        rejected_train = train_pseudo.loc[
            ~train_pseudo.index.isin(accepted_train.index)
        ].copy()
        accepted_train["acceptance_group"] = "train_missing_box"
        rejected_train["rejection_reason"] = "not_original_train_or_overlaps_val_test"
        accepted_parts.append(accepted_train)
        rejected_parts.append(rejected_train)

    accepted_unlabeled = unlabeled_pseudo.loc[
        unlabeled_pseudo["_key"].isin(unlabeled_keys)
        & ~unlabeled_pseudo["_key"].isin(blocked_keys)
    ].copy()
    rejected_unlabeled = unlabeled_pseudo.loc[
        ~unlabeled_pseudo.index.isin(accepted_unlabeled.index)
    ].copy()
    accepted_unlabeled["acceptance_group"] = "unlabeled_promoted_to_train"
    rejected_unlabeled["rejection_reason"] = "not_in_unlabeled_list_or_overlaps_val_test"
    accepted_parts.append(accepted_unlabeled)
    rejected_parts.append(rejected_unlabeled)

    accepted = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else train_pseudo.iloc[0:0].copy()
    rejected = pd.concat(rejected_parts, ignore_index=True) if rejected_parts else train_pseudo.iloc[0:0].copy()

    accepted = accepted.drop_duplicates(
        subset=["_key", "class", "xmin", "ymin", "xmax", "ymax"]
    ).reset_index(drop=True)
    return accepted, rejected


def parse_args():
    p=argparse.ArgumentParser(description='Build leakage-safe warm-start labels.')
    p.add_argument('--train-gt', required=True); p.add_argument('--val-gt', required=True); p.add_argument('--test-gt', required=True)
    p.add_argument('--unlabeled-list', required=True); p.add_argument('--train-fused', required=True); p.add_argument('--unlabeled-fused', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--minimum-confidence', type=float, default=0.50)
    p.add_argument('--exclude-train-missing-boxes', action='store_true')
    return p.parse_args()

def main() -> None:
    global ORIGINAL_TRAIN_CSV, ORIGINAL_VAL_CSV, ORIGINAL_TEST_CSV, ORIGINAL_UNLABELED_IMAGES_TXT
    global FUSED_TRAIN_PREDICTIONS_CSV, FUSED_UNLABELED_PREDICTIONS_CSV, OUTPUT_DIR, MIN_PSEUDO_CONFIDENCE, INCLUDE_TRAIN_MISSING_BOXES
    args=parse_args(); ORIGINAL_TRAIN_CSV=args.train_gt; ORIGINAL_VAL_CSV=args.val_gt; ORIGINAL_TEST_CSV=args.test_gt
    ORIGINAL_UNLABELED_IMAGES_TXT=args.unlabeled_list; FUSED_TRAIN_PREDICTIONS_CSV=args.train_fused; FUSED_UNLABELED_PREDICTIONS_CSV=args.unlabeled_fused
    OUTPUT_DIR=args.output_dir; MIN_PSEUDO_CONFIDENCE=args.minimum_confidence; INCLUDE_TRAIN_MISSING_BOXES=not args.exclude_train_missing_boxes
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    train_gt = _read_labels(ORIGINAL_TRAIN_CSV)
    val_gt = _read_labels(ORIGINAL_VAL_CSV)
    test_gt = _read_labels(ORIGINAL_TEST_CSV)
    train_pseudo = _read_labels(
        FUSED_TRAIN_PREDICTIONS_CSV,
        require_confidence=True,
    )
    unlabeled_pseudo = _read_labels(
        FUSED_UNLABELED_PREDICTIONS_CSV,
        require_confidence=True,
    )
    original_unlabeled = _read_unlabeled_names(
        ORIGINAL_UNLABELED_IMAGES_TXT
    )

    split_sets = {
        "train": set(train_gt["_key"]),
        "val": set(val_gt["_key"]),
        "test": set(test_gt["_key"]),
    }
    overlaps = {
        "train_val": split_sets["train"] & split_sets["val"],
        "train_test": split_sets["train"] & split_sets["test"],
        "val_test": split_sets["val"] & split_sets["test"],
    }
    if any(overlaps.values()):
        raise RuntimeError(
            "Original split leakage detected: "
            + ", ".join(f"{name}={len(values)}" for name, values in overlaps.items())
        )

    accepted_pseudo, rejected_pseudo = _split_allowed_pseudo(
        train_pseudo=train_pseudo,
        unlabeled_pseudo=unlabeled_pseudo,
        train_gt=train_gt,
        val_gt=val_gt,
        test_gt=test_gt,
        original_unlabeled=original_unlabeled,
    )

    warm_train = pd.concat(
        [
            train_gt[REQUIRED_COLUMNS],
            accepted_pseudo[REQUIRED_COLUMNS],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=REQUIRED_COLUMNS)

    warm_train.to_csv(output / "train_labels.csv", index=False, encoding="utf-8-sig")
    val_gt[REQUIRED_COLUMNS].to_csv(output / "val_labels.csv", index=False, encoding="utf-8-sig")
    test_gt[REQUIRED_COLUMNS].to_csv(output / "test_labels.csv", index=False, encoding="utf-8-sig")

    promoted_unlabeled_keys = set(
        accepted_pseudo.loc[
            accepted_pseudo["acceptance_group"] == "unlabeled_promoted_to_train",
            "_key",
        ]
    )
    remaining_unlabeled = [
        name for name in original_unlabeled
        if _key(name) not in promoted_unlabeled_keys
    ]
    with open(output / "unlabeled_images.txt", "w", encoding="utf-8") as file:
        for name in remaining_unlabeled:
            file.write(name + "\n")

    accepted_pseudo.drop(columns=["_key"], errors="ignore").to_csv(
        output / "accepted_pseudo_labels_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rejected_pseudo.drop(columns=["_key"], errors="ignore").to_csv(
        output / "rejected_pseudo_labels_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([
        {"item": "original_train_boxes", "count": len(train_gt)},
        {"item": "accepted_fused_pseudo_boxes", "count": len(accepted_pseudo)},
        {"item": "rejected_fused_pseudo_boxes", "count": len(rejected_pseudo)},
        {"item": "warm_train_boxes", "count": len(warm_train)},
        {"item": "original_val_boxes_kept_as_gt_only", "count": len(val_gt)},
        {"item": "original_test_boxes_kept_as_gt_only", "count": len(test_gt)},
        {"item": "remaining_unlabeled_images", "count": len(remaining_unlabeled)},
    ])
    summary.to_csv(
        output / "warmstart_labels_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Warm-start labels written to: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
