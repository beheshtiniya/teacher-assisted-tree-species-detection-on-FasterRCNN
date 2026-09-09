from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Runtime values are supplied by run_stage.py
# ---------------------------------------------------------------------

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

REQUIRED_COLUMNS = [
    "filename",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]


# =====================================================================
# Helpers
# =====================================================================

def _key(value: str) -> str:
    return os.path.basename(str(value).strip()).lower()


def _read_labels(
    path: str,
    require_confidence: bool = False,
) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path).copy()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    df["filename"] = (
        df["filename"]
        .astype(str)
        .map(lambda x: os.path.basename(x.strip()))
    )

    df["_key"] = df["filename"].map(_key)

    df["class"] = pd.to_numeric(
        df["class"],
        errors="raise",
    ).astype(int)

    for column in [
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    valid_box = (
        (df["xmax"] > df["xmin"])
        &
        (df["ymax"] > df["ymin"])
    )

    invalid_box_count = int((~valid_box).sum())

    if invalid_box_count:
        print(
            f"WARNING: {path}: "
            f"{invalid_box_count} invalid boxes removed."
        )

    invalid_classes = sorted(
        set(df.loc[
            ~df["class"].isin(ALLOWED_CLASSES),
            "class",
        ].tolist())
    )

    if invalid_classes:
        print(
            f"WARNING: {path}: "
            f"unsupported class IDs found and removed: "
            f"{invalid_classes}"
        )

    df = df.loc[
        valid_box
        &
        df["class"].isin(ALLOWED_CLASSES)
    ].copy()

    if require_confidence:

        if "confidence" not in df.columns:
            raise ValueError(
                f"{path}: confidence column is required"
            )

        df["confidence"] = pd.to_numeric(
            df["confidence"],
            errors="raise",
        )

        valid_conf = df["confidence"].between(
            0.0,
            1.0,
            inclusive="both",
        )

        if not valid_conf.all():
            bad = int((~valid_conf).sum())
            raise ValueError(
                f"{path}: {bad} confidence values "
                f"are outside [0, 1]"
            )

    return df.reset_index(drop=True)


def _read_unlabeled_names(
    path: str,
) -> List[str]:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    names: List[str] = []
    seen = set()

    with open(
        path,
        "r",
        encoding="utf-8-sig",
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            value = (
                line
                .split(",")[0]
                .strip()
                .strip('"')
                .strip("'")
            )

            if value.lower() in {
                "filename",
                "file_name",
                "image",
                "image_name",
            }:
                continue

            name = os.path.basename(value)
            key = _key(name)

            if key and key not in seen:
                seen.add(key)
                names.append(name)

    return names


# =====================================================================
# Pseudo-label selection
# =====================================================================

def _split_allowed_pseudo(
    train_pseudo: pd.DataFrame,
    unlabeled_pseudo: pd.DataFrame,
    train_gt: pd.DataFrame,
    val_gt: pd.DataFrame,
    test_gt: pd.DataFrame,
    original_unlabeled: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    train_keys = set(train_gt["_key"])

    val_keys = set(val_gt["_key"])
    test_keys = set(test_gt["_key"])

    blocked_keys = val_keys.union(test_keys)

    unlabeled_keys = {
        _key(name)
        for name in original_unlabeled
    }

    accepted_parts = []
    rejected_parts = []

    stats = {
        "input_train_fused": len(train_pseudo),
        "input_unlabeled_fused": len(unlabeled_pseudo),
        "train_fused_accepted": 0,
        "train_fused_rejected": 0,
        "unlabeled_above_confidence": 0,
        "unlabeled_below_confidence": 0,
        "unlabeled_accepted": 0,
        "unlabeled_rejected_after_confidence": 0,
    }

    # --------------------------------------------------------------
    # TRAIN FUSED PREDICTIONS
    #
    # IMPORTANT:
    # These boxes have already passed Stage-11 fusion criteria.
    # They form A2.
    #
    # DO NOT apply MIN_PSEUDO_CONFIDENCE again here.
    # Otherwise A3 may start from a weaker dataset than A2.
    # --------------------------------------------------------------

    if INCLUDE_TRAIN_MISSING_BOXES:

        accepted_train = train_pseudo.loc[
            train_pseudo["_key"].isin(train_keys)
            &
            ~train_pseudo["_key"].isin(blocked_keys)
        ].copy()

        rejected_train = train_pseudo.loc[
            ~train_pseudo.index.isin(
                accepted_train.index
            )
        ].copy()

        accepted_train[
            "acceptance_group"
        ] = "train_fused_from_A2"

        rejected_train[
            "rejection_reason"
        ] = (
            "not_original_train_or_overlaps_val_test"
        )

        stats[
            "train_fused_accepted"
        ] = len(accepted_train)

        stats[
            "train_fused_rejected"
        ] = len(rejected_train)

        accepted_parts.append(
            accepted_train
        )

        rejected_parts.append(
            rejected_train
        )

    # --------------------------------------------------------------
    # UNLABELED FUSED PREDICTIONS
    #
    # These can optionally be promoted into warm-start training.
    # Apply warm-start threshold here.
    # --------------------------------------------------------------

    if len(unlabeled_pseudo):

        unlabeled_high_conf = unlabeled_pseudo.loc[
            unlabeled_pseudo[
                "confidence"
            ] >= MIN_PSEUDO_CONFIDENCE
        ].copy()

        unlabeled_low_conf = unlabeled_pseudo.loc[
            unlabeled_pseudo[
                "confidence"
            ] < MIN_PSEUDO_CONFIDENCE
        ].copy()

    else:
        unlabeled_high_conf = (
            unlabeled_pseudo.copy()
        )

        unlabeled_low_conf = (
            unlabeled_pseudo.copy()
        )

    stats[
        "unlabeled_above_confidence"
    ] = len(unlabeled_high_conf)

    stats[
        "unlabeled_below_confidence"
    ] = len(unlabeled_low_conf)

    if len(unlabeled_low_conf):

        unlabeled_low_conf[
            "rejection_reason"
        ] = (
            f"confidence_below_"
            f"{MIN_PSEUDO_CONFIDENCE}"
        )

        rejected_parts.append(
            unlabeled_low_conf
        )

    accepted_unlabeled = (
        unlabeled_high_conf.loc[
            unlabeled_high_conf[
                "_key"
            ].isin(unlabeled_keys)
            &
            ~unlabeled_high_conf[
                "_key"
            ].isin(blocked_keys)
        ].copy()
    )

    rejected_unlabeled = (
        unlabeled_high_conf.loc[
            ~unlabeled_high_conf.index.isin(
                accepted_unlabeled.index
            )
        ].copy()
    )

    accepted_unlabeled[
        "acceptance_group"
    ] = "unlabeled_promoted_to_train"

    rejected_unlabeled[
        "rejection_reason"
    ] = (
        "not_in_unlabeled_list_or_overlaps_val_test"
    )

    stats[
        "unlabeled_accepted"
    ] = len(accepted_unlabeled)

    stats[
        "unlabeled_rejected_after_confidence"
    ] = len(rejected_unlabeled)

    accepted_parts.append(
        accepted_unlabeled
    )

    rejected_parts.append(
        rejected_unlabeled
    )

    if accepted_parts:
        accepted = pd.concat(
            accepted_parts,
            ignore_index=True,
        )
    else:
        accepted = train_pseudo.iloc[
            0:0
        ].copy()

    if rejected_parts:
        rejected = pd.concat(
            rejected_parts,
            ignore_index=True,
        )
    else:
        rejected = train_pseudo.iloc[
            0:0
        ].copy()

    accepted = (
        accepted
        .drop_duplicates(
            subset=[
                "_key",
                "class",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
        )
        .reset_index(drop=True)
    )

    rejected = rejected.reset_index(
        drop=True
    )

    return (
        accepted,
        rejected,
        stats,
    )


# =====================================================================
# Arguments
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe warm-start labels "
            "from A2 train fusion and fused "
            "unlabeled pseudo-labels."
        )
    )

    parser.add_argument(
        "--train-gt",
        required=True,
    )

    parser.add_argument(
        "--val-gt",
        required=True,
    )

    parser.add_argument(
        "--test-gt",
        required=True,
    )

    parser.add_argument(
        "--unlabeled-list",
        required=True,
    )

    parser.add_argument(
        "--train-fused",
        required=True,
    )

    parser.add_argument(
        "--unlabeled-fused",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--exclude-train-missing-boxes",
        action="store_true",
    )

    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    global ORIGINAL_TRAIN_CSV
    global ORIGINAL_VAL_CSV
    global ORIGINAL_TEST_CSV
    global ORIGINAL_UNLABELED_IMAGES_TXT

    global FUSED_TRAIN_PREDICTIONS_CSV
    global FUSED_UNLABELED_PREDICTIONS_CSV

    global OUTPUT_DIR
    global MIN_PSEUDO_CONFIDENCE
    global INCLUDE_TRAIN_MISSING_BOXES

    args = parse_args()

    ORIGINAL_TRAIN_CSV = args.train_gt
    ORIGINAL_VAL_CSV = args.val_gt
    ORIGINAL_TEST_CSV = args.test_gt

    ORIGINAL_UNLABELED_IMAGES_TXT = (
        args.unlabeled_list
    )

    FUSED_TRAIN_PREDICTIONS_CSV = (
        args.train_fused
    )

    FUSED_UNLABELED_PREDICTIONS_CSV = (
        args.unlabeled_fused
    )

    OUTPUT_DIR = args.output_dir

    MIN_PSEUDO_CONFIDENCE = (
        args.minimum_confidence
    )

    INCLUDE_TRAIN_MISSING_BOXES = (
        not args.exclude_train_missing_boxes
    )

    output = Path(OUTPUT_DIR)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Read original expert-only splits
    # --------------------------------------------------------------

    train_gt = _read_labels(
        ORIGINAL_TRAIN_CSV
    )

    val_gt = _read_labels(
        ORIGINAL_VAL_CSV
    )

    test_gt = _read_labels(
        ORIGINAL_TEST_CSV
    )

    # --------------------------------------------------------------
    # Read fused predictions
    #
    # NOTE:
    # We validate confidence but do NOT apply threshold here.
    # --------------------------------------------------------------

    train_pseudo = _read_labels(
        FUSED_TRAIN_PREDICTIONS_CSV,
        require_confidence=True,
    )

    unlabeled_pseudo = _read_labels(
        FUSED_UNLABELED_PREDICTIONS_CSV,
        require_confidence=True,
    )

    original_unlabeled = (
        _read_unlabeled_names(
            ORIGINAL_UNLABELED_IMAGES_TXT
        )
    )

    # --------------------------------------------------------------
    # Original split leakage check
    # --------------------------------------------------------------

    split_sets = {
        "train": set(train_gt["_key"]),
        "val": set(val_gt["_key"]),
        "test": set(test_gt["_key"]),
    }

    overlaps = {
        "train_val":
            split_sets["train"]
            &
            split_sets["val"],

        "train_test":
            split_sets["train"]
            &
            split_sets["test"],

        "val_test":
            split_sets["val"]
            &
            split_sets["test"],
    }

    if any(overlaps.values()):

        raise RuntimeError(
            "Original split leakage detected: "
            +
            ", ".join(
                f"{name}={len(values)}"
                for name, values
                in overlaps.items()
            )
        )

    print("=" * 72)
    print("WARM-START INPUT AUDIT")
    print("=" * 72)

    print(
        f"Original train GT boxes: "
        f"{len(train_gt)}"
    )

    print(
        f"Original validation GT boxes: "
        f"{len(val_gt)}"
    )

    print(
        f"Original test GT boxes: "
        f"{len(test_gt)}"
    )

    print(
        f"Input fused TRAIN boxes: "
        f"{len(train_pseudo)}"
    )

    print(
        f"Input fused UNLABELED boxes: "
        f"{len(unlabeled_pseudo)}"
    )

    print(
        f"Warm-start confidence threshold "
        f"for unlabeled only: "
        f"{MIN_PSEUDO_CONFIDENCE:.3f}"
    )

    # --------------------------------------------------------------
    # Select allowed pseudo-labels
    # --------------------------------------------------------------

    (
        accepted_pseudo,
        rejected_pseudo,
        stats,
    ) = _split_allowed_pseudo(
        train_pseudo=train_pseudo,
        unlabeled_pseudo=unlabeled_pseudo,
        train_gt=train_gt,
        val_gt=val_gt,
        test_gt=test_gt,
        original_unlabeled=original_unlabeled,
    )

    # --------------------------------------------------------------
    # Build final warm-start TRAIN
    # --------------------------------------------------------------

    warm_train = pd.concat(
        [
            train_gt[
                REQUIRED_COLUMNS
            ],
            accepted_pseudo[
                REQUIRED_COLUMNS
            ],
        ],
        ignore_index=True,
    )

    warm_train = (
        warm_train
        .drop_duplicates(
            subset=REQUIRED_COLUMNS
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Validation and test remain expert-only
    # --------------------------------------------------------------

    warm_train.to_csv(
        output / "train_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    val_gt[
        REQUIRED_COLUMNS
    ].to_csv(
        output / "val_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_gt[
        REQUIRED_COLUMNS
    ].to_csv(
        output / "test_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Remove promoted unlabeled images from remaining unlabeled list
    # --------------------------------------------------------------

    promoted_unlabeled_keys = set(
        accepted_pseudo.loc[
            accepted_pseudo[
                "acceptance_group"
            ]
            ==
            "unlabeled_promoted_to_train",
            "_key",
        ]
    )

    remaining_unlabeled = [
        name
        for name in original_unlabeled
        if _key(name)
        not in promoted_unlabeled_keys
    ]

    with open(
        output / "unlabeled_images.txt",
        "w",
        encoding="utf-8",
    ) as file:

        for name in remaining_unlabeled:
            file.write(
                name + "\n"
            )

    # --------------------------------------------------------------
    # Audit files
    # --------------------------------------------------------------

    accepted_pseudo.drop(
        columns=["_key"],
        errors="ignore",
    ).to_csv(
        output
        / "accepted_pseudo_labels_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rejected_pseudo.drop(
        columns=["_key"],
        errors="ignore",
    ).to_csv(
        output
        / "rejected_pseudo_labels_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Final safety checks
    # --------------------------------------------------------------

    warm_keys = set(
        warm_train[
            "filename"
        ].map(_key)
    )

    val_keys = set(
        val_gt["_key"]
    )

    test_keys = set(
        test_gt["_key"]
    )

    warm_val_overlap = (
        warm_keys
        &
        val_keys
    )

    warm_test_overlap = (
        warm_keys
        &
        test_keys
    )

    if warm_val_overlap:
        raise RuntimeError(
            "Leakage detected after warm-start: "
            f"train ∩ val = "
            f"{len(warm_val_overlap)}"
        )

    if warm_test_overlap:
        raise RuntimeError(
            "Leakage detected after warm-start: "
            f"train ∩ test = "
            f"{len(warm_test_overlap)}"
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    train_accepted_count = int(
        (
            accepted_pseudo[
                "acceptance_group"
            ]
            ==
            "train_fused_from_A2"
        ).sum()
    )

    unlabeled_accepted_count = int(
        (
            accepted_pseudo[
                "acceptance_group"
            ]
            ==
            "unlabeled_promoted_to_train"
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "item":
                    "original_train_boxes",
                "count":
                    len(train_gt),
            },
            {
                "item":
                    "input_train_fused_boxes",
                "count":
                    stats[
                        "input_train_fused"
                    ],
            },
            {
                "item":
                    "accepted_train_fused_boxes",
                "count":
                    train_accepted_count,
            },
            {
                "item":
                    "input_unlabeled_fused_boxes",
                "count":
                    stats[
                        "input_unlabeled_fused"
                    ],
            },
            {
                "item":
                    "unlabeled_above_warmstart_confidence",
                "count":
                    stats[
                        "unlabeled_above_confidence"
                    ],
            },
            {
                "item":
                    "unlabeled_below_warmstart_confidence",
                "count":
                    stats[
                        "unlabeled_below_confidence"
                    ],
            },
            {
                "item":
                    "accepted_unlabeled_fused_boxes",
                "count":
                    unlabeled_accepted_count,
            },
            {
                "item":
                    "accepted_fused_pseudo_boxes_total",
                "count":
                    len(accepted_pseudo),
            },
            {
                "item":
                    "rejected_fused_pseudo_boxes_total",
                "count":
                    len(rejected_pseudo),
            },
            {
                "item":
                    "warm_train_boxes",
                "count":
                    len(warm_train),
            },
            {
                "item":
                    "original_val_boxes_kept_as_gt_only",
                "count":
                    len(val_gt),
            },
            {
                "item":
                    "original_test_boxes_kept_as_gt_only",
                "count":
                    len(test_gt),
            },
            {
                "item":
                    "remaining_unlabeled_images",
                "count":
                    len(remaining_unlabeled),
            },
        ]
    )

    summary.to_csv(
        output
        / "warmstart_labels_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("WARM-START LABEL BUILD COMPLETE")
    print("=" * 72)

    print(
        f"Warm-start labels written to: "
        f"{output}"
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Leakage check: "
        f"warm train ∩ val = "
        f"{len(warm_val_overlap)}"
    )

    print(
        f"Leakage check: "
        f"warm train ∩ test = "
        f"{len(warm_test_overlap)}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()