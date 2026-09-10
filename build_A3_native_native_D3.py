from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# Dataset3 / Experiment 4 — Experiment3 Best-A3 replication
# Build A3_native,native manually from:
#   expert GT
#   + accepted A2-native predictions
#   + SSL predictions on labeled train
#   + SSL predictions on unlabeled images
#
# Exact rules:
#   confidence >= 0.25
#   GT-aware removal only when IoU > 0.50
#   class-agnostic greedy NMS, suppress only when IoU > 0.50
#   exact 0.25 and exact 0.50 are retained
#   expert GT is never removed/replaced
# ============================================================

MIN_CONFIDENCE = 0.25
GT_IOU = 0.50
NMS_IOU = 0.50

A2_CSV = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments"
    r"\exp4_D3_A2_native\train_A2_native.csv"
)
SSL_TRAIN_CSV = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\predictions"
    r"\exp4_D3_A2_native_ssl\train_predictions_256.csv"
)
SSL_UNLABELED_CSV = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\predictions"
    r"\exp4_D3_A2_native_ssl\unlabeled_predictions_256.csv"
)
UNLABELED_LIST = Path(
    r"E:\FASTRCNN\teacher_student\labels"
    r"\article2_hemogen_mydataset\unlabeled_images.txt"
)

OUT_DIR = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments"
    r"\exp4_D3_A3_native_native"
)

TRAIN_OUT = OUT_DIR / "train_A3_native_native.csv"
LABELED_PSEUDO_OUT = OUT_DIR / "A3_native_native_labeled_predictions.csv"
UNLABELED_PSEUDO_OUT = OUT_DIR / "A3_native_native_unlabeled_predictions.csv"
SSL_GT_REMOVED_OUT = OUT_DIR / "A3_native_native_ssl_train_removed_by_GT_overlap.csv"
SUMMARY_OUT = OUT_DIR / "A3_native_native_summary.json"

REQ7 = ["filename", "class", "xmin", "ymin", "xmax", "ymax", "confidence"]
TRAIN6 = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file:\n{path}")


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}\nFound: {list(df.columns)}")


def normalize_types(df: pd.DataFrame, with_source: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["filename"] = out["filename"].astype(str).map(lambda s: s.replace("\\", "/").split("/")[-1])
    out["class"] = pd.to_numeric(out["class"], errors="raise").astype(int)
    for c in ["xmin", "ymin", "xmax", "ymax", "confidence"]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    if with_source:
        out["source"] = out["source"].astype(str)
    return out


def iou_one_to_many(box, boxes):
    """IoU of one box against a list of boxes. Boxes are xyxy."""
    if not boxes:
        return []

    ax1, ay1, ax2, ay2 = box
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

    vals = []
    for bx1, by1, bx2, by2 in boxes:
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = a_area + b_area - inter
        vals.append(0.0 if union <= 0.0 else inter / union)
    return vals


def gt_filter(preds: pd.DataFrame, gt: pd.DataFrame):
    """
    Class-agnostic GT-aware filter.
    Remove prediction only if max IoU with expert GT in same image > 0.50.
    Exact 0.50 is retained.
    """
    gt_boxes_by_image = {}
    for fn, g in gt.groupby("filename", sort=False):
        gt_boxes_by_image[fn] = list(
            g[["xmin", "ymin", "xmax", "ymax"]].itertuples(index=False, name=None)
        )

    keep_rows = []
    removed_rows = []

    for idx, row in preds.iterrows():
        box = (row.xmin, row.ymin, row.xmax, row.ymax)
        boxes = gt_boxes_by_image.get(row.filename, [])
        vals = iou_one_to_many(box, boxes)
        max_iou = max(vals) if vals else 0.0

        rec = row.to_dict()
        rec["max_gt_iou"] = float(max_iou)

        if max_iou > GT_IOU:  # exact .50 retained
            removed_rows.append(rec)
        else:
            keep_rows.append(row.to_dict())

    kept = pd.DataFrame(keep_rows, columns=list(preds.columns))
    removed_cols = list(preds.columns) + ["max_gt_iou"]
    removed = pd.DataFrame(removed_rows, columns=removed_cols)
    return kept, removed


def greedy_class_agnostic_nms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Native-confidence, class-agnostic greedy NMS per image.
    Higher confidence wins. Ties preserve input order.
    Suppress only when IoU > 0.50; exact 0.50 is retained.
    """
    if df.empty:
        return df.copy()

    work = df.copy().reset_index(drop=True)
    work["_stable_order"] = range(len(work))

    accepted = []

    for fn, g in work.groupby("filename", sort=False):
        g = g.sort_values(
            ["confidence", "_stable_order"],
            ascending=[False, True],
            kind="mergesort",
        )

        kept_boxes = []
        kept_rows = []

        for _, row in g.iterrows():
            box = (row.xmin, row.ymin, row.xmax, row.ymax)
            overlaps = iou_one_to_many(box, kept_boxes)

            # Suppress only if IoU is strictly greater than .50.
            if overlaps and max(overlaps) > NMS_IOU:
                continue

            kept_boxes.append(box)
            kept_rows.append(row)

        accepted.extend(kept_rows)

    out = pd.DataFrame(accepted)
    if "_stable_order" in out.columns:
        out = out.drop(columns=["_stable_order"])
    return out.reset_index(drop=True)


def source_counts(df: pd.DataFrame) -> dict:
    if "source" not in df.columns or df.empty:
        return {}
    return {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()}


def class_counts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    vc = df["class"].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.to_dict().items()}


def read_unlabeled_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip().strip('"').strip("'")
        if not s:
            continue
        names.append(s.replace("\\", "/").split("/")[-1])
    return names


def main():
    for p in [A2_CSV, SSL_TRAIN_CSV, SSL_UNLABELED_CSV, UNLABELED_LIST]:
        require_file(p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a2 = pd.read_csv(A2_CSV)
    ssl_train = pd.read_csv(SSL_TRAIN_CSV)
    ssl_unl = pd.read_csv(SSL_UNLABELED_CSV)

    require_columns(a2, REQ7 + ["source"], "A2")
    require_columns(ssl_train, REQ7, "SSL train")
    require_columns(ssl_unl, REQ7, "SSL unlabeled")

    a2 = normalize_types(a2, with_source=True)
    ssl_train = normalize_types(ssl_train)
    ssl_unl = normalize_types(ssl_unl)

    # Canonical A2 checks for the current Dataset3 experiment.
    expert = a2[a2["source"].str.lower() == "expert"].copy().reset_index(drop=True)
    a2_pseudo = a2[a2["source"].str.lower() != "expert"].copy().reset_index(drop=True)

    if len(a2) != 10986:
        raise RuntimeError(f"Unexpected A2 total: {len(a2)}; expected 10986")
    if len(expert) != 6078:
        raise RuntimeError(f"Unexpected expert GT count: {len(expert)}; expected 6078")
    if len(a2_pseudo) != 4908:
        raise RuntimeError(f"Unexpected A2 pseudo count: {len(a2_pseudo)}; expected 4908")

    if (a2_pseudo["confidence"] < MIN_CONFIDENCE).any():
        n = int((a2_pseudo["confidence"] < MIN_CONFIDENCE).sum())
        raise RuntimeError(
            f"A2 canonical pseudo set contains {n} rows below confidence 0.25. "
            "Refusing to silently change canonical A2."
        )

    # SSL native-confidence filtering: exact 0.25 retained.
    ssl_train_raw_n = len(ssl_train)
    ssl_unl_raw_n = len(ssl_unl)

    ssl_train = ssl_train[ssl_train["confidence"] >= MIN_CONFIDENCE].copy().reset_index(drop=True)
    ssl_unl = ssl_unl[ssl_unl["confidence"] >= MIN_CONFIDENCE].copy().reset_index(drop=True)

    ssl_train["source"] = "et_ssl"
    ssl_unl["source"] = "et_ssl_unlabeled"

    # SSL predictions on labeled training images:
    # apply class-agnostic GT-aware filter.
    ssl_train_gtkept, ssl_train_removed = gt_filter(ssl_train, expert)

    # Re-pool canonical A2 accepted predictions + accepted SSL train predictions.
    # A2 rows come first so exact-confidence ties are deterministic.
    labeled_pool = pd.concat(
        [a2_pseudo[REQ7 + ["source"]], ssl_train_gtkept[REQ7 + ["source"]]],
        ignore_index=True,
    )
    labeled_accepted = greedy_class_agnostic_nms(labeled_pool)

    # Unlabeled predictions:
    # NO GT-aware filtering. Only confidence + class-agnostic NMS.
    unlabeled_names = read_unlabeled_names(UNLABELED_LIST)
    unlabeled_set = set(unlabeled_names)

    if len(unlabeled_set) != 7059:
        raise RuntimeError(
            f"Unexpected unique unlabeled-image count: {len(unlabeled_set)}; expected 7059"
        )

    bad_unl = sorted(set(ssl_unl["filename"]) - unlabeled_set)
    if bad_unl:
        raise RuntimeError(
            "SSL unlabeled prediction file contains filenames not present in "
            f"unlabeled_images.txt. First examples: {bad_unl[:10]}"
        )

    train_names = set(expert["filename"])
    leakage = sorted(train_names & unlabeled_set)
    if leakage:
        raise RuntimeError(
            f"Train/unlabeled filename leakage detected. First examples: {leakage[:10]}"
        )

    unlabeled_accepted = greedy_class_agnostic_nms(ssl_unl[REQ7 + ["source"]])

    # Final A3 training set.
    # Expert GT has absolute priority and is never removed/replaced.
    final_pseudo = pd.concat(
        [labeled_accepted, unlabeled_accepted],
        ignore_index=True,
    )

    final_train = pd.concat(
        [
            expert[TRAIN6],
            final_pseudo[TRAIN6],
        ],
        ignore_index=True,
    )

    # Write training CSV with only detector-training columns.
    final_train.to_csv(TRAIN_OUT, index=False)

    # Write audit CSVs with confidence/source preserved.
    labeled_accepted[REQ7 + ["source"]].to_csv(LABELED_PSEUDO_OUT, index=False)
    unlabeled_accepted[REQ7 + ["source"]].to_csv(UNLABELED_PSEUDO_OUT, index=False)
    ssl_train_removed.to_csv(SSL_GT_REMOVED_OUT, index=False)

    summary = {
        "method": "Dataset3 analogue of Experiment3 Best A3_native,native",
        "threshold_semantics": {
            "confidence": "retain confidence >= 0.25",
            "gt_overlap": "remove only if class-agnostic max GT IoU > 0.50",
            "nms": "class-agnostic greedy NMS; suppress only if IoU > 0.50",
            "expert_gt": "absolute priority; never removed or replaced",
        },
        "inputs": {
            "a2_csv": str(A2_CSV),
            "ssl_train_csv": str(SSL_TRAIN_CSV),
            "ssl_unlabeled_csv": str(SSL_UNLABELED_CSV),
            "unlabeled_list": str(UNLABELED_LIST),
        },
        "counts": {
            "a2_total": int(len(a2)),
            "expert_gt": int(len(expert)),
            "a2_accepted_pseudo": int(len(a2_pseudo)),
            "a2_pseudo_sources": source_counts(a2_pseudo),
            "ssl_train_raw": int(ssl_train_raw_n),
            "ssl_train_after_confidence": int(len(ssl_train)),
            "ssl_train_removed_by_gt_overlap": int(len(ssl_train_removed)),
            "ssl_train_after_gt_filter": int(len(ssl_train_gtkept)),
            "labeled_pool_before_nms": int(len(labeled_pool)),
            "labeled_pseudo_after_nms": int(len(labeled_accepted)),
            "unlabeled_images_in_list": int(len(unlabeled_set)),
            "ssl_unlabeled_raw": int(ssl_unl_raw_n),
            "ssl_unlabeled_after_confidence": int(len(ssl_unl)),
            "unlabeled_pseudo_after_nms": int(len(unlabeled_accepted)),
            "all_pseudo_final": int(len(final_pseudo)),
            "final_train_rows": int(len(final_train)),
        },
        "class_counts": {
            "expert_gt": class_counts(expert),
            "labeled_pseudo_final": class_counts(labeled_accepted),
            "unlabeled_pseudo_final": class_counts(unlabeled_accepted),
            "all_pseudo_final": class_counts(final_pseudo),
            "final_train": class_counts(final_train),
        },
        "source_counts": {
            "labeled_pseudo_final": source_counts(labeled_accepted),
            "unlabeled_pseudo_final": source_counts(unlabeled_accepted),
            "all_pseudo_final": source_counts(final_pseudo),
        },
        "outputs": {
            "train": str(TRAIN_OUT),
            "labeled_predictions": str(LABELED_PSEUDO_OUT),
            "unlabeled_predictions": str(UNLABELED_PSEUDO_OUT),
            "ssl_train_removed_by_gt_overlap": str(SSL_GT_REMOVED_OUT),
        },
    }

    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== A3_native,native BUILD COMPLETE ===")
    print(f"Expert GT                         : {len(expert)}")
    print(f"A2 canonical accepted pseudo      : {len(a2_pseudo)}")
    print(f"SSL train raw                     : {ssl_train_raw_n}")
    print(f"SSL train after conf >= .25       : {len(ssl_train)}")
    print(f"SSL train removed by GT IoU > .50 : {len(ssl_train_removed)}")
    print(f"SSL train after GT filter         : {len(ssl_train_gtkept)}")
    print(f"Labeled pool before NMS           : {len(labeled_pool)}")
    print(f"Labeled pseudo after NMS          : {len(labeled_accepted)}")
    print(f"SSL unlabeled raw                 : {ssl_unl_raw_n}")
    print(f"SSL unlabeled after conf >= .25   : {len(ssl_unl)}")
    print(f"Unlabeled pseudo after NMS        : {len(unlabeled_accepted)}")
    print(f"FINAL pseudo total                : {len(final_pseudo)}")
    print(f"FINAL train rows                  : {len(final_train)}")
    print("\nFinal class counts:")
    for k, v in class_counts(final_train).items():
        print(f"  C{k}: {v}")
    print("\nFinal pseudo source counts:")
    for k, v in source_counts(final_pseudo).items():
        print(f"  {k}: {v}")
    print(f"\nTRAIN CSV:\n{TRAIN_OUT}")
    print(f"\nSUMMARY:\n{SUMMARY_OUT}")


if __name__ == "__main__":
    main()
