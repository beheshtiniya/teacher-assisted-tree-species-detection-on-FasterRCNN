from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE_COLUMNS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
VALID_CLASSES = {1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Experiment-4 / Dataset-3 BEST train-fusion: RetinaNet + supervised "
            "EfficientTree + SSL-enhanced EfficientTree. Expert GT has absolute "
            "priority. Native model confidence is used; no 0.8 single-model gate."
        )
    )
    p.add_argument("--gt-train", required=True, type=Path)
    p.add_argument("--val-gt", required=True, type=Path)
    p.add_argument("--test-gt", required=True, type=Path)
    p.add_argument("--retina-train", required=True, type=Path)
    p.add_argument("--et-supervised-train", required=True, type=Path)
    p.add_argument("--et-ssl-train", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--min-confidence", type=float, default=0.25)
    p.add_argument("--gt-iou", type=float, default=0.50)
    p.add_argument("--nms-iou", type=float, default=0.50)
    return p.parse_args()


def _find_col(df: pd.DataFrame, aliases: Iterable[str], what: str) -> str:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    raise ValueError(
        f"Could not find {what} column. Available columns: {list(df.columns)}"
    )


def _normalize_filename(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace("\\", "/", regex=False)


def _validate_classes(df: pd.DataFrame, source: str) -> None:
    found = set(df["class"].astype(int).unique().tolist())
    bad = sorted(found - VALID_CLASSES)
    if bad:
        raise ValueError(
            f"{source}: class IDs {sorted(found)} are incompatible with Dataset 3. "
            "Expected only 1,2,3,4. Refusing to silently remap classes."
        )


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"GT file {path} missing columns: {missing}")
    df = df[BASE_COLUMNS].copy()
    df["filename"] = _normalize_filename(df["filename"])
    df["class"] = pd.to_numeric(df["class"], errors="raise").astype(int)
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    _validate_classes(df, "expert_gt")
    bad_box = (df["xmax"] <= df["xmin"]) | (df["ymax"] <= df["ymin"])
    if bad_box.any():
        raise ValueError(f"Expert GT contains {int(bad_box.sum())} invalid boxes")
    return df


def load_predictions(path: Path, source: str, min_conf: float) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path).copy()

    filename = _find_col(raw, ["filename", "image", "image_name", "file"], "filename")
    cls = _find_col(raw, ["class", "class_id", "label", "category_id"], "class")
    xmin = _find_col(raw, ["xmin", "x1", "left"], "xmin")
    ymin = _find_col(raw, ["ymin", "y1", "top"], "ymin")
    xmax = _find_col(raw, ["xmax", "x2", "right"], "xmax")
    ymax = _find_col(raw, ["ymax", "y2", "bottom"], "ymax")
    conf = _find_col(raw, ["confidence", "score", "conf", "probability"], "confidence")

    df = pd.DataFrame(
        {
            "filename": _normalize_filename(raw[filename]),
            "class": pd.to_numeric(raw[cls], errors="raise").astype(int),
            "xmin": pd.to_numeric(raw[xmin], errors="raise").astype(float),
            "ymin": pd.to_numeric(raw[ymin], errors="raise").astype(float),
            "xmax": pd.to_numeric(raw[xmax], errors="raise").astype(float),
            "ymax": pd.to_numeric(raw[ymax], errors="raise").astype(float),
            "confidence": pd.to_numeric(raw[conf], errors="raise").astype(float),
            "source": source,
        }
    )
    _validate_classes(df, source)

    invalid = (df["xmax"] <= df["xmin"]) | (df["ymax"] <= df["ymin"])
    if invalid.any():
        print(f"{source}: dropping {int(invalid.sum())} invalid boxes")
        df = df.loc[~invalid].copy()

    before = len(df)
    # Paper rule: predictions below 0.25 are discarded, so exactly 0.25 is kept.
    df = df.loc[df["confidence"] >= min_conf].copy()
    print(f"{source}: confidence >= {min_conf:.2f}: {len(df):,}/{before:,}")
    return df.reset_index(drop=True)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, xx2 - xx1)
    ih = np.maximum(0.0, yy2 - yy1)
    inter = iw * ih
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = area_a + area_b - inter
    out = np.zeros_like(inter, dtype=np.float64)
    good = union > 0
    out[good] = inter[good] / union[good]
    return out


def remove_gt_overlaps(
    pred: pd.DataFrame, gt: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, int]:
    gt_groups: Dict[str, np.ndarray] = {
        name: group[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        for name, group in gt.groupby("filename", sort=False)
    }
    keep = np.ones(len(pred), dtype=bool)
    removed = 0

    for i, row in enumerate(pred.itertuples(index=False)):
        gt_boxes = gt_groups.get(row.filename)
        if gt_boxes is None or len(gt_boxes) == 0:
            continue
        box = np.array([row.xmin, row.ymin, row.xmax, row.ymax], dtype=float)
        # Paper rule: remove iff max IoU > 0.50; exactly 0.50 remains.
        if np.max(iou_one_to_many(box, gt_boxes), initial=0.0) > threshold:
            keep[i] = False
            removed += 1

    return pred.loc[keep].reset_index(drop=True), removed


def class_agnostic_nms_image(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    # Native model confidences are ranked directly.
    order = np.argsort(-df["confidence"].to_numpy(dtype=float), kind="stable")
    boxes = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    kept: List[int] = []
    suppressed = np.zeros(len(df), dtype=bool)

    for pos in order:
        if suppressed[pos]:
            continue
        kept.append(int(pos))
        overlaps = iou_one_to_many(boxes[pos], boxes)
        # Paper rule: suppress iff IoU > 0.50; exactly 0.50 remains.
        suppressed |= overlaps > threshold
        suppressed[pos] = False

    result = df.iloc[kept].copy()
    return result.sort_values(
        ["filename", "confidence"],
        ascending=[True, False],
        kind="stable",
    )


def class_agnostic_nms_all(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pieces = [
        class_agnostic_nms_image(group.reset_index(drop=True), threshold)
        for _, group in df.groupby("filename", sort=False)
    ]
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(a.gt_train)
    print(f"Expert GT train: {len(gt):,} boxes")

    sources = [
        ("retinanet", a.retina_train),
        ("et_supervised", a.et_supervised_train),
        ("et_ssl", a.et_ssl_train),
    ]

    filtered_parts = []
    summary_rows = []

    for source, path in sources:
        pred = load_predictions(path, source, a.min_confidence)
        after_conf = len(pred)
        pred, removed_gt = remove_gt_overlaps(pred, gt, a.gt_iou)
        print(
            f"{source}: GT IoU > {a.gt_iou:.2f} removed={removed_gt:,}; "
            f"remaining={len(pred):,}"
        )
        filtered_parts.append(pred)
        summary_rows.append(
            {
                "source": source,
                "after_confidence": after_conf,
                "removed_gt_overlap": removed_gt,
                "after_gt_filter": len(pred),
            }
        )

    pooled = pd.concat(filtered_parts, ignore_index=True)
    pooled.to_csv(a.output_dir / "pooled_before_nms.csv", index=False)
    print(f"Pooled before global class-agnostic NMS: {len(pooled):,}")

    accepted = class_agnostic_nms_all(pooled, a.nms_iou)
    accepted.to_csv(a.output_dir / "accepted_train_predictions.csv", index=False)
    print(f"Accepted after global class-agnostic NMS: {len(accepted):,}")

    final_train = pd.concat([gt, accepted[BASE_COLUMNS]], ignore_index=True)
    final_train.to_csv(a.output_dir / "train_final_fused.csv", index=False)

    # Experiment 4 validation and test remain expert-only.
    shutil.copy2(a.val_gt, a.output_dir / "val_expert_only.csv")
    shutil.copy2(a.test_gt, a.output_dir / "test_expert_only.csv")

    summary = pd.DataFrame(summary_rows)
    summary.loc[len(summary)] = {
        "source": "ALL_AFTER_GLOBAL_NMS",
        "after_confidence": len(pooled),
        "removed_gt_overlap": np.nan,
        "after_gt_filter": len(accepted),
    }
    summary.to_csv(a.output_dir / "fusion_summary.csv", index=False)

    settings = {
        "experiment": "Experiment 4 / Dataset 3 / best train-fusion configuration",
        "prediction_sources": [
            "RetinaNet",
            "supervised EfficientTree",
            "SSL-enhanced EfficientTree",
        ],
        "minimum_confidence": a.min_confidence,
        "confidence_rule": "native model confidence; no single-model >=0.8 gate",
        "gt_filter_rule": f"remove iff max IoU with any expert GT > {a.gt_iou}",
        "gt_filter_class_agnostic": True,
        "prediction_nms_rule": (
            f"greedy class-agnostic NMS; suppress iff IoU > {a.nms_iou}; "
            "keep higher native confidence"
        ),
        "unlabeled_predictions_added_to_final_dataset": False,
        "expert_gt_boxes": len(gt),
        "accepted_prediction_boxes": len(accepted),
        "final_train_boxes": len(final_train),
    }
    (a.output_dir / "fusion_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFINAL EXPERIMENT-4 BEST DATASET CREATED")
    print("Train:", a.output_dir / "train_final_fused.csv")
    print("Val:  ", a.output_dir / "val_expert_only.csv")
    print("Test: ", a.output_dir / "test_expert_only.csv")
    print(
        f"Final train boxes = {len(gt):,} expert + "
        f"{len(accepted):,} accepted = {len(final_train):,}"
    )


if __name__ == "__main__":
    main()
