from __future__ import annotations

import argparse

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# These values are overwritten by the pipeline controller.
ORIGINAL_LABEL_FILES = {
    "train": "",
    "val": "",
    "test": "",
}
ET_LABEL_FILES = {
    "train": "",
    "val": "",
    "test": "",
}
RETINA_LABEL_FILES = {
    "train": "",
    "val": "",
    "test": "",
}
ET_UNLABELED_FILE = ""
RETINA_UNLABELED_FILE = ""
OUTPUT_DIR = ""

ALLOWED_CLASSES = [1, 2, 3, 4]
ET_CLASS_OFFSET = "auto"
RETINA_CLASS_OFFSET = 0
MIN_ET_CONFIDENCE = 0.25
MIN_RETINA_CONFIDENCE = 0.25
AGREEMENT_IOU_THRESHOLD = 0.50
MIN_AGREEMENT_CONFIDENCE = 0.25
KEEP_SINGLE_MODEL_HIGH_CONFIDENCE = True
SINGLE_MODEL_MIN_CONFIDENCE = 0.80
NMS_IOU_THRESHOLD = 0.50
GT_DUPLICATE_IOU_THRESHOLD = 0.50

REQUIRED_COLUMNS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + ["confidence", "source", "et_confidence", "retina_confidence"]


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _resolve_class_offset(values: pd.Series, requested, model_name: str) -> int:
    """Automatically map detector classes 0..3 to project classes 1..4."""
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
        unique = set(numeric.unique().tolist())
        if unique and unique.issubset({0, 1, 2, 3}) and 0 in unique:
            print(f"{model_name}: detected classes 0..3; applying class offset +1")
            return 1
        print(f"{model_name}: keeping class IDs unchanged")
        return 0
    return int(requested)


def _standardize(df: pd.DataFrame, *, class_offset, model_name: str) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    aliases = {
        "filename": ["filename", "file_name", "image", "image_name", "img", "path"],
        "class": ["class", "class_id", "label", "category_id", "cls"],
        "xmin": ["xmin", "x_min", "x1", "left"],
        "ymin": ["ymin", "y_min", "y1", "top"],
        "xmax": ["xmax", "x_max", "x2", "right"],
        "ymax": ["ymax", "y_max", "y2", "bottom"],
        "confidence": ["confidence", "conf", "score", "probability", "prob"],
        "source": ["source", "origin", "label_source", "annotation_source"],
    }
    for canonical, names in aliases.items():
        found = _find_column(df, names)
        if found is not None:
            rename_map[found] = canonical
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{model_name}: missing required columns {missing}")

    df["filename"] = df["filename"].astype(str).map(lambda x: os.path.basename(x.strip()))
    resolved_offset = _resolve_class_offset(df["class"], class_offset, model_name)
    df["class"] = pd.to_numeric(df["class"], errors="coerce") + resolved_offset
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "source" not in df.columns:
        df["source"] = ""
    else:
        df["source"] = df["source"].fillna("").astype(str)

    valid = (
        df["class"].isin(ALLOWED_CLASSES)
        & df[["xmin", "ymin", "xmax", "ymax"]].notna().all(axis=1)
        & (df["xmax"] > df["xmin"])
        & (df["ymax"] > df["ymin"])
    )
    df = df.loc[valid].copy()
    df["class"] = df["class"].astype(int)
    return df


def _read(path: str, *, class_offset, model_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{model_name} file not found: {path}")
    return _standardize(pd.read_csv(path), class_offset=class_offset, model_name=model_name)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def _strip_ground_truth(combined: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Return prediction rows from a GT+prediction CSV.

    A meaningful ``source`` column is used first.  Afterwards, exact copies of
    the original GT are removed by filename, class, and coordinates.  This is
    safer than treating every confidence value of 1.0 as GT, because a genuine
    detector prediction can occasionally round to 1.0.
    """
    if combined.empty:
        return combined.copy()

    source = combined["source"].str.lower().str.strip()
    has_meaningful_source = source.ne("").any()

    if has_meaningful_source:
        gt_terms = (
            "gt",
            "ground",
            "manual",
            "original",
            "human",
            "annotation",
        )
        is_gt_source = source.map(
            lambda value: any(term in value for term in gt_terms)
        )
        candidate = combined.loc[~is_gt_source].copy()
    else:
        candidate = combined.copy()

    if gt.empty or candidate.empty:
        print(
            f"Ground-truth stripping: input={len(combined)}, "
            f"removed={len(combined) - len(candidate)}, "
            f"candidates={len(candidate)}"
        )
        return candidate

    candidate = candidate.reset_index(drop=True)
    keep = np.ones(len(candidate), dtype=bool)
    gt_groups = {
        filename: part
        for filename, part in gt.groupby("filename", sort=False)
    }
    coordinate_columns = ["xmin", "ymin", "xmax", "ymax"]

    for index, row in candidate.iterrows():
        image_gt = gt_groups.get(row["filename"])
        if image_gt is None or image_gt.empty:
            continue

        same_class = image_gt.loc[
            image_gt["class"] == int(row["class"]),
            coordinate_columns,
        ]
        if same_class.empty:
            continue

        box = row[coordinate_columns].to_numpy(dtype=float)
        gt_boxes = same_class.to_numpy(dtype=float)

        # GT rows are copied from the original CSV into the combined files.
        # A small tolerance handles harmless CSV float formatting changes.
        exact_gt_copy = np.isclose(
            gt_boxes,
            box[None, :],
            rtol=0.0,
            atol=1e-4,
        ).all(axis=1).any()

        if exact_gt_copy:
            keep[index] = False

    result = candidate.loc[keep].copy()
    print(
        f"Ground-truth stripping: input={len(combined)}, "
        f"removed={len(combined) - len(result)}, "
        f"candidates={len(result)}"
    )
    return result


def _nms(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    kept_parts: List[pd.DataFrame] = []
    for (_, cls_id), part in df.groupby(["filename", "class"], sort=False):
        part = part.sort_values("confidence", ascending=False).reset_index(drop=True)
        boxes = part[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        order = list(range(len(part)))
        keep: List[int] = []
        while order:
            i = order.pop(0)
            keep.append(i)
            if not order:
                break
            ious = _iou_one_to_many(boxes[i], boxes[order])
            order = [j for j, iou in zip(order, ious) if iou < threshold]
        kept_parts.append(part.iloc[keep])
    return pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[0:0].copy()


def _match_image(et: pd.DataFrame, retina: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    et = et.reset_index(drop=True)
    retina = retina.reset_index(drop=True)
    used_et: set[int] = set()
    used_retina: set[int] = set()
    candidates: List[Tuple[float, float, int, int]] = []

    for cls_id in ALLOWED_CLASSES:
        et_idx = et.index[et["class"] == cls_id].tolist()
        rt_idx = retina.index[retina["class"] == cls_id].tolist()
        if not et_idx or not rt_idx:
            continue
        rt_boxes = retina.loc[rt_idx, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        for i in et_idx:
            et_box = et.loc[i, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
            for j, iou in zip(rt_idx, _iou_one_to_many(et_box, rt_boxes)):
                if iou >= AGREEMENT_IOU_THRESHOLD:
                    ec = float(et.loc[i, "confidence"])
                    rc = float(retina.loc[j, "confidence"])
                    candidates.append((float(iou), (ec + rc) / 2.0, i, j))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for iou, _, i, j in candidates:
        if i in used_et or j in used_retina:
            continue
        ec = float(et.loc[i, "confidence"])
        rc = float(retina.loc[j, "confidence"])
        fused_conf = (ec + rc) / 2.0
        if fused_conf < MIN_AGREEMENT_CONFIDENCE:
            continue
        total = max(ec + rc, 1e-12)
        ebox = et.loc[i, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        rbox = retina.loc[j, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        fbox = (ebox * ec + rbox * rc) / total
        rows.append({
            "filename": et.loc[i, "filename"],
            "class": int(et.loc[i, "class"]),
            "xmin": float(fbox[0]),
            "ymin": float(fbox[1]),
            "xmax": float(fbox[2]),
            "ymax": float(fbox[3]),
            "confidence": fused_conf,
            "source": "dual_teacher_agreement",
            "et_confidence": ec,
            "retina_confidence": rc,
            "agreement_iou": iou,
        })
        used_et.add(i)
        used_retina.add(j)

    if KEEP_SINGLE_MODEL_HIGH_CONFIDENCE:
        for i, row in et.iterrows():
            if i in used_et or float(row["confidence"]) < SINGLE_MODEL_MIN_CONFIDENCE:
                continue
            rows.append({
                "filename": row["filename"], "class": int(row["class"]),
                "xmin": float(row["xmin"]), "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]), "ymax": float(row["ymax"]),
                "confidence": float(row["confidence"]),
                "source": "et_only_high_confidence",
                "et_confidence": float(row["confidence"]),
                "retina_confidence": np.nan,
                "agreement_iou": np.nan,
            })
        for j, row in retina.iterrows():
            if j in used_retina or float(row["confidence"]) < SINGLE_MODEL_MIN_CONFIDENCE:
                continue
            rows.append({
                "filename": row["filename"], "class": int(row["class"]),
                "xmin": float(row["xmin"]), "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]), "ymax": float(row["ymax"]),
                "confidence": float(row["confidence"]),
                "source": "retina_only_high_confidence",
                "et_confidence": np.nan,
                "retina_confidence": float(row["confidence"]),
                "agreement_iou": np.nan,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["agreement_iou"])
    return _nms(out, NMS_IOU_THRESHOLD)


def _fuse(et: pd.DataFrame, retina: pd.DataFrame) -> pd.DataFrame:
    et = et.loc[et["confidence"].fillna(-1) >= MIN_ET_CONFIDENCE].copy()
    retina = retina.loc[retina["confidence"].fillna(-1) >= MIN_RETINA_CONFIDENCE].copy()
    names = sorted(set(et["filename"]).union(retina["filename"]))
    pieces: List[pd.DataFrame] = []
    for filename in names:
        part = _match_image(
            et.loc[et["filename"] == filename],
            retina.loc[retina["filename"] == filename],
        )
        if not part.empty:
            pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["agreement_iou"])
    return pd.concat(pieces, ignore_index=True)


def _remove_gt_duplicates(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or gt.empty:
        return pred.copy()
    pred = pred.reset_index(drop=True)
    keep = np.ones(len(pred), dtype=bool)
    gt_groups = {name: part for name, part in gt.groupby("filename")}
    for i, row in pred.iterrows():
        gt_img = gt_groups.get(row["filename"])
        if gt_img is None or gt_img.empty:
            continue
        box = row[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        boxes = gt_img[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        if _iou_one_to_many(box, boxes).max(initial=0.0) >= GT_DUPLICATE_IOU_THRESHOLD:
            keep[i] = False
    return pred.loc[keep].copy()


def _make_combined(gt: pd.DataFrame, fused_predictions: pd.DataFrame) -> pd.DataFrame:
    gt_out = gt[REQUIRED_COLUMNS].copy()
    gt_out["confidence"] = 1.0
    gt_out["source"] = "ground_truth"
    gt_out["et_confidence"] = np.nan
    gt_out["retina_confidence"] = np.nan
    gt_out["agreement_iou"] = np.nan
    novel = _remove_gt_duplicates(fused_predictions, gt)
    cols = REQUIRED_COLUMNS + ["confidence", "source", "et_confidence", "retina_confidence", "agreement_iou"]
    return pd.concat([gt_out[cols], novel.reindex(columns=cols)], ignore_index=True)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows: {path}")


def parse_args():
    p=argparse.ArgumentParser(description='Fuse EfficientTree and RetinaNet predictions.')
    for split in ('train','val','test'):
        p.add_argument(f'--gt-{split}', required=True)
        p.add_argument(f'--et-{split}', required=True)
        p.add_argument(f'--retina-{split}', required=True)
    p.add_argument('--et-unlabeled', required=True)
    p.add_argument('--retina-unlabeled', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--min-et-confidence', type=float, default=0.25)
    p.add_argument('--min-retina-confidence', type=float, default=0.25)
    p.add_argument('--agreement-iou', type=float, default=0.50)
    p.add_argument('--single-model-confidence', type=float, default=0.80)
    p.add_argument('--nms-iou', type=float, default=0.50)
    p.add_argument('--gt-iou', type=float, default=0.50)
    return p.parse_args()

def main() -> None:
    global ORIGINAL_LABEL_FILES, ET_LABEL_FILES, RETINA_LABEL_FILES, ET_UNLABELED_FILE, RETINA_UNLABELED_FILE, OUTPUT_DIR
    global MIN_ET_CONFIDENCE, MIN_RETINA_CONFIDENCE, AGREEMENT_IOU_THRESHOLD, SINGLE_MODEL_MIN_CONFIDENCE, NMS_IOU_THRESHOLD, GT_DUPLICATE_IOU_THRESHOLD
    args=parse_args()
    ORIGINAL_LABEL_FILES={s:getattr(args,f'gt_{s}') for s in ('train','val','test')}
    ET_LABEL_FILES={s:getattr(args,f'et_{s}') for s in ('train','val','test')}
    RETINA_LABEL_FILES={s:getattr(args,f'retina_{s}') for s in ('train','val','test')}
    ET_UNLABELED_FILE=args.et_unlabeled; RETINA_UNLABELED_FILE=args.retina_unlabeled; OUTPUT_DIR=args.output_dir
    MIN_ET_CONFIDENCE=args.min_et_confidence; MIN_RETINA_CONFIDENCE=args.min_retina_confidence
    AGREEMENT_IOU_THRESHOLD=args.agreement_iou; SINGLE_MODEL_MIN_CONFIDENCE=args.single_model_confidence
    NMS_IOU_THRESHOLD=args.nms_iou; GT_DUPLICATE_IOU_THRESHOLD=args.gt_iou
    output = Path(OUTPUT_DIR)
    predictions_dir = output / "predictions"
    combined_dir = output / "combined"

    summary_rows: List[dict] = []
    for split in ("train", "val", "test"):
        gt = _read(ORIGINAL_LABEL_FILES[split], class_offset=0, model_name=f"GT {split}")
        et_combined = _read(ET_LABEL_FILES[split], class_offset=ET_CLASS_OFFSET, model_name=f"ET {split}")
        retina_combined = _read(RETINA_LABEL_FILES[split], class_offset=RETINA_CLASS_OFFSET, model_name=f"Retina {split}")
        et_pred = _strip_ground_truth(et_combined, gt)
        retina_pred = _strip_ground_truth(retina_combined, gt)
        fused = _fuse(et_pred, retina_pred)
        combined = _make_combined(gt, fused)

        _save(fused, predictions_dir / f"{split}_fused_predictions.csv")
        _save(combined, combined_dir / f"{split}_gt_plus_fused_predictions.csv")
        summary_rows.append({
            "split": split,
            "gt_boxes": len(gt),
            "et_candidate_boxes": len(et_pred),
            "retina_candidate_boxes": len(retina_pred),
            "fused_boxes": len(fused),
            "combined_boxes": len(combined),
        })

    et_unlabeled = _read(ET_UNLABELED_FILE, class_offset=ET_CLASS_OFFSET, model_name="ET unlabeled")
    retina_unlabeled = _read(RETINA_UNLABELED_FILE, class_offset=RETINA_CLASS_OFFSET, model_name="Retina unlabeled")
    fused_unlabeled = _fuse(et_unlabeled, retina_unlabeled)
    _save(fused_unlabeled, predictions_dir / "unlabeled_fused_predictions.csv")
    summary_rows.append({
        "split": "unlabeled",
        "gt_boxes": 0,
        "et_candidate_boxes": len(et_unlabeled),
        "retina_candidate_boxes": len(retina_unlabeled),
        "fused_boxes": len(fused_unlabeled),
        "combined_boxes": len(fused_unlabeled),
    })

    _save(pd.DataFrame(summary_rows), output / "fusion_summary.csv")


if __name__ == "__main__":
    main()
