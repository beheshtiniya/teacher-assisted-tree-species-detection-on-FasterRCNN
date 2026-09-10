from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_config
from src.common.process import run


ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Experiment 4 / Dataset 3 / replication of the Experiment-3 Best-A3 pathway
#
# D3 expert GT
#   -> RetinaNet + supervised EfficientTree
#   -> A2-native fusion
#   -> EfficientTree supervised on A2-native
#   -> warm-start Teacher-Student SSL from A2-native checkpoint
#   -> native-confidence SSL predictions
#   -> A3_native,native
#   -> independently initialized Faster R-CNN
# ---------------------------------------------------------------------------

EXPECTED_D3 = {
    "train": {
        "images": 2663,
        "boxes": 6078,
        "classes": {1: 2069, 2: 1411, 3: 356, 4: 2242},
    },
    "val": {
        "images": 888,
        "boxes": 2026,
        "classes": {1: 690, 2: 470, 3: 119, 4: 747},
    },
    "test": {
        "images": 887,
        "boxes": 2024,
        "classes": {1: 690, 2: 470, 3: 118, 4: 746},
    },
}
EXPECTED_UNLABELED_IMAGES = 7059

MIN_CONFIDENCE = 0.25
GT_IOU = 0.50
NMS_IOU = 0.50

BASE_COLUMNS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
PRED_COLUMNS = BASE_COLUMNS + ["confidence", "source"]


# ===========================================================================
# General helpers
# ===========================================================================

def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        print("REMOVE DIR:", path)
        shutil.rmtree(path)
    elif path.is_file():
        print("REMOVE FILE:", path)
        path.unlink()


def _read_yaml(path: Path) -> dict[str, Any]:
    _require_file(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not a mapping: {path}")
    return data


def _yaml_run_dir(path: Path) -> Path:
    cfg = _read_yaml(path)
    project = Path(str(cfg["project"]))
    name = str(cfg["name"])
    return project / name


def _yaml_best(path: Path) -> Path:
    return _yaml_run_dir(path) / "weights" / "best.pt"


def _choose_ssl_checkpoint(run_dir: Path) -> Path:
    weights = run_dir / "weights"
    for name in ("best_ema.pt", "best.pt", "best_student.pt"):
        p = weights / name
        if p.is_file():
            print("Using SSL checkpoint:", p)
            return p
    raise FileNotFoundError(
        f"No best_ema.pt, best.pt, or best_student.pt under: {weights}"
    )


def _choose_retina_checkpoint(summary: Path, model_root: Path) -> Path:
    _require_file(summary)
    d = pd.read_csv(summary)
    metric = "best_val_map_50"
    if "run" not in d.columns or metric not in d.columns:
        raise ValueError(
            f"{summary}: expected columns 'run' and '{metric}', got {list(d.columns)}"
        )
    row = d.loc[pd.to_numeric(d[metric], errors="raise").idxmax()]
    run_id = int(row["run"])
    return _require_file(model_root / f"run_{run_id:02d}" / "best_model.pth")


def _find_retina_train_predictions(root: Path) -> Path:
    pred_dir = root / "predictions"
    exact = pred_dir / "predictions_train_after_confidence_filter.csv"
    if exact.is_file():
        return exact
    matches = sorted(pred_dir.glob("predictions_train_after_confidence_filter*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No RetinaNet confidence-filtered TRAIN predictions under {pred_dir}"
        )
    raise RuntimeError(
        "Multiple RetinaNet TRAIN confidence files found; refusing to guess:\n"
        + "\n".join(str(x) for x in matches)
    )


def _find_et_predictions(root: Path, split: str) -> Path:
    exact_names = {
        "train": ["train_predictions_256.csv"],
        "unlabeled": ["unlabeled_predictions_256.csv"],
        "val": ["val_predictions_256.csv"],
        "test": ["test_predictions_256.csv"],
    }
    for name in exact_names.get(split, []):
        p = root / name
        if p.is_file():
            return p

    matches = sorted(root.glob(f"{split}_predictions*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {split} EfficientTree prediction CSV under {root}")
    raise RuntimeError(
        f"Multiple {split} EfficientTree prediction files found; refusing to guess:\n"
        + "\n".join(str(x) for x in matches)
    )


# ===========================================================================
# Dataset-3 validation
# ===========================================================================

def _audit_d3(labels: Path) -> None:
    print("Auditing fixed Dataset 3 expert splits...")
    split_names: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        path = labels / f"{split}_labels.csv"
        _require_file(path)
        df = pd.read_csv(path)

        required = set(BASE_COLUMNS)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        df["filename"] = df["filename"].astype(str).map(lambda x: Path(x.strip()).name)
        df["class"] = pd.to_numeric(df["class"], errors="raise").astype(int)

        n_images = int(df["filename"].nunique())
        n_boxes = int(len(df))
        class_counts = {
            int(k): int(v)
            for k, v in df["class"].value_counts().sort_index().items()
        }

        exp = EXPECTED_D3[split]
        print(
            f"{split}: images={n_images}, boxes={n_boxes}, classes={class_counts}"
        )

        if n_images != exp["images"]:
            raise ValueError(
                f"{split}: expected {exp['images']} images, found {n_images}"
            )
        if n_boxes != exp["boxes"]:
            raise ValueError(
                f"{split}: expected {exp['boxes']} boxes, found {n_boxes}"
            )
        if class_counts != exp["classes"]:
            raise ValueError(
                f"{split}: expected classes {exp['classes']}, found {class_counts}"
            )

        split_names[split] = set(df["filename"])

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_names[a] & split_names[b]
        if overlap:
            raise ValueError(
                f"Filename leakage {a} vs {b}: {len(overlap)}; "
                f"examples={sorted(overlap)[:10]}"
            )

    unlabeled = labels / "unlabeled_images.txt"
    _require_file(unlabeled)
    names = [
        Path(x.strip()).name
        for x in unlabeled.read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]
    if len(names) != EXPECTED_UNLABELED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_UNLABELED_IMAGES} unlabeled images, found {len(names)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames in unlabeled_images.txt")

    labeled = split_names["train"] | split_names["val"] | split_names["test"]
    leak = labeled & set(names)
    if leak:
        raise ValueError(
            f"Labeled/unlabeled leakage: {len(leak)}; examples={sorted(leak)[:10]}"
        )

    print(f"unlabeled: images={len(names)}")
    print("Dataset 3 audit PASSED.")


# ===========================================================================
# Permanent YAML validation
# ===========================================================================

def _validate_yaml(
    path: Path,
    *,
    dataset_dir: Path,
    expected_burn: int,
    expected_train_domain: bool,
    expected_weights: Path | None = None,
) -> None:
    cfg = _read_yaml(path)
    ds = cfg.get("Dataset", {})
    ssod = cfg.get("SSOD", {})
    hyp = cfg.get("hyp", {})

    expected = {
        "train": dataset_dir / "splits" / "train.txt",
        "val": dataset_dir / "splits" / "val.txt",
        "test": dataset_dir / "splits" / "test.txt",
        "target": dataset_dir / "splits" / "target_unlabeled.txt",
    }

    for key, exp in expected.items():
        actual = Path(str(ds.get(key, "")))
        if actual != exp:
            raise ValueError(
                f"{path.name}: Dataset.{key} mismatch\n"
                f"Expected: {exp}\nActual:   {actual}"
            )

    if int(hyp.get("burn_epochs", -999)) != expected_burn:
        raise ValueError(
            f"{path.name}: expected hyp.burn_epochs={expected_burn}, "
            f"found {hyp.get('burn_epochs')}"
        )

    if bool(ssod.get("train_domain")) != expected_train_domain:
        raise ValueError(
            f"{path.name}: expected SSOD.train_domain={expected_train_domain}, "
            f"found {ssod.get('train_domain')}"
        )

    if expected_weights is not None:
        actual_weights = Path(str(cfg.get("weights", "")))
        if actual_weights != expected_weights:
            raise ValueError(
                f"{path.name}: weights mismatch\n"
                f"Expected: {expected_weights}\nActual:   {actual_weights}"
            )

    print("YAML OK:", path)


# ===========================================================================
# Prediction-table normalization
# ===========================================================================

def _pick_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _load_boxes(
    path: Path,
    *,
    require_confidence: bool,
    source: str,
) -> pd.DataFrame:
    _require_file(path)
    raw = pd.read_csv(path)

    aliases = {
        "filename": ("filename", "file_name", "image", "image_name", "img", "path"),
        "class": ("class", "class_id", "label", "category_id", "cls"),
        "xmin": ("xmin", "x_min", "x1", "left"),
        "ymin": ("ymin", "y_min", "y1", "top"),
        "xmax": ("xmax", "x_max", "x2", "right"),
        "ymax": ("ymax", "y_max", "y2", "bottom"),
        "confidence": ("confidence", "conf", "score", "probability", "prob"),
    }

    cols = {k: _pick_column(raw, v) for k, v in aliases.items()}
    required = BASE_COLUMNS + (["confidence"] if require_confidence else [])
    missing = [k for k in required if cols.get(k) is None]
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}; available={list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["filename"] = raw[cols["filename"]].astype(str).map(
        lambda x: Path(x.strip().strip('"').strip("'")).name
    )
    out["class"] = pd.to_numeric(raw[cols["class"]], errors="raise").astype(int)

    for c in ("xmin", "ymin", "xmax", "ymax"):
        out[c] = pd.to_numeric(raw[cols[c]], errors="raise").astype(float)

    if require_confidence:
        out["confidence"] = pd.to_numeric(
            raw[cols["confidence"]], errors="raise"
        ).astype(float)
    else:
        out["confidence"] = 1.0

    out["source"] = source

    if not out["class"].isin([1, 2, 3, 4]).all():
        bad = sorted(set(out.loc[~out["class"].isin([1, 2, 3, 4]), "class"]))
        raise ValueError(f"{path}: classes outside 1..4: {bad}")

    if require_confidence and not out["confidence"].between(0.0, 1.0).all():
        raise ValueError(f"{path}: confidence outside [0,1]")

    valid_box = (
        (out["xmin"] >= 0)
        & (out["ymin"] >= 0)
        & (out["xmax"] > out["xmin"])
        & (out["ymax"] > out["ymin"])
    )
    if not valid_box.all():
        raise ValueError(f"{path}: invalid bounding boxes found")

    return out[PRED_COLUMNS].reset_index(drop=True)


# ===========================================================================
# Fusion
# ===========================================================================

def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if len(others) == 0:
        return np.zeros((0,), dtype=np.float64)

    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.maximum(
        0.0,
        (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]),
    )
    union = area_a + area_b - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float64),
        where=union > 0,
    )


def _filter_confidence(pred: pd.DataFrame) -> pd.DataFrame:
    # "below 0.25 discarded" => confidence exactly 0.25 is retained.
    return pred.loc[pred["confidence"] >= MIN_CONFIDENCE].copy().reset_index(drop=True)


def _filter_against_gt(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_index = {
        fn: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
        for fn, g in gt.groupby("filename", sort=False)
    }

    keep_mask = np.ones(len(pred), dtype=bool)
    max_ious = np.zeros(len(pred), dtype=np.float64)

    for i, row in enumerate(pred.itertuples(index=False)):
        g = gt_index.get(row.filename)
        if g is None or len(g) == 0:
            continue

        box = np.array([row.xmin, row.ymin, row.xmax, row.ymax], dtype=np.float64)
        max_iou = float(_iou_one_to_many(box, g).max())
        max_ious[i] = max_iou

        # Exact methodological rule:
        # > 0.50 removed, == 0.50 retained.
        if max_iou > GT_IOU:
            keep_mask[i] = False

    audited = pred.copy()
    audited["max_gt_iou"] = max_ious

    kept = audited.loc[keep_mask].copy().reset_index(drop=True)
    removed = audited.loc[~keep_mask].copy().reset_index(drop=True)
    return kept, removed


def _class_agnostic_greedy_nms(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred.copy()

    work = pred.copy().reset_index(drop=True)
    work["_stable_order"] = np.arange(len(work))

    kept_indices: list[int] = []

    for _, group in work.groupby("filename", sort=False):
        # Native model confidence, descending. Stable tie behavior is deterministic.
        group = group.sort_values(
            ["confidence", "_stable_order"],
            ascending=[False, True],
            kind="mergesort",
        )

        candidate_indices = list(group.index)

        while candidate_indices:
            current = candidate_indices.pop(0)
            kept_indices.append(current)

            if not candidate_indices:
                break

            box = work.loc[
                current, ["xmin", "ymin", "xmax", "ymax"]
            ].to_numpy(dtype=np.float64)

            others = work.loc[
                candidate_indices, ["xmin", "ymin", "xmax", "ymax"]
            ].to_numpy(dtype=np.float64)

            ious = _iou_one_to_many(box, others)

            # Exact rule: suppress only when IoU > 0.50.
            candidate_indices = [
                idx
                for idx, iou in zip(candidate_indices, ious)
                if float(iou) <= NMS_IOU
            ]

    out = work.loc[kept_indices].copy()
    out = out.drop(columns=["_stable_order"])
    return out.reset_index(drop=True)


def _save_training_csv(path: Path, gt: pd.DataFrame, pred: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    gt6 = gt[BASE_COLUMNS].copy()
    pred6 = pred[BASE_COLUMNS].copy()

    final = pd.concat([gt6, pred6], ignore_index=True)
    final.to_csv(path, index=False, encoding="utf-8-sig")


def _counts(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "images": int(df["filename"].nunique()) if len(df) else 0,
        "classes": {
            str(k): int(v)
            for k, v in df["class"].value_counts().sort_index().items()
        } if len(df) else {},
    }


def _build_a2_native(
    *,
    gt_csv: Path,
    retina_csv: Path,
    et_csv: Path,
    output_dir: Path,
) -> Path:
    print("Building A2-native...")

    gt = _load_boxes(gt_csv, require_confidence=False, source="expert")
    retina = _load_boxes(
        retina_csv, require_confidence=True, source="retinanet"
    )
    et = _load_boxes(
        et_csv, require_confidence=True, source="et_supervised_D3_A0"
    )

    retina = _filter_confidence(retina)
    et = _filter_confidence(et)

    retina_kept, retina_removed = _filter_against_gt(retina, gt)
    et_kept, et_removed = _filter_against_gt(et, gt)

    pool = pd.concat([retina_kept, et_kept], ignore_index=True)
    accepted = _class_agnostic_greedy_nms(pool)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_A2_native.csv"
    _save_training_csv(train_csv, gt, accepted)

    accepted.to_csv(
        output_dir / "A2_native_accepted_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([retina_removed, et_removed], ignore_index=True).to_csv(
        output_dir / "A2_native_removed_by_GT_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "definition": "Dataset3 expert GT + native-confidence RetinaNet/EfficientTree fusion",
        "minimum_confidence": MIN_CONFIDENCE,
        "gt_rule": "remove only if max class-agnostic IoU with expert GT > 0.50",
        "nms_rule": "greedy class-agnostic NMS; suppress only if IoU > 0.50",
        "gt": _counts(gt),
        "retina_after_confidence": _counts(retina),
        "et_after_confidence": _counts(et),
        "retina_after_gt_filter": _counts(retina_kept),
        "et_after_gt_filter": _counts(et_kept),
        "accepted_after_nms": _counts(accepted),
        "final_train": {
            "rows": int(len(gt) + len(accepted)),
            "images": int(
                pd.concat([gt[["filename"]], accepted[["filename"]]])
                ["filename"].nunique()
            ),
        },
        "accepted_by_source": {
            str(k): int(v)
            for k, v in accepted["source"].value_counts().items()
        },
    }
    (output_dir / "A2_native_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("A2-native completed:", train_csv)
    print("Accepted pseudo-labels:", len(accepted))
    print("Final boxes:", len(gt) + len(accepted))
    return train_csv


def _build_a3_native_native(
    *,
    gt_csv: Path,
    a2_accepted_csv: Path,
    ssl_train_csv: Path,
    ssl_unlabeled_csv: Path,
    unlabeled_list: Path,
    output_dir: Path,
) -> Path:
    print("Building A3_native,native...")

    gt = _load_boxes(gt_csv, require_confidence=False, source="expert")
    a2 = _load_boxes(
        a2_accepted_csv, require_confidence=True, source="a2_native"
    )

    # Preserve the actual source names saved by A2 if available.
    a2_raw = pd.read_csv(a2_accepted_csv)
    if "source" in a2_raw.columns and len(a2_raw) == len(a2):
        a2["source"] = a2_raw["source"].astype(str).values

    ssl_train = _load_boxes(
        ssl_train_csv, require_confidence=True, source="et_ssl_train"
    )
    ssl_unlabeled = _load_boxes(
        ssl_unlabeled_csv, require_confidence=True, source="et_ssl_unlabeled"
    )

    ssl_train = _filter_confidence(ssl_train)
    ssl_unlabeled = _filter_confidence(ssl_unlabeled)

    # TRAIN predictions are GT-aware.
    ssl_train_kept, ssl_train_removed = _filter_against_gt(ssl_train, gt)

    # Re-pool A2 accepted predictions with SSL TRAIN predictions so the higher
    # native-confidence box wins when sources overlap.
    labeled_pool = pd.concat([a2, ssl_train_kept], ignore_index=True)
    accepted_labeled = _class_agnostic_greedy_nms(labeled_pool)

    # UNLABELED predictions: no GT filtering; confidence + class-agnostic NMS only.
    unlabeled_names = {
        Path(x.strip()).name
        for x in _require_file(unlabeled_list)
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if x.strip()
    }

    bad_unlabeled = set(ssl_unlabeled["filename"]) - unlabeled_names
    if bad_unlabeled:
        raise ValueError(
            "SSL unlabeled prediction CSV contains filenames not in "
            f"unlabeled_images.txt; examples={sorted(bad_unlabeled)[:10]}"
        )

    accepted_unlabeled = _class_agnostic_greedy_nms(ssl_unlabeled)

    all_pred = pd.concat(
        [accepted_labeled, accepted_unlabeled],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "train_A3_native_native.csv"
    _save_training_csv(train_csv, gt, all_pred)

    accepted_labeled.to_csv(
        output_dir / "A3_native_native_labeled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    accepted_unlabeled.to_csv(
        output_dir / "A3_native_native_unlabeled_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ssl_train_removed.to_csv(
        output_dir / "A3_native_native_ssl_train_removed_by_GT_overlap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "definition": "A3_native,native from A2-native warm-start SSL",
        "minimum_confidence": MIN_CONFIDENCE,
        "gt_rule_labeled_images": (
            "remove only if max class-agnostic IoU with expert GT > 0.50"
        ),
        "unlabeled_rule": "no GT filtering; native confidence + class-agnostic NMS",
        "nms_rule": "greedy class-agnostic NMS; suppress only if IoU > 0.50",
        "gt": _counts(gt),
        "A2_native_input": _counts(a2),
        "ssl_train_after_confidence": _counts(ssl_train),
        "ssl_train_after_gt_filter": _counts(ssl_train_kept),
        "accepted_labeled_after_repool_NMS": _counts(accepted_labeled),
        "ssl_unlabeled_after_confidence": _counts(ssl_unlabeled),
        "accepted_unlabeled_after_NMS": _counts(accepted_unlabeled),
        "final_pseudo": _counts(all_pred),
        "final_train": {
            "rows": int(len(gt) + len(all_pred)),
            "images": int(
                pd.concat([gt[["filename"]], all_pred[["filename"]]])
                ["filename"].nunique()
            ),
        },
        "accepted_labeled_by_source": {
            str(k): int(v)
            for k, v in accepted_labeled["source"].value_counts().items()
        },
    }

    (output_dir / "A3_native_native_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("A3_native,native completed:", train_csv)
    print("Accepted labeled pseudo-labels:", len(accepted_labeled))
    print("Accepted unlabeled pseudo-labels:", len(accepted_unlabeled))
    print("Final boxes:", len(gt) + len(all_pred))
    return train_csv


# ===========================================================================
# Reproducibility manifest
# ===========================================================================

def _write_manifest(
    *,
    path: Path,
    files: dict[str, Path],
    extra: dict[str, Any],
) -> None:
    manifest: dict[str, Any] = {
        "experiment": (
            "Experiment 4 / Dataset 3 / replicated Best-A3 pathway "
            "(A2-native -> A3_native,native)"
        ),
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": {
            "minimum_confidence": MIN_CONFIDENCE,
            "gt_filter": "remove prediction only when max class-agnostic GT IoU > 0.50",
            "prediction_nms": "greedy class-agnostic NMS; suppress only when IoU > 0.50",
            "A2": "native-confidence RetinaNet + supervised EfficientTree fusion",
            "A3": "A2-native warm-start source + native-confidence SSL selection",
            "validation": "fixed Dataset3 expert-only validation split",
            "test": "fixed Dataset3 expert-only test split",
        },
        "files": {},
        "extra": extra,
    }

    for key, p in files.items():
        _require_file(p)
        manifest["files"][key] = {
            "path": str(p.resolve()),
            "sha256": _sha256(p),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Manifest:", path)


# ===========================================================================
# Paths
# ===========================================================================

def _paths(c) -> dict[str, Path]:
    out = c.output_root
    repo = c.repo
    cfg = repo / "configs" / "ssod" / "custom"

    p: dict[str, Path] = {}

    p["d3_a0_dataset"] = out / "datasets" / "efficienttree_640_D3_A0"
    p["a2_dataset"] = out / "datasets" / "efficienttree_640_D3_A2_native"

    p["a0_yaml"] = cfg / "exp4_D3_A0_supervised.yaml"
    p["a2_sup_yaml"] = cfg / "exp4_D3_A2_native_supervised.yaml"
    p["a2_ssl_yaml"] = cfg / "exp4_D3_A2_native_ssl_warmstart.yaml"

    p["et_a0_pred"] = out / "predictions" / "exp4_D3_A0_et_supervised"

    p["retina_model"] = out / "models" / "exp4_D3_retinanet"
    p["retina_pred"] = out / "predictions" / "exp4_D3_retinanet"

    p["a2_exp"] = out / "experiments" / "exp4_D3_A2_native"
    p["a2_train"] = p["a2_exp"] / "train_A2_native.csv"
    p["a2_accepted"] = p["a2_exp"] / "A2_native_accepted_predictions.csv"

    p["ssl_pred"] = out / "predictions" / "exp4_D3_A2_native_ssl"

    p["a3_exp"] = out / "experiments" / "exp4_D3_A3_native_native"
    p["a3_train"] = p["a3_exp"] / "train_A3_native_native.csv"

    p["fr_final"] = (
        out / "models" / "fasterrcnn" / "exp4_D3_A3_native_native"
    )
    p["fr_baseline"] = (
        out / "models" / "fasterrcnn" / "exp4_D3_A0_baseline"
    )

    p["reports"] = out / "reports" / "experiment4_D3_A3_native_native"
    p["state"] = out / "stage_state_exp4_D3_A3_native_native"

    return p


# ===========================================================================
# Fresh cleanup
# ===========================================================================

def _fresh_cleanup(c, p: dict[str, Path]) -> None:
    print("Fresh Experiment-4 cleanup (Exp4 namespace only).")

    targets = [
        p["state"],
        p["d3_a0_dataset"],
        p["a2_dataset"],
        p["et_a0_pred"],
        p["retina_model"],
        p["retina_pred"],
        p["a2_exp"],
        p["ssl_pred"],
        p["a3_exp"],
        p["fr_final"],
        p["fr_baseline"],
        p["reports"],
    ]

    # Remove only exact EfficientTree run directories defined by permanent YAMLs.
    for key in ("a0_yaml", "a2_sup_yaml", "a2_ssl_yaml"):
        if p[key].is_file():
            targets.append(_yaml_run_dir(p[key]))

    seen: set[str] = set()
    for target in targets:
        token = str(target).lower()
        if token in seen:
            continue
        seen.add(token)
        _safe_remove(target)


# ===========================================================================
# Stages
# ===========================================================================

STAGE_NAMES = {
    1: "Preflight + Dataset3/YAML audit",
    2: "Prepare Dataset3 A0 for EfficientTree",
    3: "Train supervised EfficientTree on Dataset3 A0",
    4: "Predict with supervised EfficientTree on Dataset3",
    5: "Train RetinaNet on Dataset3 expert GT",
    6: "Predict with RetinaNet on Dataset3",
    7: "Build A2-native from RetinaNet + supervised EfficientTree",
    8: "Prepare EfficientTree dataset from A2-native",
    9: "Train supervised EfficientTree on A2-native",
    10: "Warm-start Teacher-Student SSL from A2-native checkpoint",
    11: "Generate native-confidence SSL predictions",
    12: "Build A3_native,native",
    13: "Train Faster R-CNN on A3_native,native",
    14: "Write reproducibility manifest",
}


def cmd_stage(n: int, c, a: argparse.Namespace) -> None:
    py = c.python
    labels = c.labels
    repo = c.repo
    s = c.settings
    p = _paths(c)

    a0_run = _yaml_run_dir(p["a0_yaml"]) if p["a0_yaml"].is_file() else Path()
    a0_best = a0_run / "weights" / "best.pt"

    a2_sup_run = (
        _yaml_run_dir(p["a2_sup_yaml"]) if p["a2_sup_yaml"].is_file() else Path()
    )
    a2_sup_best = a2_sup_run / "weights" / "best.pt"

    ssl_run = (
        _yaml_run_dir(p["a2_ssl_yaml"]) if p["a2_ssl_yaml"].is_file() else Path()
    )

    if n == 1:
        _audit_d3(labels)

        for script in (
            ROOT / "src" / "data" / "prepare_mydata_640.py",
            ROOT / "src" / "efficienttree" / "predict_all_splits.py",
            ROOT / "src" / "retinanet" / "train_retinanet.py",
            ROOT / "src" / "retinanet" / "predict_retinanet.py",
            ROOT / "src" / "fasterrcnn" / "train_fasterrcnn.py",
            repo / "train.py",
        ):
            _require_file(script)

        for yaml_path in (p["a0_yaml"], p["a2_sup_yaml"], p["a2_ssl_yaml"]):
            _require_file(yaml_path)

        _validate_yaml(
            p["a0_yaml"],
            dataset_dir=p["d3_a0_dataset"],
            expected_burn=1,
            expected_train_domain=False,
        )
        _validate_yaml(
            p["a2_sup_yaml"],
            dataset_dir=p["a2_dataset"],
            expected_burn=1,
            expected_train_domain=False,
        )
        _validate_yaml(
            p["a2_ssl_yaml"],
            dataset_dir=p["a2_dataset"],
            expected_burn=0,
            expected_train_domain=True,
            expected_weights=a2_sup_best,
        )

        # Pretrained weight used only for supervised initialization.
        for y in (p["a0_yaml"], p["a2_sup_yaml"]):
            cfg = _read_yaml(y)
            _require_file(Path(str(cfg["weights"])))

        print("Experiment 4 preflight PASSED.")

    elif n == 2:
        run(
            [
                py,
                ROOT / "src" / "data" / "prepare_mydata_640.py",
                "--root",
                c.data_root,
                "--images-dir",
                c.images,
                "--labels-dir",
                labels,
                "--train-csv",
                labels / "train_labels.csv",
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--output-dir",
                p["d3_a0_dataset"],
                "--size",
                str(s.get("image_size_et", 640)),
                "--overwrite",
            ],
            dry_run=a.dry_run,
        )

    elif n == 3:
        run(
            [py, repo / "train.py", "--cfg", p["a0_yaml"]],
            cwd=repo,
            dry_run=a.dry_run,
        )
        if not a.dry_run:
            _require_file(a0_best)

    elif n == 4:
        if not a.dry_run:
            _require_file(a0_best)
        run(
            [
                py,
                ROOT / "src" / "efficienttree" / "predict_all_splits.py",
                "--python",
                py,
                "--repo",
                repo,
                "--dataset",
                p["d3_a0_dataset"],
                "--labels-dir",
                labels,
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--weights",
                a0_best,
                "--output",
                p["et_a0_pred"],
                "--tag",
                "exp4_D3_A0_supervised",
                "--confidence",
                str(MIN_CONFIDENCE),
                "--gt-iou",
                str(GT_IOU),
            ],
            dry_run=a.dry_run,
        )

    elif n == 5:
        run(
            [
                py,
                ROOT / "src" / "retinanet" / "train_retinanet.py",
                "--images-dir",
                c.images,
                "--train-csv",
                labels / "train_labels.csv",
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--output-dir",
                p["retina_model"],
                "--num-runs",
                str(
                    a.retina_runs
                    if a.retina_runs is not None
                    else s.get("retina_runs", 3)
                ),
                "--max-epochs",
                str(
                    a.retina_epochs
                    if a.retina_epochs is not None
                    else s.get("retina_epochs", 80)
                ),
            ],
            dry_run=a.dry_run,
        )

    elif n == 6:
        ck = (
            p["retina_model"] / "run_01" / "best_model.pth"
            if a.dry_run
            else _choose_retina_checkpoint(
                p["retina_model"] / "all_runs_summary.csv",
                p["retina_model"],
            )
        )
        run(
            [
                py,
                ROOT / "src" / "retinanet" / "predict_retinanet.py",
                "--checkpoint",
                ck,
                "--images-dir",
                c.images,
                "--train-csv",
                labels / "train_labels.csv",
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--output-root",
                p["retina_pred"],
                "--confidence",
                str(MIN_CONFIDENCE),
                "--nms-iou",
                str(NMS_IOU),
                "--gt-iou",
                str(GT_IOU),
            ],
            dry_run=a.dry_run,
        )

    elif n == 7:
        if a.dry_run:
            print("Would build A2-native using:")
            print("  GT:", labels / "train_labels.csv")
            print("  Retina:", p["retina_pred"] / "predictions" / "predictions_train_after_confidence_filter.csv")
            print("  ET:", p["et_a0_pred"] / "train_predictions_256.csv")
        else:
            _build_a2_native(
                gt_csv=labels / "train_labels.csv",
                retina_csv=_find_retina_train_predictions(p["retina_pred"]),
                et_csv=_find_et_predictions(p["et_a0_pred"], "train"),
                output_dir=p["a2_exp"],
            )

    elif n == 8:
        if not a.dry_run:
            _require_file(p["a2_train"])
        run(
            [
                py,
                ROOT / "src" / "data" / "prepare_mydata_640.py",
                "--root",
                c.data_root,
                "--images-dir",
                c.images,
                "--labels-dir",
                labels,
                "--train-csv",
                p["a2_train"],
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--output-dir",
                p["a2_dataset"],
                "--size",
                str(s.get("image_size_et", 640)),
                "--overwrite",
            ],
            dry_run=a.dry_run,
        )

    elif n == 9:
        run(
            [py, repo / "train.py", "--cfg", p["a2_sup_yaml"]],
            cwd=repo,
            dry_run=a.dry_run,
        )
        if not a.dry_run:
            _require_file(a2_sup_best)

    elif n == 10:
        # Permanent SSL YAML already points to:
        # weights = A2-native supervised best.pt
        # train/val/test/target = efficienttree_640_D3_A2_native
        if not a.dry_run:
            _require_file(a2_sup_best)
        run(
            [py, repo / "train.py", "--cfg", p["a2_ssl_yaml"]],
            cwd=repo,
            dry_run=a.dry_run,
        )
        if not a.dry_run:
            _choose_ssl_checkpoint(ssl_run)

    elif n == 11:
        ck = (
            ssl_run / "weights" / "best_ema.pt"
            if a.dry_run
            else _choose_ssl_checkpoint(ssl_run)
        )
        run(
            [
                py,
                ROOT / "src" / "efficienttree" / "predict_all_splits.py",
                "--python",
                py,
                "--repo",
                repo,
                "--dataset",
                p["a2_dataset"],
                "--labels-dir",
                labels,
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--weights",
                ck,
                "--output",
                p["ssl_pred"],
                "--tag",
                "exp4_D3_A2_native_ssl",
                "--confidence",
                str(MIN_CONFIDENCE),
                "--gt-iou",
                str(GT_IOU),
            ],
            dry_run=a.dry_run,
        )

    elif n == 12:
        if a.dry_run:
            print("Would build A3_native,native from A2-native + SSL TRAIN + SSL UNLABELED")
        else:
            _build_a3_native_native(
                gt_csv=labels / "train_labels.csv",
                a2_accepted_csv=p["a2_accepted"],
                ssl_train_csv=_find_et_predictions(p["ssl_pred"], "train"),
                ssl_unlabeled_csv=_find_et_predictions(p["ssl_pred"], "unlabeled"),
                unlabeled_list=labels / "unlabeled_images.txt",
                output_dir=p["a3_exp"],
            )

    elif n == 13:
        if not a.dry_run:
            _require_file(p["a3_train"])

        fr_runs = (
            a.fasterrcnn_runs
            if a.fasterrcnn_runs is not None
            else s.get("fasterrcnn_runs", 5)
        )
        fr_epochs = (
            a.fasterrcnn_epochs
            if a.fasterrcnn_epochs is not None
            else s.get("fasterrcnn_epochs", 20)
        )

        run(
            [
                py,
                ROOT / "src" / "fasterrcnn" / "train_fasterrcnn.py",
                "--images-dir",
                c.images,
                "--train-csv",
                p["a3_train"],
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--output-dir",
                p["fr_final"],
                "--num-runs",
                str(fr_runs),
                "--max-epochs",
                str(fr_epochs),
            ],
            dry_run=a.dry_run,
        )

        if a.also_baseline:
            run(
                [
                    py,
                    ROOT / "src" / "fasterrcnn" / "train_fasterrcnn.py",
                    "--images-dir",
                    c.images,
                    "--train-csv",
                    labels / "train_labels.csv",
                    "--val-csv",
                    labels / "val_labels.csv",
                    "--test-csv",
                    labels / "test_labels.csv",
                    "--output-dir",
                    p["fr_baseline"],
                    "--num-runs",
                    str(fr_runs),
                    "--max-epochs",
                    str(fr_epochs),
                ],
                dry_run=a.dry_run,
            )

    elif n == 14:
        if a.dry_run:
            print("Would write reproducibility manifest.")
            return

        retina_best = _choose_retina_checkpoint(
            p["retina_model"] / "all_runs_summary.csv",
            p["retina_model"],
        )
        ssl_best = _choose_ssl_checkpoint(ssl_run)

        files = {
            "dataset3_train_gt": labels / "train_labels.csv",
            "dataset3_val_gt": labels / "val_labels.csv",
            "dataset3_test_gt": labels / "test_labels.csv",
            "dataset3_unlabeled_list": labels / "unlabeled_images.txt",
            "yaml_D3_A0_supervised": p["a0_yaml"],
            "yaml_D3_A2_native_supervised": p["a2_sup_yaml"],
            "yaml_D3_A2_native_ssl_warmstart": p["a2_ssl_yaml"],
            "D3_A0_ET_checkpoint": a0_best,
            "RetinaNet_checkpoint": retina_best,
            "A2_native_train": p["a2_train"],
            "A2_native_ET_checkpoint": a2_sup_best,
            "A2_native_SSL_checkpoint": ssl_best,
            "A3_native_native_train": p["a3_train"],
        }

        _write_manifest(
            path=p["reports"] / "reproducibility_manifest.json",
            files=files,
            extra={
                "fasterrcnn_output": str(p["fr_final"]),
                "also_baseline": bool(a.also_baseline),
            },
        )

    else:
        raise ValueError(f"Unknown stage: {n}")


# ===========================================================================
# CLI / state
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 4 / Dataset 3 pipeline replicating the Experiment-3 "
            "Best-A3 path: D3 expert GT -> RetinaNet + supervised ET -> "
            "A2-native -> ET on A2-native -> warm-start Teacher-Student SSL -> "
            "native SSL predictions -> A3_native,native -> Faster R-CNN."
        )
    )

    parser.add_argument(
        "stage",
        type=int,
        nargs="?",
        default=1,
        help="First stage to run (default: 1)",
    )
    parser.add_argument("--to", type=int, help="Last stage to run, inclusive")
    parser.add_argument("--all", action="store_true", help="Run stages 1 through 14")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected stages even when their Exp4 state file exists",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete only Experiment-4 outputs/runs/state used by this script, "
            "then execute. Permanent YAML files are never deleted."
        ),
    )

    parser.add_argument("--retina-runs", type=int, default=None)
    parser.add_argument("--retina-epochs", type=int, default=None)
    parser.add_argument("--fasterrcnn-runs", type=int, default=None)
    parser.add_argument("--fasterrcnn-epochs", type=int, default=None)
    parser.add_argument(
        "--also-baseline",
        action="store_true",
        help="At stage 13 also train an independent expert-only D3 Faster R-CNN baseline",
    )

    a = parser.parse_args()
    c = load_config()
    p = _paths(c)

    if a.all:
        start, end = 1, 14
    else:
        start = a.stage
        end = a.to if a.to is not None else start

    if not (1 <= start <= 14 and 1 <= end <= 14 and end >= start):
        raise ValueError("Stages must satisfy 1 <= start <= end <= 14")

    if a.fresh:
        if a.dry_run:
            print("DRY RUN: fresh cleanup would run before stages.")
        else:
            _fresh_cleanup(c, p)

    for n in range(start, end + 1):
        state = p["state"] / f"{n:02d}.done.json"

        if state.is_file() and not a.force and not a.dry_run:
            print(
                f"SKIP stage {n:02d} ({STAGE_NAMES[n]}): already completed. "
                "Use --force to rerun."
            )
            continue

        print("\n" + "=" * 96)
        print(f"EXPERIMENT 4 / DATASET 3 — STAGE {n:02d}: {STAGE_NAMES[n]}")
        print("=" * 96)

        cmd_stage(n, c, a)

        if not a.dry_run:
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(
                json.dumps(
                    {
                        "stage": n,
                        "name": STAGE_NAMES[n],
                        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFAILED:", exc)
        raise SystemExit(1)
