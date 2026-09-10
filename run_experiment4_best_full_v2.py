from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.common.config import load_config
from src.common.process import run

ROOT = Path(__file__).resolve().parent

# Dataset 3 / Experiment 4 fixed split expected by the paper.
EXPECTED_D3 = {
    "train": {"images": 2663, "boxes": 6078, "classes": {1: 2069, 2: 1411, 3: 356, 4: 2242}},
    "val":   {"images":  888, "boxes": 2026, "classes": {1: 690,  2: 470,  3: 119, 4: 747}},
    "test":  {"images":  887, "boxes": 2024, "classes": {1: 690,  2: 470,  3: 118, 4: 746}},
}
EXPECTED_UNLABELED_IMAGES = 7059


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


def _audit_d3_csvs(labels: Path) -> None:
    print("Auditing Dataset 3 expert splits...")
    split_names = {}
    for split in ("train", "val", "test"):
        path = labels / f"{split}_labels.csv"
        _require_file(path)
        df = pd.read_csv(path)
        required = {"filename", "class", "xmin", "ymin", "xmax", "ymax"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        n_images = int(df["filename"].astype(str).nunique())
        n_boxes = int(len(df))
        class_counts = {
            int(k): int(v)
            for k, v in df["class"].astype(int).value_counts().sort_index().items()
        }
        expected = EXPECTED_D3[split]
        print(
            f"{split}: images={n_images}, boxes={n_boxes}, classes={class_counts}"
        )
        if n_images != expected["images"] or n_boxes != expected["boxes"]:
            raise ValueError(
                f"{split}: Dataset 3 count mismatch. "
                f"Expected images={expected['images']}, boxes={expected['boxes']}"
            )
        if class_counts != expected["classes"]:
            raise ValueError(
                f"{split}: Dataset 3 class-count mismatch. "
                f"Expected {expected['classes']}, found {class_counts}"
            )
        split_names[split] = set(df["filename"].astype(str).str.strip())

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_names[left] & split_names[right]
        if overlap:
            raise ValueError(
                f"Filename leakage {left} vs {right}: {len(overlap)}; "
                f"examples={sorted(overlap)[:10]}"
            )

    unlabeled = labels / "unlabeled_images.txt"
    _require_file(unlabeled)
    names = [
        x.strip()
        for x in unlabeled.read_text(encoding="utf-8-sig").splitlines()
        if x.strip()
    ]
    if len(names) != EXPECTED_UNLABELED_IMAGES:
        raise ValueError(
            f"Unlabeled list count mismatch: expected {EXPECTED_UNLABELED_IMAGES}, "
            f"found {len(names)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames found in unlabeled_images.txt")
    all_labeled = split_names["train"] | split_names["val"] | split_names["test"]
    leak_u = set(names) & all_labeled
    if leak_u:
        raise ValueError(
            f"Unlabeled/labeled filename leakage: {len(leak_u)}; "
            f"examples={sorted(leak_u)[:10]}"
        )
    print(f"unlabeled: images={len(names)}")
    print("Dataset 3 audit PASSED.")


def _choose_retina_checkpoint(summary: Path, model_root: Path) -> Path:
    _require_file(summary)
    d = pd.read_csv(summary)
    metric = "best_val_map_50"
    if metric not in d.columns or "run" not in d.columns:
        raise ValueError(
            f"{summary}: expected columns 'run' and '{metric}', got {list(d.columns)}"
        )
    row = d.loc[d[metric].astype(float).idxmax()]
    run_id = int(row["run"])
    ck = model_root / f"run_{run_id:02d}" / "best_model.pth"
    return _require_file(ck)


def _choose_ssl_checkpoint(run_dir: Path) -> Path:
    weights = run_dir / "weights"
    # EfficientTeacher inference convention: prefer EMA teacher checkpoint.
    for name in ("best_ema.pt", "best.pt", "best_student.pt"):
        p = weights / name
        if p.is_file():
            print("Using SSL checkpoint:", p)
            return p
    raise FileNotFoundError(
        f"No best_ema.pt, best.pt, or best_student.pt found under {weights}"
    )


def _find_retina_train_confidence_file(retina_pred_root: Path) -> Path:
    pred_dir = retina_pred_root / "predictions"
    exact = pred_dir / "predictions_train_after_confidence_filter.csv"
    if exact.is_file():
        return exact
    matches = sorted(pred_dir.glob("predictions_train_after_confidence_filter*.csv"))
    if not matches:
        raise FileNotFoundError(
            "No RetinaNet TRAIN confidence-filtered prediction CSV found under "
            f"{pred_dir}"
        )
    if len(matches) > 1:
        print("WARNING: multiple Retina train prediction files; using", matches[0])
    return matches[0]


def _clone_exp4_configs(
    repo: Path,
    dataset: Path,
    generic_sup: Path,
    generic_ssl: Path,
    exp4_sup: Path,
    exp4_ssl_base: Path,
    project_dir: Path,
    sup_run_name: str,
) -> None:
    for p in (generic_sup, generic_ssl):
        _require_file(p)

    sup = yaml.safe_load(generic_sup.read_text(encoding="utf-8-sig"))
    sup["project"] = str(project_dir.resolve())
    sup["name"] = sup_run_name
    sup["resume"] = False
    sup["exist_ok"] = True
    ds = sup.setdefault("Dataset", {})
    ds["train"] = str((dataset / "splits" / "train.txt").resolve())
    ds["val"] = str((dataset / "splits" / "val.txt").resolve())
    ds["test"] = str((dataset / "splits" / "test.txt").resolve())
    ds["target"] = str((dataset / "splits" / "target_unlabeled.txt").resolve())

    exp4_sup.parent.mkdir(parents=True, exist_ok=True)
    exp4_sup.write_text(
        yaml.safe_dump(sup, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    ssl = yaml.safe_load(generic_ssl.read_text(encoding="utf-8-sig"))
    ssl["project"] = str(project_dir.resolve())
    ssl["resume"] = False
    ssl["exist_ok"] = True
    ds2 = ssl.setdefault("Dataset", {})
    ds2["train"] = str((dataset / "splits" / "train.txt").resolve())
    ds2["val"] = str((dataset / "splits" / "val.txt").resolve())
    ds2["test"] = str((dataset / "splits" / "test.txt").resolve())
    ds2["target"] = str((dataset / "splits" / "target_unlabeled.txt").resolve())

    exp4_ssl_base.write_text(
        yaml.safe_dump(ssl, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("Wrote:", exp4_sup)
    print("Wrote:", exp4_ssl_base)


def _write_manifest(
    out_path: Path,
    labels: Path,
    sup_best: Path,
    retina_best: Path,
    ssl_best: Path,
    final_train: Path,
    val_gt: Path,
    test_gt: Path,
    settings: dict,
) -> None:
    files = {
        "dataset3_train_gt": labels / "train_labels.csv",
        "dataset3_val_gt": val_gt,
        "dataset3_test_gt": test_gt,
        "et_supervised_checkpoint": sup_best,
        "retinanet_checkpoint": retina_best,
        "et_ssl_checkpoint": ssl_best,
        "final_train_dataset": final_train,
    }
    manifest = {
        "experiment": "Experiment 4 - Dataset 3 - best train-fusion configuration",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "final_dataset_definition": "G_D3 union P_train",
        "prediction_sources": [
            "RetinaNet",
            "supervised EfficientTree",
            "SSL-enhanced EfficientTree",
        ],
        "fusion": {
            "minimum_confidence": 0.25,
            "confidence_mode": "native model confidence",
            "single_model_confidence_gate": None,
            "gt_overlap_remove_rule": "max IoU with any expert GT > 0.50",
            "prediction_duplicate_rule": "greedy class-agnostic NMS; suppress IoU > 0.50",
            "unlabeled_pseudo_labels_added_to_final_dataset": False,
        },
        "settings": settings,
        "files": {},
    }
    for key, path in files.items():
        _require_file(path)
        manifest["files"][key] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Manifest:", out_path)


def cmd_stage(n: int, c, a: argparse.Namespace) -> None:
    dry = a.dry_run
    py = c.python
    s = c.settings
    labels = c.labels
    out = c.output_root
    repo = c.repo

    et_epochs = int(a.et_epochs if a.et_epochs is not None else s.get("et_epochs", 45))
    ssl_epochs = int(a.ssl_epochs)
    retina_runs = int(
        a.retina_runs if a.retina_runs is not None else s.get("retina_runs", 3)
    )
    retina_epochs = int(
        a.retina_epochs if a.retina_epochs is not None else s.get("retina_epochs", 80)
    )
    fr_runs = int(
        a.fasterrcnn_runs
        if a.fasterrcnn_runs is not None
        else s.get("fasterrcnn_runs", 5)
    )
    fr_epochs = int(
        a.fasterrcnn_epochs
        if a.fasterrcnn_epochs is not None
        else s.get("fasterrcnn_epochs", 20)
    )

    # -------------------------------------------------------------------------
    # Experiment-4-only namespace. These names do NOT collide with Experiment 3.
    # -------------------------------------------------------------------------
    d3_dataset = out / "datasets" / "exp4_D3_A0_efficienttree_640"

    et_cfg = repo / "configs" / "ssod" / "custom"
    generic_sup = et_cfg / "mydata_full_supervised_4class_staged70.yaml"
    generic_ssl = et_cfg / "mydata_full_ssl_4class_staged70.yaml"
    exp4_sup_yaml = et_cfg / "exp4_D3_A0_supervised.yaml"
    exp4_ssl_base_yaml = et_cfg / "exp4_D3_A0_ssl_base.yaml"
    exp4_ssl_warm_yaml = et_cfg / "exp4_D3_A0_ssl_warmstart_burn0.yaml"

    exp4_et_project = repo / "runs" / "experiment4_D3"
    sup_run_name = "exp4_D3_A0_et_supervised"
    ssl_run_name = f"exp4_D3_A0_et_ssl_ep{ssl_epochs}"
    sup_run = exp4_et_project / sup_run_name
    ssl_run = exp4_et_project / ssl_run_name
    sup_best = sup_run / "weights" / "best.pt"

    et_sup_pred = out / "predictions" / "exp4_D3_et_supervised"
    et_ssl_pred = out / "predictions" / f"exp4_D3_et_ssl_ep{ssl_epochs}"

    retina_model = out / "models" / "exp4_D3_retinanet"
    retina_pred = out / "predictions" / "exp4_D3_retinanet"

    final_exp = out / "experiments" / "exp4_D3_best_train_fusion"
    fr_root = out / "models" / "fasterrcnn_exp4_D3"
    reports = out / "reports" / "experiment4_D3"

    if n == 1:
        _audit_d3_csvs(labels)
        _require_file(ROOT / "src" / "data" / "prepare_mydata_640.py")
        _require_file(ROOT / "src" / "efficienttree" / "make_configs.py")
        _require_file(ROOT / "src" / "efficienttree" / "make_warmstart_config.py")
        _require_file(ROOT / "src" / "efficienttree" / "predict_all_splits.py")
        _require_file(ROOT / "src" / "retinanet" / "train_retinanet.py")
        _require_file(ROOT / "src" / "retinanet" / "predict_retinanet.py")
        _require_file(ROOT / "src" / "fusion" / "three_source_train_fusion_exp4.py")
        _require_file(ROOT / "src" / "fasterrcnn" / "train_fasterrcnn.py")
        print("Experiment 4 preflight PASSED.")

    elif n == 2:
        # Prepare A0 for EfficientTeacher. The unlabeled split is prepared only
        # because the SSL branch needs it; it is NOT added to the final dataset.
        run(
            [
                py,
                ROOT / "src/data/prepare_mydata_640.py",
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
                d3_dataset,
                "--size",
                str(s.get("image_size_et", 640)),
                "--overwrite",
            ],
            dry_run=dry,
        )

    elif n == 3:
        # Generate verified base configs, then immediately clone them to
        # Experiment-4-specific YAML files and output paths.
        run(
            [
                py,
                ROOT / "src/efficienttree/make_configs.py",
                "--repo",
                repo,
                "--dataset-dir",
                d3_dataset,
                "--batch-size",
                "4",
                "--workers",
                "0",
                "--img-size",
                str(s.get("image_size_et", 640)),
                "--epochs",
                str(et_epochs),
                "--burn-epochs",
                "1",
                "--class-names",
                *c.raw["classes"],
            ],
            dry_run=dry,
        )
        if not dry:
            _clone_exp4_configs(
                repo=repo,
                dataset=d3_dataset,
                generic_sup=generic_sup,
                generic_ssl=generic_ssl,
                exp4_sup=exp4_sup_yaml,
                exp4_ssl_base=exp4_ssl_base_yaml,
                project_dir=exp4_et_project,
                sup_run_name=sup_run_name,
            )

    elif n == 4:
        # Supervised EfficientTree trained ONLY on Dataset-3 expert A0 labels.
        run(
            [
                py,
                repo / "train.py",
                "--cfg",
                exp4_sup_yaml,
                "epochs",
                str(et_epochs),
                "name",
                sup_run_name,
                "exist_ok",
                "True",
            ],
            cwd=repo,
            dry_run=dry,
        )
        if not dry:
            _require_file(sup_best)

    elif n == 5:
        # Predictions from supervised ET. Helper predicts all splits, but final
        # Experiment-4 fusion consumes TRAIN predictions only.
        _require_file(sup_best)
        run(
            [
                py,
                ROOT / "src/efficienttree/predict_all_splits.py",
                "--python",
                py,
                "--repo",
                repo,
                "--dataset",
                d3_dataset,
                "--labels-dir",
                labels,
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--weights",
                sup_best,
                "--output",
                et_sup_pred,
                "--tag",
                "exp4_D3_supervised",
                "--confidence",
                "0.25",
                "--gt-iou",
                "0.5",
            ],
            dry_run=dry,
        )

    elif n == 6:
        # RetinaNet trained ONLY on Dataset-3 labeled training subset.
        run(
            [
                py,
                ROOT / "src/retinanet/train_retinanet.py",
                "--images-dir",
                c.images,
                "--train-csv",
                labels / "train_labels.csv",
                "--val-csv",
                labels / "val_labels.csv",
                "--test-csv",
                labels / "test_labels.csv",
                "--output-dir",
                retina_model,
                "--num-runs",
                str(retina_runs),
                "--max-epochs",
                str(retina_epochs),
            ],
            dry_run=dry,
        )

    elif n == 7:
        ck = (
            retina_model / "run_01" / "best_model.pth"
            if dry
            else _choose_retina_checkpoint(
                retina_model / "all_runs_summary.csv", retina_model
            )
        )
        run(
            [
                py,
                ROOT / "src/retinanet/predict_retinanet.py",
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
                retina_pred,
                "--confidence",
                "0.25",
                "--nms-iou",
                "0.5",
                "--gt-iou",
                "0.5",
            ],
            dry_run=dry,
        )

    elif n == 8:
        # SSL branch:
        # warm-start from D3 supervised ET, source=A0 labeled, target=7059 unlabeled.
        _require_file(sup_best)
        run(
            [
                py,
                ROOT / "src/efficienttree/make_warmstart_config.py",
                "--source-yaml",
                exp4_ssl_base_yaml,
                "--dataset",
                d3_dataset,
                "--weights",
                sup_best,
                "--output-yaml",
                exp4_ssl_warm_yaml,
                "--run-name",
                ssl_run_name,
                "--burn-epochs",
                "0",
            ],
            dry_run=dry,
        )
        if not dry:
            # make_warmstart_config preserves the base project path.
            cfg = yaml.safe_load(exp4_ssl_warm_yaml.read_text(encoding="utf-8-sig"))
            cfg["project"] = str(exp4_et_project.resolve())
            cfg["name"] = ssl_run_name
            cfg["epochs"] = ssl_epochs
            cfg["resume"] = False
            cfg["exist_ok"] = True
            exp4_ssl_warm_yaml.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            print("Patched SSL project/name/epochs:", exp4_ssl_warm_yaml)

    elif n == 9:
        # Natural 20-epoch completion by default so final result files are written.
        run(
            [
                py,
                repo / "train.py",
                "--cfg",
                exp4_ssl_warm_yaml,
                "epochs",
                str(ssl_epochs),
                "name",
                ssl_run_name,
                "exist_ok",
                "True",
            ],
            cwd=repo,
            dry_run=dry,
        )
        if not dry:
            _choose_ssl_checkpoint(ssl_run)

    elif n == 10:
        ck = (
            ssl_run / "weights" / "best_ema.pt"
            if dry
            else _choose_ssl_checkpoint(ssl_run)
        )
        run(
            [
                py,
                ROOT / "src/efficienttree/predict_all_splits.py",
                "--python",
                py,
                "--repo",
                repo,
                "--dataset",
                d3_dataset,
                "--labels-dir",
                labels,
                "--unlabeled-list",
                labels / "unlabeled_images.txt",
                "--weights",
                ck,
                "--output",
                et_ssl_pred,
                "--tag",
                f"exp4_D3_ssl_ep{ssl_epochs}",
                "--confidence",
                "0.25",
                "--gt-iou",
                "0.5",
            ],
            dry_run=dry,
        )

    elif n == 11:
        # BEST Experiment-4 configuration:
        # three prediction sources on LABELED TRAIN only.
        # No single-model 0.8 gate. Native score >=0.25 enters the pool.
        retina_train = _find_retina_train_confidence_file(retina_pred)
        _require_file(et_sup_pred / "train_predictions_256.csv")
        _require_file(et_ssl_pred / "train_predictions_256.csv")
        run(
            [
                py,
                ROOT / "src/fusion/three_source_train_fusion_exp4.py",
                "--gt-train",
                labels / "train_labels.csv",
                "--val-gt",
                labels / "val_labels.csv",
                "--test-gt",
                labels / "test_labels.csv",
                "--retina-train",
                retina_train,
                "--et-supervised-train",
                et_sup_pred / "train_predictions_256.csv",
                "--et-ssl-train",
                et_ssl_pred / "train_predictions_256.csv",
                "--output-dir",
                final_exp,
                "--min-confidence",
                "0.25",
                "--gt-iou",
                "0.5",
                "--nms-iou",
                "0.5",
            ],
            dry_run=dry,
        )

    elif n == 12:
        # Optional full Experiment-4 validation:
        # baseline and best train-fusion configuration, same expert-only val/test.
        for name, train_csv in (
            ("baseline_A0", labels / "train_labels.csv"),
            ("best_train_fusion", final_exp / "train_final_fused.csv"),
        ):
            run(
                [
                    py,
                    ROOT / "src/fasterrcnn/train_fasterrcnn.py",
                    "--images-dir",
                    c.images,
                    "--train-csv",
                    train_csv,
                    "--val-csv",
                    labels / "val_labels.csv",
                    "--test-csv",
                    labels / "test_labels.csv",
                    "--output-dir",
                    fr_root / name,
                    "--num-runs",
                    str(fr_runs),
                    "--max-epochs",
                    str(fr_epochs),
                ],
                dry_run=dry,
            )

    elif n == 13:
        run(
            [
                py,
                ROOT / "src/reports/summarize_results.py",
                "--root",
                fr_root,
                "--output",
                reports / "fasterrcnn_baseline_vs_best_train_fusion_mean_std.csv",
            ],
            dry_run=dry,
        )

    elif n == 14:
        # Reproducibility manifest with exact checkpoints/dataset hashes.
        retina_best = (
            retina_model / "run_01" / "best_model.pth"
            if dry
            else _choose_retina_checkpoint(
                retina_model / "all_runs_summary.csv", retina_model
            )
        )
        ssl_best = (
            ssl_run / "weights" / "best_ema.pt"
            if dry
            else _choose_ssl_checkpoint(ssl_run)
        )
        if not dry:
            _write_manifest(
                reports / "reproducibility_manifest.json",
                labels=labels,
                sup_best=sup_best,
                retina_best=retina_best,
                ssl_best=ssl_best,
                final_train=final_exp / "train_final_fused.csv",
                val_gt=labels / "val_labels.csv",
                test_gt=labels / "test_labels.csv",
                settings={
                    "et_supervised_epochs": et_epochs,
                    "et_ssl_epochs": ssl_epochs,
                    "retinanet_runs": retina_runs,
                    "retinanet_max_epochs": retina_epochs,
                    "fasterrcnn_runs": fr_runs,
                    "fasterrcnn_max_epochs": fr_epochs,
                },
            )

    else:
        raise ValueError(f"Unknown Experiment-4 stage: {n}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Reproducible Experiment 4 / Dataset 3 BEST-CONFIGURATION pipeline. "
            "Trains RetinaNet and supervised EfficientTree from Dataset3 A0, "
            "warm-starts SSL-enhanced EfficientTree with A0 + unlabeled imagery, "
            "then builds G_D3 ∪ P_train from 3-source TRAIN predictions. "
            "Unlabeled pseudo-labels are NOT added to the final dataset."
        )
    )
    p.add_argument("stage", type=int, help="First stage to run (1-14)")
    p.add_argument("--to", type=int, help="Last stage to run, inclusive")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")

    p.add_argument("--et-epochs", type=int, default=None)
    p.add_argument("--ssl-epochs", type=int, default=20)
    p.add_argument("--retina-runs", type=int, default=None)
    p.add_argument("--retina-epochs", type=int, default=None)
    p.add_argument("--fasterrcnn-runs", type=int, default=None)
    p.add_argument("--fasterrcnn-epochs", type=int, default=None)

    a = p.parse_args()
    c = load_config()
    end = a.to or a.stage
    if not (1 <= a.stage <= 14 and 1 <= end <= 14 and end >= a.stage):
        raise ValueError("Stages must satisfy 1 <= stage <= to <= 14")

    for n in range(a.stage, end + 1):
        state = c.out("stage_state_exp4_D3_best", f"{n:02d}.done.json")
        if state.is_file() and not a.force and not a.dry_run:
            print(
                f"SKIP Experiment-4 stage {n:02d}: already completed. "
                "Use --force to rerun."
            )
            continue

        print("\n" + "=" * 88)
        print(f"EXPERIMENT 4 / DATASET 3 / BEST CONFIGURATION — STAGE {n:02d}")
        print("=" * 88)

        cmd_stage(n, c, a)

        if not a.dry_run:
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(
                json.dumps(
                    {
                        "stage": n,
                        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        raise SystemExit(1)
