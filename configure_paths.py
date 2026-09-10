from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the local path configuration for "
                    "Teacher_Assisted_Tree_Detection."
    )

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\FASTRCNN\teacher_student"),
        help="Main data root."
    )

    parser.add_argument(
        "--python",
        dest="python_executable",
        default="python",
        help="Python executable or full path to python.exe."
    )

    parser.add_argument(
        "--efficienttree-repo",
        type=Path,
        default=None,
        help="Path to EfficientTree-master."
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for pipeline outputs."
    )

    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not create output directories."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Project root
    #
    # configure_paths.py is located at:
    #
    # E:\FASTRCNN\FASTRCNN\Teacher_Assisted_Tree_Detection
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # Data root
    #
    # E:\FASTRCNN\teacher_student
    # ------------------------------------------------------------------
    data_root = args.data_root.expanduser().resolve()

    # ------------------------------------------------------------------
    # Images
    #
    # E:\FASTRCNN\teacher_student\images
    # ------------------------------------------------------------------
    images_dir = (
        data_root
        / "images"
    ).resolve()

    # ------------------------------------------------------------------
    # Labels
    #
    # E:\FASTRCNN\teacher_student\labels\article2_hemogen_mydataset
    # ------------------------------------------------------------------
    labels_dir = (
        data_root
        / "labels"
        / "article2_hemogen_mydataset"
    ).resolve()

    train_csv = labels_dir / "train_labels.csv"
    val_csv = labels_dir / "val_labels.csv"
    test_csv = labels_dir / "test_labels.csv"
    unlabeled_txt = labels_dir / "unlabeled_images.txt"

    # ------------------------------------------------------------------
    # EfficientTree repository
    #
    # Default:
    # E:\FASTRCNN\FASTRCNN\Teacher_Assisted_Tree_Detection\
    # EfficientTree-master
    # ------------------------------------------------------------------
    if args.efficienttree_repo is None:
        repo = project_root / "EfficientTree-master"
    else:
        repo = args.efficienttree_repo

    repo = repo.expanduser().resolve()

    # ------------------------------------------------------------------
    # Output root
    #
    # Default:
    # E:\FASTRCNN\teacher_student\tatd_outputs
    # ------------------------------------------------------------------
    if args.output_root is None:
        output_root = data_root / "tatd_outputs"
    else:
        output_root = args.output_root

    output_root = output_root.expanduser().resolve()

    # ------------------------------------------------------------------
    # Python executable
    # ------------------------------------------------------------------
    py_value = args.python_executable.strip()

    if py_value.lower() not in {
        "python",
        "python3",
        "py",
    }:
        py_path = Path(py_value).expanduser().resolve()

        if not py_path.is_file():
            raise FileNotFoundError(
                f"Python executable not found: {py_path}"
            )

        py_value = str(py_path)

    # ------------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------------
    required = {
        "Project root": project_root,

        "Images directory": images_dir,

        "Labels directory": labels_dir,

        "train_labels.csv": train_csv,
        "val_labels.csv": val_csv,
        "test_labels.csv": test_csv,

        "unlabeled_images.txt": unlabeled_txt,

        "EfficientTree-master": repo,
        "EfficientTree train.py": repo / "train.py",
        "EfficientTree detect.py": repo / "detect.py",

        "src": project_root / "src",

        "src/data": project_root / "src" / "data",
        "src/common": project_root / "src" / "common",
        "src/efficienttree": project_root / "src" / "efficienttree",
        "src/experiments": project_root / "src" / "experiments",
        "src/fusion": project_root / "src" / "fusion",
        "src/retinanet": project_root / "src" / "retinanet",
        "src/fasterrcnn": project_root / "src" / "fasterrcnn",
        "src/reports": project_root / "src" / "reports",
    }

    missing = []

    for name, path in required.items():
        if not path.exists():
            missing.append(
                f"{name}: {path}"
            )

    if missing:
        print()
        print("=" * 72)
        print("MISSING REQUIRED INPUTS")
        print("=" * 72)

        for item in missing:
            print(item)

        raise FileNotFoundError(
            "Required project inputs are missing."
        )

    # ------------------------------------------------------------------
    # Create output folders
    # ------------------------------------------------------------------
    if not args.no_create:

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_subdirs = [
            "datasets",
            "models",
            "predictions",
            "fusion",
            "experiments",
            "reports",
            "stage_state",
        ]

        for name in output_subdirs:
            (
                output_root
                / name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # CONFIG
    #
    # Important:
    #
    # run_stage.py uses:
    #
    # c.python
    # c.settings
    # c.labels
    # c.output_root
    # c.repo
    # c.data_root
    # c.images
    # c.raw
    #
    # Therefore explicit "images" and "labels" paths are included.
    # ------------------------------------------------------------------
    config = {
        "schema_version": 1,

        "project_root": project_root.as_posix(),

        "data_root": data_root.as_posix(),

        # IMPORTANT FOR run_stage.py
        "images": images_dir.as_posix(),
        "labels": labels_dir.as_posix(),

        # Explicit individual files
        "train_labels_csv": train_csv.as_posix(),
        "val_labels_csv": val_csv.as_posix(),
        "test_labels_csv": test_csv.as_posix(),
        "unlabeled_images_txt": unlabeled_txt.as_posix(),

        # Python
        "python_executable": py_value,

        # EfficientTree
        "efficienttree_repo": repo.as_posix(),

        # Outputs
        "output_root": output_root.as_posix(),

        # Classes
        "classes": [
            "class1",
            "class2",
            "class3",
            "class4",
        ],

        # Experimental settings
        "settings": {
            "image_size_et": 640,
            "image_size_csv": 256,

            "confidence": 0.25,

            "gt_iou": 0.50,
            "agreement_iou": 0.50,

            "single_model_confidence": 0.80,

            "warmstart_confidence": 0.50,

            "et_epochs": 45,

            "retina_runs": 3,
            "retina_epochs": 80,

            "fasterrcnn_runs": 5,
            "fasterrcnn_epochs": 20,
        }
    }

    # ------------------------------------------------------------------
    # Save paths.local.json
    #
    # E:\FASTRCNN\FASTRCNN\Teacher_Assisted_Tree_Detection\
    # config\paths.local.json
    # ------------------------------------------------------------------
    config_dir = (
        project_root
        / "config"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        config_dir
        / "paths.local.json"
    )

    out.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("PATH CONFIGURATION CREATED")
    print("=" * 72)

    print()
    print("PROJECT")
    print("-" * 72)

    print(f"Project root:")
    print(f"  {project_root}")

    print()
    print("DATA")
    print("-" * 72)

    print(f"Data root:")
    print(f"  {data_root}")

    print(f"Images:")
    print(f"  {images_dir}")

    print(f"Labels:")
    print(f"  {labels_dir}")

    print()
    print("LABEL FILES")
    print("-" * 72)

    print(f"Train:")
    print(f"  {train_csv}")

    print(f"Validation:")
    print(f"  {val_csv}")

    print(f"Test:")
    print(f"  {test_csv}")

    print(f"Unlabeled:")
    print(f"  {unlabeled_txt}")

    print()
    print("EFFICIENTTREE")
    print("-" * 72)

    print(f"Repository:")
    print(f"  {repo}")

    print()
    print("OUTPUT")
    print("-" * 72)

    print(f"Output root:")
    print(f"  {output_root}")

    print()
    print("PYTHON")
    print("-" * 72)

    print(f"Python:")
    print(f"  {py_value}")

    print()
    print("CONFIG FILE")
    print("-" * 72)

    print(f"  {out}")

    print()
    print("=" * 72)
    print("ALL REQUIRED PATHS ARE VALID")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())