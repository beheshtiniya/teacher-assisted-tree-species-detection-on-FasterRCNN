from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PERCENT_PATTERN = re.compile(r"^labeled_(\d+)_percent$", re.IGNORECASE)
PAVILION_CANDIDATES = (
    "pavilon",
    "pavilion",
    "pavillon",
    "Pavilon",
    "Pavilion",
    "Pavillon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Hall overlap-0 EfficientTree experiments for every available "
            "labeled_X_percent folder, with robust Pavilion folder discovery."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            r"E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            r"E:\FASTRCNN\FASTRCNN\EfficientTree-paper\EfficientTree-master"
            r"\data\hall\tree_overlap_0_patch_640"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(r"configs\ssod\custom\yolov8s_tree_ssod.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--burn-epochs",
        type=int,
        default=None,
        help="Optional override; otherwise preserve the author's config.",
    )
    return parser.parse_args()


def find_pavilion_dir(images_root: Path) -> Path | None:
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    directories = [path for path in images_root.iterdir() if path.is_dir()]

    # First: exact common spellings.
    by_name = {path.name.lower(): path.resolve() for path in directories}
    for candidate in PAVILION_CANDIDATES:
        match = by_name.get(candidate.lower())
        if match is not None:
            return match

    # Second: fuzzy name containing "pavil".
    fuzzy = [path.resolve() for path in directories if "pavil" in path.name.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise RuntimeError(
            "Multiple possible Pavilion directories were found:\n"
            + "\n".join(str(path) for path in fuzzy)
        )

    return None


def list_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.stem.lower()
            if key in result:
                raise RuntimeError(
                    f"Duplicate image stem {path.stem!r} in {directory}: "
                    f"{result[key].name}, {path.name}"
                )
            result[key] = path.resolve()

    if not result:
        raise RuntimeError(f"No supported images found in: {directory}")
    return result


def list_labels(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory not found: {directory}")

    result: dict[str, Path] = {}
    for path in sorted(directory.glob("*.txt")):
        key = path.stem.lower()
        if key in result:
            raise RuntimeError(f"Duplicate label stem {path.stem!r} in {directory}")
        result[key] = path.resolve()

    if not result:
        raise RuntimeError(f"No label TXT files found in: {directory}")
    return result


def write_paired_list(
    image_map: dict[str, Path],
    label_map: dict[str, Path],
    output: Path,
    *,
    split_name: str,
) -> dict:
    missing_images = sorted(set(label_map) - set(image_map))
    if missing_images:
        raise FileNotFoundError(
            f"{split_name}: {len(missing_images)} labels have no matching image. "
            f"Examples: {missing_images[:20]}"
        )

    rows = [f"{image_map[stem]} {label_map[stem]}" for stem in sorted(label_map)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "split": split_name,
        "pairs": len(rows),
        "list_file": str(output.resolve()),
    }


def write_image_list(paths: list[Path], output: Path, split_name: str) -> dict:
    unique: list[Path] = []
    seen: set[str] = set()

    for path in paths:
        normalized = str(path.resolve()).lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(str(path) for path in unique) + ("\n" if unique else ""),
        encoding="utf-8",
    )
    return {
        "split": split_name,
        "images": len(unique),
        "list_file": str(output.resolve()),
    }


def read_dataset_yaml(path: Path) -> tuple[int, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"dataset.yaml root must be a mapping: {path}")

    names_raw = data.get("names", [])
    if isinstance(names_raw, dict):
        names = [
            str(names_raw[key])
            for key in sorted(names_raw, key=lambda value: int(value))
        ]
    elif isinstance(names_raw, list):
        names = [str(value) for value in names_raw]
    else:
        names = []

    nc = int(data.get("nc", len(names)))
    if not names:
        names = [f"class{i}" for i in range(nc)]
    if len(names) != nc:
        raise ValueError(f"nc={nc}, but {len(names)} class names were found.")
    return nc, names


def discover_percent_folders(labels_root: Path) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []

    for path in labels_root.iterdir():
        if not path.is_dir():
            continue
        match = PERCENT_PATTERN.fullmatch(path.name)
        if match:
            percent = int(match.group(1))
            if not 1 <= percent <= 99:
                raise ValueError(f"Invalid percentage folder: {path}")
            discovered.append((percent, path.resolve()))

    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise RuntimeError(
            f"No labeled_X_percent directories were found under: {labels_root}"
        )
    return discovered


def mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise TypeError(f"Config key {key!r} must be a mapping.")
    return value


def deep_copy_yaml(value: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(value, sort_keys=False))


def write_config(
    base: dict,
    output: Path,
    *,
    repo: Path,
    run_name: str,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    target_list: Path,
    nc: int,
    names: list[str],
    ssl_enabled: bool,
    epochs: int,
    batch_size: int,
    workers: int,
    device: int,
    burn_epochs: int | None,
) -> dict:
    cfg = deep_copy_yaml(base)
    dataset = mapping(cfg, "Dataset")
    ssod = mapping(cfg, "SSOD")
    hyp = mapping(cfg, "hyp")

    dataset["data_name"] = "hall_overlap0"
    dataset["train"] = str(train_list.resolve())
    dataset["val"] = str(val_list.resolve())
    dataset["test"] = str(test_list.resolve())
    dataset["target"] = str(target_list.resolve())
    dataset["nc"] = int(nc)
    dataset["names"] = list(names)
    dataset["img_size"] = 640
    dataset["batch_size"] = int(batch_size)
    dataset["workers"] = int(workers)

    cfg["epochs"] = int(epochs)
    cfg["device"] = int(device)
    cfg["project"] = str((repo / "runs" / "hall_overlap0_reproduction").resolve())
    cfg["name"] = run_name
    cfg["exist_ok"] = False

    ssod["train_domain"] = bool(ssl_enabled)

    if "debug" in cfg:
        cfg["debug"] = False
    if "debug" in ssod:
        ssod["debug"] = False

    if burn_epochs is not None:
        hyp["burn_epochs"] = int(burn_epochs)

    current_burn = int(hyp.get("burn_epochs", 0))
    if ssl_enabled and current_burn >= epochs:
        raise ValueError(
            f"{run_name}: burn_epochs={current_burn} must be less than epochs={epochs}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            cfg,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=180,
        )

    return {
        "name": run_name,
        "config": str(output.resolve()),
        "ssl": ssl_enabled,
        "train": str(train_list.resolve()),
        "target": str(target_list.resolve()),
        "epochs": epochs,
        "burn_epochs": current_burn,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    root = args.dataset_root.resolve()

    if not repo.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    labels_root = root / "labels"
    images_root = root / "images"
    split_root = root / "reproduction_splits_v3"
    config_root = repo / "configs" / "ssod" / "custom"

    print("Available image directories:")
    for directory in sorted(path for path in images_root.iterdir() if path.is_dir()):
        print(f"  - {directory.name}: {directory}")

    pavilion_dir = find_pavilion_dir(images_root)
    if pavilion_dir is None:
        print(
            "\nWARNING: No Pavilion directory was found. "
            "The script will create supervised and ssl_remaining experiments only."
        )
    else:
        print(f"\nDetected Pavilion directory: {pavilion_dir}")

    nc, names = read_dataset_yaml(root / "dataset.yaml")

    train_images = list_images(images_root / "train")
    val_images = list_images(images_root / "val")
    test_images = list_images(images_root / "test")
    pavilion_images = list_images(pavilion_dir) if pavilion_dir is not None else {}

    full_train_labels = list_labels(labels_root / "train")
    val_labels = list_labels(labels_root / "val")
    test_labels = list_labels(labels_root / "test")

    unknown_full = set(full_train_labels) - set(train_images)
    if unknown_full:
        raise RuntimeError(
            f"Full train labels without image: {sorted(unknown_full)[:20]}"
        )

    train_100_list = split_root / "train_100_percent.txt"
    val_list = split_root / "val.txt"
    test_list = split_root / "test.txt"
    pavilion_list = split_root / "target_pavilion.txt"

    fixed_reports: dict[str, dict] = {}
    fixed_reports["train_100"] = write_paired_list(
        train_images,
        full_train_labels,
        train_100_list,
        split_name="train_100_percent",
    )
    fixed_reports["val"] = write_paired_list(
        val_images,
        val_labels,
        val_list,
        split_name="val",
    )
    fixed_reports["test"] = write_paired_list(
        test_images,
        test_labels,
        test_list,
        split_name="test",
    )

    if pavilion_images:
        fixed_reports["pavilion"] = write_image_list(
            [pavilion_images[key] for key in sorted(pavilion_images)],
            pavilion_list,
            "target_pavilion",
        )

    base_config_path = (
        args.base_config.resolve()
        if args.base_config.is_absolute()
        else (repo / args.base_config).resolve()
    )
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with base_config_path.open("r", encoding="utf-8-sig") as handle:
        base_config = yaml.safe_load(handle)
    if not isinstance(base_config, dict):
        raise TypeError(f"Invalid base config: {base_config_path}")

    percentage_folders = discover_percent_folders(labels_root)
    config_reports: list[dict] = []
    percentage_reports: dict[str, dict] = {}

    full_train_stems = set(full_train_labels)

    for percent, folder in percentage_folders:
        subset_labels = list_labels(folder)
        subset_stems = set(subset_labels)

        unknown = subset_stems - full_train_stems
        if unknown:
            raise RuntimeError(
                f"{folder.name}: labels not present in labels/train: "
                f"{sorted(unknown)[:20]}"
            )

        labeled_list = split_root / f"train_{percent:02d}_percent.txt"
        remaining_list = (
            split_root / f"target_remaining_train_{percent:02d}_percent.txt"
        )

        labeled_report = write_paired_list(
            train_images,
            subset_labels,
            labeled_list,
            split_name=f"train_{percent}_percent",
        )

        remaining_stems = sorted(full_train_stems - subset_stems)
        remaining_paths = [train_images[stem] for stem in remaining_stems]
        remaining_report = write_image_list(
            remaining_paths,
            remaining_list,
            f"remaining_train_for_{percent}_percent",
        )

        percentage_reports[str(percent)] = {
            "folder": str(folder),
            "labeled": labeled_report,
            "remaining_train": remaining_report,
        }

        common = dict(
            base=base_config,
            repo=repo,
            train_list=labeled_list,
            val_list=val_list,
            test_list=test_list,
            nc=nc,
            names=names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            burn_epochs=args.burn_epochs,
        )

        supervised_name = f"hall_overlap0_{percent:02d}pct_supervised_patience7"
        config_reports.append(
            write_config(
                output=config_root / f"{supervised_name}.yaml",
                run_name=supervised_name,
                target_list=remaining_list,
                ssl_enabled=False,
                **common,
            )
        )

        remaining_name = f"hall_overlap0_{percent:02d}pct_ssl_remaining_patience7"
        config_reports.append(
            write_config(
                output=config_root / f"{remaining_name}.yaml",
                run_name=remaining_name,
                target_list=remaining_list,
                ssl_enabled=True,
                **common,
            )
        )

        if pavilion_images:
            pavilion_name = f"hall_overlap0_{percent:02d}pct_ssl_pavilion_patience7"
            config_reports.append(
                write_config(
                    output=config_root / f"{pavilion_name}.yaml",
                    run_name=pavilion_name,
                    target_list=pavilion_list,
                    ssl_enabled=True,
                    **common,
                )
            )

            combined_list = (
                split_root
                / f"target_remaining_plus_pavilion_{percent:02d}_percent.txt"
            )
            combined_paths = remaining_paths + [
                pavilion_images[key] for key in sorted(pavilion_images)
            ]
            combined_report = write_image_list(
                combined_paths,
                combined_list,
                f"remaining_plus_pavilion_for_{percent}_percent",
            )
            percentage_reports[str(percent)]["combined"] = combined_report

            combined_name = (
                f"hall_overlap0_{percent:02d}pct_ssl_combined_patience7"
            )
            config_reports.append(
                write_config(
                    output=config_root / f"{combined_name}.yaml",
                    run_name=combined_name,
                    target_list=combined_list,
                    ssl_enabled=True,
                    **common,
                )
            )

    # 100% supervised; external-domain SSL only when Pavilion is available.
    common_100 = dict(
        base=base_config,
        repo=repo,
        train_list=train_100_list,
        val_list=val_list,
        test_list=test_list,
        nc=nc,
        names=names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        burn_epochs=args.burn_epochs,
    )

    name_100 = "hall_overlap0_100pct_supervised_patience7"
    config_reports.append(
        write_config(
            output=config_root / f"{name_100}.yaml",
            run_name=name_100,
            target_list=train_100_list,
            ssl_enabled=False,
            **common_100,
        )
    )

    if pavilion_images:
        name_100_ssl = "hall_overlap0_100pct_ssl_pavilion_patience7"
        config_reports.append(
            write_config(
                output=config_root / f"{name_100_ssl}.yaml",
                run_name=name_100_ssl,
                target_list=pavilion_list,
                ssl_enabled=True,
                **common_100,
            )
        )

    leakage = {
        "train_val": len(set(train_images) & set(val_images)),
        "train_test": len(set(train_images) & set(test_images)),
        "val_test": len(set(val_images) & set(test_images)),
    }
    if pavilion_images:
        leakage.update(
            {
                "pavilion_train": len(set(pavilion_images) & set(train_images)),
                "pavilion_val": len(set(pavilion_images) & set(val_images)),
                "pavilion_test": len(set(pavilion_images) & set(test_images)),
            }
        )

    report = {
        "repo": str(repo),
        "dataset_root": str(root),
        "pavilion_directory": str(pavilion_dir) if pavilion_dir else None,
        "nc": nc,
        "names": names,
        "available_percentages": [value for value, _ in percentage_folders],
        "fixed_splits": fixed_reports,
        "percentage_splits": percentage_reports,
        "filename_leakage": leakage,
        "configs": config_reports,
    }

    report_path = split_root / "hall_percent_experiments_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("HALL OVERLAP-0 EXPERIMENT SETUP COMPLETE")
    print("=" * 92)
    print(f"Available percentages: {[value for value, _ in percentage_folders]}")
    print(f"Classes: {names}")
    print(f"Full train pairs: {len(full_train_labels)}")
    print(f"Validation pairs: {len(val_labels)}")
    print(f"Test pairs: {len(test_labels)}")
    print(f"Pavilion images: {len(pavilion_images)}")
    print(f"Filename leakage: {leakage}")
    print(f"Created configs: {len(config_reports)}")
    print(f"Report: {report_path}")

    print("\nRecommended smoke-test configs:")
    print("  hall_overlap0_10pct_supervised_patience7.yaml")
    print("  hall_overlap0_10pct_ssl_remaining_patience7.yaml")
    if pavilion_images:
        print("  hall_overlap0_10pct_ssl_pavilion_patience7.yaml")


if __name__ == "__main__":
    main()
