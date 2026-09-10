from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the public Hall/overlap-0 EfficientTree dataset for "
            "10%, 20%, and 100% supervised/SSL experiments."
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
        help=(
            "Optional burn-in override. Omit to preserve the author's config. "
            "The script refuses a config where burn_epochs >= total epochs."
        ),
    )
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate.resolve()

    lowered = stem.lower()
    for candidate in images_dir.iterdir():
        if (
            candidate.is_file()
            and candidate.suffix.lower() in IMAGE_EXTENSIONS
            and candidate.stem.lower() == lowered
        ):
            return candidate.resolve()
    return None


def label_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory not found: {directory}")
    return sorted(path for path in directory.glob("*.txt") if path.is_file())


def create_paired_list(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
) -> dict:
    labels = label_files(labels_dir)
    if not labels:
        raise RuntimeError(f"No label TXT files found: {labels_dir}")

    rows: list[str] = []
    missing: list[str] = []

    for label_path in labels:
        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            missing.append(label_path.stem)
            continue
        rows.append(f"{image_path} {label_path.resolve()}")

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} labels have no matching image in {images_dir}. "
            f"Examples: {missing[:20]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    return {
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "list_file": str(output_path.resolve()),
        "pairs": len(rows),
    }


def create_unlabeled_list(images_dir: Path, output_path: Path) -> dict:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Unlabeled image directory not found: {images_dir}")

    images = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No unlabeled images found: {images_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(path) for path in images) + "\n",
        encoding="utf-8",
    )
    return {
        "images_dir": str(images_dir.resolve()),
        "list_file": str(output_path.resolve()),
        "images": len(images),
    }


def read_dataset_metadata(dataset_yaml: Path) -> tuple[int, list[str]]:
    with dataset_yaml.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"Invalid dataset YAML: {dataset_yaml}")

    names_raw = data.get("names")
    nc_raw = data.get("nc")

    if isinstance(names_raw, dict):
        names = [str(names_raw[key]) for key in sorted(names_raw, key=lambda x: int(x))]
    elif isinstance(names_raw, list):
        names = [str(name) for name in names_raw]
    else:
        names = []

    nc = int(nc_raw) if nc_raw is not None else len(names)
    if not names:
        names = [f"class{i}" for i in range(nc)]
    if len(names) != nc:
        raise ValueError(f"dataset.yaml says nc={nc}, but contains {len(names)} names.")

    return nc, names


def ensure_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if value is None:
        value = {}
        config[key] = value
    if not isinstance(value, dict):
        raise TypeError(f"Config key {key!r} must be a mapping.")
    return value


def write_config(
    base: dict,
    output_path: Path,
    *,
    repo: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    target_list: Path,
    nc: int,
    names: list[str],
    epochs: int,
    batch_size: int,
    workers: int,
    device: int,
    run_name: str,
    ssl_enabled: bool,
    burn_epochs: int | None,
) -> dict:
    # Round-trip through YAML to make an independent deep copy.
    config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    dataset = ensure_mapping(config, "Dataset")
    ssod = ensure_mapping(config, "SSOD")
    hyp = ensure_mapping(config, "hyp")

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

    config["epochs"] = int(epochs)
    config["device"] = int(device)
    config["project"] = str((repo / "runs" / "hall_overlap0_reproduction").resolve())
    config["name"] = run_name
    config["exist_ok"] = False

    ssod["train_domain"] = bool(ssl_enabled)
    if "debug" in ssod:
        ssod["debug"] = False
    if "debug" in config:
        config["debug"] = False

    if burn_epochs is not None:
        hyp["burn_epochs"] = int(burn_epochs)

    current_burn = int(hyp.get("burn_epochs", 0))
    if ssl_enabled and current_burn >= epochs:
        raise ValueError(
            f"{run_name}: hyp.burn_epochs={current_burn} but epochs={epochs}. "
            "SSL would never start. Re-run with --burn-epochs smaller than --epochs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=160,
        )

    return {
        "config": str(output_path.resolve()),
        "name": run_name,
        "ssl": ssl_enabled,
        "epochs": epochs,
        "burn_epochs": current_burn,
        "train": str(train_list.resolve()),
        "target": str(target_list.resolve()),
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    root = args.dataset_root.resolve()

    if not repo.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    dataset_yaml = root / "dataset.yaml"
    nc, names = read_dataset_metadata(dataset_yaml)

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

    images = root / "images"
    labels = root / "labels"
    split_dir = root / "reproduction_splits"

    reports: dict[str, dict] = {}
    reports["train_100"] = create_paired_list(
        images / "train",
        labels / "train",
        split_dir / "train_100_percent.txt",
    )
    reports["train_10"] = create_paired_list(
        images / "train",
        labels / "labeled_10_percent",
        split_dir / "train_10_percent.txt",
    )
    reports["train_20"] = create_paired_list(
        images / "train",
        labels / "labeled_20_percent",
        split_dir / "train_20_percent.txt",
    )
    reports["val"] = create_paired_list(
        images / "val",
        labels / "val",
        split_dir / "val.txt",
    )
    reports["test"] = create_paired_list(
        images / "test",
        labels / "test",
        split_dir / "test.txt",
    )
    reports["pavilon"] = create_unlabeled_list(
        images / "pavilon",
        split_dir / "target_pavilon.txt",
    )

    train_stems = {
        Path(line.split(" ", 1)[0]).stem.lower()
        for line in (split_dir / "train_100_percent.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    val_stems = {
        Path(line.split(" ", 1)[0]).stem.lower()
        for line in (split_dir / "val.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    test_stems = {
        Path(line.split(" ", 1)[0]).stem.lower()
        for line in (split_dir / "test.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    target_stems = {
        Path(line.strip()).stem.lower()
        for line in (split_dir / "target_pavilon.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    overlaps = {
        "train_val": len(train_stems & val_stems),
        "train_test": len(train_stems & test_stems),
        "val_test": len(val_stems & test_stems),
        "target_train": len(target_stems & train_stems),
        "target_val": len(target_stems & val_stems),
        "target_test": len(target_stems & test_stems),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Filename leakage detected: {overlaps}")

    config_dir = repo / "configs" / "ssod" / "custom"
    config_reports: list[dict] = []

    experiments = [
        ("10_percent", split_dir / "train_10_percent.txt"),
        ("20_percent", split_dir / "train_20_percent.txt"),
        ("100_percent", split_dir / "train_100_percent.txt"),
    ]

    for label_name, train_list in experiments:
        for ssl_enabled in (False, True):
            mode = "ssl" if ssl_enabled else "supervised"
            run_name = f"hall_overlap0_{label_name}_{mode}_patience7"
            output_path = config_dir / f"{run_name}.yaml"
            config_reports.append(
                write_config(
                    base_config,
                    output_path,
                    repo=repo,
                    train_list=train_list,
                    val_list=split_dir / "val.txt",
                    test_list=split_dir / "test.txt",
                    target_list=split_dir / "target_pavilon.txt",
                    nc=nc,
                    names=names,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    device=args.device,
                    run_name=run_name,
                    ssl_enabled=ssl_enabled,
                    burn_epochs=args.burn_epochs,
                )
            )

    report = {
        "repo": str(repo),
        "dataset_root": str(root),
        "dataset_yaml": str(dataset_yaml),
        "nc": nc,
        "names": names,
        "splits": reports,
        "filename_overlap": overlaps,
        "configs": config_reports,
    }
    report_path = split_dir / "hall_reproduction_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 84)
    print("HALL OVERLAP-0 REPRODUCTION SETUP CREATED")
    print("=" * 84)
    print(f"Classes: {nc} -> {names}")
    for key, value in reports.items():
        count = value.get("pairs", value.get("images"))
        print(f"{key:>12}: {count}")
    print(f"Leakage check: {overlaps}")
    print(f"Report: {report_path}")
    print("\nCreated configs:")
    for item in config_reports:
        print(
            f"  {Path(item['config']).name} | "
            f"SSL={item['ssl']} | burn={item['burn_epochs']} | "
            f"epochs={item['epochs']}"
        )

    print("\nRecommended first smoke test:")
    first_config = config_dir / "hall_overlap0_10_percent_ssl_patience7.yaml"
    print(
        f'  & "C:\\ProgramData\\Anaconda3\\envs\\p311cuda\\python.exe" '
        f'train.py --cfg "{first_config.relative_to(repo)}" '
        f'epochs 2 name hall_overlap0_10_percent_ssl_smoke'
    )


if __name__ == "__main__":
    main()
