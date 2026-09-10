from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml


CLASS_MAP = {2: 0, 3: 1, 4: 2, 5: 3}
CLASS_NAMES = [
    "china cedar",
    "masson pine",
    "golden larch",
    "ginkgo",
]
PAIR_PATTERN = re.compile(
    r"^(?P<image>.+\.(?:jpg|jpeg|png|tif|tiff|bmp|webp))\s+"
    r"(?P<label>.+\.txt)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive four-class Hall dataset/config variant by "
            "remapping YOLO class IDs 2,3,4,5 to 0,1,2,3."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="EfficientTree repository root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Hall tree_overlap_0_patch_640 directory.",
    )
    parser.add_argument(
        "--source-splits",
        default="reproduction_splits_v3",
        help="Existing split-list directory name under dataset root.",
    )
    parser.add_argument(
        "--output-splits",
        default="reproduction_splits_v4_4class",
        help="New remapped split-list directory name under dataset root.",
    )
    parser.add_argument(
        "--output-labels",
        default="labels_remapped_4class",
        help="New remapped label directory name under dataset root.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            width=180,
        ),
        encoding="utf-8",
    )


def parse_split_line(raw: str) -> tuple[Path, Path | None]:
    line = raw.strip()
    if not line:
        raise ValueError("empty split line")

    match = PAIR_PATTERN.match(line)
    if match:
        return Path(match.group("image")), Path(match.group("label"))

    # Unlabeled target lists contain image paths only.
    return Path(line), None


def remap_label_file(
    source: Path,
    destination: Path,
) -> tuple[Counter[int], Counter[int], int]:
    source_counts: Counter[int] = Counter()
    output_counts: Counter[int] = Counter()
    boxes = 0
    output_lines: list[str] = []

    if not source.is_file():
        raise FileNotFoundError(f"Referenced label file not found: {source}")

    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) < 5:
            raise ValueError(
                f"Malformed YOLO label at {source}:{line_number}: {line!r}"
            )

        try:
            source_class = int(float(fields[0]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid class ID at {source}:{line_number}: {fields[0]!r}"
            ) from exc

        if source_class not in CLASS_MAP:
            raise ValueError(
                f"Unexpected class ID {source_class} at "
                f"{source}:{line_number}; expected only {sorted(CLASS_MAP)}."
            )

        output_class = CLASS_MAP[source_class]
        source_counts[source_class] += 1
        output_counts[output_class] += 1
        boxes += 1

        fields[0] = str(output_class)
        output_lines.append(" ".join(fields))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )
    return source_counts, output_counts, boxes


def make_output_config_name(source_name: str) -> str:
    suffix = "_patience7.yaml"
    if source_name.endswith(suffix):
        return source_name[: -len(suffix)] + "_4class_patience7.yaml"
    return Path(source_name).stem + "_4class.yaml"


def audit_paired_split(path: Path) -> dict:
    class_counts: Counter[int] = Counter()
    rows = 0
    empty_labels = 0
    image_only_rows = 0

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            continue
        rows += 1
        _, label_path = parse_split_line(raw)
        if label_path is None:
            image_only_rows += 1
            continue

        label_lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not label_lines:
            empty_labels += 1
            continue

        for line in label_lines:
            class_id = int(float(line.split()[0]))
            class_counts[class_id] += 1

    unexpected = sorted(set(class_counts) - {0, 1, 2, 3})
    return {
        "rows": rows,
        "image_only_rows": image_only_rows,
        "empty_label_files": empty_labels,
        "boxes": int(sum(class_counts.values())),
        "class_counts": {
            str(key): int(value)
            for key, value in sorted(class_counts.items())
        },
        "unexpected_class_ids": unexpected,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    dataset_root = args.dataset_root.resolve()

    source_split_root = dataset_root / args.source_splits
    output_split_root = dataset_root / args.output_splits
    source_labels_root = dataset_root / "labels"
    output_labels_root = dataset_root / args.output_labels
    config_root = repo / "configs" / "ssod" / "custom"

    for required in (
        repo,
        dataset_root,
        source_split_root,
        source_labels_root,
        config_root,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    split_files = sorted(source_split_root.glob("*.txt"))
    if not split_files:
        raise FileNotFoundError(
            f"No split TXT files found in: {source_split_root}"
        )

    output_split_root.mkdir(parents=True, exist_ok=True)
    output_labels_root.mkdir(parents=True, exist_ok=True)

    remapped_cache: dict[Path, Path] = {}
    source_counts_total: Counter[int] = Counter()
    output_counts_total: Counter[int] = Counter()
    split_reports: dict[str, dict] = {}

    for source_split in split_files:
        output_split = output_split_root / source_split.name
        output_rows: list[str] = []
        paired_rows = 0
        image_only_rows = 0

        for line_number, raw in enumerate(
            source_split.read_text(encoding="utf-8-sig").splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue

            image_path, label_path = parse_split_line(raw)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image missing in {source_split}:{line_number}: {image_path}"
                )

            if label_path is None:
                image_only_rows += 1
                output_rows.append(str(image_path.resolve()))
                continue

            paired_rows += 1
            label_path = label_path.resolve()

            try:
                relative_label = label_path.relative_to(source_labels_root)
            except ValueError as exc:
                raise ValueError(
                    f"Label path is outside {source_labels_root}: {label_path}"
                ) from exc

            output_label = (output_labels_root / relative_label).resolve()

            if label_path not in remapped_cache:
                src_counts, dst_counts, _ = remap_label_file(
                    label_path,
                    output_label,
                )
                source_counts_total.update(src_counts)
                output_counts_total.update(dst_counts)
                remapped_cache[label_path] = output_label

            output_rows.append(
                f"{image_path.resolve()} {remapped_cache[label_path]}"
            )

        output_split.write_text(
            "\n".join(output_rows) + ("\n" if output_rows else ""),
            encoding="utf-8",
        )

        split_reports[source_split.name] = {
            "source": str(source_split.resolve()),
            "output": str(output_split.resolve()),
            "rows": len(output_rows),
            "paired_rows": paired_rows,
            "image_only_rows": image_only_rows,
        }

    # Create a four-class dataset metadata file without modifying dataset.yaml.
    source_dataset_yaml = dataset_root / "dataset.yaml"
    dataset_4class = (
        read_yaml(source_dataset_yaml)
        if source_dataset_yaml.is_file()
        else {}
    )
    dataset_4class["nc"] = 4
    dataset_4class["names"] = CLASS_NAMES
    dataset_4class_path = dataset_root / "dataset_4class.yaml"
    write_yaml(dataset_4class_path, dataset_4class)

    source_configs = sorted(
        path
        for path in config_root.glob("hall_overlap0_*patience7.yaml")
        if "_4class" not in path.stem
    )
    if not source_configs:
        raise FileNotFoundError(
            f"No original Hall patience7 configs found in: {config_root}"
        )

    generated_configs: list[str] = []
    for source_config in source_configs:
        cfg = read_yaml(source_config)
        dataset_cfg = cfg.get("Dataset")
        if not isinstance(dataset_cfg, dict):
            raise ValueError(
                f"Dataset section missing or invalid: {source_config}"
            )

        for key in ("train", "val", "test", "target"):
            old_value = dataset_cfg.get(key)
            if not isinstance(old_value, str) or not old_value:
                raise ValueError(
                    f"Dataset.{key} missing in: {source_config}"
                )
            replacement = output_split_root / Path(old_value).name
            if not replacement.is_file():
                raise FileNotFoundError(
                    f"Remapped split for Dataset.{key} not found: {replacement}"
                )
            dataset_cfg[key] = str(replacement.resolve())

        dataset_cfg["data_name"] = "hall_overlap0_4class"
        dataset_cfg["nc"] = 4
        dataset_cfg["names"] = CLASS_NAMES

        # Preserve the successful behavior used in this repository:
        # omit device from YAML and let the project default choose GPU 0.
        cfg.pop("device", None)

        if isinstance(cfg.get("SSOD"), dict):
            cfg["SSOD"]["debug"] = False
        if "debug" in cfg:
            cfg["debug"] = False

        output_name = make_output_config_name(source_config.name)
        output_config = config_root / output_name
        cfg["name"] = output_config.stem
        write_yaml(output_config, cfg)
        generated_configs.append(str(output_config.resolve()))

    # Final audit of every generated split.
    audit_reports = {
        path.name: audit_paired_split(path)
        for path in sorted(output_split_root.glob("*.txt"))
    }
    bad_splits = {
        name: report
        for name, report in audit_reports.items()
        if report["unexpected_class_ids"]
    }
    if bad_splits:
        raise RuntimeError(
            "Unexpected class IDs remain after remapping: "
            + json.dumps(bad_splits, ensure_ascii=False)
        )

    report = {
        "repo": str(repo),
        "dataset_root": str(dataset_root),
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "class_names": CLASS_NAMES,
        "source_split_root": str(source_split_root),
        "output_split_root": str(output_split_root),
        "output_labels_root": str(output_labels_root),
        "unique_label_files_remapped": len(remapped_cache),
        "unique_source_class_counts": {
            str(key): int(value)
            for key, value in sorted(source_counts_total.items())
        },
        "unique_output_class_counts": {
            str(key): int(value)
            for key, value in sorted(output_counts_total.items())
        },
        "split_generation": split_reports,
        "split_audit": audit_reports,
        "generated_configs": generated_configs,
        "dataset_yaml": str(dataset_4class_path.resolve()),
    }
    report_path = output_split_root / "four_class_remap_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 92)
    print("HALL FOUR-CLASS REMAP COMPLETE")
    print("=" * 92)
    print(f"Class mapping          : {CLASS_MAP}")
    print(f"Class names            : {CLASS_NAMES}")
    print(f"Unique labels remapped : {len(remapped_cache)}")
    print(f"New split directory    : {output_split_root}")
    print(f"New label directory    : {output_labels_root}")
    print(f"Generated configs      : {len(generated_configs)}")
    print(f"Report                 : {report_path}")
    print("\nRecommended SSL smoke-test config:")
    print(
        config_root
        / "hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml"
    )


if __name__ == "__main__":
    main()
