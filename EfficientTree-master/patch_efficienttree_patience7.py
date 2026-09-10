from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_REPO = Path(r"E:\FASTRCNN\FASTRCNN\EfficientTree-master")
DEFAULT_CONFIG = Path(r"configs\ssod\custom\yolov8s_tree_ssod.yaml")
DEFAULT_OUTPUT_CONFIG = Path(
    r"configs\ssod\custom\yolov8s_tree_ssod_patience7.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely set EfficientTree early-stopping patience and create a "
            "separate practical reproduction config."
        )
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--name",
        default="EfficientTree_public_patience7",
    )
    return parser.parse_args()


def absolute_under_repo(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def backup_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def patch_early_stopping(repo: Path, patience: int) -> list[tuple[Path, Path]]:
    """
    Patch only assignments such as:
        self.stopper = EarlyStopping(patience=30)
        self.stopper = EarlyStopping(30)

    It intentionally does not inject new stopping logic. If the local source does
    not already contain an EarlyStopping assignment, it exits without altering it.
    """
    candidates = [
        repo / "trainer" / "trainer.py",
        repo / "trainer" / "ssod_trainer.py",
    ]

    named_pattern = re.compile(
        r"(?P<prefix>\b(?:self\.)?(?:stopper|early_stopper)\s*=\s*"
        r"EarlyStopping\s*\(\s*patience\s*=\s*)"
        r"(?P<value>[^,\)\r\n]+)"
    )
    positional_pattern = re.compile(
        r"(?P<prefix>\b(?:self\.)?(?:stopper|early_stopper)\s*=\s*"
        r"EarlyStopping\s*\(\s*)"
        r"(?P<value>\d+)"
    )

    changed: list[tuple[Path, Path]] = []
    diagnostic_lines: list[str] = []

    for path in candidates:
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in ("EarlyStopping", "stopper", "patience")):
                diagnostic_lines.append(f"{path}:{line_number}: {line.strip()}")

        updated, count_named = named_pattern.subn(
            lambda match: f"{match.group('prefix')}{patience}",
            text,
        )
        updated, count_positional = positional_pattern.subn(
            lambda match: f"{match.group('prefix')}{patience}",
            updated,
        )

        if count_named + count_positional:
            backup = backup_file(path)
            path.write_text(updated, encoding="utf-8")
            changed.append((path, backup))

    if not changed:
        details = "\n".join(diagnostic_lines) or "No matching lines were found."
        raise RuntimeError(
            "The local repository did not contain a safely patchable "
            "EarlyStopping assignment.\n\n"
            "Relevant lines found:\n"
            f"{details}\n\n"
            "No trainer file was changed."
        )

    return changed


def create_config(
    repo: Path,
    source_config: Path,
    output_config: Path,
    max_epochs: int,
    batch_size: int,
    workers: int,
    device: int,
    name: str,
) -> dict:
    if not source_config.is_file():
        raise FileNotFoundError(f"Training config not found: {source_config}")

    with source_config.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping: {source_config}")

    config["epochs"] = int(max_epochs)
    config["device"] = int(device)
    config["name"] = str(name)
    config["project"] = str(repo / "runs" / "paper_public_reproduction")
    config["exist_ok"] = False

    dataset = config.setdefault("Dataset", {})
    if not isinstance(dataset, dict):
        raise TypeError("Config key Dataset must be a mapping.")
    dataset["batch_size"] = int(batch_size)
    dataset["workers"] = int(workers)

    ssod = config.setdefault("SSOD", {})
    if not isinstance(ssod, dict):
        raise TypeError("Config key SSOD must be a mapping.")
    ssod["train_domain"] = True

    # The released code contains hard-coded pseudo-label debug image writes.
    # Disabling debug prevents those writes during the full training run.
    if "debug" in ssod:
        ssod["debug"] = False
    elif "debug" in config:
        config["debug"] = False

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=140,
        )

    return config


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()

    if not repo.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo}")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1.")

    source_config = absolute_under_repo(repo, args.config)
    output_config = absolute_under_repo(repo, args.output_config)

    changed = patch_early_stopping(repo, args.patience)
    config = create_config(
        repo=repo,
        source_config=source_config,
        output_config=output_config,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        name=args.name,
    )

    print("=" * 78)
    print("EFFICIENTTREE EARLY STOPPING PATCH COMPLETE")
    print("=" * 78)
    print(f"Patience: {args.patience} consecutive validations without improvement")
    print(f"Maximum epochs: {args.max_epochs}")
    print("\nChanged trainer files:")
    for path, backup in changed:
        print(f"  Changed: {path}")
        print(f"  Backup : {backup}")

    print(f"\nNew config: {output_config}")
    print("\nDataset entries preserved from the author's config:")
    dataset = config.get("Dataset", {})
    for key in ("train", "all", "val", "test", "target", "nc", "names"):
        if key in dataset:
            print(f"  Dataset.{key}: {dataset[key]}")

    print("\nIMPORTANT:")
    print("1. Confirm every displayed Dataset path exists on this computer.")
    print("2. Test data must not be used by early stopping.")
    print("3. Run a 2-epoch smoke test before the full run.")
    print("\nFull training command:")
    print(
        f'  & "C:\\ProgramData\\Anaconda3\\envs\\p311cuda\\python.exe" '
        f'train.py --cfg "{output_config.relative_to(repo)}"'
    )


if __name__ == "__main__":
    main()
