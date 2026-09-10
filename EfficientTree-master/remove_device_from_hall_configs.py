from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the root-level device key from generated Hall YAML configs."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="EfficientTree repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    config_dir = repo / "configs" / "ssod" / "custom"
    files = sorted(config_dir.glob("hall_overlap0_*patience7.yaml"))

    if not files:
        raise FileNotFoundError(f"No generated configs found in: {config_dir}")

    changed = 0
    for path in files:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML mapping: {path}")

        if "device" in cfg:
            cfg.pop("device", None)
            changed += 1

        path.write_text(
            yaml.safe_dump(
                cfg,
                sort_keys=False,
                allow_unicode=True,
                width=180,
            ),
            encoding="utf-8",
        )

    remaining = []
    for path in files:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and "device" in cfg:
            remaining.append(path)

    print(f"configs found          = {len(files)}")
    print(f"device keys removed    = {changed}")
    print(f"configs without device = {len(files) - len(remaining)}")

    if remaining:
        for path in remaining:
            print(f"STILL HAS DEVICE: {path}")
        raise RuntimeError("Some configs still contain a root-level device key.")

    sample = files[0]
    print(f"sample                 = {sample}")
    print("sample has device key  = False")


if __name__ == "__main__":
    main()
