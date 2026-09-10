from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set device: 0 as an integer in all generated Hall experiment configs."
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
        raise FileNotFoundError(
            f"No matching configs were found in: {config_dir}"
        )

    changed = 0
    for path in files:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML mapping: {path}")

        if config.get("device") != 0 or type(config.get("device")) is not int:
            config["device"] = 0
            path.write_text(
                yaml.safe_dump(
                    config,
                    sort_keys=False,
                    allow_unicode=True,
                    width=180,
                ),
                encoding="utf-8",
            )
            changed += 1

    bad = []
    for path in files:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        value = config.get("device")
        if type(value) is not int or value != 0:
            bad.append((path, value, type(value).__name__))

    print(f"configs found   = {len(files)}")
    print(f"configs changed = {changed}")
    print(f"configs valid   = {len(files) - len(bad)}")

    if bad:
        for path, value, type_name in bad:
            print(f"INVALID: {path} -> {value!r} ({type_name})")
        raise RuntimeError("Some configs still have an invalid device value.")

    sample = files[0]
    sample_cfg = yaml.safe_load(sample.read_text(encoding="utf-8"))
    print(f"sample          = {sample}")
    print(
        f"sample device   = {sample_cfg['device']!r} "
        f"({type(sample_cfg['device']).__name__})"
    )


if __name__ == "__main__":
    main()
