from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create full-data natural/unbalanced Supervised and SSL configs from the verified Hall configs."
    )
    p.add_argument(
        "--repo",
        required=True,
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
    )
    p.add_argument(
        "--base-supervised",
        default=r"configs\ssod\custom\hall_overlap0_10pct_supervised_4class_patience7.yaml",
    )
    p.add_argument(
        "--base-ssl",
        default=r"configs\ssod\custom\hall_overlap0_10pct_ssl_pavilion_4class_patience7.yaml",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--burn-epochs", type=int, default=1)
    p.add_argument(
        "--class-names",
        nargs=4,
        default=["class_0", "class_1", "class_2", "class_3"],
    )
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root is not a mapping: {path}")
    return data


def patch_common(
    cfg: Dict[str, Any],
    dataset_dir: Path,
    class_names: list[str],
    batch_size: int,
    workers: int,
    img_size: int,
    epochs: int,
    project: Path,
    name: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    ds = cfg.setdefault("Dataset", {})
    ds.update(
        {
            "data_name": "user_tree_full_unbalanced_4class",
            "nc": 4,
            "names": class_names,
            "batch_size": batch_size,
            "workers": workers,
            "img_size": img_size,
            "train": str((dataset_dir / "splits" / "train.txt").resolve()),
            "val": str((dataset_dir / "splits" / "val.txt").resolve()),
            "test": str((dataset_dir / "splits" / "test.txt").resolve()),
            "target": str((dataset_dir / "splits" / "target_unlabeled.txt").resolve()),
        }
    )
    cfg["epochs"] = epochs
    cfg["project"] = str(project.resolve())
    cfg["name"] = name
    cfg["exist_ok"] = True
    cfg["resume"] = False
    # Avoid the old device-as-integer config problem; CLI/GPU auto-selection remains valid.
    cfg["device"] = ""
    return cfg


def patch_smoke(cfg: Dict[str, Any], dataset_dir: Path, name: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    ds = out["Dataset"]
    ds["train"] = str((dataset_dir / "splits" / "smoke_train.txt").resolve())
    ds["val"] = str((dataset_dir / "splits" / "smoke_val.txt").resolve())
    ds["test"] = str((dataset_dir / "splits" / "smoke_test.txt").resolve())
    ds["target"] = str(
        (dataset_dir / "splits" / "smoke_target_unlabeled.txt").resolve()
    )
    out["name"] = name
    return out


def write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    dataset_dir = Path(args.dataset_dir)
    config_dir = repo / "configs" / "ssod" / "custom"
    project = repo / "runs" / "mydata_full_unbalanced"

    required = [
        dataset_dir / "splits" / "train.txt",
        dataset_dir / "splits" / "val.txt",
        dataset_dir / "splits" / "test.txt",
        dataset_dir / "splits" / "target_unlabeled.txt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepared dataset files are missing. Run prepare_mydata_640.py first:\n"
            + "\n".join(missing)
        )

    base_sup = Path(args.base_supervised)
    if not base_sup.is_absolute():
        base_sup = repo / base_sup
    base_ssl = Path(args.base_ssl)
    if not base_ssl.is_absolute():
        base_ssl = repo / base_ssl

    sup = patch_common(
        load_yaml(base_sup),
        dataset_dir,
        list(args.class_names),
        args.batch_size,
        args.workers,
        args.img_size,
        args.epochs,
        project,
        "mydata_full_supervised_4class_staged70",
    )
    ssl = patch_common(
        load_yaml(base_ssl),
        dataset_dir,
        list(args.class_names),
        args.batch_size,
        args.workers,
        args.img_size,
        args.epochs,
        project,
        "mydata_full_ssl_4class_staged70",
    )
    ssl.setdefault("hyp", {})["burn_epochs"] = args.burn_epochs
    ssl.setdefault("SSOD", {})["debug"] = False

    paths = {
        "supervised": config_dir / "mydata_full_supervised_4class_staged70.yaml",
        "ssl": config_dir / "mydata_full_ssl_4class_staged70.yaml",
        "smoke_ssl": config_dir / "mydata_smoke_ssl_4class.yaml",
    }
    write_yaml(paths["supervised"], sup)
    write_yaml(paths["ssl"], ssl)
    write_yaml(paths["smoke_ssl"], patch_smoke(ssl, dataset_dir, "mydata_smoke_ssl_4class"))

    print("Configs created:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print("\nThe full train split is used. No train_10_percent.txt is referenced.")


if __name__ == "__main__":
    main()
