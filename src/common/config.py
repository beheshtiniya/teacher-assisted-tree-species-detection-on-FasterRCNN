from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(self.raw["project_root"])

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"])

    @property
    def repo(self) -> Path:
        return Path(self.raw["efficienttree_repo"])

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"])

    @property
    def python(self) -> str:
        return str(self.raw["python_executable"])

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self.raw.get("settings", {}))

    # ---------------------------------------------------------
    # Explicit dataset paths
    # ---------------------------------------------------------

    @property
    def labels(self) -> Path:
        return Path(self.raw["labels"])

    @property
    def images(self) -> Path:
        return Path(self.raw["images"])

    # ---------------------------------------------------------
    # Explicit label files
    # ---------------------------------------------------------

    @property
    def train_csv(self) -> Path:
        return Path(self.raw["train_labels_csv"])

    @property
    def val_csv(self) -> Path:
        return Path(self.raw["val_labels_csv"])

    @property
    def test_csv(self) -> Path:
        return Path(self.raw["test_labels_csv"])

    @property
    def unlabeled_list(self) -> Path:
        return Path(self.raw["unlabeled_images_txt"])

    # ---------------------------------------------------------
    # Output helper
    # ---------------------------------------------------------

    def out(self, *parts: str) -> Path:
        return self.output_root.joinpath(*parts)


def load_config(path: Path | None = None) -> ProjectConfig:

    if path is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "paths.local.json"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Local config not found: {path}\n"
            f"Run configure_paths.py first."
        )

    raw = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if raw.get("schema_version") != 1:
        raise ValueError(
            "Unsupported config schema."
        )

    # ---------------------------------------------------------
    # Validate important keys
    # ---------------------------------------------------------

    required_keys = [
        "project_root",
        "data_root",
        "images",
        "labels",
        "train_labels_csv",
        "val_labels_csv",
        "test_labels_csv",
        "unlabeled_images_txt",
        "efficienttree_repo",
        "output_root",
        "python_executable",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in raw
    ]

    if missing_keys:
        raise KeyError(
            "Missing required config keys:\n"
            + "\n".join(missing_keys)
        )

    return ProjectConfig(
        path.resolve(),
        raw,
    )