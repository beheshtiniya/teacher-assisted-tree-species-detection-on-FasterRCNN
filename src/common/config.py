
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
    def project_root(self) -> Path: return Path(self.raw['project_root'])
    @property
    def data_root(self) -> Path: return Path(self.raw['data_root'])
    @property
    def repo(self) -> Path: return Path(self.raw['efficienttree_repo'])
    @property
    def output_root(self) -> Path: return Path(self.raw['output_root'])
    @property
    def python(self) -> str: return str(self.raw['python_executable'])
    @property
    def settings(self) -> dict[str, Any]: return dict(self.raw.get('settings', {}))
    @property
    def labels(self) -> Path: return self.data_root/'labels'
    @property
    def images(self) -> Path: return self.data_root/'images'
    def out(self, *parts: str) -> Path: return self.output_root.joinpath(*parts)

def load_config(path: Path | None = None) -> ProjectConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2]/'config'/'paths.local.json'
    if not path.is_file():
        raise FileNotFoundError(f'Local config not found: {path}\nRun configure_paths.cmd first.')
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    if raw.get('schema_version') != 1:
        raise ValueError('Unsupported config schema.')
    return ProjectConfig(path.resolve(), raw)
