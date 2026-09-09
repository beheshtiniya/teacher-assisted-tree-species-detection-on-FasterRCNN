
from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import Iterable, Mapping

def run(command: Iterable[object], *, cwd: Path | None = None, env: Mapping[str,str] | None = None, dry_run: bool=False) -> None:
    cmd=[str(x) for x in command]
    print('\nRUN:', subprocess.list2cmdline(cmd))
    if cwd: print('CWD:', cwd)
    if dry_run: return
    merged=os.environ.copy()
    if env: merged.update({str(k):str(v) for k,v in env.items()})
    result=subprocess.run(cmd, cwd=cwd, env=merged, check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit code {result.returncode}')
