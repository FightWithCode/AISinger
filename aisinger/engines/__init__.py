"""Voice-conversion engines.

Each engine is cloned at a pinned version into ``AISINGER_CACHE/engines/<name>``
with its *own* virtualenv (created with ``uv``), so their conflicting
dependencies never clash with each other or with Colab's preinstalled packages.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import config
from ..utils import AISingerError, log, run


def uv_cmd() -> list[str]:
    exe = shutil.which("uv")
    return [exe] if exe else [sys.executable, "-m", "uv"]


def git_checkout(repo: str, ref: str, dest: Path) -> None:
    marker = dest / ".aisinger-ref"
    if marker.exists() and marker.read_text().strip() == ref:
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Cloning {repo} @ {ref[:12]} ...")
    run(["git", "init", "-q", str(dest)])
    run(["git", "-C", str(dest), "remote", "add", "origin", repo])
    run(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", ref])
    run(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"])
    marker.write_text(ref)


def make_venv(dest: Path, python: str) -> Path:
    venv = dest / ".venv"
    py = venv / "bin" / "python"
    if not py.exists():
        log(f"Creating Python {python} environment for {dest.name} ...")
        run([*uv_cmd(), "venv", "--python", python, str(venv)])
    return py


def uv_pip(py: Path, args: list[str]) -> None:
    run([*uv_cmd(), "pip", "install", "--python", str(py), *args])


def torch_index(gpu: bool, cuda: str) -> str:
    return f"https://download.pytorch.org/whl/{cuda if gpu else 'cpu'}"


def default_gpu() -> bool:
    if os.environ.get("AISINGER_FORCE_CPU"):
        return False
    return config.has_nvidia_gpu()
