"""Run provenance: code version, environment and hardware."""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@cache
def git_info() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": bool(status) if status is not None else None,
    }


def environment_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 2**30, 1),
            cuda=torch.version.cuda,
        )
    return info
