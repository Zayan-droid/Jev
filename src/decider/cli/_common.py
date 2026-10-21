"""Shared CLI setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from decider.utils.provenance import REPO_ROOT


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal ``.env`` support (KEY=VALUE lines); real environment variables win.

    Runs at import time: cache locations such as ``HF_HOME`` must be set before the Hugging
    Face libraries are first imported.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()

CONFIG_DIR = str(Path(os.environ.get("DECIDER_CONFIG_DIR", REPO_ROOT / "configs")).resolve())

_QUIET_DEFAULTS = {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",  # Windows without developer mode
    "MLFLOW_DISABLE_AGENT_HINT": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def prepare_environment() -> None:
    for key, value in _QUIET_DEFAULTS.items():
        os.environ.setdefault(key, value)
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "alembic", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_logging(level: int = logging.INFO) -> None:
    """Rich console logging for the non-Hydra entry points."""
    from rich.logging import RichHandler

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    prepare_environment()
