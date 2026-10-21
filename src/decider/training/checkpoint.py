"""Resumable training checkpoints (model, optimizer, scheduler, data stream and RNG state)."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: Path, keep: int = 2) -> None:
        self.directory = directory
        self.keep = keep
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def last(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best(self) -> Path:
        return self.directory / "best.pt"

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)  # atomic: a crash mid-write never corrupts the previous checkpoint

    def save(self, payload: dict[str, Any], step: int) -> Path:
        path = self.directory / f"step_{step:08d}.pt"
        self._write(path, payload)
        self._write(self.last, payload)
        for old in sorted(self.directory.glob("step_*.pt"))[: -self.keep]:
            old.unlink()
        log.info("checkpoint saved at step %d", step)
        return path

    def save_best(self, model_state: dict[str, torch.Tensor], step: int, metric: float) -> None:
        self._write(self.best, {"model": model_state, "step": step, "metric": metric})

    def load(self, path: Path | None = None, map_location: str | torch.device = "cpu") -> dict:
        target = path or self.last
        payload: dict = torch.load(target, map_location=map_location, weights_only=False)
        return payload
