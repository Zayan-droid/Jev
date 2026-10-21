"""Self-contained model bundles: weights, config, tokenizer and calibration in one directory.

A bundle is what the trainer exports, what the registry stores and what the server loads::

    bundle/
      config.json         model + packing config, temperatures, provenance
      model.safetensors   weights (including the fitted temperatures)
      tokenizer/          tokenizer files
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from decider.model.config import ModelConfig
from decider.model.decider import DeciderModel
from decider.model.factory import build_model
from decider.packing import PackingLimits
from decider.tokenization import Tokenizer, load_tokenizer

BUNDLE_FORMAT = 1


@dataclasses.dataclass
class Bundle:
    model: DeciderModel
    tokenizer: Tokenizer
    limits: PackingLimits
    metadata: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unversioned"))


def save_bundle(
    directory: Path,
    model: DeciderModel,
    tokenizer: Tokenizer,
    limits: PackingLimits,
    metadata: dict[str, Any] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    tokenizer.save(directory / "tokenizer")
    config = {
        "format": BUNDLE_FORMAT,
        "model": model.cfg.to_dict(),
        "packing": dataclasses.asdict(limits),
        "temperatures": model.temperatures(),
        "metadata": metadata or {},
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return directory


def load_bundle(directory: Path | str, device: str | torch.device = "cpu") -> Bundle:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {config.get('format')!r}")
    tokenizer = load_tokenizer(directory / "tokenizer")
    model_cfg = ModelConfig.from_dict(config["model"])
    model, _ = build_model(model_cfg, tokenizer=tokenizer, load_weights=False)
    model.load_state_dict(load_file(str(directory / "model.safetensors")))
    model.to(device).eval()
    return Bundle(model, tokenizer, PackingLimits(**config["packing"]), config["metadata"])
