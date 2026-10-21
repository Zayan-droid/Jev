"""Training health monitors: non-finite losses, loss spikes and update/weight ratios."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


class NonFiniteLossError(RuntimeError):
    pass


@dataclass
class SpikeDetector:
    """Flags steps whose loss is ``threshold`` standard deviations above an EMA baseline."""

    beta: float = 0.98
    threshold: float = 4.0
    warmup: int = 50
    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    spikes: int = 0

    def update(self, loss: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.mean = loss
            return False
        std = math.sqrt(max(self.var, 1e-12))
        spike = self.n > self.warmup and loss > self.mean + self.threshold * std
        if spike:
            self.spikes += 1
        else:  # spikes do not move the baseline
            delta = loss - self.mean
            self.mean += (1 - self.beta) * delta
            self.var = self.beta * (self.var + (1 - self.beta) * delta * delta)
        return spike


def check_finite(loss: torch.Tensor, step: int) -> None:
    if not torch.isfinite(loss):
        raise NonFiniteLossError(f"non-finite loss {loss.item()} at step {step}")


@dataclass
class UpdateRatioTracker:
    """||Δw|| / ||w|| for a few representative tensors.

    A healthy ratio is around 1e-3; much larger means the learning rate is too high for that
    layer, much smaller means it is barely training. Only a handful of tensors are tracked
    so the snapshot costs almost no memory even for a large model.
    """

    names: list[str] = field(default_factory=list)
    _before: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def for_model(cls, model: nn.Module) -> UpdateRatioTracker:
        params = [n for n, p in model.named_parameters() if p.ndim == 2 and p.requires_grad]
        enc = [n for n in params if n.startswith("encoder.layers.") and "attn.qkv" in n]
        picks = [enc[0], enc[len(enc) // 2], enc[-1]] if enc else []
        picks += [n for n in params if n.startswith("heads.") and n.endswith("proj.weight")]
        picks.append("encoder.tok_embeddings.weight")
        return cls(names=sorted(set(picks) & set(params)))

    def snapshot(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        self._before = {n: params[n].detach().clone() for n in self.names}

    def ratios(self, model: nn.Module) -> dict[str, float]:
        if not self._before:
            return {}
        params = dict(model.named_parameters())
        out = {}
        for n, before in self._before.items():
            after = params[n].detach()
            ratio = (after - before).float().norm() / before.float().norm().clamp(min=1e-12)
            out[n.replace("encoder.", "").replace(".weight", "")] = float(ratio)
        self._before = {}
        return out
