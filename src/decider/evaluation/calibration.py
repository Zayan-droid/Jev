"""Post-hoc temperature scaling, one temperature per question type.

A single temperature per type is the deliberate choice: task families are unknown at inference
time (zero-shot), so per-family temperatures cannot be applied in production. Per-family ECE is
reported instead, to show where one temperature is not enough.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from decider.evaluation.metrics import QuestionResult
from decider.packing import TYPES

MIN_T, MAX_T = 0.05, 20.0


def _nll(results: Sequence[QuestionResult], log_t: torch.Tensor) -> torch.Tensor:
    t = log_t.exp()
    if results[0].type == "noul":
        z = torch.tensor([r.logits[0] for r in results], dtype=torch.float64)
        y = torch.tensor([r.target[0] for r in results], dtype=torch.float64)
        return F.binary_cross_entropy_with_logits(z / t, y)
    k = max(len(r.logits) for r in results)
    z = torch.zeros((len(results), k), dtype=torch.float64)
    y = torch.zeros((len(results), k), dtype=torch.float64)
    mask = torch.zeros((len(results), k), dtype=torch.bool)
    for i, r in enumerate(results):
        z[i, : len(r.logits)] = torch.from_numpy(r.logits.astype("float64"))
        y[i, : len(r.target)] = torch.from_numpy(r.target.astype("float64"))
        mask[i, : len(r.logits)] = True
    # Mask *after* scaling: padding as -inf before dividing by t would make d/dt infinite.
    scaled = torch.where(mask, z / t, torch.tensor(float("-inf"), dtype=torch.float64))
    logp = torch.log_softmax(scaled, dim=-1)
    return -(torch.where(mask, y * logp, torch.zeros_like(logp))).sum(-1).mean()


def fit_temperature(results: Sequence[QuestionResult]) -> float:
    """Temperature minimising the log loss of ``results`` (all of one question type)."""
    if len(results) < 20:
        return 1.0
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _nll(results, log_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(MIN_T, MAX_T))


def fit_temperatures(results: Sequence[QuestionResult]) -> dict[str, float]:
    return {t: fit_temperature([r for r in results if r.type == t]) for t in TYPES}
