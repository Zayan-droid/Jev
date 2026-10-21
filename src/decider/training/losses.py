"""Proper scoring rules over typed questions.

For every question: log score (cross-entropy against the target distribution) plus a weighted
Brier score; ``score`` questions add the ranked probability score, which also rewards putting
mass on *nearby* levels. All three are strictly proper, so the minimiser is the true
conditional distribution: the objective itself asks for calibrated outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from decider.packing import TYPES


@dataclass(frozen=True)
class LossWeights:
    brier: float = 0.25
    rps: float = 0.5


DEFAULT_WEIGHTS = LossWeights()


def _masked_softmax_terms(
    z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp = torch.log_softmax(z, dim=-1)
    zero = torch.zeros_like(logp)
    ce = -torch.where(mask, t * logp, zero).sum(-1)
    p = torch.where(mask, logp.exp(), zero)
    brier = ((p - t) ** 2).sum(-1)
    return ce, brier, p


def decision_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    option_mask: dict[str, torch.Tensor],
    weights: LossWeights = DEFAULT_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean loss over all questions in the batch, plus per-type means for logging."""
    per_question: list[torch.Tensor] = []
    parts: dict[str, torch.Tensor] = {}
    for t in TYPES:
        z, y = logits[t].float(), targets[t].float()
        if z.numel() == 0:
            continue
        if t == "noul":
            ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
            brier = 2.0 * (torch.sigmoid(z) - y) ** 2
            loss = ce + weights.brier * brier
        else:
            mask = option_mask[t]
            ce, brier, p = _masked_softmax_terms(z, y, mask)
            loss = ce + weights.brier * brier
            if t == "score":
                k = mask.sum(-1).clamp(min=2).float()
                cdf_gap = (p.cumsum(-1) - y.cumsum(-1)) ** 2
                rps = torch.where(mask, cdf_gap, torch.zeros_like(cdf_gap)).sum(-1) / (k - 1)
                loss = loss + weights.rps * rps
                parts["rps"] = rps.mean().detach()
        per_question.append(loss)
        parts[f"loss_{t}"] = loss.mean().detach()
        parts[f"ce_{t}"] = ce.mean().detach()
    if not per_question:
        raise ValueError("batch contains no questions")
    total = torch.cat(per_question).mean()
    return total, parts
