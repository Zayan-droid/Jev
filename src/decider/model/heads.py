from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from decider.packing import TYPES


class ReadoutHead(nn.Module):
    """Maps the hidden state at a readout token to one logit."""

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(F.gelu(self.proj(h)))).squeeze(-1)


class DecisionHeads(nn.Module):
    """One shared head per question type.

    * noul: a logit at the ``[Q_NOUL]`` token, ``p = sigmoid(z)``.
    * choice / score: a logit at every ``[OPT]`` / ``[LVL]`` token, softmax within the question.

    The last layer starts at zero, so every distribution starts uniform (noul at 0.5): a
    well-defined, maximally uncertain starting point for a proper scoring rule.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({t: ReadoutHead(d_model, hidden, dropout) for t in TYPES})

    def forward(
        self,
        hidden: torch.Tensor,
        index: dict[str, torch.Tensor],
        option_mask: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``hidden`` is flattened to ``(B*L, D)``; returns float32 logits per type.

        Padded option slots get ``-inf`` so they receive exactly zero probability.
        """
        logits: dict[str, torch.Tensor] = {}
        for t in TYPES:
            idx = index[t]
            if idx.numel() == 0:
                logits[t] = hidden.new_zeros(idx.shape, dtype=torch.float32)
                continue
            z = self.heads[t](hidden[idx.reshape(-1)]).float().view(idx.shape)
            if t != "noul":
                z = z.masked_fill(~option_mask[t], float("-inf"))
            logits[t] = z
        return logits
