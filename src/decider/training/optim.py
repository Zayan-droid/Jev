from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def parameter_groups(
    model: nn.Module, lr: float, weight_decay: float, head_lr_multiplier: float
) -> list[dict]:
    """AdamW groups: no decay on norms, biases and embeddings; a larger LR for the freshly
    initialised heads when the encoder is pretrained."""
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith("heads.")
        no_decay = param.ndim < 2 or "embeddings" in name or "norm" in name
        groups.setdefault((is_head, no_decay), []).append(param)
    out = []
    for (is_head, no_decay), params in sorted(groups.items()):
        out.append(
            {
                "params": params,
                "lr": lr * (head_lr_multiplier if is_head else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{'heads' if is_head else 'encoder'}{'_no_decay' if no_decay else ''}",
            }
        )
    return out


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    head_lr_multiplier: float = 1.0,
    betas: tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    groups = parameter_groups(model, lr, weight_decay, head_lr_multiplier)
    fused = all(p.is_cuda for g in groups for p in g["params"])
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)


def warmup_cosine(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1
) -> LambdaLR:
    """Linear warmup, then cosine decay to ``min_ratio`` of the peak learning rate."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)
