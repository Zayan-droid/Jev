from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from decider.model.config import EncoderConfig

KV = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


def make_norm(cfg: EncoderConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model, cfg.norm_eps)
    return nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps, bias=cfg.norm_bias)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding evaluated at arbitrary (tree-structured) position ids."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)  # (B, 1, L, head_dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    return (x32 * cos + _rotate_half(x32) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.attention_dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.attention_bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        prefix: KV | None = None,
    ) -> tuple[torch.Tensor, KV]:
        """``prefix`` holds cached keys/values (already rotated) prepended to the key axis.

        Returns the output and this call's own keys/values so a caller can cache them.
        """
        b, length, _ = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cos, sin = rope
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        new_kv = (k, v)
        if prefix is not None:
            k = torch.cat((prefix[0].to(k.dtype), k), dim=2)
            v = torch.cat((prefix[1].to(v.dtype), v), dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, length, -1)
        return self.out(out), new_kv


class GatedMLP(nn.Module):
    """GLU feed-forward: SwiGLU with ``silu``, GeGLU with ``gelu``."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=cfg.mlp_bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.mlp_bias)
        self.act = F.silu if cfg.activation == "silu" else F.gelu
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(self.drop(self.act(value) * gate))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, index: int) -> None:
        super().__init__()
        self.is_global = cfg.is_global(index)
        use_attn_norm = index > 0 or cfg.first_layer_attn_norm
        self.attn_norm = make_norm(cfg) if use_attn_norm else nn.Identity()
        self.attn = Attention(cfg)
        self.mlp_norm = make_norm(cfg)
        self.mlp = GatedMLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        prefix: KV | None = None,
    ) -> tuple[torch.Tensor, KV]:
        attn_out, kv = self.attn(self.attn_norm(x), rope, mask, prefix)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x, kv
