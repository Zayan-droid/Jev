from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from decider.model.config import EncoderConfig
from decider.model.layers import KV, EncoderLayer, RotaryEmbedding, make_norm


class Encoder(nn.Module):
    """Token + level embeddings, a stack of encoder layers and a final norm.

    ``masks`` holds one boolean attention mask per layer (see :mod:`decider.masks`); layers
    usually share a few distinct tensors. ``prefix`` supplies cached per-layer keys/values for
    tokens that precede ``input_ids`` (the encoded state).
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Index 0 = "not a score level". Zero-initialised so a pretrained backbone computes
        # exactly its original function at step 0.
        self.level_embeddings = nn.Embedding(cfg.max_levels + 1, cfg.d_model)
        self.embedding_norm = make_norm(cfg) if cfg.embedding_norm else nn.Identity()
        self.embedding_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = make_norm(cfg)
        self.rope_global = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.rope_local = RotaryEmbedding(cfg.head_dim, cfg.local_rope_theta)
        self.reset_parameters()

    @property
    def has_local_layers(self) -> bool:
        return not all(self.cfg.is_global(i) for i in range(self.cfg.n_layers))

    def reset_parameters(self) -> None:
        std = 0.02
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_residual = name.endswith(("attn.out", "mlp.down"))
                nn.init.normal_(module.weight, std=residual_std if is_residual else std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)
        nn.init.zeros_(self.level_embeddings.weight)

    def resize_token_embeddings(self, vocab_size: int) -> None:
        old = self.tok_embeddings
        if vocab_size == old.num_embeddings:
            return
        new = nn.Embedding(vocab_size, self.cfg.d_model).to(old.weight.device, old.weight.dtype)
        n = min(vocab_size, old.num_embeddings)
        with torch.no_grad():
            new.weight.normal_(std=0.02)
            new.weight[:n] = old.weight[:n]
        self.tok_embeddings = new
        self.cfg.vocab_size = vocab_size

    def embed(self, input_ids: torch.Tensor, level_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(input_ids) + self.level_embeddings(level_ids)
        return self.embedding_drop(self.embedding_norm(x))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        level_ids: torch.Tensor,
        masks: Sequence[torch.Tensor],
        prefix: list[KV] | None = None,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, list[KV]]:
        if len(masks) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} masks, got {len(masks)}")
        x = self.embed(input_ids, level_ids)
        rope_g = self.rope_global(position_ids)
        rope_l = self.rope_local(position_ids) if self.has_local_layers else rope_g
        kvs: list[KV] = []
        for i, (layer, mask) in enumerate(zip(self.layers, masks, strict=True)):
            rope = rope_g if self.cfg.is_global(i) else rope_l
            layer_prefix = prefix[i] if prefix is not None else None
            if self.cfg.gradient_checkpointing and self.training and layer_prefix is None:
                x, kv = checkpoint(layer, x, rope, mask, use_reentrant=False)
            else:
                x, kv = layer(x, rope, mask, layer_prefix)
            if return_kv:
                kvs.append(kv)
        return self.final_norm(x), kvs
