"""Loading ModernBERT checkpoints into :class:`decider.model.encoder.Encoder`.

The encoder re-implements ModernBERT (pre-norm LayerNorm without bias, GeGLU, RoPE with separate
global/local bases, alternating global and sliding-window attention) so it can take the packed
segment masks, tree position ids and a state KV cache. ``tests/test_pretrained_parity.py``
checks that the hidden states match the reference ``transformers`` implementation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from decider.model.config import EncoderConfig
from decider.model.encoder import Encoder
from decider.tokenization import SPECIAL_TOKEN_SEEDS, Tokenizer

log = logging.getLogger(__name__)

_LAYER_KEYS = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.Wqkv.weight": "attn.qkv.weight",
    "attn.Wo.weight": "attn.out.weight",
    "mlp_norm.weight": "mlp_norm.weight",
    "mlp.Wi.weight": "mlp.up.weight",
    "mlp.Wo.weight": "mlp.down.weight",
}


def resolve_checkpoint(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name_or_path, allow_patterns=["*.json", "*.safetensors"]))


def modernbert_encoder_config(hf_config: dict[str, Any], **overrides: Any) -> EncoderConfig:
    cfg = {
        "vocab_size": hf_config["vocab_size"],
        "d_model": hf_config["hidden_size"],
        "n_layers": hf_config["num_hidden_layers"],
        "n_heads": hf_config["num_attention_heads"],
        "d_ff": hf_config["intermediate_size"],
        "norm": "layernorm",
        "norm_eps": hf_config.get("norm_eps", 1e-5),
        "norm_bias": hf_config.get("norm_bias", False),
        "activation": "gelu" if hf_config.get("hidden_activation", "gelu") == "gelu" else "silu",
        "attention_bias": hf_config.get("attention_bias", False),
        "mlp_bias": hf_config.get("mlp_bias", False),
        "embedding_norm": True,
        "first_layer_attn_norm": False,
        "global_every": hf_config.get("global_attn_every_n_layers", 3),
        "local_window": hf_config.get("local_attention", 128) // 2,
        "rope_theta": hf_config.get("global_rope_theta", 160_000.0),
        "local_rope_theta": hf_config.get("local_rope_theta", 10_000.0),
    }
    cfg.update(overrides)
    return EncoderConfig(**cfg)


def read_hf_config(name_or_path: str) -> dict[str, Any]:
    config: dict[str, Any] = json.loads(
        (resolve_checkpoint(name_or_path) / "config.json").read_text()
    )
    if config.get("model_type") != "modernbert":
        raise ValueError(f"expected a ModernBERT checkpoint, got {config.get('model_type')!r}")
    return config


def load_modernbert_weights(encoder: Encoder, name_or_path: str, tokenizer: Tokenizer) -> None:
    """Copy ModernBERT weights into ``encoder`` and initialise the decision tokens.

    New special tokens start from the mean embedding of a short descriptive phrase
    (``SPECIAL_TOKEN_SEEDS``), which is a better starting point than random rows.
    """
    from safetensors.torch import load_file

    directory = resolve_checkpoint(name_or_path)
    files = sorted(directory.glob("*.safetensors"))
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(str(f)))
    prefix = "model." if any(k.startswith("model.") for k in state) else ""

    mapped: dict[str, torch.Tensor] = {
        "embedding_norm.weight": state[f"{prefix}embeddings.norm.weight"],
        "final_norm.weight": state[f"{prefix}final_norm.weight"],
    }
    for i in range(encoder.cfg.n_layers):
        for src, dst in _LAYER_KEYS.items():
            key = f"{prefix}layers.{i}.{src}"
            if key in state:
                mapped[f"layers.{i}.{dst}"] = state[key]

    pretrained_emb = state[f"{prefix}embeddings.tok_embeddings.weight"]
    vocab = max(tokenizer.vocab_size, pretrained_emb.shape[0])
    encoder.resize_token_embeddings(vocab)
    with torch.no_grad():
        encoder.tok_embeddings.weight[: pretrained_emb.shape[0]] = pretrained_emb
        for token, seed_text in SPECIAL_TOKEN_SEEDS.items():
            seed_ids = tokenizer.encode(seed_text)
            if seed_ids:
                encoder.tok_embeddings.weight[tokenizer.token_id(token)] = pretrained_emb[
                    seed_ids
                ].mean(0)

    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    allowed_missing = {"tok_embeddings.weight", "level_embeddings.weight"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    log.info("loaded %d tensors from %s", len(mapped), directory)
