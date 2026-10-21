from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class EncoderConfig:
    """Bidirectional pre-norm transformer encoder.

    The defaults describe a small from-scratch model (RMSNorm, SwiGLU, global attention).
    :func:`decider.model.pretrained.modernbert_encoder_config` produces the settings that make
    this same module numerically equivalent to ModernBERT.
    """

    vocab_size: int = 264
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 768
    norm: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    norm_bias: bool = False
    activation: Literal["silu", "gelu"] = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    embedding_norm: bool = False
    first_layer_attn_norm: bool = True
    # Layer i uses global attention when i % global_every == 0, otherwise a sliding window.
    global_every: int = 1
    local_window: int = 64
    rope_theta: float = 10_000.0
    local_rope_theta: float = 10_000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    max_levels: int = 10
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_global(self, layer: int) -> bool:
        return layer % self.global_every == 0


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    tokenizer: str = "byte"
    # Hugging Face id or local path of a ModernBERT checkpoint to initialise the encoder from.
    pretrained: str | None = None
    head_hidden: int | None = None
    head_dropout: float = 0.1
    # Options of the same question attend to each other only in the last N layers. Earlier
    # layers keep options isolated so every [OPT] token first builds an option-specific
    # representation: with shared positions and sibling attention from layer 0, all [OPT]
    # tokens would have identical queries and identical visible keys, hence identical scores.
    cross_option_layers: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.cross_option_layers <= self.encoder.n_layers:
            raise ValueError("cross_option_layers must be in [0, n_layers]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        data = dict(data)
        encoder = EncoderConfig(**data.pop("encoder", {}))
        return cls(encoder=encoder, **data)
