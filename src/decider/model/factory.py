from __future__ import annotations

import dataclasses
import logging

from decider.model.config import ModelConfig
from decider.model.decider import DeciderModel
from decider.tokenization import Tokenizer, build_tokenizer

log = logging.getLogger(__name__)

# Fields of EncoderConfig that are training choices rather than architecture; these are kept
# from the experiment config when the architecture comes from a pretrained checkpoint.
_TRAINING_FIELDS = ("dropout", "attention_dropout", "gradient_checkpointing", "max_levels")


def build_model(
    cfg: ModelConfig, tokenizer: Tokenizer | None = None, load_weights: bool = True
) -> tuple[DeciderModel, Tokenizer]:
    """Build the model and tokenizer described by ``cfg``.

    With ``cfg.pretrained`` set, the encoder architecture is read from the checkpoint and
    ``cfg.encoder`` is replaced by the resolved settings, so the saved config is
    self-contained. ``load_weights=False`` skips the download (used when restoring a bundle).
    """
    tokenizer = tokenizer or build_tokenizer(cfg.tokenizer)
    if cfg.pretrained and load_weights:
        from decider.model.pretrained import (
            load_modernbert_weights,
            modernbert_encoder_config,
            read_hf_config,
        )

        keep = {f: getattr(cfg.encoder, f) for f in _TRAINING_FIELDS}
        cfg.encoder = modernbert_encoder_config(read_hf_config(cfg.pretrained), **keep)
        model = DeciderModel(cfg)
        load_modernbert_weights(model.encoder, cfg.pretrained, tokenizer)
    else:
        if cfg.encoder.vocab_size < tokenizer.vocab_size:
            cfg.encoder = dataclasses.replace(cfg.encoder, vocab_size=tokenizer.vocab_size)
        model = DeciderModel(cfg)
    log.info(
        "model: %.1fM parameters (%d layers, d_model=%d)",
        model.num_parameters() / 1e6,
        cfg.encoder.n_layers,
        cfg.encoder.d_model,
    )
    return model, tokenizer
