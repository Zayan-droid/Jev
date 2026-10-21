"""Our encoder loaded with ModernBERT weights must reproduce the reference implementation."""

from __future__ import annotations

import pytest
import torch

from decider.masks import sliding_window
from decider.model.encoder import Encoder
from decider.model.pretrained import (
    load_modernbert_weights,
    modernbert_encoder_config,
    read_hf_config,
)
from decider.tokenization import HFTokenizer

CHECKPOINT = "answerdotai/ModernBERT-base"

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_hidden_states_match_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    tokenizer = HFTokenizer.from_pretrained(CHECKPOINT)
    text = " ".join(f"Sentence number {i} talks about refunds and parcels." for i in range(30))
    ids = [tokenizer.cls_id, *tokenizer.encode(text), tokenizer.sep_id]
    assert len(ids) > 200, "sequence must be longer than the local window"
    input_ids = torch.tensor([ids])
    positions = torch.arange(len(ids)).unsqueeze(0)

    reference = transformers.AutoModel.from_pretrained(CHECKPOINT, attn_implementation="sdpa")
    reference.eval()
    with torch.no_grad():
        expected = reference(input_ids=input_ids).last_hidden_state

    encoder = Encoder(modernbert_encoder_config(read_hf_config(CHECKPOINT)))
    load_modernbert_weights(encoder, CHECKPOINT, tokenizer)
    encoder.eval()
    full = torch.ones(1, 1, len(ids), len(ids), dtype=torch.bool)
    local = sliding_window(full, positions, encoder.cfg.local_window)
    masks = [full if layer.is_global else local for layer in encoder.layers]
    with torch.no_grad():
        actual, _ = encoder(input_ids, positions, torch.zeros_like(input_ids), masks)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-4)
