"""ModernBERT loading without downloads: a tiny random checkpoint saved by ``transformers``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from decider.masks import sliding_window
from decider.model.encoder import Encoder
from decider.model.pretrained import (
    load_modernbert_weights,
    modernbert_encoder_config,
    read_hf_config,
)
from decider.tokenization import ByteTokenizer

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    config = transformers.ModernBertConfig(
        vocab_size=300,  # covers byte ids, so the byte tokenizer can seed decision tokens
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        local_attention=8,
        global_attn_every_n_layers=3,
        global_rope_theta=160_000.0,
        local_rope_theta=10_000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    path = tmp_path_factory.mktemp("tiny-modernbert")
    transformers.ModernBertModel(config).save_pretrained(path)
    return path


def test_layout_is_read_from_the_saved_config(checkpoint: Path) -> None:
    cfg = modernbert_encoder_config(read_hf_config(str(checkpoint)))
    assert [cfg.is_global(i) for i in range(cfg.n_layers)] == [True, False, False, True]
    assert cfg.local_window == 4
    assert (cfg.rope_theta, cfg.local_rope_theta) == (160_000.0, 10_000.0)


def test_both_config_formats_give_the_same_layout() -> None:
    old = {
        "vocab_size": 10,
        "hidden_size": 8,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "intermediate_size": 8,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
        "global_rope_theta": 160_000.0,
        "local_rope_theta": 10_000.0,
    }
    new = {
        k: v for k, v in old.items() if "rope_theta" not in k and k != "global_attn_every_n_layers"
    }
    new["layer_types"] = ["full_attention", "sliding_attention", "sliding_attention"] * 2
    new["rope_parameters"] = {
        "full_attention": {"rope_theta": 160_000.0},
        "sliding_attention": {"rope_theta": 10_000.0},
    }
    assert modernbert_encoder_config(old) == modernbert_encoder_config(new)
    new["layer_types"] = [
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    with pytest.raises(ValueError, match="layer layout"):
        modernbert_encoder_config(new)


def test_hidden_states_match_transformers(checkpoint: Path) -> None:
    reference = transformers.ModernBertModel.from_pretrained(checkpoint, attn_implementation="sdpa")
    reference.eval()
    encoder = Encoder(modernbert_encoder_config(read_hf_config(str(checkpoint))))
    load_modernbert_weights(encoder, str(checkpoint), ByteTokenizer())
    encoder.eval()

    input_ids = torch.randint(3, 256, (1, 40))  # longer than the 2*4+1 local window
    positions = torch.arange(40).unsqueeze(0)
    full = torch.ones(1, 1, 40, 40, dtype=torch.bool)
    local = sliding_window(full, positions, encoder.cfg.local_window)
    masks = [full if encoder.cfg.is_global(i) else local for i in range(encoder.cfg.n_layers)]
    with torch.no_grad():
        expected = reference(input_ids=input_ids).last_hidden_state
        actual, _ = encoder(input_ids, positions, torch.zeros_like(input_ids), masks)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_decision_tokens_are_seeded_from_descriptions(checkpoint: Path) -> None:
    tokenizer = ByteTokenizer()
    encoder = Encoder(modernbert_encoder_config(read_hf_config(str(checkpoint))))
    load_modernbert_weights(encoder, str(checkpoint), tokenizer)
    emb = encoder.tok_embeddings.weight
    assert emb.shape[0] == max(tokenizer.vocab_size, 300)
    seed = emb[tokenizer.encode("option")].mean(0)
    torch.testing.assert_close(emb[tokenizer.option_id], seed)
