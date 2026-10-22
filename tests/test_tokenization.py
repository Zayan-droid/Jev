from pathlib import Path

import pytest
from tokenizers import Tokenizer as Backend
from tokenizers import models, pre_tokenizers

from decider.tokenization import (
    DECISION_TOKENS,
    ByteTokenizer,
    HFTokenizer,
    build_tokenizer,
    load_tokenizer,
)


def tiny_backend() -> Backend:
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "the": 4, "card": 5, "arrived": 6}
    backend = Backend(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return backend


def test_hf_tokenizer_adds_decision_tokens_once() -> None:
    tok = HFTokenizer(tiny_backend())
    ids = {t: tok.token_id(t) for t in DECISION_TOKENS}
    assert len(set(ids.values())) == len(DECISION_TOKENS)
    assert tok.vocab_size == 7 + len(DECISION_TOKENS)
    assert (tok.cls_id, tok.sep_id, tok.pad_id) == (1, 2, 0)
    assert tok.encode("the card arrived") == [4, 5, 6]
    # Decision tokens are atomic even inside text.
    assert tok.encode("[OPT] card") == [tok.option_id, 5]


def test_hf_tokenizer_roundtrip(tmp_path: Path) -> None:
    tok = HFTokenizer(tiny_backend())
    tok.save(tmp_path / "tok")
    loaded = load_tokenizer(tmp_path / "tok")
    assert isinstance(loaded, HFTokenizer)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.level_id == tok.level_id
    assert loaded.encode_batch(["the card", "arrived"]) == [[4, 5], [6]]
    assert isinstance(HFTokenizer.from_pretrained(str(tmp_path / "tok")), HFTokenizer)


def test_byte_tokenizer(tmp_path: Path) -> None:
    tok = build_tokenizer("byte")
    assert isinstance(tok, ByteTokenizer)
    assert tok.encode("é") == [0xC3, 0xA9]
    assert tok.vocab_size == 256 + 3 + len(DECISION_TOKENS)
    assert tok.question_token_id("score") == tok.token_id("[Q_SCORE]")
    tok.save(tmp_path / "byte")
    assert isinstance(load_tokenizer(tmp_path / "byte"), ByteTokenizer)


def test_unknown_token_is_an_error() -> None:
    with pytest.raises(KeyError):
        HFTokenizer(tiny_backend()).token_id("[NOPE]")
