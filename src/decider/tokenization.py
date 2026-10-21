"""Tokenizers with the decision-specific special tokens.

Two implementations share one interface:

* :class:`ByteTokenizer` - UTF-8 bytes plus special tokens. No download, fully deterministic;
  used for from-scratch models, unit tests and CI.
* :class:`HFTokenizer` - a Hugging Face ``tokenizers`` tokenizer (e.g. ModernBERT's) with the
  special tokens appended to its vocabulary.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

QUESTION_TOKENS = {"noul": "[Q_NOUL]", "choice": "[Q_CHOICE]", "score": "[Q_SCORE]"}
OPTION_TOKEN = "[OPT]"
LEVEL_TOKEN = "[LVL]"
DECISION_TOKENS = (*QUESTION_TOKENS.values(), OPTION_TOKEN, LEVEL_TOKEN)

# Text used to initialise the embeddings of the new tokens in a pretrained vocabulary.
SPECIAL_TOKEN_SEEDS = {
    "[Q_NOUL]": "yes or no question",
    "[Q_CHOICE]": "multiple choice question",
    "[Q_SCORE]": "rating scale question",
    "[OPT]": "option",
    "[LVL]": "level",
}


class Tokenizer(ABC):
    kind: str
    cls_id: int
    sep_id: int
    pad_id: int

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def token_id(self, token: str) -> int: ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    def encode(self, text: str) -> list[int]:
        return self.encode_batch([text])[0]

    def question_token_id(self, question_type: str) -> int:
        return self.token_id(QUESTION_TOKENS[question_type])

    @property
    def option_id(self) -> int:
        return self.token_id(OPTION_TOKEN)

    @property
    def level_id(self) -> int:
        return self.token_id(LEVEL_TOKEN)


class ByteTokenizer(Tokenizer):
    kind = "byte"
    _specials = ("[PAD]", "[CLS]", "[SEP]", *DECISION_TOKENS)

    def __init__(self) -> None:
        self._special_ids = {tok: 256 + i for i, tok in enumerate(self._specials)}
        self.pad_id = self._special_ids["[PAD]"]
        self.cls_id = self._special_ids["[CLS]"]
        self.sep_id = self._special_ids["[SEP]"]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self._specials)

    def token_id(self, token: str) -> int:
        return self._special_ids[token]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(t.encode("utf-8")) for t in texts]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer_config.json").write_text(json.dumps({"kind": self.kind}))


class HFTokenizer(Tokenizer):
    kind = "hf"

    def __init__(
        self,
        backend: object,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
    ) -> None:
        from tokenizers import AddedToken
        from tokenizers import Tokenizer as Backend

        assert isinstance(backend, Backend)
        backend.no_padding()
        backend.no_truncation()
        missing = [t for t in DECISION_TOKENS if backend.token_to_id(t) is None]
        if missing:
            backend.add_special_tokens(
                [AddedToken(t, special=True, normalized=False) for t in missing]
            )
        self._backend = backend
        self._tokens = {"cls": cls_token, "sep": sep_token, "pad": pad_token}
        self.cls_id = self.token_id(cls_token)
        self.sep_id = self.token_id(sep_token)
        self.pad_id = self.token_id(pad_token)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> HFTokenizer:
        from tokenizers import Tokenizer as Backend

        path = Path(name_or_path)
        if path.is_dir():
            return cls.load(path)
        return cls(Backend.from_pretrained(name_or_path))

    @property
    def vocab_size(self) -> int:
        return int(self._backend.get_vocab_size(with_added_tokens=True))

    def token_id(self, token: str) -> int:
        idx = self._backend.token_to_id(token)
        if idx is None:
            raise KeyError(f"token {token!r} is not in the vocabulary")
        return int(idx)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        encodings = self._backend.encode_batch(texts, add_special_tokens=False)
        return [e.ids for e in encodings]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(directory / "tokenizer.json"))
        config = {"kind": self.kind, **self._tokens}
        (directory / "tokenizer_config.json").write_text(json.dumps(config))

    @classmethod
    def load(cls, directory: Path) -> HFTokenizer:
        from tokenizers import Tokenizer as Backend

        config = json.loads((directory / "tokenizer_config.json").read_text())
        backend = Backend.from_file(str(directory / "tokenizer.json"))
        return cls(backend, config["cls"], config["sep"], config["pad"])


def build_tokenizer(name: str) -> Tokenizer:
    """``"byte"`` or a Hugging Face model id / local directory."""
    return ByteTokenizer() if name == "byte" else HFTokenizer.from_pretrained(name)


def load_tokenizer(directory: Path) -> Tokenizer:
    config = json.loads((directory / "tokenizer_config.json").read_text())
    return ByteTokenizer() if config["kind"] == "byte" else HFTokenizer.load(directory)
