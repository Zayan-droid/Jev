"""Tokenized datasets, multi-source mixture sampling and token-budget batching."""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import pickle
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decider.data.io import iter_records, sha256_file, split_path
from decider.packing import (
    Batch,
    Example,
    PackingLimits,
    RequestEncoder,
    TokenizedQuestion,
    TokenizedRequest,
    augment,
    collate,
)
from decider.schema import QuestionType, Record, target_vector

log = logging.getLogger(__name__)


def record_to_example_targets(record: Record) -> dict[str, list[float]]:
    targets: dict[str, list[float]] = {}
    for name, question in record.questions.items():
        target = record.targets[name]
        if question.type == "noul":
            assert isinstance(target, int | float)
            targets[name] = [float(target)]
        else:
            order = list(question.criteria) if question.type == "choice" else []
            targets[name] = target_vector(question, target, order)
    return targets


def record_meta(record: Record) -> dict[str, Any]:
    meta: dict[str, Any] = {"id": record.id, "source": record.source, "family": record.family}
    for key in ("injected", "negation_pair", "templates"):
        if key in record.meta:
            meta[key] = record.meta[key]
    return meta


def encode_records(
    records: Sequence[Record], encoder: RequestEncoder, chunk: int = 4096
) -> list[Example]:
    examples: list[Example] = []
    for start in range(0, len(records), chunk):
        part = records[start : start + chunk]
        tokenized = encoder.encode_many(part)
        examples.extend(
            Example(t, record_to_example_targets(r), record_meta(r))
            for t, r in zip(tokenized, part, strict=True)
        )
    return examples


@dataclass(frozen=True, slots=True)
class _Packed:
    """One example in compact form: every token id in one int32 array plus the layout.

    Python lists of ints cost ~28 bytes per token; mix-v0's training split took ~3 GB that way
    and ~10x less like this. Examples are rebuilt on access, which costs microseconds.
    """

    tokens: np.ndarray  # state, then each question's header followed by its options
    state_len: int
    # (name, type, header length, option lengths, option names) per question
    questions: tuple[tuple[str, QuestionType, int, tuple[int, ...], tuple[str, ...]], ...]
    targets: dict[str, list[float]] | None
    meta: dict[str, Any]


def _pack(example: Example) -> _Packed:
    request = example.request
    parts: list[Sequence[int]] = [request.state]
    layout = []
    for q in request.questions:
        parts.append(q.header)
        parts.extend(q.options)
        option_lengths = tuple(len(o) for o in q.options)
        layout.append((q.name, q.type, len(q.header), option_lengths, tuple(q.option_names)))
    tokens = np.fromiter(itertools.chain.from_iterable(parts), dtype=np.int32)
    return _Packed(tokens, len(request.state), tuple(layout), example.targets, example.meta)


def _unpack(packed: _Packed) -> Example:
    flat = packed.tokens.tolist()
    pos = packed.state_len
    questions = []
    for name, qtype, header_len, option_lengths, option_names in packed.questions:
        header = flat[pos : pos + header_len]
        pos += header_len
        options = []
        for n in option_lengths:
            options.append(flat[pos : pos + n])
            pos += n
        questions.append(TokenizedQuestion(name, qtype, header, options, list(option_names)))
    request = TokenizedRequest(flat[: packed.state_len], questions)
    return Example(request, packed.targets, packed.meta)


class EncodedSplit:
    """One split of a built dataset, tokenized once, stored compactly and cached on disk."""

    CACHE_FORMAT = "packed-v1"

    def __init__(self, examples: list[Example], name: str) -> None:
        self._init([_pack(ex) for ex in examples], name)

    def _init(self, packed: list[_Packed], name: str) -> None:
        self._packed = packed
        self.name = name
        self.lengths = np.array([len(p.tokens) for p in packed], dtype=np.int64)
        self.sources = [p.meta.get("source", "unknown") for p in packed]

    @classmethod
    def _from_packed(cls, packed: list[_Packed], name: str) -> EncodedSplit:
        split = cls.__new__(cls)
        split._init(packed, name)
        return split

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, i: int) -> Example:
        return _unpack(self._packed[i])

    @property
    def examples(self) -> list[Example]:
        return [_unpack(p) for p in self._packed]

    @classmethod
    def load(
        cls,
        data_dir: Path,
        split: str,
        encoder: RequestEncoder,
        limit: int | None = None,
        use_cache: bool = True,
    ) -> EncodedSplit:
        path = split_path(data_dir, split)
        if not path.exists():
            return cls([], split)
        tok = encoder.tokenizer
        key = (sha256_file(path), tok.kind, tok.vocab_size, encoder.limits, limit, cls.CACHE_FORMAT)
        fingerprint = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        # Outside the dataset directory, so DVC-tracked outputs are never modified.
        cache_root = Path(os.environ.get("DECIDER_CACHE_DIR", ".cache"))
        cache = cache_root / "tokenized" / data_dir.name / f"{split}-{fingerprint}.pkl"
        if use_cache and cache.exists():
            with cache.open("rb") as f:
                packed = pickle.load(f)
            log.info("%s: %d examples (cached)", split, len(packed))
            return cls._from_packed(packed, split)
        records = []
        for record in iter_records(path):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        result = cls(encode_records(records, encoder), split)
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as f:
                pickle.dump(result._packed, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("%s: %d examples", split, len(result))
        return result


class MixtureSampler:
    """Infinite stream of example indices drawn from several sources.

    Source ``s`` is picked with probability proportional to ``n_s ** alpha`` (``alpha=1``:
    proportional to size, ``alpha=0``: uniform over sources), optionally scaled by
    ``weights``; within a source, examples are visited in a fresh random order each pass.
    The full state is checkpointable, so a resumed run continues the exact same stream.
    """

    def __init__(
        self,
        sources: Sequence[str],
        alpha: float = 0.5,
        seed: int = 0,
        weights: dict[str, float] | None = None,
    ) -> None:
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(sources):
            groups.setdefault(s, []).append(i)
        self.names = sorted(groups)
        self.groups = [groups[n] for n in self.names]
        sizes = np.array([len(g) for g in self.groups], dtype=np.float64)
        scale = np.array([(weights or {}).get(n, 1.0) for n in self.names])
        p = sizes**alpha * scale
        self.probs = p / p.sum()
        self.rng = random.Random(seed)
        self.perms = [self._shuffled(g) for g in self.groups]
        self.cursors = [0] * len(self.groups)

    def _shuffled(self, group: list[int]) -> list[int]:
        perm = group[:]
        self.rng.shuffle(perm)
        return perm

    def mixture(self) -> dict[str, float]:
        return {n: round(float(p), 4) for n, p in zip(self.names, self.probs, strict=True)}

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        s = self.rng.choices(range(len(self.groups)), weights=self.probs.tolist())[0]
        if self.cursors[s] >= len(self.perms[s]):
            self.perms[s] = self._shuffled(self.groups[s])
            self.cursors[s] = 0
        idx = self.perms[s][self.cursors[s]]
        self.cursors[s] += 1
        return idx

    def state_dict(self) -> dict[str, Any]:
        # Copies: the live lists keep changing as sampling continues.
        return {
            "rng": self.rng.getstate(),
            "perms": [p[:] for p in self.perms],
            "cursors": self.cursors[:],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["rng"])
        self.perms = [p[:] for p in state["perms"]]
        self.cursors = state["cursors"][:]


@dataclass
class BatchStats:
    examples: int
    questions: int
    tokens: int
    padded_tokens: int

    @property
    def packing_efficiency(self) -> float:
        return self.tokens / max(1, self.padded_tokens)


class TrainBatches:
    """Draws examples until a token budget is reached, augments them and packs them into rows.

    The budget is soft: the example that crosses it is still included.
    """

    def __init__(
        self,
        split: EncodedSplit,
        sampler: MixtureSampler,
        limits: PackingLimits,
        max_batch_tokens: int,
        pad_id: int,
        seed: int = 0,
        max_options: int | None = None,
        augment_examples: bool = True,
    ) -> None:
        self.split = split
        self.sampler = sampler
        self.limits = limits
        self.max_batch_tokens = max_batch_tokens
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.max_options = max_options
        self.augment_examples = augment_examples

    def __iter__(self) -> Iterator[tuple[Batch, BatchStats]]:
        return self

    def __next__(self) -> tuple[Batch, BatchStats]:
        examples: list[Example] = []
        used = 0
        while used < self.max_batch_tokens:
            ex = self.split[next(self.sampler)]
            if self.augment_examples:
                ex = augment(ex, self.rng, self.max_options)
            examples.append(ex)
            used += ex.request.length
        batch = collate(examples, self.limits.max_seq_len, pad_id=self.pad_id)
        rows, width = batch.shape
        n_questions = sum(len(v) for v in batch.readouts.values())
        stats = BatchStats(len(examples), n_questions, batch.num_tokens, rows * width)
        return batch, stats

    def state_dict(self) -> dict[str, Any]:
        return {"sampler": self.sampler.state_dict(), "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sampler.load_state_dict(state["sampler"])
        self.rng.setstate(state["rng"])
