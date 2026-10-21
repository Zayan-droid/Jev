"""Turning requests into packed token sequences and batches.

Layout of one request (one *document*)::

    [CLS] state ... [SEP] | [Q_CHOICE] instr ... [OPT] a ... [OPT] b ... | [Q_NOUL] instr ... | ...

Every token carries three segment labels, which :mod:`decider.masks` turns into the attention
mask, plus a *tree* position id:

* ``question_ids``: 0 for the state, ``k`` for the k-th question (1-based).
* ``option_ids``: 0 for the state or a question header, ``m`` for the m-th option/level (1-based).
* ``position_ids``: the state uses ``0..S-1``; every question restarts at ``S``; every option of a
  question restarts right after that question's header. Questions are parallel branches, and
  options within a question are indistinguishable by position, which makes ``choice`` answers
  exactly permutation-equivariant.

Several documents can share one row (``doc_ids``); the mask keeps them apart.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from decider.schema import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    Question,
    QuestionType,
    ScoreQuestion,
    serialize_state,
)
from decider.tokenization import Tokenizer

TYPES: tuple[QuestionType, ...] = ("noul", "choice", "score")


@dataclass(frozen=True)
class PackingLimits:
    max_seq_len: int = 1024
    max_state_tokens: int = 512
    max_question_tokens: int = 96
    max_option_tokens: int = 48

    def __post_init__(self) -> None:
        if self.max_state_tokens + self.max_question_tokens > self.max_seq_len:
            raise ValueError("max_state_tokens + max_question_tokens must fit in max_seq_len")


@dataclass(frozen=True)
class TokenizedQuestion:
    name: str
    type: QuestionType
    header: list[int]  # [Q_TYPE] + instruction tokens
    options: list[list[int]]  # [OPT]/[LVL] + text tokens, one list per option or level
    option_names: list[str]  # choice: option names; score: "0".."L-1"; noul: empty

    @property
    def length(self) -> int:
        return len(self.header) + sum(len(o) for o in self.options)


@dataclass(frozen=True)
class TokenizedRequest:
    state: list[int]  # [CLS] + state tokens + [SEP]
    questions: list[TokenizedQuestion]

    @property
    def length(self) -> int:
        return len(self.state) + sum(q.length for q in self.questions)


def _option_text(name: str, description: str) -> str:
    return f"{name}: {description}" if description else name


class RequestEncoder:
    """Tokenizes requests once; packing then only rearranges token lists."""

    def __init__(self, tokenizer: Tokenizer, limits: PackingLimits) -> None:
        self.tokenizer = tokenizer
        self.limits = limits

    def encode(self, request: DecisionRequest) -> TokenizedRequest:
        return self.encode_many([request])[0]

    def encode_many(self, requests: Sequence[DecisionRequest]) -> list[TokenizedRequest]:
        # One encode_batch call for every piece of text across all requests.
        texts: list[str] = []
        for request in requests:
            texts.append(serialize_state(request.state))
            for question in request.questions.values():
                texts.append(question.instructions)
                texts.extend(self._option_texts(question))
        ids = iter(self.tokenizer.encode_batch(texts))
        return [self._assemble(request, ids) for request in requests]

    @staticmethod
    def _option_texts(question: Question) -> list[str]:
        if isinstance(question, ChoiceQuestion):
            return [_option_text(k, v) for k, v in question.criteria.items()]
        if isinstance(question, ScoreQuestion):
            return list(question.levels)
        return []

    def _assemble(self, request: DecisionRequest, it: Iterator[list[int]]) -> TokenizedRequest:
        tok, lim = self.tokenizer, self.limits
        state = next(it)
        questions = []
        for name, question in request.questions.items():
            header = [tok.question_token_id(question.type), *next(it)][: lim.max_question_tokens]
            if isinstance(question, NoulQuestion):
                options, names = [], []
            elif isinstance(question, ChoiceQuestion):
                options = [
                    [tok.option_id, *next(it)][: lim.max_option_tokens] for _ in question.criteria
                ]
                names = list(question.criteria)
            else:
                options = [
                    [tok.level_id, *next(it)][: lim.max_option_tokens] for _ in question.levels
                ]
                names = [str(i) for i in range(len(question.levels))]
            questions.append(TokenizedQuestion(name, question.type, header, options, names))

        question_len = sum(q.length for q in questions)
        budget = min(lim.max_state_tokens, lim.max_seq_len - question_len) - 2
        if budget < 0:
            raise ValueError(
                f"questions need {question_len} tokens, more than max_seq_len={lim.max_seq_len}"
            )
        return TokenizedRequest([tok.cls_id, *state[:budget], tok.sep_id], questions)


# --------------------------------------------------------------------------------------
# Examples (tokenized request + dense targets) and augmentation
# --------------------------------------------------------------------------------------


@dataclass
class Example:
    request: TokenizedRequest
    # Dense target per question, aligned with that question's options (noul: [p_yes]).
    targets: dict[str, list[float]] | None = None
    meta: dict[str, str] = field(default_factory=dict)


def augment(example: Example, rng: random.Random, max_options: int | None = None) -> Example:
    """Shuffle question order and choice option order; optionally subsample large option sets.

    Subsampling keeps every option that carries target mass and fills up with random negatives,
    so the target distribution is unchanged on the kept options. Score levels are ordered and
    are never shuffled or dropped.
    """
    questions = list(example.request.questions)
    rng.shuffle(questions)
    targets = dict(example.targets) if example.targets is not None else None
    out = []
    for q in questions:
        if q.type != "choice":
            out.append(q)
            continue
        order = list(range(len(q.options)))
        if max_options is not None and len(order) > max_options and targets is not None:
            t = targets[q.name]
            positive = [i for i in order if t[i] > 0]
            negative = [i for i in order if t[i] <= 0]
            rng.shuffle(negative)
            order = positive + negative[: max(0, max_options - len(positive))]
        rng.shuffle(order)
        out.append(
            TokenizedQuestion(
                q.name,
                q.type,
                q.header,
                [q.options[i] for i in order],
                [q.option_names[i] for i in order],
            )
        )
        if targets is not None:
            kept = [targets[q.name][i] for i in order]
            total = sum(kept)
            targets[q.name] = [v / total for v in kept]
    return Example(TokenizedRequest(example.request.state, out), targets, example.meta)


# --------------------------------------------------------------------------------------
# Packing into rows and batches
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Readout:
    example: int  # index of the example within the batch
    question: str
    type: QuestionType
    option_names: tuple[str, ...]


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, L)
    position_ids: torch.Tensor  # (B, L)
    doc_ids: torch.Tensor  # (B, L), -1 on padding
    question_ids: torch.Tensor  # (B, L)
    option_ids: torch.Tensor  # (B, L)
    level_ids: torch.Tensor  # (B, L), 0 = not a level token, m+1 = level m
    # Flat indices into the (B*L) token axis where each question is read out.
    index: dict[str, torch.Tensor]  # noul: (N,), choice/score: (N, K)
    option_mask: dict[str, torch.Tensor]  # choice/score: (N, K) bool
    targets: dict[str, torch.Tensor] | None
    readouts: dict[str, list[Readout]]
    num_tokens: int  # real (non-padding) tokens

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.input_ids.shape)  # type: ignore[return-value]

    def to(self, device: torch.device | str) -> Batch:
        def move(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {k: v.to(device, non_blocking=True) for k, v in d.items()}

        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            position_ids=self.position_ids.to(device, non_blocking=True),
            doc_ids=self.doc_ids.to(device, non_blocking=True),
            question_ids=self.question_ids.to(device, non_blocking=True),
            option_ids=self.option_ids.to(device, non_blocking=True),
            level_ids=self.level_ids.to(device, non_blocking=True),
            index=move(self.index),
            option_mask=move(self.option_mask),
            targets=move(self.targets) if self.targets is not None else None,
            readouts=self.readouts,
            num_tokens=self.num_tokens,
        )


@dataclass
class _Row:
    input_ids: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    doc_ids: list[int] = field(default_factory=list)
    question_ids: list[int] = field(default_factory=list)
    option_ids: list[int] = field(default_factory=list)
    level_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.input_ids)

    def extend(self, ids: list[int], pos: int, doc: int, q: int, o: int, level: int = 0) -> int:
        n = len(ids)
        self.input_ids += ids
        self.position_ids += range(pos, pos + n)
        self.doc_ids += [doc] * n
        self.question_ids += [q] * n
        self.option_ids += [o] * n
        self.level_ids += [level] * n
        return len(self.input_ids) - n  # offset of the first appended token


def assign_rows(lengths: Sequence[int], max_len: int) -> list[list[int]]:
    """First-fit-decreasing bin packing of sequences into rows of at most ``max_len`` tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    rows: list[list[int]] = []
    free: list[int] = []
    for i in order:
        if lengths[i] > max_len:
            raise ValueError(f"sequence of {lengths[i]} tokens exceeds the row length {max_len}")
        for r, space in enumerate(free):
            if lengths[i] <= space:
                rows[r].append(i)
                free[r] -= lengths[i]
                break
        else:
            rows.append([i])
            free.append(max_len - lengths[i])
    return rows


def collate(
    examples: Sequence[Example],
    max_seq_len: int,
    pack: bool = True,
    pad_id: int = 0,
    pad_to_multiple: int = 8,
) -> Batch:
    """Pack examples into rows and build the readout index tensors."""
    lengths = [ex.request.length for ex in examples]
    rows_of = assign_rows(lengths, max_seq_len) if pack else [[i] for i in range(len(examples))]

    rows = [_Row() for _ in rows_of]
    # Per type: list of (row, token offsets) and dense targets.
    positions: dict[str, list[list[tuple[int, int]]]] = {t: [] for t in TYPES}
    target_lists: dict[str, list[list[float]]] = {t: [] for t in TYPES}
    readouts: dict[str, list[Readout]] = {t: [] for t in TYPES}
    with_targets = all(ex.targets is not None for ex in examples)

    for r, members in enumerate(rows_of):
        row = rows[r]
        for doc, ex_idx in enumerate(members):
            example = examples[ex_idx]
            req = example.request
            row.extend(req.state, 0, doc, 0, 0)
            branch = len(req.state)
            for k, q in enumerate(req.questions, start=1):
                head = row.extend(q.header, branch, doc, k, 0)
                opt_start = branch + len(q.header)
                if q.type == "noul":
                    slots = [(r, head)]
                else:
                    slots = []
                    for m, opt in enumerate(q.options, start=1):
                        level = m if q.type == "score" else 0
                        slots.append((r, row.extend(opt, opt_start, doc, k, m, level)))
                positions[q.type].append(slots)
                readouts[q.type].append(Readout(ex_idx, q.name, q.type, tuple(q.option_names)))
                if with_targets:
                    assert example.targets is not None
                    target_lists[q.type].append(example.targets[q.name])

    width = max(len(row) for row in rows)
    width = -(-width // pad_to_multiple) * pad_to_multiple

    def tensor(attr: str, fill: int) -> torch.Tensor:
        data = [getattr(row, attr) + [fill] * (width - len(row)) for row in rows]
        return torch.tensor(data, dtype=torch.long)

    index: dict[str, torch.Tensor] = {}
    option_mask: dict[str, torch.Tensor] = {}
    targets: dict[str, torch.Tensor] = {}
    for t in TYPES:
        slots_per_q = positions[t]
        if t == "noul":
            index[t] = torch.tensor(
                [s[0][0] * width + s[0][1] for s in slots_per_q], dtype=torch.long
            )
            if with_targets:
                targets[t] = torch.tensor([v[0] for v in target_lists[t]], dtype=torch.float32)
            continue
        k_max = max((len(s) for s in slots_per_q), default=0)
        idx = torch.zeros(len(slots_per_q), k_max, dtype=torch.long)
        mask = torch.zeros(len(slots_per_q), k_max, dtype=torch.bool)
        tgt = torch.zeros(len(slots_per_q), k_max, dtype=torch.float32)
        for n, slots in enumerate(slots_per_q):
            idx[n, : len(slots)] = torch.tensor([r * width + p for r, p in slots])
            mask[n, : len(slots)] = True
            if with_targets:
                tgt[n, : len(slots)] = torch.tensor(target_lists[t][n])
        index[t], option_mask[t] = idx, mask
        if with_targets:
            targets[t] = tgt

    return Batch(
        input_ids=tensor("input_ids", pad_id),
        position_ids=tensor("position_ids", 0),
        doc_ids=tensor("doc_ids", -1),
        question_ids=tensor("question_ids", 0),
        option_ids=tensor("option_ids", 0),
        level_ids=tensor("level_ids", 0),
        index=index,
        option_mask=option_mask,
        targets=targets if with_targets else None,
        readouts=readouts,
        num_tokens=sum(lengths),
    )
