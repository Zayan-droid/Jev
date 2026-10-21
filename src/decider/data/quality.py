"""Deduplication, split-leakage checks and dataset statistics."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from decider.schema import ChoiceQuestion, Record, ScoreQuestion, serialize_state

_NON_WORD = re.compile(r"[^0-9a-z]+")

# Order matters: a record that collides with an earlier split is dropped from the later one.
SPLIT_PRIORITY = ("train", "val", "calib", "test", "heldout", "gold")


def normalize(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def state_key(record: Record) -> str:
    return hashlib.sha1(normalize(serialize_state(record.state)).encode()).hexdigest()


def content_key(record: Record) -> str:
    """Normalised state + questions. Two records with the same key are the same example."""
    parts = [normalize(serialize_state(record.state))]
    for name in sorted(record.questions):
        q = record.questions[name]
        parts.append(f"{q.type}:{normalize(q.instructions)}")
        if isinstance(q, ChoiceQuestion):
            parts.extend(sorted(normalize(f"{k} {v}") for k, v in q.criteria.items()))
        elif isinstance(q, ScoreQuestion):
            parts.extend(normalize(level) for level in q.levels)
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()


@dataclass
class QualityReport:
    duplicates_removed: dict[str, int] = field(default_factory=dict)
    leakage_removed: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deduplicate(
    splits: dict[str, list[Record]], leakage_key: str = "state"
) -> tuple[dict[str, list[Record]], QualityReport]:
    """Drop exact (normalised) duplicates within a split and any evaluation record whose state
    (``leakage_key="state"``) or full content also appears in an earlier split.

    Programmatic families share states by design only within a split, so leakage on states is
    the right default: a model must never be evaluated on a passage it was trained on.
    """
    report = QualityReport()
    seen_state: set[str] = set()
    seen_content: set[str] = set()
    out: dict[str, list[Record]] = {}
    for split in sorted(splits, key=lambda s: SPLIT_PRIORITY.index(s)):
        kept: list[Record] = []
        local: set[str] = set()
        dupes = leaks = 0
        for record in splits[split]:
            ckey = content_key(record)
            if ckey in local:
                dupes += 1
                continue
            skey = state_key(record)
            leaked = ckey in seen_content or (leakage_key == "state" and skey in seen_state)
            if leaked and split != "train":
                leaks += 1
                continue
            local.add(ckey)
            kept.append(record)
        for record in kept:
            seen_content.add(content_key(record))
            seen_state.add(state_key(record))
        out[split] = kept
        report.duplicates_removed[split] = dupes
        report.leakage_removed[split] = leaks
    return out, report


def assert_no_leakage(splits: dict[str, list[Record]]) -> None:
    train_states = {state_key(r) for r in splits.get("train", [])}
    for split, records in splits.items():
        if split == "train":
            continue
        leaked = [r.id for r in records if state_key(r) in train_states]
        if leaked:
            raise AssertionError(f"{len(leaked)} {split} records share a state with train")


def summarize(records: list[Record]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    options: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_source[r.source] += 1
        by_family[r.family] += 1
        for q in r.questions.values():
            by_type[q.type] += 1
            if isinstance(q, ChoiceQuestion):
                options["choice"].append(len(q.criteria))
            elif isinstance(q, ScoreQuestion):
                options["score"].append(len(q.levels))
    n_questions = sum(by_type.values())
    return {
        "records": len(records),
        "questions": n_questions,
        "by_source": dict(sorted(by_source.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_type": dict(sorted(by_type.items())),
        "mean_options": {k: round(sum(v) / len(v), 2) for k, v in options.items() if v},
    }
