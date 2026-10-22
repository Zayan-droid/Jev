"""Paired comparison of two models evaluated on the same questions.

``evaluate`` writes one row per question (``predictions_<split>.jsonl.gz``). Joining two such
files on ``(record, question)`` gives paired outcomes, so the difference between models can be
tested directly instead of comparing two unpaired averages. Confidence intervals come from a
paired bootstrap that resamples whole *records*: questions about the same state are
correlated, and resampling them independently would make the intervals too narrow.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decider.evaluation.metrics import (
    QuestionResult,
    is_correct,
    log_loss,
    probabilities,
    target_distribution,
)


def prediction_rows(
    results: Sequence[QuestionResult], temperatures: dict[str, float]
) -> Iterable[dict[str, Any]]:
    for r in results:
        p = probabilities(r, temperatures.get(r.type, 1.0))
        t = target_distribution(r)
        yield {
            "record": r.record_id,
            "question": r.question,
            "type": r.type,
            "family": r.family,
            "source": r.source,
            "correct": is_correct(p, t),
            "log_loss": log_loss(p, t),
            "confidence": float(p.max()),
        }


def write_predictions(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_predictions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["record"], r["question"]): r for r in rows}


@dataclass(frozen=True)
class Delta:
    group: str
    n: int
    a: float
    b: float
    diff: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_deltas(
    a: dict[tuple[str, str], dict[str, Any]],
    b: dict[tuple[str, str], dict[str, Any]],
    metric: str = "correct",
    by: str = "type",
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[Delta]:
    """Mean of ``metric`` for A and B on their shared questions, per group and overall, with a
    95 % record-level paired bootstrap interval for B - A."""
    keys = sorted(set(a) & set(b))
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        groups[str(a[key][by])].append(key)
    groups["all"] = keys
    rng = np.random.default_rng(seed)
    out = []
    for group, members in sorted(groups.items(), key=lambda kv: (kv[0] == "all", kv[0])):
        if not members:
            continue
        by_record: dict[str, list[float]] = defaultdict(list)
        for key in members:
            by_record[key[0]].append(float(b[key][metric]) - float(a[key][metric]))
        diffs = [np.asarray(v) for v in by_record.values()]
        sums = np.array([d.sum() for d in diffs])
        counts = np.array([len(d) for d in diffs])
        idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        mean_a = float(np.mean([float(a[k][metric]) for k in members]))
        mean_b = float(np.mean([float(b[k][metric]) for k in members]))
        out.append(
            Delta(
                group,
                len(members),
                mean_a,
                mean_b,
                mean_b - mean_a,
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            )
        )
    return out


def render(deltas: Sequence[Delta], metric: str, names: tuple[str, str] = ("A", "B")) -> str:
    scale, unit = (100.0, " pt") if metric == "correct" else (1.0, "")
    label = "accuracy" if metric == "correct" else metric.replace("_", " ")
    lines = [
        f"| group | n | {names[0]} {label} | {names[1]} {label} | difference | 95 % CI |",
        "|---|---|---|---|---|---|",
    ]
    for d in deltas:
        mark = " *" if d.significant else ""
        lines.append(
            f"| {d.group} | {d.n} | {d.a * scale:.2f} | {d.b * scale:.2f} | "
            f"{d.diff * scale:+.2f}{unit}{mark} | [{d.low * scale:+.2f}, {d.high * scale:+.2f}] |"
        )
    lines.append("")
    lines.append("`*`: the 95 % interval excludes zero (record-level paired bootstrap).")
    return "\n".join(lines)
