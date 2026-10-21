"""Quality and calibration metrics over per-question predictions.

All functions are plain NumPy. A *result* is one question of one record: the predicted
distribution, the target distribution and grouping metadata. noul questions are treated as
two-class distributions ``[p_no, p_yes]`` so every metric has one definition.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decider.schema import QuestionType

EPS = 1e-12


@dataclass
class QuestionResult:
    record_id: str
    question: str
    type: QuestionType
    source: str
    family: str
    option_names: tuple[str, ...]
    logits: np.ndarray  # noul: (1,), others: (K,)
    target: np.ndarray  # noul: (1,) = p_yes, others: (K,)
    meta: dict[str, Any] = field(default_factory=dict)


def probabilities(result: QuestionResult, temperature: float = 1.0) -> np.ndarray:
    """Full predicted distribution (noul as ``[p_no, p_yes]``)."""
    z = result.logits / temperature
    if result.type == "noul":
        p = 1.0 / (1.0 + np.exp(-z[0]))
        return np.array([1.0 - p, p])
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def target_distribution(result: QuestionResult) -> np.ndarray:
    if result.type == "noul":
        return np.array([1.0 - result.target[0], result.target[0]])
    return result.target


# --------------------------------------------------------------------------------------
# per-question scores
# --------------------------------------------------------------------------------------


def log_loss(p: np.ndarray, t: np.ndarray) -> float:
    return float(-(t * np.log(np.clip(p, EPS, 1.0))).sum())


def brier(p: np.ndarray, t: np.ndarray) -> float:
    return float(((p - t) ** 2).sum())


def rps(p: np.ndarray, t: np.ndarray) -> float:
    """Ranked probability score (discrete CRPS) for ordered levels, normalised to [0, 1]."""
    if len(p) < 2:
        return 0.0
    return float(((np.cumsum(p) - np.cumsum(t)) ** 2)[:-1].sum() / (len(p) - 1))


def is_correct(p: np.ndarray, t: np.ndarray) -> bool:
    """argmax(p) is one of the (possibly tied) argmaxes of the target."""
    return bool(t[int(p.argmax())] >= t.max() - 1e-9)


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> list[ReliabilityBin]:
    """Equal-width bins on [0, 1], or equal-mass bins when ``adaptive``."""
    if len(confidence) == 0:
        return []
    if adaptive:
        edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        sel = ids == b
        if sel.any():
            bins.append(
                ReliabilityBin(
                    float(edges[b]),
                    float(edges[b + 1]),
                    int(sel.sum()),
                    float(confidence[sel].mean()),
                    float(correct[sel].mean()),
                )
            )
    return bins


def ece(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> float:
    """Expected calibration error of the top-label confidence."""
    n = len(confidence)
    if n == 0:
        return float("nan")
    bins = reliability_bins(confidence, correct, n_bins, adaptive)
    return float(sum(b.count / n * abs(b.accuracy - b.confidence) for b in bins))


# --------------------------------------------------------------------------------------
# selective prediction
# --------------------------------------------------------------------------------------


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy when answering only the most confident fraction of questions."""
    order = np.argsort(-confidence, kind="stable")
    hits = np.cumsum(correct[order])
    n = np.arange(1, len(order) + 1)
    return n / len(order), hits / n


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    if len(confidence) == 0:
        return float("nan")
    _, acc = risk_coverage(confidence, correct)
    return float((1.0 - acc).mean())


def coverage_at_accuracy(confidence: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Largest fraction of questions that can be answered automatically at ``target`` accuracy."""
    if len(confidence) == 0:
        return float("nan")
    coverage, acc = risk_coverage(confidence, correct)
    ok = np.nonzero(acc >= target)[0]
    return float(coverage[ok.max()]) if len(ok) else 0.0


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


@dataclass
class Scored:
    """Arrays over a group of results, computed once and reused by every metric."""

    types: list[str]
    log_loss: np.ndarray
    brier: np.ndarray
    confidence: np.ndarray
    correct: np.ndarray
    rps: np.ndarray  # nan for non-score questions
    score_error: np.ndarray  # |E_p[level] - E_t[level]|, nan for non-score
    y_true: list[str]
    y_pred: list[str]


def score_results(
    results: Sequence[QuestionResult], temperatures: dict[str, float] | None = None
) -> Scored:
    temperatures = temperatures or {}
    ll, br, conf, corr, rp, se, yt, yp = [], [], [], [], [], [], [], []
    types: list[str] = []
    for r in results:
        p = probabilities(r, temperatures.get(r.type, 1.0))
        t = target_distribution(r)
        names = ("no", "yes") if r.type == "noul" else r.option_names
        ll.append(log_loss(p, t))
        br.append(brier(p, t))
        conf.append(float(p.max()))
        corr.append(is_correct(p, t))
        types.append(r.type)
        yt.append(names[int(t.argmax())])
        yp.append(names[int(p.argmax())])
        if r.type == "score":
            levels = np.arange(len(p))
            rp.append(rps(p, t))
            se.append(abs(float((levels * p).sum() - (levels * t).sum())))
        else:
            rp.append(np.nan)
            se.append(np.nan)
    return Scored(
        types,
        np.array(ll),
        np.array(br),
        np.array(conf),
        np.array(corr, dtype=np.float64),
        np.array(rp),
        np.array(se),
        yt,
        yp,
    )


def summarize_scored(s: Scored, n_bins: int = 15) -> dict[str, float]:
    n = len(s.log_loss)
    if n == 0:
        return {"n": 0}
    out = {
        "n": n,
        "accuracy": float(s.correct.mean()),
        "log_loss": float(s.log_loss.mean()),
        "brier": float(s.brier.mean()),
        "ece": ece(s.confidence, s.correct, n_bins),
        "adaptive_ece": ece(s.confidence, s.correct, n_bins, adaptive=True),
        "mean_confidence": float(s.confidence.mean()),
        "aurc": aurc(s.confidence, s.correct),
        "coverage_at_95": coverage_at_accuracy(s.confidence, s.correct, 0.95),
        "macro_f1": macro_f1(s.y_true, s.y_pred),
    }
    if not np.isnan(s.rps).all():
        out["rps"] = float(np.nanmean(s.rps))
        out["score_mae"] = float(np.nanmean(s.score_error))
    return out


def subset(s: Scored, mask: np.ndarray) -> Scored:
    idx = np.nonzero(mask)[0]
    return Scored(
        [s.types[i] for i in idx],
        s.log_loss[idx],
        s.brier[idx],
        s.confidence[idx],
        s.correct[idx],
        s.rps[idx],
        s.score_error[idx],
        [s.y_true[i] for i in idx],
        [s.y_pred[i] for i in idx],
    )


def grouped(
    results: Sequence[QuestionResult],
    scored: Scored,
    key: Callable[[QuestionResult], str],
    n_bins: int = 15,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        groups[key(r)].append(i)
    out = {}
    for name, idx in sorted(groups.items()):
        mask = np.zeros(len(results), dtype=bool)
        mask[idx] = True
        out[name] = summarize_scored(subset(scored, mask), n_bins)
    return out


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    stats = np.array([statistic(values[row]) for row in idx])
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def bootstrap_ece_ci(
    confidence: np.ndarray, correct: np.ndarray, n_resamples: int = 500, seed: int = 0
) -> tuple[float, float]:
    if len(confidence) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(confidence)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats.append(ece(confidence[idx], correct[idx]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
