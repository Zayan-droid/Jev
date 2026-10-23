"""Familiarity: is this question like the ones the model was calibrated on?

Temperature scaling makes confidences honest *in distribution*. On question templates the model
has never seen, it can be confidently wrong (docs/MODEL_CARD.md), and confidence alone cannot
reveal that. Familiarity is measured on the representation the answer is read from: a Gaussian
is fitted per question type to the readout features of the calibration split, and a question's
score is its Mahalanobis distance to that Gaussian. Questions beyond the distance that covers
``quantile`` of the calibration questions are *unfamiliar*.

The threshold is set on calibration records held out from the fit. In 768 dimensions, distances
of the points a Gaussian was fitted on are biased low, and a threshold taken from them flags
far more than ``1 - quantile`` of new in-distribution questions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from decider.inference import token_budget_batches
from decider.model.decider import DeciderModel
from decider.packing import TYPES, Example, collate


@dataclass
class QuestionFeatures:
    """Per question: identity, readout feature, confidence and correctness, row by row."""

    record_ids: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)  # calibrated top-label probability
    correct: list[bool] = field(default_factory=list)

    def of_type(self, question_type: str) -> np.ndarray:
        rows = [f for f, t in zip(self.features, self.types, strict=True) if t == question_type]
        return np.stack(rows) if rows else np.zeros((0, 0))

    def column(self, name: str, question_type: str) -> np.ndarray:
        values = getattr(self, name)
        return np.array([v for v, t in zip(values, self.types, strict=True) if t == question_type])


@torch.inference_mode()
def collect_features(
    model: DeciderModel,
    examples: Sequence[Example],
    max_seq_len: int,
    device: torch.device,
    max_batch_tokens: int = 16_384,
) -> QuestionFeatures:
    """Readout features and correctness (calibrated argmax vs target) for every question."""
    model = model.to(device).eval()
    out = QuestionFeatures()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    lengths = [ex.request.length for ex in examples]
    for group in token_budget_batches(lengths, max_batch_tokens):
        chunk = [examples[i] for i in group]
        batch = collate([Example(ex.request) for ex in chunk], max_seq_len).to(device)
        with autocast:
            logits, features = model.forward_with_features(batch)
        probs = model.probabilities(logits)
        for t in TYPES:
            feats = features[t].float().cpu().numpy()
            p = probs[t].float().cpu().numpy()
            for row, readout in enumerate(batch.readouts[t]):
                example = chunk[readout.example]
                assert example.targets is not None, "familiarity needs labelled examples"
                target = np.asarray(example.targets[readout.question])
                if t == "noul":
                    correct = (p[row] >= 0.5) == (target[0] >= 0.5)
                    confidence = max(float(p[row]), 1.0 - float(p[row]))
                else:
                    k = len(readout.option_names)
                    correct = target[int(p[row, :k].argmax())] >= target.max() - 1e-9
                    confidence = float(p[row, :k].max())
                out.record_ids.append(str(example.meta.get("id", "")))
                out.questions.append(readout.question)
                out.types.append(t)
                out.families.append(str(example.meta.get("family", "unknown")))
                out.features.append(feats[row])
                out.confidence.append(confidence)
                out.correct.append(bool(correct))
    return out


@dataclass
class Familiarity:
    """Per question type: mean, shrunk precision matrix and the unfamiliarity threshold."""

    mean: dict[str, np.ndarray]
    precision: dict[str, np.ndarray]
    threshold: dict[str, float]

    @classmethod
    def fit(
        cls,
        features: QuestionFeatures,
        quantile: float = 0.95,
        shrinkage: float = 0.1,
        threshold_share: float = 0.3,
    ) -> Familiarity:
        """Fit on ``1 - threshold_share`` of the records, set thresholds on the rest.

        Records, not questions, are split: questions about the same state are correlated.
        """
        held = np.array([_bucket(r) < threshold_share for r in features.record_ids])
        types = np.array(features.types)
        mean: dict[str, np.ndarray] = {}
        precision: dict[str, np.ndarray] = {}
        threshold: dict[str, float] = {}
        for t in TYPES:
            x = features.of_type(t).astype(np.float64)
            h = held[types == t]
            fit, rest = (x[~h], x[h]) if h.any() and (~h).sum() >= 2 else (x, x)
            if len(fit) < 2:
                continue
            mu = fit.mean(0)
            cov = np.cov(fit - mu, rowvar=False)
            # Shrink towards a scaled identity: the feature dimension (768) is not small next to
            # the number of calibration questions, and the raw covariance is ill-conditioned.
            target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
            prec = np.linalg.inv((1 - shrinkage) * cov + shrinkage * target)
            mean[t], precision[t] = mu, prec
            threshold[t] = float(np.quantile(_mahalanobis(rest, mu, prec), quantile))
        return cls(mean, precision, threshold)

    def score(self, question_type: str, x: np.ndarray) -> np.ndarray:
        if question_type not in self.mean:
            return np.zeros(len(x))
        x = np.asarray(x, dtype=np.float64)
        return _mahalanobis(x, self.mean[question_type], self.precision[question_type])

    def unfamiliar(self, question_type: str, x: np.ndarray) -> np.ndarray:
        if question_type not in self.threshold:
            return np.zeros(len(x), dtype=bool)
        return self.score(question_type, x) > self.threshold[question_type]

    def save(self, path: Path) -> None:
        arrays: dict[str, Any] = {}
        for t in self.mean:
            arrays[f"{t}.mean"] = self.mean[t]
            arrays[f"{t}.precision"] = self.precision[t]
            arrays[f"{t}.threshold"] = np.array(self.threshold[t])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Familiarity:
        data = np.load(path)
        types = {key.split(".")[0] for key in data.files}
        return cls(
            {t: data[f"{t}.mean"] for t in types},
            {t: data[f"{t}.precision"] for t in types},
            {t: float(data[f"{t}.threshold"]) for t in types},
        )


def _bucket(record_id: str) -> float:
    """A stable pseudo-random number in [0, 1) per record.

    Salted: dataset splits are assigned by the unsalted hash of the record id
    (``decider.data.build``), so every public-source calib record already has an unsalted hash
    in one narrow range. Reusing it would split calib by source instead of at random.
    """
    digest = hashlib.sha256(b"familiarity-threshold:" + record_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _mahalanobis(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    d = x - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d).clip(min=0.0))


def auroc(negatives: np.ndarray, positives: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if len(negatives) == 0 or len(positives) == 0:
        return float("nan")
    scores = np.concatenate([negatives, positives])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    pos_ranks = ranks[len(negatives) :].sum()
    n_pos, n_neg = len(positives), len(negatives)
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
