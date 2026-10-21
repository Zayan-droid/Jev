"""Batched inference: requests in, distributions and typed answers out."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from decider.model.decider import DeciderModel
from decider.packing import TYPES, Batch, Example, PackingLimits, RequestEncoder, collate
from decider.schema import (
    Answer,
    ChoiceAnswer,
    DecisionRequest,
    NoulAnswer,
    QuestionType,
    ScoreAnswer,
)
from decider.tokenization import Tokenizer


def normalized_confidence(p: np.ndarray) -> float:
    """``1 - H(p) / log K``: 1 for a one-hot distribution, 0 for a uniform one."""
    k = len(p)
    if k < 2:
        return 1.0
    q = np.clip(p, 1e-12, 1.0)
    entropy = float(-(p * np.log(q)).sum())
    return max(0.0, 1.0 - entropy / math.log(k))


@dataclass(frozen=True)
class Prediction:
    """Model output for one question of one request."""

    question: str
    type: QuestionType
    option_names: tuple[str, ...]
    logits: np.ndarray  # noul: shape (1,); choice/score: (K,)
    probs: np.ndarray  # noul: [p_yes]; choice/score: distribution over options

    def distribution(self) -> np.ndarray:
        """Full distribution: noul as ``[p_no, p_yes]``."""
        if self.type == "noul":
            return np.array([1.0 - self.probs[0], self.probs[0]])
        return self.probs

    def answer(self) -> Answer:
        conf = normalized_confidence(self.distribution())
        if self.type == "noul":
            p = float(self.probs[0])
            return NoulAnswer(p_yes=p, answer=p >= 0.5, confidence=conf)
        if self.type == "choice":
            dist = {n: float(v) for n, v in zip(self.option_names, self.probs, strict=True)}
            best = max(dist, key=dist.__getitem__)
            return ChoiceAnswer(distribution=dist, choice=best, confidence=conf)
        levels = np.arange(len(self.probs))
        return ScoreAnswer(
            distribution=[float(v) for v in self.probs],
            score=float((levels * self.probs).sum()),
            level=int(self.probs.argmax()),
            confidence=conf,
        )


def split_predictions(
    batch: Batch, logits: dict[str, torch.Tensor], probs: dict[str, torch.Tensor], n: int
) -> list[dict[str, Prediction]]:
    """Regroup flat per-type outputs into one ``{question: Prediction}`` dict per example."""
    out: list[dict[str, Prediction]] = [{} for _ in range(n)]
    for t in TYPES:
        z_all = logits[t].float().cpu().numpy()
        p_all = probs[t].float().cpu().numpy()
        for row, readout in enumerate(batch.readouts[t]):
            k = len(readout.option_names)
            z = z_all[row : row + 1] if t == "noul" else z_all[row, :k]
            p = p_all[row : row + 1] if t == "noul" else p_all[row, :k]
            out[readout.example][readout.question] = Prediction(
                readout.question, t, readout.option_names, z, p
            )
    return out


def token_budget_batches(lengths: Sequence[int], max_tokens: int) -> Iterator[list[int]]:
    """Consecutive groups of indices whose total length stays within ``max_tokens``."""
    group: list[int] = []
    used = 0
    for i, n in enumerate(lengths):
        if group and used + n > max_tokens:
            yield group
            group, used = [], 0
        group.append(i)
        used += n
    if group:
        yield group


class Predictor:
    def __init__(
        self,
        model: DeciderModel,
        tokenizer: Tokenizer,
        limits: PackingLimits,
        device: str | torch.device = "cpu",
        max_batch_tokens: int = 16_384,
        calibrated: bool = True,
        precision: str = "bf16",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.limits = limits
        self.encoder = RequestEncoder(tokenizer, limits)
        self.device = torch.device(device)
        self.max_batch_tokens = max_batch_tokens
        self.calibrated = calibrated
        self.precision = precision  # "bf16" (autocast on CUDA) or "fp32"

    def _autocast(self) -> torch.autocast:
        enabled = self.device.type == "cuda" and self.precision == "bf16"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def with_precision(self, precision: str) -> Predictor:
        """A predictor sharing this one's model and settings at another precision."""
        return Predictor(
            self.model,
            self.tokenizer,
            self.limits,
            self.device,
            self.max_batch_tokens,
            self.calibrated,
            precision,
        )

    @torch.inference_mode()
    def predict_examples(self, examples: Sequence[Example]) -> list[dict[str, Prediction]]:
        results: list[dict[str, Prediction]] = []
        lengths = [ex.request.length for ex in examples]
        for group in token_budget_batches(lengths, self.max_batch_tokens):
            chunk = [Example(examples[i].request) for i in group]
            batch = collate(chunk, self.limits.max_seq_len, pad_id=self.tokenizer.pad_id)
            batch = batch.to(self.device)
            with self._autocast():
                logits = self.model(batch)
            probs = self.model.probabilities(logits, calibrated=self.calibrated)
            results.extend(split_predictions(batch, logits, probs, len(chunk)))
        return results

    def predict(self, requests: Sequence[DecisionRequest]) -> list[dict[str, Prediction]]:
        encoded = self.encoder.encode_many(requests)
        return self.predict_examples([Example(r) for r in encoded])

    def answer(self, request: DecisionRequest) -> dict[str, Answer]:
        predictions = self.predict([request])[0]
        return {name: predictions[name].answer() for name in request.questions}
