"""Production monitoring: Prometheus metrics and confidence-drift detection.

Labels are bounded on purpose (question *type*, never question name or option), so metric
cardinality does not grow with traffic.

Drift is measured as the population stability index (PSI) between the live distribution of
top-label confidence and the reference distribution measured on the test split at training
time (``reference.json`` in the bundle). Without labels in production, a shift in how
confident the model is serves as the earliest available signal that inputs have moved.
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 drifting, > 0.25 investigate.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from decider.packing import TYPES

REQUESTS = Counter("decider_requests_total", "Decision requests", ["status"])
LATENCY = Histogram(
    "decider_request_latency_seconds",
    "End-to-end latency of /v1/decide",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
)
MODEL_LATENCY = Histogram(
    "decider_model_latency_seconds",
    "Model forward time, by state cache outcome",
    ["cache"],
    buckets=(0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
QUESTIONS = Histogram(
    "decider_questions_per_request", "Questions per request", buckets=(1, 2, 4, 8, 16, 32)
)
STATE_TOKENS = Histogram(
    "decider_state_tokens", "State length in tokens", buckets=(32, 64, 128, 256, 512, 1024, 2048)
)
CONFIDENCE = Histogram(
    "decider_confidence",
    "Top-label confidence of answers",
    ["type"],
    buckets=tuple(np.round(np.linspace(0.1, 1.0, 10), 2)),
)
LOW_CONFIDENCE = Counter(
    "decider_low_confidence_total", "Answers below the review threshold", ["type"]
)
NOUL_YES = Counter("decider_noul_answers_total", "noul answers", ["answer"])
CACHE_EVENTS = Counter("decider_state_cache_total", "State cache lookups", ["result"])
CACHE_BYTES = Gauge("decider_state_cache_bytes", "Bytes held by the state cache")
PSI = Gauge("decider_confidence_psi", "PSI of live vs reference confidence", ["type"])
MODEL_INFO = Gauge("decider_model_info", "Loaded model", ["version"])
LOG_WRITE_ERRORS = Counter(
    "decider_log_write_errors_total",
    "Failed writes to the prediction log or review queue",
    ["sink"],
)

# Create every bounded label value up front, so series exist at zero before the first event
# (otherwise an error-rate query has no data until the first error, and alerts cannot fire on
# rates computed from a series that appeared mid-window).
for _status in ("ok", "rejected", "error"):
    REQUESTS.labels(status=_status)
for _result in ("hit", "miss"):
    CACHE_EVENTS.labels(result=_result)
    MODEL_LATENCY.labels(cache=_result)
for _answer in ("yes", "no"):
    NOUL_YES.labels(answer=_answer)
for _sink in ("predictions", "feedback"):
    LOG_WRITE_ERRORS.labels(sink=_sink)
for _type in TYPES:
    CONFIDENCE.labels(type=_type)
    LOW_CONFIDENCE.labels(type=_type)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    """Population stability index between two binned distributions."""
    e = np.clip(np.asarray(expected, dtype=np.float64), eps, None)
    a = np.clip(np.asarray(actual, dtype=np.float64), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(((a - e) * np.log(a / e)).sum())


class DriftMonitor:
    """Rolling window of confidences per question type, compared against a reference."""

    def __init__(
        self, reference: dict[str, list[float]], window: int = 1000, min_samples: int = 200
    ) -> None:
        self.reference = {t: np.asarray(h) for t, h in reference.items() if sum(h) > 0}
        self.bins = len(next(iter(self.reference.values()))) if self.reference else 10
        self.window: dict[str, deque[float]] = {t: deque(maxlen=window) for t in TYPES}
        self.min_samples = min_samples
        self._lock = threading.Lock()

    def observe(self, question_type: str, confidence: float) -> None:
        with self._lock:
            self.window[question_type].append(confidence)

    def current(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with self._lock:
            for t, values in self.window.items():
                if t not in self.reference or len(values) < self.min_samples:
                    continue
                hist, _ = np.histogram(list(values), bins=self.bins, range=(0.0, 1.0))
                live = hist / hist.sum()
                out[t] = {"psi": psi(self.reference[t], live), "n": len(values)}
        return out

    def export(self) -> dict[str, Any]:
        state = self.current()
        for t, v in state.items():
            PSI.labels(type=t).set(v["psi"])
        return state
