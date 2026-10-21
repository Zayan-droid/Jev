"""The inference engine behind the API: bundle, state cache, monitoring and logging."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from decider.bundle import Bundle, load_bundle
from decider.inference import split_predictions
from decider.model.decider import StateCache
from decider.packing import Example, RequestEncoder, collate
from decider.schema import Answer, DecisionRequest, serialize_state
from decider.serving import monitoring as m
from decider.serving.redaction import redact

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    bundle: str
    device: str = "auto"
    cache_max_bytes: int = 256 * 2**20
    low_confidence_threshold: float = 0.6
    prediction_log: str | None = "logs/predictions.jsonl"
    feedback_queue: str | None = "data/feedback/queue.jsonl"
    log_states: bool = True  # redacted copies; set False to log only a state hash
    drift_window: int = 1000
    drift_min_samples: int = 200


class LRUStateCache:
    """Byte-budgeted LRU of encoded states, keyed by the state's token ids."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.bytes = 0
        self._items: OrderedDict[str, StateCache] = OrderedDict()

    @staticmethod
    def key(state_ids: list[int]) -> str:
        return hashlib.sha256(json.dumps(state_ids).encode()).hexdigest()

    def get(self, key: str) -> StateCache | None:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: str, value: StateCache) -> None:
        size = value.nbytes()
        if size > self.max_bytes:
            return
        if key in self._items:
            self.bytes -= self._items.pop(key).nbytes()
        self._items[key] = value
        self.bytes += size
        while self.bytes > self.max_bytes:
            _, evicted = self._items.popitem(last=False)
            self.bytes -= evicted.nbytes()

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class DecisionResult:
    answers: dict[str, Answer]
    cache_hit: bool
    state_tokens: int
    model_ms: float
    review: list[str] = field(default_factory=list)  # questions below the confidence threshold


class DecisionEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.bundle: Bundle = load_bundle(cfg.bundle, self.device)
        self.encoder = RequestEncoder(self.bundle.tokenizer, self.bundle.limits)
        self.cache = LRUStateCache(cfg.cache_max_bytes)
        self._lock = threading.Lock()
        reference_path = Path(cfg.bundle) / "reference.json"
        reference = (
            json.loads(reference_path.read_text()).get("histograms", {})
            if reference_path.exists()
            else {}
        )
        self.drift = m.DriftMonitor(reference, cfg.drift_window, cfg.drift_min_samples)
        self._log_lock = threading.Lock()
        m.MODEL_INFO.labels(version=self.version).set(1)
        log.info("loaded %s on %s", self.version, self.device)

    @property
    def version(self) -> str:
        return self.bundle.version

    def info(self) -> dict[str, Any]:
        meta = self.bundle.metadata
        return {
            "version": self.version,
            "device": str(self.device),
            "temperatures": self.bundle.model.temperatures(),
            "packing": self.bundle.limits.__dict__,
            "parameters": self.bundle.model.num_parameters(),
            "data": meta.get("data"),
            "code": meta.get("code"),
            "summary": meta.get("summary"),
        }

    def _autocast(self) -> torch.autocast:
        cuda = self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cuda)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        tokenized = self.encoder.encode(request)
        batch = collate([Example(tokenized)], self.bundle.limits.max_seq_len, pack=False)
        batch = batch.to(self.device)
        key = LRUStateCache.key(tokenized.state)
        model = self.bundle.model
        with self._lock, torch.inference_mode(), self._autocast():
            start = time.perf_counter()
            cache = self.cache.get(key)
            hit = cache is not None
            if cache is None:
                logits, cache = model.forward_and_cache(batch, len(tokenized.state))
                self.cache.put(key, cache)
            else:
                logits = model.forward_cached(cache, batch)
            probs = model.probabilities(logits)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - start) * 1000
        predictions = split_predictions(batch, logits, probs, 1)[0]
        answers = {name: predictions[name].answer() for name in request.questions}

        m.CACHE_EVENTS.labels(result="hit" if hit else "miss").inc()
        m.CACHE_BYTES.set(self.cache.bytes)
        m.MODEL_LATENCY.labels(cache="hit" if hit else "miss").observe(model_ms / 1000)
        m.QUESTIONS.observe(len(request.questions))
        m.STATE_TOKENS.observe(len(tokenized.state))
        review = []
        for name, answer in answers.items():
            confidence = max(predictions[name].distribution())
            m.CONFIDENCE.labels(type=answer.type).observe(confidence)
            self.drift.observe(answer.type, confidence)
            if answer.type == "noul":
                m.NOUL_YES.labels(answer="yes" if answer.answer else "no").inc()
            if confidence < self.cfg.low_confidence_threshold:
                m.LOW_CONFIDENCE.labels(type=answer.type).inc()
                review.append(name)
        self.drift.export()
        return DecisionResult(answers, hit, len(tokenized.state), model_ms, review)

    # ------------------------------------------------------------------ logging

    def record(
        self, request_id: str, request: DecisionRequest, result: DecisionResult, latency_ms: float
    ) -> None:
        """Append to the prediction log; queue low-confidence questions for labelling."""
        state_text = serialize_state(request.state)
        entry = {
            "request_id": request_id,
            "time": time.time(),
            "model_version": self.version,
            "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
            "state": redact(request.state) if self.cfg.log_states else None,
            "questions": {k: v.model_dump() for k, v in request.questions.items()},
            "answers": {k: v.model_dump() for k, v in result.answers.items()},
            "latency_ms": round(latency_ms, 3),
            "cache_hit": result.cache_hit,
        }
        with self._log_lock:
            if self.cfg.prediction_log:
                _append(Path(self.cfg.prediction_log), entry)
            if self.cfg.feedback_queue and result.review:
                queued = {**entry, "review": result.review, "state": redact(request.state)}
                _append(Path(self.cfg.feedback_queue), queued)


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
