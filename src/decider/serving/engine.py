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
from decider.evaluation.familiarity import Familiarity
from decider.inference import Prediction, split_predictions
from decider.model.decider import StateCache, readout_features
from decider.packing import Batch, Example, RequestEncoder, collate
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
    # Dynamic batching (see decider.serving.batching); 1 disables it and keeps the state cache.
    max_batch_size: int = 1
    max_wait_ms: float = 5.0


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
    unfamiliar: list[str] = field(default_factory=list)  # flagged by the familiarity detector


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
        # Optional: a bundle carries a detector once `decider-familiarity --save` has fitted
        # one (training does it automatically). Without it, no question is ever flagged.
        familiarity_path = Path(cfg.bundle) / "familiarity.npz"
        self.familiarity = Familiarity.load(familiarity_path) if familiarity_path.exists() else None
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
            "familiarity": self.familiarity is not None,
        }

    def _autocast(self) -> torch.autocast:
        cuda = self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=cuda)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        tokenized = self.encoder.encode(request)
        batch = collate(
            [Example(tokenized)],
            self.bundle.limits.max_seq_len,
            pack=False,
            pad_id=self.bundle.tokenizer.pad_id,
        ).to(self.device)
        key = LRUStateCache.key(tokenized.state)
        model = self.bundle.model
        with self._lock, torch.inference_mode(), self._autocast():
            start = time.perf_counter()
            cache = self.cache.get(key)
            hit = cache is not None
            if cache is None:
                flat, index, cache = model.encode_and_cache(batch, len(tokenized.state))
                self.cache.put(key, cache)
            else:
                flat, index = model.encode_cached(cache, batch)
            logits = model.heads(flat, index, batch.option_mask)
            probs = model.probabilities(logits)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - start) * 1000
            unfamiliar = self._unfamiliar(batch, flat, index, 1)[0]
        predictions = split_predictions(batch, logits, probs, 1)[0]
        m.CACHE_EVENTS.labels(result="hit" if hit else "miss").inc()
        m.CACHE_BYTES.set(self.cache.bytes)
        m.MODEL_LATENCY.labels(cache="hit" if hit else "miss").observe(model_ms / 1000)
        state_tokens = len(tokenized.state)
        return self._result(request, predictions, state_tokens, model_ms, hit, unfamiliar)

    def decide_many(self, requests: list[DecisionRequest]) -> list[DecisionResult | Exception]:
        """Answer several requests with one packed forward pass.

        Requests share rows (the masks keep them apart), so a batch costs about as much as its
        total token count. Each request either gets a result or the exception that rejected
        it, for example a request too long for the model. The state cache is not used.
        """
        outcomes: list[DecisionResult | Exception | None] = [None] * len(requests)
        accepted: list[tuple[int, Example]] = []
        for i, request in enumerate(requests):
            try:
                accepted.append((i, Example(self.encoder.encode(request))))
            except ValueError as exc:
                outcomes[i] = exc
        if accepted:
            self._run_batch(requests, accepted, outcomes)
        return [o for o in outcomes if o is not None]

    def _run_batch(
        self,
        requests: list[DecisionRequest],
        accepted: list[tuple[int, Example]],
        outcomes: list[DecisionResult | Exception | None],
    ) -> None:
        examples = [ex for _, ex in accepted]
        batch = collate(
            examples, self.bundle.limits.max_seq_len, pad_id=self.bundle.tokenizer.pad_id
        )
        batch = batch.to(self.device)
        model = self.bundle.model
        with self._lock, torch.inference_mode(), self._autocast():
            start = time.perf_counter()
            flat = model.encode(batch)
            logits = model.heads(flat, batch.index, batch.option_mask)
            probs = model.probabilities(logits)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            model_ms = (time.perf_counter() - start) * 1000
            unfamiliar = self._unfamiliar(batch, flat, batch.index, len(examples))
        m.BATCH_SIZE.observe(len(examples))
        m.MODEL_LATENCY.labels(cache="batched").observe(model_ms / 1000)
        per_example = split_predictions(batch, logits, probs, len(examples))
        for j, ((i, example), predictions) in enumerate(zip(accepted, per_example, strict=True)):
            state_tokens = len(example.request.state)
            outcomes[i] = self._result(
                requests[i], predictions, state_tokens, model_ms, False, unfamiliar[j]
            )

    def _unfamiliar(
        self, batch: Batch, flat: torch.Tensor, index: dict[str, torch.Tensor], n: int
    ) -> list[set[str]]:
        """Per example in the batch, the questions the familiarity detector flags."""
        flagged: list[set[str]] = [set() for _ in range(n)]
        if self.familiarity is None:
            return flagged
        features = readout_features(flat, index, batch.option_mask)
        for t, x in features.items():
            readouts = batch.readouts[t]
            if not readouts:
                continue
            flags = self.familiarity.unfamiliar(t, x[: len(readouts)].cpu().numpy())
            for readout, flag in zip(readouts, flags, strict=True):
                if flag:
                    flagged[readout.example].add(readout.question)
        return flagged

    def _result(
        self,
        request: DecisionRequest,
        predictions: dict[str, Prediction],
        state_tokens: int,
        model_ms: float,
        cache_hit: bool,
        unfamiliar: set[str],
    ) -> DecisionResult:
        answers = {name: predictions[name].answer() for name in request.questions}
        m.QUESTIONS.observe(len(request.questions))
        m.STATE_TOKENS.observe(state_tokens)
        review = []
        flagged = [name for name in request.questions if name in unfamiliar]
        for name, answer in answers.items():
            confidence = max(predictions[name].distribution())
            m.CONFIDENCE.labels(type=answer.type).observe(confidence)
            self.drift.observe(answer.type, confidence)
            if answer.type == "noul":
                m.NOUL_YES.labels(answer="yes" if answer.answer else "no").inc()
            if confidence < self.cfg.low_confidence_threshold:
                m.LOW_CONFIDENCE.labels(type=answer.type).inc()
                review.append(name)
            if name in unfamiliar:
                m.UNFAMILIAR.labels(type=answer.type).inc()
        self.drift.export()
        return DecisionResult(answers, cache_hit, state_tokens, model_ms, review, flagged)

    # ------------------------------------------------------------------ logging

    def record(
        self, request_id: str, request: DecisionRequest, result: DecisionResult, latency_ms: float
    ) -> None:
        """Append to the prediction log; queue low-confidence and unfamiliar questions.

        Confidence alone misses questions the model is confidently wrong about, which is what
        unfamiliar questions tend to be (docs/EXPERIMENTS.md, E4).
        """
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
            "low_confidence": result.review,
            "unfamiliar": result.unfamiliar,
        }
        targets = []
        if self.cfg.prediction_log:
            targets.append(("predictions", Path(self.cfg.prediction_log), entry))
        to_label = list(dict.fromkeys(result.review + result.unfamiliar))
        if self.cfg.feedback_queue and to_label:
            queued = {**entry, "review": to_label, "state": redact(request.state)}
            targets.append(("feedback", Path(self.cfg.feedback_queue), queued))
        # A failed log write must never fail the request it describes: it is reported and
        # counted (decider_log_write_errors_total) instead.
        with self._log_lock:
            for sink, path, payload in targets:
                try:
                    _append(path, payload)
                except OSError:
                    m.LOG_WRITE_ERRORS.labels(sink=sink).inc()
                    log.exception("could not write the %s log at %s", sink, path)


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
