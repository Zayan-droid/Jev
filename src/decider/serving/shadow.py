"""Shadow deployment: a candidate model answers live traffic without affecting it.

Every request the live model answers (or a fixed fraction of them, chosen by request id) is also
handed to the shadow model on a background thread. The response never waits for it and never
changes: when the shadow falls behind, requests are dropped (and counted), and a shadow failure
is logged, not raised. Per question, the two answers are compared, counted in Prometheus
(agreement per question type) and appended to a shadow log keyed by request id.

Not waiting is not enough. On a shared GPU the shadow's forward passes slow the live ones, and on
the CPU they take the cores the server needs (docs/EXPERIMENTS.md, E8). So by default the shadow
only runs while no live request is in flight: it uses spare capacity, and under saturation it
drops requests instead of taking capacity from live traffic. A live request that arrives during
a shadow pass still waits for that one pass to finish.

Agreement alone cannot say which model is right. ``decider-shadow`` joins the log with labels
(records whose id was sent as the request id, e.g. by ``decider-loadtest --tag-ids``, or
production requests labelled later) and gives a paired accuracy comparison on the same live
questions: the evidence for promoting the shadow, or for not doing so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from decider.evaluation.metrics import is_correct
from decider.schema import Answer, ChoiceAnswer, DecisionRequest, NoulAnswer, Target
from decider.serving import monitoring as m
from decider.serving.engine import DecisionEngine, EngineConfig

log = logging.getLogger(__name__)


@dataclass
class ShadowConfig:
    bundle: str
    device: str = "auto"
    log: str | None = "logs/shadow.jsonl"
    max_queue: int = 256  # requests waiting for the shadow; beyond this they are dropped
    cache_max_bytes: int = 128 * 2**20
    fraction: float = 1.0  # share of requests shadowed, chosen by a hash of the request id
    idle_only: bool = True  # run only while no live request is in flight


# ---------------------------------------------------------------------------- comparison


def distribution(answer: Answer) -> Target:
    """An answer's distribution in the same shape as a label: p(yes), option -> p, or levels."""
    if isinstance(answer, NoulAnswer):
        return answer.p_yes
    if isinstance(answer, ChoiceAnswer):
        return dict(answer.distribution)
    return list(answer.distribution)


def vector(value: Target, order: list[str] | None = None) -> np.ndarray:
    """Dense distribution: noul as ``[p_no, p_yes]``, choice in ``order``, score as is."""
    if isinstance(value, dict):
        keys = order if order is not None else list(value)
        return np.array([float(value.get(k, 0.0)) for k in keys])
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    return np.array([1.0 - float(value), float(value)])


def compare(primary: Answer, shadow: Answer) -> dict[str, Any]:
    """Same top answer or not, and the total variation distance between the distributions."""
    a, b = distribution(primary), distribution(shadow)
    order = list(a) if isinstance(a, dict) else None
    p, q = vector(a, order), vector(b, order)
    return {
        "type": primary.type,
        "primary": a,
        "shadow": b,
        "agree": bool(p.argmax() == q.argmax()),
        "distance": round(float(0.5 * np.abs(p - q).sum()), 6),
    }


# ---------------------------------------------------------------------------- runner


class ShadowRunner:
    """Answers copies of live requests with a second engine on a background thread."""

    def __init__(
        self,
        engine: DecisionEngine,
        log_path: str | None = None,
        max_queue: int = 256,
        fraction: float = 1.0,
        idle_only: bool = True,
    ) -> None:
        if engine.cfg.emit_metrics:
            raise ValueError("a shadow engine must not emit live metrics (emit_metrics=False)")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.engine = engine
        self.log_path = Path(log_path) if log_path else None
        self.fraction = fraction
        self.idle_only = idle_only
        self._live = 0
        self._idle = threading.Event()
        self._idle.set()
        self._queue: queue.Queue[tuple[str, DecisionRequest, str, dict[str, Answer]]] = queue.Queue(
            maxsize=max_queue
        )
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "agree": 0})
        self._status = {"ok": 0, "error": 0, "dropped": 0}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="decider-shadow", daemon=True)
        self._thread.start()
        m.SHADOW_INFO.labels(version=self.version).set(1)
        log.info("shadowing with %s", self.version)

    @classmethod
    def from_config(cls, cfg: ShadowConfig) -> ShadowRunner:
        engine = DecisionEngine(
            EngineConfig(
                bundle=cfg.bundle,
                device=cfg.device,
                cache_max_bytes=cfg.cache_max_bytes,
                prediction_log=None,
                feedback_queue=None,
                emit_metrics=False,
            )
        )
        return cls(engine, cfg.log, cfg.max_queue, cfg.fraction, cfg.idle_only)

    @property
    def version(self) -> str:
        return self.engine.version

    @contextmanager
    def live(self) -> Iterator[None]:
        """Wrap each live forward pass: the shadow waits while any is in flight."""
        with self._lock:
            self._live += 1
            self._idle.clear()
        try:
            yield
        finally:
            with self._lock:
                self._live -= 1
                if self._live == 0:
                    self._idle.set()

    def sampled(self, request_id: str) -> bool:
        """Deterministic in the request id, so replayed traffic shadows the same requests."""
        if self.fraction >= 1.0:
            return True
        digest = hashlib.sha256(request_id.encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.fraction

    def submit(
        self,
        request_id: str,
        request: DecisionRequest,
        primary_version: str,
        primary: dict[str, Answer],
    ) -> bool:
        """Queue a request the live model answered; never blocks. False if it was not queued:
        outside the sampled fraction (not counted), or dropped because the queue is full."""
        if not self.sampled(request_id):
            return False
        try:
            self._queue.put_nowait((request_id, request, primary_version, primary))
        except queue.Full:
            self._count("dropped")
            return False
        return True

    def wait(self) -> None:
        """Block until every queued request has been answered (for tests and shutdown)."""
        self._queue.join()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            types = {
                t: {**c, "agreement": c["agree"] / c["n"] if c["n"] else None}
                for t, c in sorted(self._counts.items())
            }
            return {"version": self.version, "requests": dict(self._status), "types": types}

    def _count(self, status: str) -> None:
        m.SHADOW_REQUESTS.labels(status=status).inc()
        with self._lock:
            self._status[status] += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.idle_only:
                    while not self._idle.wait(0.05) and not self._stop.is_set():
                        pass
                self._handle(*item)
            except Exception:  # a shadow failure must never reach the live path
                log.exception("shadow model failed on a request")
                self._count("error")
            finally:
                self._queue.task_done()

    def _handle(
        self,
        request_id: str,
        request: DecisionRequest,
        primary_version: str,
        primary: dict[str, Answer],
    ) -> None:
        result = self.engine.decide(request)
        m.SHADOW_LATENCY.observe(result.model_ms / 1000)
        questions = {name: compare(primary[name], result.answers[name]) for name in primary}
        with self._lock:
            for c in questions.values():
                counts = self._counts[c["type"]]
                counts["n"] += 1
                counts["agree"] += c["agree"]
        for c in questions.values():
            m.SHADOW_ANSWERS.labels(type=c["type"], agree="yes" if c["agree"] else "no").inc()
            m.SHADOW_DISTANCE.labels(type=c["type"]).observe(c["distance"])
        self._count("ok")
        if self.log_path is None:
            return
        entry = {
            "request_id": request_id,
            "time": time.time(),
            "primary_version": primary_version,
            "shadow_version": self.version,
            "shadow_ms": round(result.model_ms, 3),
            "questions": questions,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            m.LOG_WRITE_ERRORS.labels(sink="shadow").inc()
            log.exception("could not write the shadow log at %s", self.log_path)


# ---------------------------------------------------------------------------- offline report


def agreement(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Questions, agreement and mean distance per question type, and the versions compared."""
    per_type: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    versions: set[tuple[str, str]] = set()
    requests = 0
    for e in entries:
        requests += 1
        versions.add((e["primary_version"], e["shadow_version"]))
        for q in e["questions"].values():
            per_type[q["type"]].append((bool(q["agree"]), float(q["distance"])))
    types = {
        t: {
            "n": len(rows),
            "agreement": float(np.mean([a for a, _ in rows])),
            "mean_distance": float(np.mean([d for _, d in rows])),
        }
        for t, rows in sorted(per_type.items())
    }
    return {"requests": requests, "versions": sorted(versions), "types": types}


def labelled_outcomes(
    entries: Iterable[dict[str, Any]], labels: dict[str, dict[str, Target]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Per labelled question, whether each model was right, keyed like ``decider-compare`` rows.

    A request answered twice keeps its last answers: replayed traffic repeats records.
    """
    live: dict[tuple[str, str], dict[str, Any]] = {}
    shadow: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entries:
        targets = labels.get(e["request_id"])
        if targets is None:
            continue
        for name, q in e["questions"].items():
            if name not in targets:
                continue
            order = list(q["primary"]) if isinstance(q["primary"], dict) else None
            t = vector(targets[name], order)
            key = (e["request_id"], name)
            common = {"type": q["type"], "agree": bool(q["agree"])}
            live[key] = {**common, "correct": is_correct(vector(q["primary"], order), t)}
            shadow[key] = {**common, "correct": is_correct(vector(q["shadow"], order), t)}
    return live, shadow


def disagreements(
    live: dict[tuple[str, str], dict[str, Any]], shadow: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, int]:
    """Who was right on the labelled questions where the two models' answers differ."""
    out = {"live_right": 0, "shadow_right": 0, "both_right": 0, "both_wrong": 0}
    for key, a in live.items():
        if a["agree"]:
            continue
        b = shadow[key]
        if a["correct"] and b["correct"]:  # possible when the label is a tie
            out["both_right"] += 1
        elif a["correct"]:
            out["live_right"] += 1
        elif b["correct"]:
            out["shadow_right"] += 1
        else:
            out["both_wrong"] += 1
    return out


def render_agreement(report: dict[str, Any]) -> str:
    pairs = ", ".join(f"{a} (live) vs {b} (shadow)" for a, b in report["versions"])
    lines = [
        f"{report['requests']} requests: {pairs}",
        "",
        "| type | questions | same answer | mean distance |",
        "|---|---|---|---|",
    ]
    for t, r in report["types"].items():
        lines.append(f"| {t} | {r['n']} | {r['agreement']:.1%} | {r['mean_distance']:.3f} |")
    return "\n".join(lines)
