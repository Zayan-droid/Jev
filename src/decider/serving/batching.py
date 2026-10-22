"""Dynamic micro-batching for concurrent requests.

Each caller hands its request to a worker thread and blocks on a future. The worker waits up to
``max_wait_ms`` for more requests (at most ``max_batch_size``), answers them with one packed
forward pass (``DecisionEngine.decide_many``) and resolves the futures. Under load this trades a
few milliseconds of queueing for far fewer forward passes; with a single client it adds at most
``max_wait_ms``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future

from decider.schema import DecisionRequest
from decider.serving.engine import DecisionEngine, DecisionResult

log = logging.getLogger(__name__)

_Item = tuple[DecisionRequest, "Future[DecisionResult]"]


class MicroBatcher:
    def __init__(
        self, engine: DecisionEngine, max_batch_size: int = 16, max_wait_ms: float = 5.0
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000
        self._queue: queue.Queue[_Item] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="decider-batcher", daemon=True)
        self._thread.start()

    def submit(self, request: DecisionRequest, timeout: float = 30.0) -> DecisionResult:
        future: Future[DecisionResult] = Future()
        self._queue.put((request, future))
        return future.result(timeout=timeout)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _collect(self, first: _Item) -> list[_Item]:
        items = [first]
        deadline = time.monotonic() + self.max_wait
        while len(items) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                items.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return items

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            items = self._collect(first)
            try:
                outcomes = self.engine.decide_many([request for request, _ in items])
            except Exception as exc:  # a failed pass fails every request in it, not the worker
                log.exception("batched forward pass failed")
                for _, future in items:
                    future.set_exception(exc)
                continue
            for (_, future), outcome in zip(items, outcomes, strict=True):
                if isinstance(outcome, Exception):
                    future.set_exception(outcome)
                else:
                    future.set_result(outcome)
