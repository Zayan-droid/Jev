"""FastAPI application.

Endpoints:

* ``POST /v1/decide`` - answer the questions of one request
* ``GET  /v1/model``  - model version, calibration and evaluation summary
* ``GET  /v1/drift``  - live confidence drift against the training-time reference
* ``GET  /healthz``   - liveness; ``GET /readyz`` - readiness (model loaded)
* ``GET  /metrics``   - Prometheus
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import make_asgi_app

from decider.schema import DecisionRequest, DecisionResponse
from decider.serving import monitoring as m
from decider.serving.batching import MicroBatcher
from decider.serving.engine import DecisionEngine, EngineConfig

log = logging.getLogger("decider.access")


def create_app(cfg: EngineConfig, engine: DecisionEngine | None = None) -> FastAPI:
    state: dict[str, Any] = {"engine": engine, "batcher": None}
    batcher_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state["engine"] is None:
            state["engine"] = DecisionEngine(cfg)
        yield
        if state["batcher"] is not None:
            state["batcher"].close()

    app = FastAPI(title="decider", version="0.1.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    def get_engine() -> DecisionEngine:
        engine = state["engine"]
        if engine is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return engine

    def get_batcher() -> MicroBatcher | None:
        if cfg.max_batch_size <= 1:
            return None
        with batcher_lock:
            if state["batcher"] is None:
                state["batcher"] = MicroBatcher(get_engine(), cfg.max_batch_size, cfg.max_wait_ms)
            return state["batcher"]

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        if request.url.path != "/metrics":
            log.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round(elapsed, 2),
                    }
                )
            )
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        get_engine()
        return {"status": "ready"}

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        return get_engine().info()

    @app.get("/v1/drift")
    def drift() -> dict[str, Any]:
        return get_engine().drift.export()

    # A plain `def` endpoint runs in FastAPI's thread pool, keeping the event loop free while
    # the model runs. Requests either go through the micro-batcher or straight to the engine,
    # which serialises access to the device.
    @app.post("/v1/decide", response_model=DecisionResponse)
    def decide(body: DecisionRequest, request: Request) -> DecisionResponse:
        engine = get_engine()
        batcher = get_batcher()
        start = time.perf_counter()
        try:
            result = batcher.submit(body) if batcher is not None else engine.decide(body)
        except ValueError as exc:
            m.REQUESTS.labels(status="rejected").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            m.REQUESTS.labels(status="error").inc()
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        m.LATENCY.observe(latency_ms / 1000)
        m.REQUESTS.labels(status="ok").inc()
        request_id = request.state.request_id
        engine.record(request_id, body, result, latency_ms)
        return DecisionResponse(
            answers=result.answers,
            model_version=engine.version,
            request_id=request_id,
            latency_ms=round(latency_ms, 3),
            cache_hit=result.cache_hit,
            review=result.review,
            unfamiliar=result.unfamiliar,
        )

    return app
