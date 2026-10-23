"""End-to-end: build data -> train -> resume -> evaluate -> bundle -> serve."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from decider.bundle import load_bundle
from decider.data.build import BuildConfig, build_dataset
from decider.data.dataset import TrainBatches
from decider.evaluation.familiarity import Familiarity
from decider.evaluation.report import EvalConfig
from decider.model import EncoderConfig, ModelConfig
from decider.packing import PackingLimits
from decider.schema import DecisionRequest
from decider.serving.analysis import analyse, read_log, render
from decider.serving.app import create_app
from decider.serving.engine import DecisionEngine, EngineConfig
from decider.training.telemetry import TelemetryConfig
from decider.training.trainer import RunConfig, TrainConfig, Trainer


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    cfg = BuildConfig.from_dict(
        {
            "name": "pipe",
            "output_root": str(root),
            "synthetic": {
                "families": ["inventory", "arithmetic", "dates"],
                "heldout_families": ["dates"],
            },
            "synthetic_sizes": {"train": 300, "val": 40, "calib": 60, "test": 60, "heldout": 30},
        }
    )
    return build_dataset(cfg)


def run_config(run_dir: Path, data_dir: Path, steps: int) -> RunConfig:
    return RunConfig(
        name="pipe",
        run_dir=str(run_dir),
        data_dir=str(data_dir),
        model=ModelConfig(
            encoder=EncoderConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64),
            cross_option_layers=1,
            head_dropout=0.0,
        ),
        packing=PackingLimits(max_seq_len=768, max_state_tokens=384),
        train=TrainConfig(
            max_steps=steps,
            max_batch_tokens=2048,
            warmup_steps=2,
            eval_every=3,
            eval_examples=20,
            checkpoint_every=3,
            log_every=1,
            update_ratio_every=2,
            device="cpu",
        ),
        telemetry=TelemetryConfig(mlflow=False, system_metrics=False),
        evaluation=EvalConfig(
            bootstrap=20, robustness_examples=10, latency_state_tokens=[64], latency_questions=[1]
        ),
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> Path:
    run_dir = tmp_path_factory.mktemp("run")
    Trainer(run_config(run_dir, data_dir, steps=6)).fit()
    return run_dir


def test_training_produces_report_and_bundle(trained: Path) -> None:
    assert (trained / "eval" / "report.json").exists()
    assert (trained / "eval" / "figures" / "reliability_test_calibrated.png").exists()
    assert (trained / "metrics.jsonl").read_text().count("\n") >= 6
    bundle = load_bundle(trained / "bundle")
    assert set(bundle.model.temperatures()) == {"noul", "choice", "score"}
    assert (trained / "bundle" / "reference.json").exists()
    detector = Familiarity.load(trained / "bundle" / "familiarity.npz")
    assert set(detector.threshold) == {"noul", "choice", "score"}


def test_resume_after_interruption_is_exact(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    straight = tmp_path / "straight"
    Trainer(run_config(straight, data_dir, steps=6)).fit()

    interrupted = tmp_path / "interrupted"
    original = TrainBatches.__next__
    calls = {"n": 0}

    def interrupt_on_fourth_batch(self: TrainBatches) -> tuple:
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt
        return original(self)

    monkeypatch.setattr(TrainBatches, "__next__", interrupt_on_fourth_batch)
    with pytest.raises(KeyboardInterrupt):
        Trainer(run_config(interrupted, data_dir, steps=6)).fit()
    monkeypatch.setattr(TrainBatches, "__next__", original)
    Trainer(run_config(interrupted, data_dir, steps=6)).fit()

    a = torch.load(straight / "checkpoints" / "last.pt", weights_only=False)
    b = torch.load(interrupted / "checkpoints" / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)


def test_stop_at_step_keeps_the_schedule(tmp_path: Path, data_dir: Path) -> None:
    full = tmp_path / "full"
    Trainer(run_config(full, data_dir, steps=6)).fit()
    cfg = run_config(tmp_path / "stopped", data_dir, steps=6)
    cfg.train.stop_at_step = 3
    Trainer(cfg).fit()
    a = torch.load(full / "checkpoints" / "step_00000003.pt", weights_only=False)
    b = torch.load(tmp_path / "stopped" / "checkpoints" / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == 3
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
    assert b["scheduler"] == a["scheduler"], "the LR schedule follows max_steps, not the stop"


def test_api_serves_the_bundle(trained: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        bundle=str(trained / "bundle"),
        device="cpu",
        prediction_log=str(tmp_path / "predictions.jsonl"),
        feedback_queue=str(tmp_path / "queue.jsonl"),
        low_confidence_threshold=1.01,  # queue everything
        drift_min_samples=1,
    )
    engine = DecisionEngine(cfg)
    client = TestClient(create_app(cfg, engine))
    body = {
        "state": {"items": [{"name": "lamp", "color": "red", "qty": 2}], "email": "a@b.com"},
        "questions": {
            "red": {"type": "noul", "instructions": "Is there a red item?"},
            "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": "y"}},
            "level": {"type": "score", "instructions": "How many?", "levels": ["0", "1", "2"]},
        },
    }
    first = client.post("/v1/decide", json=body)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert set(payload["answers"]) == {"red", "pick", "level"}
    assert payload["cache_hit"] is False
    assert sum(payload["answers"]["pick"]["distribution"].values()) == pytest.approx(1.0)
    assert client.post("/v1/decide", json=body).json()["cache_hit"] is True

    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/v1/model").json()["version"]
    metrics = client.get("/metrics/").text
    assert "decider_requests_total" in metrics and "decider_confidence_bucket" in metrics

    log_text = (tmp_path / "predictions.jsonl").read_text()
    assert "a@b.com" not in log_text and "<EMAIL>" in log_text
    assert (tmp_path / "queue.jsonl").exists()

    bad = client.post("/v1/decide", json={"state": "x", "questions": {}})
    assert bad.status_code == 422

    reference = json.loads((trained / "bundle" / "reference.json").read_text())["histograms"]
    report = analyse(read_log(tmp_path / "predictions.jsonl"), reference)
    (day,) = report["days"].values()
    assert day["requests"] == 2 and day["cache_hit_rate"] == 0.5
    assert {"noul", "choice", "score"} <= set(day["types"])
    assert render(report).startswith("| day |")


def test_log_write_failures_do_not_fail_requests(trained: Path, tmp_path: Path) -> None:
    cfg = EngineConfig(
        bundle=str(trained / "bundle"),
        device="cpu",
        prediction_log=str(tmp_path),  # a directory: every write fails
        feedback_queue=None,
    )
    client = TestClient(create_app(cfg, DecisionEngine(cfg)))
    body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it x?"}}}
    assert client.post("/v1/decide", json=body).status_code == 200
    metrics = client.get("/metrics/").text
    assert 'decider_log_write_errors_total{sink="predictions"} 1.0' in metrics


def test_python_client_against_a_live_server(trained: Path, tmp_path: Path) -> None:
    import socket
    import threading

    import uvicorn

    from decider.client import DeciderClient, DeciderError, choice, noul, score

    cfg = EngineConfig(
        bundle=str(trained / "bundle"), device="cpu", prediction_log=None, feedback_queue=None
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(cfg), port=port, log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    client = DeciderClient(f"http://127.0.0.1:{port}")
    try:
        for _ in range(100):
            if client.ready():
                break
            threading.Event().wait(0.1)
        result = client.decide(
            {"ticket": "parcel arrived broken"},
            refund=noul("Does `ticket` ask for a refund?"),
            team=choice("Which team?", billing="payments", returns="damaged items"),
            urgency=score("How urgent?", ["low", "medium", "high"]),
        )
        assert result.answers["team"].type == "choice"
        assert len(result.answers["urgency"].distribution) == 3
        assert client.model()["version"] == result.model_version
        with pytest.raises(DeciderError) as err:
            client._request("POST", "/v1/decide", {"state": "x", "questions": {}})
        assert err.value.status == 422
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _requests() -> list[DecisionRequest]:
    base = {
        "red": {"type": "noul", "instructions": "Is there a red item?"},
        "pick": {"type": "choice", "instructions": "Which?", "criteria": {"a": "x", "b": "y"}},
        "level": {"type": "score", "instructions": "How many?", "levels": ["0", "1", "2"]},
    }
    too_long = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {f"o{i}": "x" * 60 for i in range(40)},
    }
    return [
        DecisionRequest(state={"items": [{"color": c, "qty": q}]}, questions=base)
        for c, q in (("red", 1), ("blue", 7), ("green", 3))
    ] + [DecisionRequest(state="x", questions={"big": too_long})]


def test_batched_answers_match_single_requests(trained: Path) -> None:
    cfg = EngineConfig(
        bundle=str(trained / "bundle"), device="cpu", prediction_log=None, feedback_queue=None
    )
    engine = DecisionEngine(cfg)
    requests = _requests()
    batched = engine.decide_many(requests)
    assert isinstance(batched[-1], ValueError), "an oversized request fails on its own"
    for request, outcome in zip(requests[:-1], batched[:-1], strict=True):
        single = engine.decide(request)
        assert not isinstance(outcome, Exception)
        for name, answer in single.answers.items():
            a, b = answer.model_dump(), outcome.answers[name].model_dump()
            key = {"noul": "p_yes", "choice": "distribution", "score": "distribution"}[answer.type]
            assert a[key] == pytest.approx(b[key], abs=1e-5)


def test_unfamiliar_questions_are_flagged_and_queued(trained: Path, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(trained / "bundle", bundle)
    detector = Familiarity.load(bundle / "familiarity.npz")
    # Flag every noul question and nothing else, whatever the features.
    detector.threshold = {"noul": -1.0, "choice": float("inf"), "score": float("inf")}
    detector.save(bundle / "familiarity.npz")
    cfg = EngineConfig(
        bundle=str(bundle),
        device="cpu",
        prediction_log=None,
        feedback_queue=str(tmp_path / "queue.jsonl"),
        low_confidence_threshold=0.0,  # nothing is low-confidence
    )
    engine = DecisionEngine(cfg)
    request = _requests()[0]
    miss, hit = engine.decide(request), engine.decide(request)
    assert (miss.cache_hit, hit.cache_hit) == (False, True)
    batched = engine.decide_many([request])[0]
    assert not isinstance(batched, Exception)
    for result in (miss, hit, batched):
        assert result.unfamiliar == ["red"]
        assert result.review == []
    engine.record("r1", request, miss, 1.0)
    queued = json.loads((tmp_path / "queue.jsonl").read_text())
    assert queued["review"] == ["red"], "unfamiliar questions are queued for labelling"
    assert queued["unfamiliar"] == ["red"]


def test_shadow_model_sees_live_traffic_without_changing_it(trained: Path, tmp_path: Path) -> None:
    from prometheus_client import REGISTRY

    from decider.serving.shadow import ShadowConfig, ShadowRunner

    cfg = EngineConfig(
        bundle=str(trained / "bundle"), device="cpu", prediction_log=None, feedback_queue=None
    )
    log_path = tmp_path / "shadow.jsonl"
    # The same bundle in shadow: every answer must agree, which tests the plumbing end to end.
    shadow = ShadowRunner.from_config(
        ShadowConfig(bundle=str(trained / "bundle"), device="cpu", log=str(log_path))
    )
    plain = TestClient(create_app(cfg, DecisionEngine(cfg)))
    shadowed = TestClient(create_app(cfg, DecisionEngine(cfg), shadow=shadow))
    observed = "decider_questions_per_request_count"
    before = REGISTRY.get_sample_value(observed) or 0.0
    try:
        for i, request in enumerate(_requests()[:3]):
            body = request.model_dump(mode="json")
            a = plain.post("/v1/decide", json=body)
            b = shadowed.post("/v1/decide", json=body, headers={"x-request-id": f"r{i}"})
            assert a.json()["answers"] == b.json()["answers"], "the shadow never changes answers"
        shadow.wait()
    finally:
        shadow.close()
    assert REGISTRY.get_sample_value(observed) - before == 6, "the shadow emits no live metrics"

    summary = shadowed.get("/v1/shadow").json()
    assert summary["requests"] == {"ok": 3, "error": 0, "dropped": 0}
    assert {t: v["agreement"] for t, v in summary["types"].items()} == {
        "choice": 1.0,
        "noul": 1.0,
        "score": 1.0,
    }
    assert shadowed.get("/v1/model").json()["shadow"] == shadow.version
    assert plain.get("/v1/shadow").status_code == 404
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [e["request_id"] for e in entries] == ["r0", "r1", "r2"]
    assert all(q["distance"] == 0.0 for e in entries for q in e["questions"].values())


def test_micro_batcher_groups_concurrent_requests(trained: Path) -> None:
    import threading

    from decider.serving.batching import MicroBatcher

    cfg = EngineConfig(
        bundle=str(trained / "bundle"), device="cpu", prediction_log=None, feedback_queue=None
    )
    engine = DecisionEngine(cfg)
    sizes: list[int] = []
    original = engine.decide_many

    def spy(requests: list[DecisionRequest]) -> list:
        sizes.append(len(requests))
        return original(requests)

    engine.decide_many = spy  # type: ignore[method-assign]
    batcher = MicroBatcher(engine, max_batch_size=8, max_wait_ms=300)
    requests = _requests()[:3] * 3
    results: list = [None] * len(requests)
    errors: list[Exception] = []

    def call(i: int) -> None:
        try:
            results[i] = batcher.submit(requests[i])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(requests))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with pytest.raises(ValueError):
        batcher.submit(_requests()[-1])
    batcher.close()
    assert not errors and all(r is not None for r in results)
    assert sum(sizes) == len(requests) + 1 and max(sizes) > 1


def test_load_test_against_a_live_server(trained: Path) -> None:
    import socket
    import threading

    import uvicorn

    from decider.cli.loadtest import run_load

    cfg = EngineConfig(
        bundle=str(trained / "bundle"),
        device="cpu",
        prediction_log=None,
        feedback_queue=None,
        max_batch_size=4,
        max_wait_ms=2,
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(cfg), port=port, log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        from decider.client import DeciderClient

        client = DeciderClient(f"http://127.0.0.1:{port}")
        for _ in range(100):
            if client.ready():
                break
            threading.Event().wait(0.1)
        result = run_load(f"http://127.0.0.1:{port}", _requests()[:3] * 4, concurrency=3, warmup=1)
        assert result["errors"] == 0 and result["requests"] == 12
        assert result["requests_per_s"] > 0 and result["p50_ms"] <= result["p99_ms"]
        paced = run_load(
            f"http://127.0.0.1:{port}", _requests()[:3] * 4, concurrency=3, warmup=0, rate=20
        )
        assert paced["errors"] == 0 and paced["seconds"] >= 11 / 20, "open loop keeps the schedule"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
