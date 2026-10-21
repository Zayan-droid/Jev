"""End-to-end: build data -> train -> resume -> evaluate -> bundle -> serve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from decider.bundle import load_bundle
from decider.data.build import BuildConfig, build_dataset
from decider.data.dataset import TrainBatches
from decider.evaluation.report import EvalConfig
from decider.model import EncoderConfig, ModelConfig
from decider.packing import PackingLimits
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
