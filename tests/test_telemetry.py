import json
import socket
import urllib.request
from pathlib import Path

import pytest
import torch

from decider.training.health import (
    NonFiniteLossError,
    SpikeDetector,
    UpdateRatioTracker,
    check_finite,
)
from decider.training.telemetry import Telemetry, TelemetryConfig, flatten
from tests.conftest import make_model


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_metrics_go_to_jsonl_and_prometheus(tmp_path: Path) -> None:
    port = _free_port()
    telemetry = Telemetry(TelemetryConfig(mlflow=False, prometheus_port=port), tmp_path, "run-a")
    telemetry.log_metrics({"train/loss": 0.5, "val/ece": 0.02, "skip": float("nan")}, step=7)
    telemetry.finish()

    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert record["step"] == 7 and record["train/loss"] == 0.5 and "skip" not in record
    scraped = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
    assert 'decider_train_loss{run="run-a"} 0.5' in scraped
    assert 'decider_step{run="run-a"} 7.0' in scraped


def test_flatten_params() -> None:
    flat = flatten({"model": {"encoder": {"d_model": 64}}, "betas": [0.9, 0.98]})
    assert flat == {"model.encoder.d_model": 64, "betas": "[0.9, 0.98]"}


def test_spike_detector() -> None:
    detector = SpikeDetector(warmup=20)
    torch.manual_seed(0)
    for v in (1.0 + 0.01 * torch.randn(100)).tolist():
        assert not detector.update(v)
    assert detector.update(5.0)
    assert detector.spikes == 1
    assert detector.mean == pytest.approx(1.0, abs=0.05), "a spike must not move the baseline"


def test_non_finite_loss_stops_training() -> None:
    with pytest.raises(NonFiniteLossError):
        check_finite(torch.tensor(float("nan")), step=3)


def test_update_ratio_tracker() -> None:
    model = make_model()
    tracker = UpdateRatioTracker.for_model(model)
    assert any("heads" in n for n in tracker.names) and len(tracker.names) >= 4
    tracker.snapshot(model)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(1.01)
    ratios = tracker.ratios(model)
    assert ratios and all(r == pytest.approx(0.01, rel=1e-3) for r in ratios.values())
    assert tracker.ratios(model) == {}, "ratios are consumed once per snapshot"
