"""Release gate, model registry and production-monitoring building blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from decider.evaluation.gate import check_gate
from decider.model.decider import StateCache
from decider.serving.engine import LRUStateCache
from decider.serving.monitoring import DriftMonitor, psi
from decider.serving.redaction import redact

GATE = {
    "split": "test",
    "max_ece": 0.05,
    "min_accuracy": 0.5,
    "max_permutation_diff": 0.005,
    "max_independence_diff": 0.005,
    "max_full_p95_ms": 100.0,
}


def report(ece: float = 0.02, permutation: float | None = 0.0) -> dict:
    split = {
        "calibrated": {
            "overall": {"accuracy": 0.8},
            "by_type": {"noul": {"ece": ece}, "choice": {"ece": 0.01}},
        },
        "independence": {"n": 10, "max_abs_diff": 0.0},
    }
    if permutation is not None:
        split["permutation"] = {"n": 10, "max_abs_diff": permutation}
    return {"splits": {"test": split}, "latency": [{"full_p95_ms": 20.0}]}


def test_gate_passes_and_fails_on_thresholds() -> None:
    assert check_gate(report(), GATE).passed
    failed = check_gate(report(ece=0.2), GATE)
    assert not failed.passed
    assert [c.name for c in failed.checks if not c.passed] == ["ece.noul"]


def test_gate_fails_when_a_metric_is_missing() -> None:
    result = check_gate(report(permutation=None), GATE)
    assert not result.passed
    assert "missing" in result.summary()


def test_registry_roundtrip(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from decider import registry

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment("t", artifact_location=(tmp_path / "a").as_uri())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.log_artifacts(str(bundle), "bundle")

    version, passed = registry.register_bundle(
        run.info.run_id, "m", report(), GATE, tracking_uri=uri
    )
    assert passed
    client = mlflow.MlflowClient()
    assert str(client.get_model_version_by_alias("m", registry.CHALLENGER).version) == version
    registry.promote("m", version, tracking_uri=uri)
    assert str(client.get_model_version_by_alias("m", registry.CHAMPION).version) == version

    bad, passed = registry.register_bundle(
        run.info.run_id, "m", report(ece=0.5), GATE, tracking_uri=uri
    )
    assert not passed
    with pytest.raises(PermissionError):
        registry.promote("m", bad, tracking_uri=uri)
    downloaded = registry.download("models:/m@champion", tmp_path / "dl", uri)
    assert (downloaded / "config.json").exists()


def test_redaction() -> None:
    state = {
        "note": "mail jane.doe@example.com or call +1 (555) 010-9999",
        "card": "4111 1111 1111 1111",
        "nested": ["from 10.0.0.1"],
        "qty": 3,
    }
    out = redact(state)
    text = str(out)
    for secret in ("jane.doe", "4111", "555", "10.0.0.1"):
        assert secret not in text
    assert out["qty"] == 3 and "<EMAIL>" in out["note"]


def test_psi_and_drift_monitor() -> None:
    reference = np.full(10, 0.1)
    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    shifted = np.array([0.0] * 8 + [0.5, 0.5])
    assert psi(reference, shifted) > 0.25

    monitor = DriftMonitor({"noul": reference.tolist()}, window=500, min_samples=100)
    rng = np.random.default_rng(0)
    for c in rng.uniform(0, 1, 500):
        monitor.observe("noul", float(c))
    assert monitor.current()["noul"]["psi"] < 0.1
    for c in rng.uniform(0.9, 1.0, 500):
        monitor.observe("noul", float(c))
    assert monitor.current()["noul"]["psi"] > 0.25


def test_state_cache_respects_byte_budget() -> None:
    def entry() -> StateCache:
        k = torch.zeros(1, 1, 10, 4)  # 160 bytes each for k and v
        return StateCache(kv=[(k, k.clone())], length=10)

    cache = LRUStateCache(max_bytes=700)
    for key in "abc":
        cache.put(key, entry())
    assert len(cache) == 2 and cache.get("a") is None
    assert cache.get("b") is not None
    cache.put("d", entry())  # evicts "c", the least recently used
    assert cache.get("c") is None and cache.get("b") is not None
    assert cache.bytes <= 700
