"""Where training signals go.

Every metric is written to all enabled sinks:

* ``metrics.jsonl`` in the run directory - always on, the offline source of truth;
* MLflow - params, tags, metrics, system metrics (GPU/CPU/memory) and artifacts;
* a Prometheus exporter - live gauges scraped by the Grafana training dashboard;
* the console - a progress bar plus a table at every evaluation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    mlflow: bool = True
    tracking_uri: str = "sqlite:///mlflow/mlflow.db"
    experiment: str = "decider"
    artifact_root: str = "mlflow/artifacts"
    system_metrics: bool = True
    system_metrics_interval_s: int = 10
    prometheus_port: int | None = None
    console: bool = True
    tags: dict[str, str] = field(default_factory=dict)


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key))
        elif isinstance(v, list | tuple):
            out[key] = json.dumps(v, default=str)[:500]
        else:
            out[key] = v
    return out


class _Prometheus:
    def __init__(self, port: int) -> None:
        from prometheus_client import CollectorRegistry, Gauge, start_http_server

        self._gauge_cls = Gauge
        self.registry = CollectorRegistry()
        self.gauges: dict[str, Any] = {}
        start_http_server(port, registry=self.registry)
        log.info("prometheus exporter on :%d/metrics", port)

    def set(self, metrics: Mapping[str, float], run: str) -> None:
        for key, value in metrics.items():
            name = "decider_" + re.sub(r"[^a-zA-Z0-9_]", "_", key)
            if name not in self.gauges:
                self.gauges[name] = self._gauge_cls(name, key, ["run"], registry=self.registry)
            self.gauges[name].labels(run=run).set(value)


class Telemetry:
    def __init__(self, cfg: TelemetryConfig, run_dir: Path, run_name: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_name = run_name
        self.run_id: str | None = None
        run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._prom = _Prometheus(cfg.prometheus_port) if cfg.prometheus_port else None
        self._mlflow: Any = None
        self._started = time.time()

    # ---------------------------------------------------------------- lifecycle

    def start(
        self,
        params: Mapping[str, Any],
        tags: Mapping[str, Any],
        resume_run_id: str | None = None,
    ) -> None:
        if not self.cfg.mlflow:
            return
        import mlflow

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.cfg.experiment)
        if experiment is None:
            artifact_root = Path(self.cfg.artifact_root).resolve().as_uri()
            experiment_id = mlflow.create_experiment(
                self.cfg.experiment, artifact_location=artifact_root
            )
        else:
            experiment_id = experiment.experiment_id
        if self.cfg.system_metrics:
            mlflow.set_system_metrics_sampling_interval(self.cfg.system_metrics_interval_s)
        run = mlflow.start_run(
            run_id=resume_run_id,
            experiment_id=None if resume_run_id else experiment_id,
            run_name=None if resume_run_id else self.run_name,
            tags={**self.cfg.tags, **{k: str(v) for k, v in tags.items()}},
            log_system_metrics=self.cfg.system_metrics,
        )
        self._mlflow = mlflow
        self.run_id = run.info.run_id
        if not resume_run_id:
            flat = {k: str(v)[:6000] for k, v in flatten(params).items()}
            mlflow.log_params(flat)
        log.info("mlflow run %s (%s)", self.run_id, self.cfg.tracking_uri)

    def finish(self, status: str = "FINISHED") -> None:
        self._jsonl.close()
        if self._mlflow is not None:
            self._mlflow.end_run(status=status)

    # ---------------------------------------------------------------- logging

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
        record = {"step": step, "time": round(time.time() - self._started, 2), **clean}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean, step=step, synchronous=False)
        if self._prom is not None:
            self._prom.set({**clean, "step": step}, self.run_name)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self._mlflow is None:
            return
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path)
        else:
            self._mlflow.log_artifact(str(path), artifact_path)

    def log_dict(self, data: Mapping[str, Any], artifact_file: str) -> None:
        path = self.run_dir / artifact_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        if self._mlflow is not None:
            self._mlflow.log_dict(dict(data), artifact_file)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        path = self.run_dir / artifact_file
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120, bbox_inches="tight")
        if self._mlflow is not None:
            parent = str(Path(artifact_file).parent)
            self._mlflow.log_artifact(str(path), None if parent == "." else parent)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
