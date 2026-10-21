"""MLflow model registry: versions, gate results and aliases.

Every trained bundle can be registered as a new version of a registered model. Versions that
pass the release gate get the ``challenger`` alias; promotion to ``champion`` is an explicit,
separate step (``decider-register promote``), and clients pin either an alias or a version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from decider.evaluation.gate import check_gate

log = logging.getLogger(__name__)

CHALLENGER, CHAMPION = "challenger", "champion"


def _client(tracking_uri: str | None = None) -> Any:
    import mlflow
    from mlflow import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def register_bundle(
    run_id: str,
    model_name: str,
    report: dict[str, Any],
    gate: dict[str, Any],
    artifact_path: str = "bundle",
    tracking_uri: str | None = None,
) -> tuple[str, bool]:
    """Register the run's bundle; returns ``(version, gate_passed)``."""
    client = _client(tracking_uri)
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(
            model_name, description="Calibrated single-pass decision model"
        )
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/{artifact_path}"
    version = client.create_model_version(model_name, source=source, run_id=run_id)
    result = check_gate(report, gate)
    log.info("%s v%s\n%s", model_name, version.version, result.summary())
    client.set_model_version_tag(model_name, version.version, "gate.passed", str(result.passed))
    for check in result.checks:
        client.set_model_version_tag(
            model_name, version.version, f"gate.{check.name}", f"{check.value}"
        )
    if result.passed:
        client.set_registered_model_alias(model_name, CHALLENGER, version.version)
    return str(version.version), result.passed


def promote(
    model_name: str,
    version: str,
    alias: str = CHAMPION,
    force: bool = False,
    tracking_uri: str | None = None,
) -> None:
    client = _client(tracking_uri)
    mv = client.get_model_version(model_name, version)
    if mv.tags.get("gate.passed") != "True" and not force:
        raise PermissionError(f"{model_name} v{version} did not pass the release gate")
    client.set_registered_model_alias(model_name, alias, version)
    log.info("%s@%s -> v%s", model_name, alias, version)


def download(model_uri: str, dst: Path, tracking_uri: str | None = None) -> Path:
    """Fetch a registered bundle, e.g. ``models:/decider@champion``, into ``dst``."""
    import mlflow

    _client(tracking_uri)
    dst.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst))
    return Path(path)
