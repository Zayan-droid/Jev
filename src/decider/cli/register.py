"""``decider-register``: manage model versions in the MLflow registry.

decider-register add --run-id <id>             register a finished run's bundle
decider-register promote --version 3           move `champion` to v3 (gate must pass)
decider-register list                          versions, aliases and gate results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

from decider.cli._common import configure_logging

log = logging.getLogger(__name__)

DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tracking-uri", default=DEFAULT_URI)
    p.add_argument("--name", default="decider", help="registered model name")
    sub = p.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--run-id", required=True)
    add.add_argument("--gate", type=Path, default=Path("configs/gate/default.yaml"))
    promote = sub.add_parser("promote")
    promote.add_argument("--version", required=True)
    promote.add_argument("--alias", default="champion")
    promote.add_argument("--force", action="store_true", help="promote despite a failed gate")
    sub.add_parser("list")
    args = p.parse_args(argv)

    from decider import registry

    if args.command == "add":
        import mlflow
        from omegaconf import OmegaConf

        mlflow.set_tracking_uri(args.tracking_uri)
        with tempfile.TemporaryDirectory() as tmp:
            path = mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path="eval/report.json", dst_path=tmp
            )
            report = json.loads(Path(path).read_text())
        gate = OmegaConf.to_container(OmegaConf.load(args.gate), resolve=True)
        assert isinstance(gate, dict)
        thresholds = {str(k): v for k, v in gate.items()}
        version, passed = registry.register_bundle(
            args.run_id, args.name, report, thresholds, tracking_uri=args.tracking_uri
        )
        log.info(
            "registered %s v%s (gate %s)", args.name, version, "passed" if passed else "failed"
        )
    elif args.command == "promote":
        try:
            registry.promote(args.name, args.version, args.alias, args.force, args.tracking_uri)
        except PermissionError as exc:
            raise SystemExit(f"{exc} (use --force to override)") from None
    else:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(args.tracking_uri)
        client = MlflowClient()
        model = client.get_registered_model(args.name)
        aliases = model.aliases
        for mv in client.search_model_versions(f"name='{args.name}'"):
            tags = [a for a, v in aliases.items() if v == mv.version]
            print(
                f"v{mv.version}  gate={mv.tags.get('gate.passed', '?'):5}  "
                f"aliases={','.join(tags) or '-'}  run={mv.run_id}"
            )


if __name__ == "__main__":
    main()
