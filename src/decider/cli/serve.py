"""``decider-serve``: serve a bundle (or a registered model) over HTTP."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from decider.cli._common import configure_logging

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--bundle", type=Path, default=os.environ.get("DECIDER_BUNDLE"))
    src.add_argument("--model-uri", help="e.g. models:/decider@champion")
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db"),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--cache-mb", type=int, default=256)
    p.add_argument(
        "--low-confidence",
        type=float,
        default=0.6,
        help=(
            "queue a request for review when an answer's top-label probability (the calibrated "
            "chance it is right) is below this value"
        ),
    )
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument(
        "--feedback-queue",
        type=Path,
        default=Path("data/feedback/queue.jsonl"),
        help="where low-confidence requests are queued for decider-label",
    )
    p.add_argument(
        "--no-state-logging",
        action="store_true",
        help="log only a hash of each state, never its (redacted) content",
    )
    args = p.parse_args(argv)
    if not args.bundle and not args.model_uri:
        p.error("one of --bundle, --model-uri or $DECIDER_BUNDLE is required")

    import uvicorn

    from decider.serving.app import create_app
    from decider.serving.engine import EngineConfig

    bundle = args.bundle
    if args.model_uri:
        from decider.registry import download

        name = args.model_uri.split("/")[-1].replace("@", "-")
        bundle = download(args.model_uri, Path("bundles") / name, args.tracking_uri)
    cfg = EngineConfig(
        bundle=str(bundle),
        device=args.device,
        cache_max_bytes=args.cache_mb * 2**20,
        low_confidence_threshold=args.low_confidence,
        prediction_log=str(args.log_dir / "predictions.jsonl"),
        feedback_queue=str(args.feedback_queue),
        log_states=not args.no_state_logging,
    )
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
