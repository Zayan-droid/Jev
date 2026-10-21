"""``decider-eval``: evaluate a bundle (or a baseline) on a built dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from decider.cli._common import configure_logging

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", type=Path, help="model bundle directory")
    src.add_argument("--baseline", choices=["nli", "embedding"], help="evaluate a baseline")
    p.add_argument("--data", type=Path, required=True, help="built dataset directory")
    p.add_argument("--splits", nargs="+", default=["test", "heldout", "gold"])
    p.add_argument("--out", type=Path, help="output directory (default: next to the bundle)")
    p.add_argument(
        "--refit-calibration",
        action="store_true",
        help="refit temperatures on the calib split instead of using the bundle's",
    )
    p.add_argument("--max-examples", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--no-latency", action="store_true")
    p.add_argument(
        "--gate",
        type=Path,
        default=Path("configs/gate/default.yaml"),
        help="release-gate thresholds to check the report against",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    from omegaconf import OmegaConf

    from decider.evaluation.gate import check_gate
    from decider.evaluation.report import EvalConfig

    cfg = EvalConfig(
        splits=args.splits,
        max_examples=args.max_examples,
        robustness=not args.no_robustness and args.bundle is not None,
        latency=not args.no_latency and args.bundle is not None,
    )
    if args.baseline:
        from decider.evaluation.baselines import evaluate_baseline

        out = args.out or Path("runs") / "baselines" / f"{args.baseline}-{args.data.name}"
        report = evaluate_baseline(args.baseline, args.data, cfg, out, args.device)
    else:
        from decider.bundle import load_bundle
        from decider.evaluation.report import evaluate
        from decider.inference import Predictor
        from decider.training.trainer import resolve_device

        device = resolve_device(args.device)
        bundle = load_bundle(args.bundle, device)
        predictor = Predictor(bundle.model, bundle.tokenizer, bundle.limits, device)
        out = args.out or args.bundle.parent / f"eval-{args.data.name}"
        report = evaluate(predictor, args.data, cfg, out, fit_calibration=args.refit_calibration)

    if args.gate.exists() and not args.baseline:
        gate = OmegaConf.to_container(OmegaConf.load(args.gate), resolve=True)
        assert isinstance(gate, dict)
        result = check_gate(report, gate)  # type: ignore[arg-type]
        (out / "gate.json").write_text(json.dumps(result.to_dict(), indent=2))
        log.info(result.summary())
    log.info("report written to %s", out / "report.md")


if __name__ == "__main__":
    main()
