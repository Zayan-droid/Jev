"""``decider-familiarity``: fit and evaluate the question-familiarity detector of a bundle.

    decider-familiarity --bundle runs/<name>/<ts>/bundle --data data/processed/mix-v1 --save

Fits the detector on the calibration split, then reports for each evaluation split how many
questions are flagged unfamiliar, how accurate familiar and unfamiliar questions are, and how
well the score separates in-distribution (test) from held-out questions (AUROC). ``--save``
stores the detector in the bundle (``familiarity.npz``), where the server picks it up.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from decider.cli._common import configure_logging

log = logging.getLogger(__name__)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def render(report: dict[str, Any]) -> list[str]:
    lines = [
        f"Familiarity detector fitted on `{report['fit_split']}` "
        f"({report['quantile']:.0%} quantile of its scores)",
        "",
        "| split | type | n | unfamiliar | accuracy (familiar) | accuracy (unfamiliar) |",
        "|---|---|---|---|---|---|",
    ]
    for split, entry in report["splits"].items():
        for qtype, m in entry.items():
            lines.append(
                f"| {split} | {qtype} | {m['n']} | {_pct(m['unfamiliar_share'])} | "
                f"{_pct(m['accuracy_familiar'])} | {_pct(m['accuracy_unfamiliar'])} |"
            )
    if "auroc_test_vs_heldout" in report:
        values = ", ".join(f"{k} {v:.3f}" for k, v in report["auroc_test_vs_heldout"].items())
        lines += ["", f"AUROC, test (in distribution) vs heldout: {values}"]
    return lines


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fit-split", default="calib")
    p.add_argument("--splits", nargs="+", default=["test", "heldout"])
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--max-examples", type=int, default=6000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, help="default: <bundle>/../familiarity-<data name>")
    p.add_argument("--save", action="store_true", help="write familiarity.npz into the bundle")
    args = p.parse_args(argv)

    import numpy as np

    from decider.bundle import load_bundle
    from decider.data.dataset import EncodedSplit
    from decider.evaluation.familiarity import Familiarity, auroc, collect_features
    from decider.evaluation.report import sample_examples
    from decider.packing import TYPES, RequestEncoder
    from decider.training.trainer import resolve_device

    device = resolve_device(args.device)
    bundle = load_bundle(args.bundle, device)
    encoder = RequestEncoder(bundle.tokenizer, bundle.limits)

    def features(split: str) -> Any:
        examples = sample_examples(
            EncodedSplit.load(args.data, split, encoder), args.max_examples, seed=0
        )
        return collect_features(bundle.model, examples, bundle.limits.max_seq_len, device)

    detector = Familiarity.fit(features(args.fit_split), quantile=args.quantile)
    report: dict[str, Any] = {"fit_split": args.fit_split, "quantile": args.quantile, "splits": {}}
    scores: dict[str, dict[str, np.ndarray]] = {}
    for split in args.splits:
        f = features(split)
        entry: dict[str, Any] = {}
        scores[split] = {}
        for t in TYPES:
            x = f.of_type(t)
            if not len(x):
                continue
            correct = np.array([c for c, ty in zip(f.correct, f.types, strict=True) if ty == t])
            s = detector.score(t, x)
            flagged = s > detector.threshold[t]
            scores[split][t] = s
            entry[t] = {
                "n": len(x),
                "unfamiliar_share": float(flagged.mean()),
                "accuracy_familiar": float(correct[~flagged].mean()) if (~flagged).any() else None,
                "accuracy_unfamiliar": float(correct[flagged].mean()) if flagged.any() else None,
            }
        report["splits"][split] = entry
    if "test" in scores and "heldout" in scores:
        report["auroc_test_vs_heldout"] = {
            t: auroc(scores["test"][t], scores["heldout"][t])
            for t in TYPES
            if t in scores["test"] and t in scores["heldout"]
        }

    lines = render(report)
    markdown = "\n".join(lines)
    print(markdown)

    out = args.out or args.bundle.parent / f"familiarity-{args.data.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(markdown + "\n")
    if args.save:
        detector.save(args.bundle / "familiarity.npz")
        log.info("saved %s", args.bundle / "familiarity.npz")


if __name__ == "__main__":
    main()
