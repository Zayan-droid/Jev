"""``decider-monitor``: offline drift and traffic report over the prediction log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decider.cli._common import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=Path("logs/predictions.jsonl"))
    p.add_argument("--bundle", type=Path, help="bundle whose reference.json is the baseline")
    p.add_argument("--low-confidence", type=float, default=0.6)
    p.add_argument("--out", type=Path, help="write the report as JSON")
    args = p.parse_args(argv)

    from decider.serving.analysis import analyse, read_log, render

    reference = None
    if args.bundle and (args.bundle / "reference.json").exists():
        reference = json.loads((args.bundle / "reference.json").read_text())["histograms"]
    report = analyse(read_log(args.log), reference, args.low_confidence)
    print(render(report))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
