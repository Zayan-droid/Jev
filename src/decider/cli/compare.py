"""``decider-compare``: paired comparison of two evaluation runs on the same questions.

decider-compare runs/a/eval runs/b/eval --split heldout --by family
"""

from __future__ import annotations

import argparse
from pathlib import Path

from decider.cli._common import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("a", type=Path, help="evaluation directory of the baseline model")
    p.add_argument("b", type=Path, help="evaluation directory of the candidate model")
    p.add_argument("--split", default="test")
    p.add_argument("--by", default="type", choices=["type", "family", "source"])
    p.add_argument("--metric", default="correct", choices=["correct", "log_loss"])
    p.add_argument("--names", nargs=2, default=["A", "B"], metavar=("A", "B"))
    p.add_argument("--out", type=Path, help="also write the table to this file")
    args = p.parse_args(argv)

    from decider.evaluation.compare import paired_deltas, read_predictions, render

    a = read_predictions(args.a / f"predictions_{args.split}.jsonl.gz")
    b = read_predictions(args.b / f"predictions_{args.split}.jsonl.gz")
    shared = len(set(a) & set(b))
    if not shared:
        raise SystemExit("the two runs share no questions; evaluate them on the same dataset")
    table = render(paired_deltas(a, b, args.metric, args.by), args.metric, tuple(args.names))
    header = f"{args.split}: {shared} shared questions (A: {len(a)}, B: {len(b)})\n\n"
    print(header + table)
    if args.out:
        args.out.write_text(header + table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
