"""``decider-shadow``: compare the live and shadow models from the shadow log.

    decider-shadow --log logs/shadow.jsonl
    decider-shadow --log logs/shadow.jsonl --labels data/processed/mix-v0 --split heldout

Without labels: how often the two models give the same answer, per question type. With labels
(a dataset whose record ids were sent as request ids, e.g. by ``decider-loadtest --tag-ids``):
the paired accuracy difference on the same live questions, and who was right where they differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from decider.cli._common import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--log", type=Path, default=Path("logs/shadow.jsonl"))
    p.add_argument("--labels", type=Path, help="built dataset directory holding the labels")
    p.add_argument("--split", nargs="+", default=["test", "heldout"])
    p.add_argument("--out", type=Path, help="write the report as JSON")
    args = p.parse_args(argv)

    from decider.evaluation.compare import paired_deltas, render
    from decider.serving.analysis import read_log
    from decider.serving.shadow import (
        agreement,
        disagreements,
        labelled_outcomes,
        render_agreement,
    )

    entries = list(read_log(args.log))
    report: dict[str, Any] = {"agreement": agreement(entries)}
    print(render_agreement(report["agreement"]))
    if args.labels:
        from decider.data.io import iter_records, split_path

        labels = {
            r.id: r.targets
            for split in args.split
            for r in iter_records(split_path(args.labels, split))
        }
        live, shadow = labelled_outcomes(entries, labels)
        if not live:
            print("\nNo logged request id matches a labelled record.")
        else:
            deltas = paired_deltas(live, shadow)
            print(f"\nOn {len(live)} labelled questions (B - A = shadow - live):\n")
            print(render(deltas, "correct", names=("live", "shadow")))
            split = disagreements(live, shadow)
            print(
                f"\nWhere the answers differ: live right {split['live_right']}, shadow right "
                f"{split['shadow_right']}, both wrong {split['both_wrong']}."
            )
            report["accuracy"] = [d.__dict__ for d in deltas]
            report["disagreements"] = split
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
