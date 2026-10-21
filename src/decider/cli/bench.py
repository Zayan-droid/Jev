"""``decider-bench``: latency and memory of a bundle across request sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decider.cli._common import configure_logging


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--state-tokens", type=int, nargs="+", default=[128, 512])
    p.add_argument("--questions", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.table import Table

    from decider.bundle import load_bundle
    from decider.evaluation.latency import benchmark
    from decider.training.trainer import resolve_device

    device = resolve_device(args.device)
    bundle = load_bundle(args.bundle, device)
    rows = benchmark(
        bundle.model,
        bundle.tokenizer,
        bundle.limits,
        device,
        args.state_tokens,
        args.questions,
        repeats=args.repeats,
    )
    table = Table(title=f"{bundle.version} on {device}")
    columns = [
        "state_tokens",
        "questions",
        "full_p50_ms",
        "full_p95_ms",
        "state_encode_p50_ms",
        "cached_p50_ms",
        "peak_memory_mb",
    ]
    for c in columns:
        table.add_column(c, justify="right")
    for row in rows:
        table.add_row(
            *[f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        )
    Console().print(table)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
