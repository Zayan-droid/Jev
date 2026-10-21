"""``decider-bench``: latency and memory of a bundle across request sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decider.cli._common import configure_logging


def _cell(value: object) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


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
    table = Table(title=f"{bundle.version} on {device}, milliseconds")
    columns = {
        "state_tokens": "state tokens",
        "questions": "questions",
        "full_p50_ms": "full p50",
        "full_p95_ms": "full p95",
        "state_encode_p50_ms": "encode state",
        "cached_p50_ms": "cached p50",
        "peak_memory_mb": "peak MB",
    }
    for title in columns.values():
        table.add_column(title, justify="right")
    for row in rows:
        table.add_row(*(_cell(row[key]) for key in columns))
    Console().print(table)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
