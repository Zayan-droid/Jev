"""``decider-loadtest``: replay dataset records against a running server.

    decider-loadtest --data data/processed/mix-v1 --split test --concurrency 16 --requests 500

Each worker thread sends requests back to back; the report gives throughput, latency
percentiles (end to end, as the client sees them) and errors. Run it against servers started
with and without ``--max-batch-size`` to measure dynamic batching on your hardware.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from decider.cli._common import configure_logging


def run_load(url: str, requests: list[Any], concurrency: int, warmup: int = 5) -> dict[str, Any]:
    import numpy as np

    from decider.client import DeciderClient, DeciderError

    client = DeciderClient(url, timeout=60)
    for request in requests[:warmup]:
        client.decide(request.state, **request.questions)

    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    cursor = iter(range(len(requests)))

    def worker() -> None:
        while True:
            with lock:
                i = next(cursor, None)
            if i is None:
                return
            request = requests[i]
            start = time.perf_counter()
            try:
                client.decide(request.state, **request.questions)
            except (DeciderError, OSError) as exc:
                with lock:
                    errors.append(type(exc).__name__)
                continue
            with lock:
                latencies.append((time.perf_counter() - start) * 1000)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    lat = np.asarray(latencies) if latencies else np.zeros(1)
    return {
        "concurrency": concurrency,
        "requests": len(requests),
        "errors": len(errors),
        "seconds": round(elapsed, 2),
        "requests_per_s": round(len(latencies) / elapsed, 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
    }


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--data", type=Path, required=True, help="built dataset directory")
    p.add_argument("--split", default="test")
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the results as JSON")
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.table import Table

    from decider.data.io import iter_records, split_path

    records = list(iter_records(split_path(args.data, args.split)))
    rng = random.Random(args.seed)
    sample = [rng.choice(records).request() for _ in range(args.requests)]
    results = [run_load(args.url, sample, c) for c in args.concurrency]

    table = Table(title=f"{args.url}: {args.requests} requests from {args.data.name}/{args.split}")
    columns = ("concurrency", "requests_per_s", "p50_ms", "p95_ms", "p99_ms", "errors")
    for column in columns:
        table.add_column(column, justify="right")
    for r in results:
        table.add_row(*(str(r[c]) for c in columns))
    Console().print(table)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
