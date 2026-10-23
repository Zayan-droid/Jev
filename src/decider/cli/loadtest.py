"""``decider-loadtest``: replay dataset records against a running server.

    decider-loadtest --data data/processed/mix-v1 --split test --concurrency 16 --requests 500

Each worker thread sends requests back to back; the report gives throughput, latency
percentiles (end to end, as the client sees them) and errors. Run it against servers started
with and without ``--max-batch-size`` to measure dynamic batching on your hardware.

By default each worker sends its next request as soon as the last one returns (closed loop),
which measures capacity. ``--rate`` sends requests on a fixed schedule instead (open loop, at
most ``--concurrency`` in flight), like real traffic that leaves the server idle between
requests. ``--tag-ids`` sends each record's id as the request id, so the server's logs can be
joined with the labels afterwards (``decider-shadow --labels``).
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


def run_load(
    url: str,
    requests: list[Any],
    concurrency: int,
    warmup: int = 5,
    request_ids: list[str] | None = None,
    rate: float | None = None,
) -> dict[str, Any]:
    """Closed loop by default; with ``rate`` (requests/s), request ``i`` is sent no earlier
    than ``i / rate`` seconds after the start."""
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
            if rate:
                delay = start + i / rate - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            request = requests[i]
            request_id = request_ids[i] if request_ids else None
            sent = time.perf_counter()
            try:
                client.send(request, request_id)
            except (DeciderError, OSError) as exc:
                with lock:
                    errors.append(type(exc).__name__)
                continue
            with lock:
                latencies.append((time.perf_counter() - sent) * 1000)

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
    p.add_argument("--rate", type=float, help="requests/s on a fixed schedule (open loop)")
    p.add_argument("--tag-ids", action="store_true", help="send record ids as request ids")
    p.add_argument("--out", type=Path, help="write the results as JSON")
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.table import Table

    from decider.data.io import iter_records, split_path

    records = list(iter_records(split_path(args.data, args.split)))
    rng = random.Random(args.seed)
    chosen = [rng.choice(records) for _ in range(args.requests)]
    sample = [r.request() for r in chosen]
    ids = [r.id for r in chosen] if args.tag_ids else None
    results = [
        run_load(args.url, sample, c, request_ids=ids, rate=args.rate) for c in args.concurrency
    ]

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
