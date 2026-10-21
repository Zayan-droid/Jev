"""Offline analysis of the prediction log.

Prometheus answers "what is happening now"; this answers "how has traffic changed since the
model shipped": per-day volume, latency percentiles, cache hit rate, low-confidence share,
noul yes-rate and confidence drift (PSI) per question type against the bundle's reference.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from decider.serving.monitoring import psi


@dataclass
class _Day:
    requests: int = 0
    cache_hits: int = 0
    noul: int = 0
    yes: int = 0
    latency: list[float] = field(default_factory=list)
    versions: set[str] = field(default_factory=set)
    confidence: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def _confidence(answer: dict[str, Any]) -> float:
    if answer["type"] == "noul":
        p = float(answer["p_yes"])
        return max(p, 1.0 - p)
    distribution = answer["distribution"]
    values = distribution.values() if isinstance(distribution, dict) else distribution
    return float(max(values))


def read_log(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def analyse(
    entries: Iterable[dict[str, Any]],
    reference: dict[str, list[float]] | None = None,
    low_confidence: float = 0.6,
) -> dict[str, Any]:
    days: dict[str, _Day] = defaultdict(_Day)
    for e in entries:
        d = days[dt.datetime.fromtimestamp(e["time"], dt.UTC).date().isoformat()]
        d.requests += 1
        d.latency.append(e["latency_ms"])
        d.cache_hits += bool(e.get("cache_hit"))
        d.versions.add(e.get("model_version", "?"))
        for answer in e["answers"].values():
            d.confidence[answer["type"]].append(_confidence(answer))
            if answer["type"] == "noul":
                d.noul += 1
                d.yes += bool(answer["answer"])

    bins = len(next(iter(reference.values()))) if reference else 10
    report: dict[str, Any] = {"days": {}}
    for day in sorted(days):
        d = days[day]
        lat = np.array(d.latency)
        per_type = {}
        for t, values in d.confidence.items():
            conf = np.array(values)
            entry = {
                "n": len(conf),
                "mean_confidence": float(conf.mean()),
                "low_confidence_share": float((conf < low_confidence).mean()),
            }
            if reference and t in reference:
                hist, _ = np.histogram(conf, bins=bins, range=(0.0, 1.0))
                entry["psi"] = psi(np.array(reference[t]), hist / max(1, hist.sum()))
            per_type[t] = entry
        report["days"][day] = {
            "requests": d.requests,
            "model_versions": sorted(d.versions),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "cache_hit_rate": d.cache_hits / d.requests,
            "noul_yes_rate": d.yes / d.noul if d.noul else None,
            "types": per_type,
        }
    return report


def render(report: dict[str, Any]) -> str:
    columns = [
        "day",
        "requests",
        "p50 ms",
        "p95 ms",
        "cache hits",
        "yes rate",
        "type",
        "mean conf",
        "low conf",
        "PSI",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for day, d in report["days"].items():
        yes = "-" if d["noul_yes_rate"] is None else f"{d['noul_yes_rate']:.2f}"
        head = (
            f"| {day} | {d['requests']} | {d['latency_p50_ms']:.1f} | "
            f"{d['latency_p95_ms']:.1f} | {d['cache_hit_rate']:.2f} | {yes} |"
        )
        for i, (t, e) in enumerate(sorted(d["types"].items())):
            prefix = head if i == 0 else "| | | | | | |"
            value = e.get("psi")
            psi_text = "-" if value is None else f"{value:.3f}"
            conf, low = e["mean_confidence"], e["low_confidence_share"]
            lines.append(f"{prefix} {t} | {conf:.3f} | {low:.2f} | {psi_text} |")
    return "\n".join(lines)
