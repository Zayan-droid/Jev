"""Release gate: automated checks a model must pass before it can be promoted."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    threshold: float
    passed: bool


@dataclass
class GateResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        lines = [f"release gate: {'PASSED' if self.passed else 'FAILED'}"]
        for c in self.checks:
            value = "missing" if c.value is None else f"{c.value:.4f}"
            mark = "ok  " if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {value} (threshold {c.threshold})")
        return "\n".join(lines)


def _upper(name: str, value: float | None, threshold: float) -> Check:
    return Check(name, value, threshold, value is not None and value <= threshold)


def _lower(name: str, value: float | None, threshold: float) -> Check:
    return Check(name, value, threshold, value is not None and value >= threshold)


def check_gate(report: dict[str, Any], gate: dict[str, Any]) -> GateResult:
    """Evaluate the thresholds in ``gate`` (see ``register.gate`` in configs/config.yaml).

    A check whose metric is missing from the report fails: a gate that cannot be evaluated
    must not let a model through.
    """
    split = report.get("splits", {}).get(gate.get("split", "test"), {})
    cal = split.get("calibrated", {})
    result = GateResult()
    if "max_ece" in gate:
        for t, metrics in cal.get("by_type", {}).items():
            result.checks.append(_upper(f"ece.{t}", metrics.get("ece"), gate["max_ece"]))
    if "min_accuracy" in gate:
        acc = cal.get("overall", {}).get("accuracy")
        result.checks.append(_lower("accuracy", acc, gate["min_accuracy"]))
    if "max_permutation_diff" in gate:
        value = split.get("permutation", {}).get("max_abs_diff")
        result.checks.append(_upper("permutation", value, gate["max_permutation_diff"]))
    if "max_independence_diff" in gate:
        value = split.get("independence", {}).get("max_abs_diff")
        result.checks.append(_upper("independence", value, gate["max_independence_diff"]))
    if "max_injection_false_yes_increase" in gate:
        value = split.get("injection", {}).get("false_yes_increase")
        if value is not None:
            threshold = gate["max_injection_false_yes_increase"]
            result.checks.append(_upper("injection", value, threshold))
    if "max_full_p95_ms" in gate and report.get("latency"):
        worst = max(row["full_p95_ms"] for row in report["latency"])
        result.checks.append(_upper("latency_p95_ms", worst, gate["max_full_p95_ms"]))
    return result
