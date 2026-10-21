"""The evaluation harness: one entry point that produces the full report for a model.

Produces ``report.json`` (every number), ``report.md`` (the tables a reader wants) and figures.
The same function runs at the end of training, from ``decider-eval``, and for baselines.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from decider.data.dataset import EncodedSplit
from decider.evaluation import plots
from decider.evaluation.calibration import fit_temperatures
from decider.evaluation.collect import collect_results
from decider.evaluation.latency import benchmark
from decider.evaluation.metrics import (
    QuestionResult,
    bootstrap_ci,
    bootstrap_ece_ci,
    grouped,
    score_results,
    subset,
    summarize_scored,
)
from decider.evaluation.robustness import (
    independence_gap,
    injection_effect,
    negation_consistency,
    permutation_gap,
)
from decider.inference import Predictor
from decider.packing import TYPES, Example

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    splits: list[str] = field(default_factory=lambda: ["test", "heldout"])
    calibration_split: str = "calib"
    max_examples: int | None = None
    n_bins: int = 15
    bootstrap: int = 1000
    robustness: bool = True
    robustness_examples: int = 200
    latency: bool = True
    latency_state_tokens: list[int] = field(default_factory=lambda: [128, 512])
    latency_questions: list[int] = field(default_factory=lambda: [1, 8, 32])
    seed: int = 0


def sample_examples(split: EncodedSplit, n: int | None, seed: int) -> list[Example]:
    examples = list(split.examples)
    if n is not None and len(examples) > n:
        examples = random.Random(seed).sample(examples, n)
    return examples


def summarize_results(
    results: Sequence[QuestionResult],
    temperatures: dict[str, float] | None,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    scored = score_results(results, temperatures)
    by_type = grouped(results, scored, lambda r: r.type, n_bins)
    ci: dict[str, Any] = {}
    if n_bootstrap:
        types = np.array(scored.types)
        for t in by_type:
            s = subset(scored, types == t)
            ci[t] = {
                "accuracy": bootstrap_ci(s.correct, n_resamples=n_bootstrap),
                "ece": bootstrap_ece_ci(s.confidence, s.correct, n_resamples=n_bootstrap // 2),
            }
    return {
        "overall": summarize_scored(scored, n_bins),
        "by_type": by_type,
        "by_family": grouped(results, scored, lambda r: r.family, n_bins),
        "by_source": grouped(results, scored, lambda r: r.source, n_bins),
        "ci95": ci,
    }


HISTOGRAM_BINS = 10


def confidence_histograms(
    results: Sequence[QuestionResult], temperatures: dict[str, float] | None
) -> dict[str, list[float]]:
    """Share of questions per top-label confidence bin on [0, 1], per question type."""
    out = {}
    for t, (conf, _) in _series(results, temperatures).items():
        counts, _ = np.histogram(conf, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
        out[t] = (counts / max(1, counts.sum())).tolist()
    return out


def _series(results: Sequence[QuestionResult], temperatures: dict[str, float] | None) -> dict:
    scored = score_results(results, temperatures)
    types = np.array(scored.types)
    return {
        t: (scored.confidence[types == t], scored.correct[types == t])
        for t in TYPES
        if (types == t).any()
    }


def evaluate(
    predictor: Predictor,
    data_dir: Path,
    cfg: EvalConfig,
    out_dir: Path,
    temperatures: dict[str, float] | None = None,
    fit_calibration: bool = True,
) -> dict[str, Any]:
    """Evaluate ``predictor`` on the configured splits and write the report to ``out_dir``.

    With ``fit_calibration`` the temperatures are (re)fitted on the calibration split and
    written into the predictor's model, so the caller can bundle the calibrated model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    encoder = predictor.encoder
    report: dict[str, Any] = {"config": asdict(cfg), "splits": {}}

    if fit_calibration:
        calib = EncodedSplit.load(data_dir, cfg.calibration_split, encoder)
        if len(calib):
            calib_results = collect_results(predictor, sample_examples(calib, 20_000, cfg.seed))
            temperatures = fit_temperatures(calib_results)
            predictor.model.set_temperatures(temperatures)
            report["calibration"] = {"split": cfg.calibration_split, "n": len(calib_results)}
    temperatures = temperatures or predictor.model.temperatures()
    report["temperatures"] = temperatures
    log.info("temperatures: %s", {k: round(v, 3) for k, v in temperatures.items()})

    for split_name in cfg.splits:
        split = EncodedSplit.load(data_dir, split_name, encoder)
        if not len(split):
            continue
        examples = sample_examples(split, cfg.max_examples, cfg.seed)
        results = collect_results(predictor, examples, split_name)
        entry: dict[str, Any] = {
            "examples": len(examples),
            "questions": len(results),
            "uncalibrated": summarize_results(results, None, cfg.n_bins, 0),
            "calibrated": summarize_results(results, temperatures, cfg.n_bins, cfg.bootstrap),
            "confidence_histogram": confidence_histograms(results, temperatures),
            "negation": negation_consistency(results, temperatures),
            "injection": injection_effect(results, temperatures),
        }
        if cfg.robustness:
            n = cfg.robustness_examples
            exact = predictor.with_precision("fp32")
            entry["permutation"] = permutation_gap(exact, examples, n, cfg.seed)
            entry["independence"] = independence_gap(exact, examples, n, cfg.seed)
            if predictor.precision != "fp32" and predictor.device.type == "cuda":
                entry["permutation_bf16"] = permutation_gap(predictor, examples, n, cfg.seed)
                entry["independence_bf16"] = independence_gap(predictor, examples, n, cfg.seed)
        report["splits"][split_name] = entry

        for label, temps in (("uncalibrated", None), ("calibrated", temperatures)):
            fig = plots.reliability_figure(_series(results, temps), f"{split_name} ({label})")
            fig.savefig(fig_dir / f"reliability_{split_name}_{label}.png", dpi=120)
            plots.close(fig)
        fig = plots.risk_coverage_figure(_series(results, temperatures), f"{split_name}")
        fig.savefig(fig_dir / f"risk_coverage_{split_name}.png", dpi=120)
        plots.close(fig)
        log.info(
            "%s: acc=%.4f ece=%.4f log_loss=%.4f",
            split_name,
            entry["calibrated"]["overall"]["accuracy"],
            entry["calibrated"]["overall"]["ece"],
            entry["calibrated"]["overall"]["log_loss"],
        )

    if cfg.latency:
        report["latency"] = benchmark(
            predictor.model,
            predictor.tokenizer,
            predictor.limits,
            predictor.device,
            cfg.latency_state_tokens,
            cfg.latency_questions,
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _table(rows: dict[str, dict[str, Any]], columns: list[str], first: str) -> list[str]:
    lines = [f"| {first} | " + " | ".join(columns) + " |"]
    lines.append("|" + "---|" * (len(columns) + 1))
    for name, metrics in rows.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(metrics.get(c)) for c in columns) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    cols = [
        "n",
        "accuracy",
        "macro_f1",
        "log_loss",
        "brier",
        "ece",
        "adaptive_ece",
        "coverage_at_95",
    ]
    lines = ["# Evaluation report", ""]
    temps = report.get("temperatures", {})
    lines.append("Temperatures: " + ", ".join(f"{k}={v:.3f}" for k, v in temps.items()))
    lines.append("")
    for split, entry in report["splits"].items():
        cal, unc = entry["calibrated"], entry["uncalibrated"]
        summary = f"{entry['examples']} records, {entry['questions']} questions."
        lines += [f"## {split}", "", summary, ""]
        lines += ["### By question type (calibrated)", ""]
        lines += _table(cal["by_type"], cols, "type")
        lines += ["", "### Calibration effect", ""]
        rows = {
            "uncalibrated": unc["overall"],
            "calibrated": cal["overall"],
        }
        lines += _table(rows, ["log_loss", "brier", "ece", "adaptive_ece"], "")
        if cal.get("ci95"):
            lines += ["", "95% bootstrap intervals:", ""]
            for t, ci in cal["ci95"].items():
                acc, e = ci["accuracy"], ci["ece"]
                acc_ci = f"[{_fmt(acc[0])}, {_fmt(acc[1])}]"
                lines.append(f"- {t}: accuracy {acc_ci}, ECE [{_fmt(e[0])}, {_fmt(e[1])}]")
        lines += ["", "### By family", ""]
        lines += _table(cal["by_family"], cols, "family")
        lines += ["", "### Robustness", ""]
        for key in (
            "permutation",
            "independence",
            "permutation_bf16",
            "independence_bf16",
            "negation",
        ):
            if key in entry and entry[key].get("n"):
                values = ", ".join(f"{k}={_fmt(v, 6)}" for k, v in entry[key].items())
                lines.append(f"- {key}: {values}")
        inj = entry.get("injection", {})
        if "accuracy_drop" in inj:
            lines.append(
                f"- injection: accuracy drop {_fmt(inj['accuracy_drop'])}, "
                f"false-yes increase {_fmt(inj['false_yes_increase'])}"
            )
        lines.append("")
    if report.get("latency"):
        lines += ["## Latency", ""]
        lat_cols = [
            "state_tokens",
            "questions",
            "total_tokens",
            "full_p50_ms",
            "full_p95_ms",
            "cached_p50_ms",
            "peak_memory_mb",
        ]
        rows = {str(i): r for i, r in enumerate(report["latency"])}
        lines += _table(rows, lat_cols, "#")
        lines.append("")
    return "\n".join(lines)
