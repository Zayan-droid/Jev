"""Reliability diagrams and risk-coverage curves."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from decider.evaluation.metrics import ece, reliability_bins, risk_coverage

Series = tuple[np.ndarray, np.ndarray]  # (confidence, correct)


def reliability_figure(series: Mapping[str, Series], title: str = "", n_bins: int = 15) -> Figure:
    """One panel per series: accuracy per confidence bin against the diagonal, with the share
    of questions per bin underneath."""
    names = [n for n, (c, _) in series.items() if len(c)]
    fig, axes = plt.subplots(
        2,
        max(1, len(names)),
        figsize=(4 * max(1, len(names)), 5),
        gridspec_kw={"height_ratios": [3, 1]},
        squeeze=False,
    )
    for col, name in enumerate(names):
        conf, correct = series[name]
        bins = reliability_bins(conf, correct, n_bins)
        top, bottom = axes[0, col], axes[1, col]
        centers = [(b.lower + b.upper) / 2 for b in bins]
        width = 1.0 / n_bins
        accuracy = [b.accuracy for b in bins]
        top.bar(centers, accuracy, width=width * 0.9, color="#4C72B0", label="accuracy")
        top.bar(
            centers,
            [b.confidence - b.accuracy for b in bins],
            bottom=accuracy,
            width=width * 0.9,
            color="#DD8452",
            alpha=0.5,
            label="gap",
        )
        top.plot([0, 1], [0, 1], "k--", linewidth=1)
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.set_title(f"{name}  ECE={ece(conf, correct, n_bins):.3f}  n={len(conf)}")
        top.set_ylabel("accuracy")
        top.legend(loc="upper left", fontsize=8)
        bottom.bar(centers, [b.count / len(conf) for b in bins], width=width * 0.9, color="#55A868")
        bottom.set_xlim(0, 1)
        bottom.set_xlabel("confidence")
        bottom.set_ylabel("share")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def risk_coverage_figure(series: Mapping[str, Series], title: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (conf, correct) in series.items():
        if len(conf) == 0:
            continue
        coverage, acc = risk_coverage(conf, correct)
        ax.plot(coverage, acc, label=name)
    ax.axhline(0.95, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("accuracy on answered")
    ax.set_xlim(0, 1)
    ax.set_title(title or "Selective prediction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def close(fig: Figure) -> None:
    plt.close(fig)
