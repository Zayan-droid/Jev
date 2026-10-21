import numpy as np
import pytest

from decider.evaluation.calibration import fit_temperature
from decider.evaluation.metrics import (
    QuestionResult,
    aurc,
    brier,
    coverage_at_accuracy,
    ece,
    is_correct,
    log_loss,
    macro_f1,
    probabilities,
    reliability_bins,
    rps,
    score_results,
    summarize_scored,
)


def test_ece_matches_hand_computation() -> None:
    conf = np.array([0.95, 0.95, 0.65, 0.65, 0.65, 0.65])
    correct = np.array([1, 1, 1, 0, 0, 1], dtype=float)
    # bin [0.933, 1.0]: acc 1.0 vs conf 0.95 -> 0.05 * 2/6
    # bin [0.6, 0.667]: acc 0.5 vs conf 0.65 -> 0.15 * 4/6
    assert ece(conf, correct, n_bins=15) == pytest.approx(0.05 * 2 / 6 + 0.15 * 4 / 6)


def test_perfectly_calibrated_predictions_have_low_ece() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 200_000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    assert ece(conf, correct) < 0.005
    assert ece(conf, correct, adaptive=True) < 0.005


def test_reliability_bins_cover_all_samples() -> None:
    conf = np.linspace(0, 1, 101)
    bins = reliability_bins(conf, np.ones_like(conf), n_bins=10)
    assert sum(b.count for b in bins) == 101


def test_scoring_rules() -> None:
    t = np.array([0.0, 1.0, 0.0])
    assert log_loss(t, t) == pytest.approx(0.0, abs=1e-9)
    assert brier(np.array([1 / 3] * 3), t) == pytest.approx(2 / 3)
    # RPS rewards mass on a neighbouring level more than on a distant one.
    near = np.array([0.5, 0.5, 0.0])
    far = np.array([0.0, 0.5, 0.5])
    target = np.array([1.0, 0.0, 0.0])
    assert rps(near, target) < rps(far, target)


def test_ties_in_the_target_count_as_correct() -> None:
    assert is_correct(np.array([0.1, 0.5, 0.4]), np.array([0.0, 0.5, 0.5]))
    assert not is_correct(np.array([0.6, 0.2, 0.2]), np.array([0.0, 0.5, 0.5]))


def test_selective_prediction() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0], dtype=float)
    assert coverage_at_accuracy(conf, correct, 0.95) == pytest.approx(0.5)
    assert aurc(conf, correct) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 2]))


def test_macro_f1() -> None:
    assert macro_f1(["a", "a", "b"], ["a", "a", "b"]) == pytest.approx(1.0)
    assert macro_f1(["a", "b"], ["b", "a"]) == pytest.approx(0.0)


def _noul_results(logits: np.ndarray, labels: np.ndarray) -> list[QuestionResult]:
    return [
        QuestionResult("r", f"q{i}", "noul", "s", "f", (), np.array([z]), np.array([y]))
        for i, (z, y) in enumerate(zip(logits, labels, strict=True))
    ]


def test_temperature_scaling_recovers_true_temperature() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20_000)
    labels = (rng.uniform(size=true_logits.size) < 1 / (1 + np.exp(-true_logits))).astype(float)
    overconfident = true_logits * 3.0  # a model that is 3x too sharp
    t = fit_temperature(_noul_results(overconfident, labels))
    assert t == pytest.approx(3.0, rel=0.05)


def test_temperature_scaling_for_choice_with_varying_option_counts() -> None:
    rng = np.random.default_rng(1)
    results = []
    for i in range(3000):
        k = int(rng.integers(2, 6))
        z = rng.normal(0, 1.5, k)
        p = np.exp(z) / np.exp(z).sum()
        y = np.zeros(k)
        y[rng.choice(k, p=p)] = 1.0
        results.append(
            QuestionResult("r", f"q{i}", "choice", "s", "f", tuple("abcde"[:k]), z * 2.0, y)
        )
    assert fit_temperature(results) == pytest.approx(2.0, rel=0.1)


def test_score_results_summary() -> None:
    results = _noul_results(np.array([5.0, -5.0, 5.0]), np.array([1.0, 0.0, 0.0]))
    summary = summarize_scored(score_results(results))
    assert summary["n"] == 3
    assert summary["accuracy"] == pytest.approx(2 / 3)
    p = probabilities(results[0])
    assert p.sum() == pytest.approx(1.0)
