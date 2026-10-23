from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from decider.data.dataset import record_to_example_targets
from decider.data.synthetic import SyntheticConfig, SyntheticGenerator
from decider.evaluation.familiarity import Familiarity, QuestionFeatures, auroc, collect_features
from decider.packing import Example, RequestEncoder, collate
from tests.conftest import LIMITS, encode, make_model, sample_request


def test_features_come_with_identical_logits(encoder: RequestEncoder) -> None:
    model = make_model()
    batch = collate([encode(encoder, sample_request(i)) for i in range(3)], LIMITS.max_seq_len)
    with torch.no_grad():
        logits = model(batch)
        logits_f, features = model.forward_with_features(batch)
    for t in logits:
        torch.testing.assert_close(logits[t], logits_f[t])
    assert features["noul"].shape == (len(batch.readouts["noul"]), 64)
    assert features["choice"].shape == (len(batch.readouts["choice"]), 64)


def _features(x: np.ndarray, question_type: str = "noul") -> QuestionFeatures:
    f = QuestionFeatures()
    for i, row in enumerate(x):
        f.record_ids.append(f"r{i}")
        f.questions.append("q")
        f.types.append(question_type)
        f.families.append("f")
        f.features.append(row)
        f.confidence.append(0.9)
        f.correct.append(True)
    return f


def test_shifted_questions_are_unfamiliar(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    calib = rng.normal(size=(2000, 16))
    familiar = rng.normal(size=(500, 16))
    shifted = rng.normal(loc=1.5, size=(500, 16))
    detector = Familiarity.fit(_features(calib), quantile=0.95)
    assert detector.unfamiliar("noul", familiar).mean() == pytest.approx(0.05, abs=0.03)
    assert detector.unfamiliar("noul", shifted).mean() > 0.9
    assert auroc(detector.score("noul", familiar), detector.score("noul", shifted)) > 0.95
    assert not detector.unfamiliar("choice", shifted).any(), "unfitted types are never flagged"

    detector.save(tmp_path / "f.npz")
    loaded = Familiarity.load(tmp_path / "f.npz")
    np.testing.assert_allclose(loaded.score("noul", shifted), detector.score("noul", shifted))
    assert loaded.threshold == detector.threshold


def test_threshold_from_held_back_records_keeps_the_nominal_rate() -> None:
    """In high dimensions, distances of the fitted points are biased low: a threshold taken
    from them flags several times the nominal share of new in-distribution questions."""
    rng = np.random.default_rng(0)
    calib, new = rng.normal(size=(1500, 128)), rng.normal(size=(2000, 128))
    held_back = Familiarity.fit(_features(calib), quantile=0.95)
    in_sample = Familiarity.fit(_features(calib), quantile=0.95, threshold_share=0.0)
    assert held_back.unfamiliar("noul", new).mean() == pytest.approx(0.05, abs=0.025)
    assert in_sample.unfamiliar("noul", new).mean() > 0.15


def test_threshold_split_is_independent_of_the_dataset_split() -> None:
    """Calib records are exactly those whose unsalted id hash falls in calib's range, so the
    fit/threshold split must not reuse that hash (it would split by source, not at random)."""
    from decider.data.build import _bucket as split_bucket
    from decider.evaluation.familiarity import _bucket

    ids = [f"banking77-train-{i:07d}" for i in range(20_000)]
    calib_like = [r for r in ids if split_bucket(r) < 0.05][:500]
    share = np.mean([_bucket(r) < 0.3 for r in calib_like])
    assert share == pytest.approx(0.3, abs=0.07)


def test_auroc_matches_brute_force_with_ties() -> None:
    neg = np.array([0.1, 0.4, 0.4, 0.8])
    pos = np.array([0.4, 0.9, 0.3])
    pairs = [(1.0 if p > n else 0.5 if p == n else 0.0) for n, p in product(neg, pos)]
    assert auroc(neg, pos) == pytest.approx(np.mean(pairs))
    assert np.isnan(auroc(np.array([]), pos))


def test_collect_features_aligns_rows(encoder: RequestEncoder) -> None:
    records = list(SyntheticGenerator(SyntheticConfig()).generate(12, "test", seed=0))
    tokenized = encoder.encode_many(records)
    examples = [
        Example(t, record_to_example_targets(r), {"id": r.id, "family": r.family})
        for t, r in zip(tokenized, records, strict=True)
    ]
    features = collect_features(make_model(), examples, LIMITS.max_seq_len, torch.device("cpu"))
    n_questions = sum(len(r.questions) for r in records)
    assert len(features.features) == len(features.correct) == n_questions
    assert len(features.confidence) == n_questions
    assert all(0.0 < c <= 1.0 for c in features.confidence)
    assert set(features.record_ids) == {r.id for r in records}
    assert all(f.shape == (64,) for f in features.features)
