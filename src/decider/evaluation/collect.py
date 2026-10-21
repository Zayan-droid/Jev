"""Running a model over an encoded split and collecting per-question results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from decider.evaluation.metrics import QuestionResult
from decider.inference import Predictor
from decider.packing import Example


def collect_results(
    predictor: Predictor, examples: Sequence[Example], split: str | None = None
) -> list[QuestionResult]:
    """Uncalibrated logits + targets for every question of every example."""
    calibrated = predictor.calibrated
    predictor.calibrated = False
    try:
        predictions = predictor.predict_examples(examples)
    finally:
        predictor.calibrated = calibrated
    results: list[QuestionResult] = []
    for example, preds in zip(examples, predictions, strict=True):
        assert example.targets is not None, "evaluation needs labelled examples"
        meta = dict(example.meta)
        if split is not None:
            meta["split"] = split
        for name, pred in preds.items():
            results.append(
                QuestionResult(
                    record_id=str(meta.get("id", "")),
                    question=name,
                    type=pred.type,
                    source=str(meta.get("source", "unknown")),
                    family=str(meta.get("family", "unknown")),
                    option_names=pred.option_names,
                    logits=pred.logits.astype(np.float64),
                    target=np.asarray(example.targets[name], dtype=np.float64),
                    meta=meta,
                )
            )
    return results
