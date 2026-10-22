from pathlib import Path

import numpy as np
import pytest

from decider.cli import compare as compare_cli
from decider.evaluation.compare import paired_deltas, read_predictions, render, write_predictions


def rows(correct: list[bool], per_record: int = 1, kind: str = "noul") -> list[dict]:
    return [
        {
            "record": f"r{i // per_record}",
            "question": f"q{i % per_record}",
            "type": kind,
            "family": "f",
            "source": "s",
            "correct": c,
            "log_loss": 0.0 if c else 1.0,
            "confidence": 0.9,
        }
        for i, c in enumerate(correct)
    ]


def index(rs: list[dict]) -> dict:
    return {(r["record"], r["question"]): r for r in rs}


def test_identical_models_show_no_difference() -> None:
    rng = np.random.default_rng(0)
    same = index(rows(list(rng.random(400) < 0.7)))
    (overall,) = [d for d in paired_deltas(same, same) if d.group == "all"]
    assert overall.diff == 0 and overall.low == 0 and overall.high == 0
    assert not overall.significant


def test_a_clear_improvement_is_significant() -> None:
    rng = np.random.default_rng(1)
    a_correct = rng.random(500) < 0.6
    b_correct = a_correct | (rng.random(500) < 0.3)  # B fixes some of A's errors
    deltas = paired_deltas(index(rows(list(a_correct))), index(rows(list(b_correct))))
    overall = deltas[-1]
    assert overall.group == "all" and overall.diff > 0 and overall.significant
    assert "*" in render(deltas, "correct")


def test_record_level_bootstrap_is_wider_for_correlated_questions() -> None:
    rng = np.random.default_rng(2)
    record_effect = rng.random(100) < 0.5  # all four questions of a record move together
    a = index(rows([False] * 400, per_record=4))
    b = index(rows([bool(record_effect[i // 4]) for i in range(400)], per_record=4))
    clustered = paired_deltas(a, b)[-1]
    a1 = index(rows([False] * 400, per_record=1))
    b1 = index(rows([bool(record_effect[i // 4]) for i in range(400)], per_record=1))
    independent = paired_deltas(a1, b1)[-1]
    assert clustered.diff == pytest.approx(independent.diff)
    assert (clustered.high - clustered.low) > 1.5 * (independent.high - independent.low)


def test_cli_compares_two_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name, correct in (("a", [True, False, False, True]), ("b", [True, True, True, True])):
        (tmp_path / name).mkdir()
        write_predictions(tmp_path / name / "predictions_test.jsonl.gz", rows(correct))
    assert len(read_predictions(tmp_path / "a" / "predictions_test.jsonl.gz")) == 4
    compare_cli.main([str(tmp_path / "a"), str(tmp_path / "b"), "--names", "v1", "v2"])
    out = capsys.readouterr().out
    assert "4 shared questions" in out and "| all | 4 | 50.00 | 100.00 | +50.00 pt" in out
