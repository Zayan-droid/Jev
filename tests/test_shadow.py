"""Shadow deployment: answer comparison, the runner's isolation from the live path, the report."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from decider.schema import ChoiceAnswer, DecisionRequest, NoulAnswer, ScoreAnswer
from decider.serving.shadow import (
    ShadowRunner,
    agreement,
    compare,
    disagreements,
    labelled_outcomes,
)


def test_compare_uses_top_answer_and_total_variation() -> None:
    a = NoulAnswer(p_yes=0.8, answer=True, confidence=0.3)
    b = NoulAnswer(p_yes=0.4, answer=False, confidence=0.1)
    c = compare(a, b)
    assert c["agree"] is False and c["distance"] == pytest.approx(0.4)

    a = ChoiceAnswer(distribution={"x": 0.7, "y": 0.3}, choice="x", confidence=0.1)
    # Same options in another order: the comparison is by name, not position.
    b = ChoiceAnswer(distribution={"y": 0.4, "x": 0.6}, choice="x", confidence=0.1)
    c = compare(a, b)
    assert c["agree"] is True and c["distance"] == pytest.approx(0.1)

    a = ScoreAnswer(distribution=[0.1, 0.9, 0.0], score=0.9, level=1, confidence=0.5)
    b = ScoreAnswer(distribution=[0.0, 0.0, 1.0], score=2.0, level=2, confidence=1.0)
    c = compare(a, b)
    assert c["agree"] is False and c["distance"] == pytest.approx(1.0)


def _entry(request_id: str, live: float, shadow: float) -> dict[str, Any]:
    q = compare(
        NoulAnswer(p_yes=live, answer=live >= 0.5, confidence=0.0),
        NoulAnswer(p_yes=shadow, answer=shadow >= 0.5, confidence=0.0),
    )
    return {
        "request_id": request_id,
        "primary_version": "v1",
        "shadow_version": "v2",
        "questions": {"q": q},
    }


def test_report_pairs_outcomes_with_labels() -> None:
    entries = [
        _entry("r1", 0.9, 0.8),  # agree, both right
        _entry("r2", 0.9, 0.2),  # live right
        _entry("r3", 0.3, 0.7),  # shadow right
        _entry("r4", 0.6, 0.1),  # shadow right
        _entry("unlabelled", 0.9, 0.1),
    ]
    labels: dict[str, Any] = {
        "r1": {"q": 1.0},
        "r2": {"q": 1.0},
        "r3": {"q": 1.0},
        "r4": {"q": 0.0},
    }
    summary = agreement(entries)
    assert summary["requests"] == 5 and summary["versions"] == [("v1", "v2")]
    assert summary["types"]["noul"]["agreement"] == pytest.approx(0.2)

    live, shadow = labelled_outcomes(entries, labels)
    assert set(live) == {("r1", "q"), ("r2", "q"), ("r3", "q"), ("r4", "q")}
    assert [live[k]["correct"] for k in sorted(live)] == [True, True, False, False]
    assert [shadow[k]["correct"] for k in sorted(shadow)] == [True, False, True, True]
    assert disagreements(live, shadow) == {
        "live_right": 1,
        "shadow_right": 2,
        "both_right": 0,
        "both_wrong": 0,
    }


class _FakeEngine:
    """Stands in for a DecisionEngine: answers every noul question with a fixed p(yes)."""

    def __init__(self, p_yes: float, gate: threading.Event | None = None, fail: bool = False):
        self.cfg = SimpleNamespace(emit_metrics=False)
        self.version = "fake"
        self.p_yes = p_yes
        self.gate = gate
        self.fail = fail

    def decide(self, request: DecisionRequest) -> Any:
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("shadow broke")
        answer = NoulAnswer(p_yes=self.p_yes, answer=self.p_yes >= 0.5, confidence=0.0)
        return SimpleNamespace(answers=dict.fromkeys(request.questions, answer), model_ms=1.0)


_REQUEST = DecisionRequest.model_validate(
    {"state": "x", "questions": {"q": {"type": "noul", "instructions": "Is it x?"}}}
)
_LIVE = {"q": NoulAnswer(p_yes=0.9, answer=True, confidence=0.5)}


def test_runner_counts_agreement_and_logs(tmp_path: Any) -> None:
    runner = ShadowRunner(_FakeEngine(0.2), str(tmp_path / "shadow.jsonl"))  # type: ignore[arg-type]
    try:
        for i in range(3):
            assert runner.submit(f"r{i}", _REQUEST, "live", _LIVE)
        runner.wait()
        summary = runner.summary()
        assert summary["requests"] == {"ok": 3, "error": 0, "dropped": 0}
        assert summary["types"]["noul"] == {"n": 3, "agree": 0, "agreement": 0.0}
        assert len((tmp_path / "shadow.jsonl").read_text().splitlines()) == 3
    finally:
        runner.close()


def test_runner_drops_when_behind_and_survives_failures() -> None:
    gate = threading.Event()
    runner = ShadowRunner(_FakeEngine(0.9, gate=gate), max_queue=1)  # type: ignore[arg-type]
    try:
        # The worker takes the first request and blocks; the second fills the queue; the third
        # is dropped instead of making the caller wait.
        assert runner.submit("r0", _REQUEST, "live", _LIVE)
        for _ in range(100):
            if runner._queue.empty():
                break
            threading.Event().wait(0.01)
        assert runner.submit("r1", _REQUEST, "live", _LIVE)
        assert not runner.submit("r2", _REQUEST, "live", _LIVE)
        gate.set()
        runner.wait()
        assert runner.summary()["requests"] == {"ok": 2, "error": 0, "dropped": 1}
    finally:
        runner.close()

    broken = ShadowRunner(_FakeEngine(0.9, fail=True))  # type: ignore[arg-type]
    try:
        broken.submit("r0", _REQUEST, "live", _LIVE)
        broken.wait()
        assert broken.summary()["requests"]["error"] == 1
    finally:
        broken.close()


def test_runner_refuses_an_engine_that_emits_live_metrics() -> None:
    engine = _FakeEngine(0.5)
    engine.cfg.emit_metrics = True
    with pytest.raises(ValueError, match="emit_metrics"):
        ShadowRunner(engine)  # type: ignore[arg-type]
