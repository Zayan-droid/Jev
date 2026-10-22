import pytest
from pydantic import ValidationError

from decider.schema import (
    MAX_STATE_CHARS,
    MAX_TEXT_CHARS,
    DecisionRequest,
    Record,
    serialize_state,
)

NOUL = {"type": "noul", "instructions": "Is it urgent?"}


def test_state_serialisation_is_canonical() -> None:
    assert serialize_state({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert serialize_state({"a": [1, 2], "b": 1}) == serialize_state({"b": 1, "a": [1, 2]})
    assert serialize_state("plain text") == "plain text"


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {f"q{i}": NOUL for i in range(33)},
        {"c": {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}},
        {"c": {"type": "choice", "instructions": "x", "criteria": {"": "a", "b": "b"}}},
        {"s": {"type": "score", "instructions": "x", "levels": ["one"]}},
        {"s": {"type": "score", "instructions": "x", "levels": [str(i) for i in range(11)]}},
        {"n": {"type": "noul", "instructions": ""}},
        {"n": {"type": "maybe", "instructions": "x"}},
    ],
    ids=[
        "none",
        "too-many",
        "one-option",
        "empty-name",
        "one-level",
        "eleven-levels",
        "empty-instructions",
        "unknown-type",
    ],
)
def test_invalid_questions_are_rejected(questions: dict) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(state="s", questions=questions)


def test_size_limits() -> None:
    long_text = "x" * (MAX_TEXT_CHARS + 1)
    with pytest.raises(ValidationError, match="characters"):
        DecisionRequest(state="x" * (MAX_STATE_CHARS + 1), questions={"q": NOUL})
    with pytest.raises(ValidationError):
        DecisionRequest(state="s", questions={"q": {"type": "noul", "instructions": long_text}})
    with pytest.raises(ValidationError, match="exceeds"):
        DecisionRequest(
            state="s",
            questions={
                "c": {"type": "choice", "instructions": "x", "criteria": {"a": long_text, "b": "b"}}
            },
        )
    DecisionRequest(state="x" * MAX_STATE_CHARS, questions={"q": NOUL})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionRequest(state="s", questions={"q": NOUL}, debug=True)  # type: ignore[call-arg]


def _record(**changes: object) -> dict:
    base = {
        "id": "r1",
        "source": "s",
        "family": "f",
        "split": "train",
        "state": "s",
        "questions": {
            "n": NOUL,
            "c": {"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}},
            "s": {"type": "score", "instructions": "x", "levels": ["lo", "hi"]},
        },
        "targets": {"n": 1.0, "c": {"a": 0.25, "b": 0.75}, "s": [0.5, 0.5]},
    }
    base.update(changes)
    return base


def test_record_targets_are_validated() -> None:
    Record(**_record())
    bad_targets = [
        {"n": 1.0, "c": {"a": 0.25, "b": 0.75}},  # missing a question
        {"n": 1.5, "c": {"a": 0.25, "b": 0.75}, "s": [0.5, 0.5]},  # not a probability
        {"n": 1.0, "c": {"a": 0.5, "z": 0.5}, "s": [0.5, 0.5]},  # unknown option
        {"n": 1.0, "c": {"a": 0.2, "b": 0.2}, "s": [0.5, 0.5]},  # does not sum to one
        {"n": 1.0, "c": {"a": 0.25, "b": 0.75}, "s": [1.0]},  # wrong number of levels
    ]
    for targets in bad_targets:
        with pytest.raises(ValidationError):
            Record(**_record(targets=targets))
