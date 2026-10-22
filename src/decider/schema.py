"""Request, response and training-record schemas.

A request is a *state* (free text or a JSON object) plus a set of named, typed
questions. Every question is answered with a probability distribution:

* ``noul``   - a yes/no proposition, answered with ``p(yes)``.
* ``choice`` - an unordered set of named options, answered with one probability per option.
* ``score``  - an ordered set of levels ``0..L-1``, answered with one probability per level.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 32
MAX_OPTIONS = 255
MIN_LEVELS, MAX_LEVELS = 2, 10
# Size limits keep a single request from tying up the tokenizer: text beyond what the model reads
# is truncated anyway (the state to ~512 tokens, instructions and options to <100 tokens).
MAX_STATE_CHARS = 100_000
MAX_TEXT_CHARS = 2_000
_PROB_TOL = 1e-3

QuestionType = Literal["noul", "choice", "score"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoulQuestion(_Strict):
    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ChoiceQuestion(_Strict):
    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    criteria: dict[str, str] = Field(description="Option name -> description of when it applies.")

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, value: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(value) <= MAX_OPTIONS:
            raise ValueError(f"choice questions need 2..{MAX_OPTIONS} options, got {len(value)}")
        if any(not k for k in value):
            raise ValueError("option names must be non-empty")
        if any(len(k) + len(v) > MAX_TEXT_CHARS for k, v in value.items()):
            raise ValueError(f"an option name plus description exceeds {MAX_TEXT_CHARS} characters")
        return value


class ScoreQuestion(_Strict):
    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    levels: list[str] = Field(description="Description of each level, lowest first.")

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, value: list[str]) -> list[str]:
        if not MIN_LEVELS <= len(value) <= MAX_LEVELS:
            raise ValueError(f"score questions need {MIN_LEVELS}..{MAX_LEVELS} levels")
        if any(len(level) > MAX_TEXT_CHARS for level in value):
            raise ValueError(f"a level description exceeds {MAX_TEXT_CHARS} characters")
        return value


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]
State = str | dict[str, Any]


class DecisionRequest(_Strict):
    state: State
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, value: dict[str, Question]) -> dict[str, Question]:
        if not 1 <= len(value) <= MAX_QUESTIONS:
            raise ValueError(f"requests need 1..{MAX_QUESTIONS} questions, got {len(value)}")
        return value

    @field_validator("state")
    @classmethod
    def _check_state(cls, value: State) -> State:
        if len(serialize_state(value)) > MAX_STATE_CHARS:
            raise ValueError(f"the state exceeds {MAX_STATE_CHARS} characters when serialised")
        return value


def serialize_state(state: State) -> str:
    """Canonical text form of a state: JSON objects get sorted keys and compact separators,
    so semantically equal states always tokenize identically (and hit the same cache entry)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


# --------------------------------------------------------------------------------------
# Training records
# --------------------------------------------------------------------------------------

# noul: p(yes); choice: option name -> probability; score: probability per level.
Target = float | dict[str, float] | list[float]

Split = Literal["train", "val", "calib", "test", "heldout", "gold"]


class Record(DecisionRequest):
    """A labelled request. ``targets`` holds a distribution for every question."""

    id: str
    source: str
    family: str
    split: Split
    targets: dict[str, Target]
    label_kind: Literal["hard", "soft"] = "hard"
    label_source: str = "human"
    gen_version: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_targets(self) -> Record:
        if set(self.targets) != set(self.questions):
            raise ValueError(f"{self.id}: targets must cover exactly the questions")
        for name, question in self.questions.items():
            _validate_target(self.id, name, question, self.targets[name])
        return self

    def request(self) -> DecisionRequest:
        return DecisionRequest(state=self.state, questions=self.questions)


def _validate_target(record_id: str, name: str, question: Question, target: Target) -> None:
    where = f"{record_id}/{name}"
    if isinstance(question, NoulQuestion):
        if not isinstance(target, int | float) or not 0.0 <= target <= 1.0:
            raise ValueError(f"{where}: noul target must be a probability, got {target!r}")
        return
    if isinstance(question, ChoiceQuestion):
        if not isinstance(target, dict):
            raise ValueError(f"{where}: choice target must map option -> probability")
        unknown = set(target) - set(question.criteria)
        if unknown:
            raise ValueError(f"{where}: target names unknown options {sorted(unknown)}")
        values = list(target.values())
    else:
        if not isinstance(target, list) or len(target) != len(question.levels):
            raise ValueError(f"{where}: score target must have one probability per level")
        values = target
    if any(v < 0 for v in values) or not math.isclose(sum(values), 1.0, abs_tol=_PROB_TOL):
        raise ValueError(f"{where}: target must be a distribution, got {target!r}")


def target_vector(question: Question, target: Target, option_order: list[str]) -> list[float]:
    """Dense target in the given option order (choice) or level order (score)."""
    if isinstance(question, ChoiceQuestion):
        assert isinstance(target, dict)
        return [float(target.get(k, 0.0)) for k in option_order]
    assert isinstance(target, list)
    return [float(v) for v in target]


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class NoulAnswer(_Strict):
    type: Literal["noul"] = "noul"
    p_yes: float
    answer: bool
    confidence: float


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    distribution: dict[str, float]
    choice: str
    confidence: float


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    distribution: list[float]
    score: float
    level: int
    confidence: float


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionResponse(_Strict):
    answers: dict[str, Answer]
    model_version: str
    request_id: str
    latency_ms: float
    cache_hit: bool
    # Questions whose confidence fell below the server's review threshold.
    review: list[str]
