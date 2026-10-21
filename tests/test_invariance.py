"""The structural guarantees of the architecture, checked on a randomly initialised model.

These hold by construction (masks + tree positions), so they must hold for *any* weights.
"""

from __future__ import annotations

import random

import pytest
import torch

from decider.model import DeciderModel
from decider.packing import Example, RequestEncoder, collate
from decider.schema import DecisionRequest
from tests.conftest import LIMITS, encode, make_model, sample_request

ATOL = 1e-5


def outputs(model: DeciderModel, examples: list[Example], pack: bool = True) -> list[dict]:
    """Per example: {question: probability tensor}."""
    batch = collate(examples, LIMITS.max_seq_len, pack=pack)
    with torch.no_grad():
        probs = model.probabilities(model(batch))
    result: list[dict] = [{} for _ in examples]
    for t, readouts in batch.readouts.items():
        for n, r in enumerate(readouts):
            p = probs[t][n]
            result[r.example][r.question] = p if t == "noul" else p[: len(r.option_names)]
    return result


def with_questions(request: DecisionRequest, **changes: dict | None) -> DecisionRequest:
    questions = {k: v.model_dump() for k, v in request.questions.items()}
    for name, q in changes.items():
        if q is None:
            questions.pop(name)
        else:
            questions[name] = q
    return DecisionRequest(state=request.state, questions=questions)


def test_changing_one_question_leaves_the_others_untouched(
    model: DeciderModel, encoder: RequestEncoder
) -> None:
    base = sample_request()
    edited = with_questions(
        base, refund={"type": "noul", "instructions": "Completely different text, much longer."}
    )
    a, b = outputs(model, [encode(encoder, base), encode(encoder, edited)], pack=False)
    assert not torch.allclose(a["refund"], b["refund"])
    for name in ("color", "severity", "escalate"):
        torch.testing.assert_close(a[name], b[name], atol=ATOL, rtol=0)


def test_adding_and_removing_questions(model: DeciderModel, encoder: RequestEncoder) -> None:
    base = sample_request()
    fewer = with_questions(base, color=None, refund=None)
    more = with_questions(
        base,
        extra={"type": "choice", "instructions": "Pick one", "criteria": {"a": "x", "b": "y"}},
    )
    a, b, c = outputs(model, [encode(encoder, r) for r in (base, fewer, more)], pack=False)
    for name in ("severity", "escalate"):
        torch.testing.assert_close(a[name], b[name], atol=ATOL, rtol=0)
        torch.testing.assert_close(a[name], c[name], atol=ATOL, rtol=0)


def test_question_order_does_not_matter(model: DeciderModel, encoder: RequestEncoder) -> None:
    base = sample_request()
    names = list(base.questions)[::-1]
    reordered = DecisionRequest(state=base.state, questions={n: base.questions[n] for n in names})
    a, b = outputs(model, [encode(encoder, base), encode(encoder, reordered)], pack=False)
    for name in names:
        torch.testing.assert_close(a[name], b[name], atol=ATOL, rtol=0)


@pytest.mark.parametrize("cross_option_layers", [0, 1, 2])
@pytest.mark.parametrize("local", [False, True], ids=["global", "sliding"])
def test_choice_is_permutation_equivariant(
    encoder: RequestEncoder, cross_option_layers: int, local: bool
) -> None:
    model = make_model(local=local, cross_option_layers=cross_option_layers)
    base = sample_request()
    criteria = dict(base.questions["color"].criteria)  # type: ignore[union-attr]
    names = list(criteria)
    shuffled = names[:]
    random.Random(1).shuffle(shuffled)
    permuted = with_questions(
        base,
        color={
            "type": "choice",
            "instructions": base.questions["color"].instructions,
            "criteria": {n: criteria[n] for n in shuffled},
        },
    )
    a, b = outputs(model, [encode(encoder, base), encode(encoder, permuted)], pack=False)
    p_a = dict(zip(names, a["color"].tolist(), strict=True))
    p_b = dict(zip(shuffled, b["color"].tolist(), strict=True))
    for n in names:
        assert p_a[n] == pytest.approx(p_b[n], abs=ATOL)
    assert len(set(p_a.values())) > 1, "test is vacuous if all options score the same"


def test_packed_rows_match_separate_rows(model: DeciderModel, encoder: RequestEncoder) -> None:
    examples = [encode(encoder, sample_request(i)) for i in range(4)]
    packed = outputs(model, examples, pack=True)
    alone = outputs(model, examples, pack=False)
    for p, s in zip(packed, alone, strict=True):
        for name in p:
            torch.testing.assert_close(p[name], s[name], atol=ATOL, rtol=0)


def test_state_hidden_states_do_not_depend_on_questions(
    model: DeciderModel, encoder: RequestEncoder
) -> None:
    base = sample_request()
    other = with_questions(base, color=None, severity=None)
    hidden = []
    for request in (base, other):
        ex = encode(encoder, request)
        batch = collate([ex], LIMITS.max_seq_len, pack=False)
        masks = model.layer_masks(batch)
        with torch.no_grad():
            h, _ = model.encoder(batch.input_ids, batch.position_ids, batch.level_ids, masks)
        hidden.append(h[0, : len(ex.request.state)])
    torch.testing.assert_close(hidden[0], hidden[1], atol=ATOL, rtol=0)


def test_cached_state_matches_full_forward(model: DeciderModel, encoder: RequestEncoder) -> None:
    ex = encode(encoder, sample_request())
    batch = collate([ex], LIMITS.max_seq_len, pack=False)
    with torch.no_grad():
        full = model(batch)
    cache = model.encode_state(ex.request.state)
    cached = model.forward_cached(cache, batch)
    for t in full:
        torch.testing.assert_close(full[t], cached[t], atol=ATOL, rtol=0)
