import random

import pytest

from decider.packing import (
    Example,
    PackingLimits,
    RequestEncoder,
    TokenizedQuestion,
    TokenizedRequest,
    assign_rows,
    augment,
    collate,
)
from decider.schema import DecisionRequest
from decider.tokenization import ByteTokenizer
from tests.conftest import encode, sample_request


def test_tree_positions_restart_per_question_and_option(encoder: RequestEncoder) -> None:
    example = encode(encoder, sample_request())
    batch = collate([example], max_seq_len=768)
    pos, q, o = batch.position_ids[0], batch.question_ids[0], batch.option_ids[0]
    s = len(example.request.state)
    assert pos[:s].tolist() == list(range(s))
    for k, question in enumerate(example.request.questions, start=1):
        first = int((q == k).nonzero()[0])
        assert int(pos[first]) == s, "every question starts at the end of the state"
        for m in range(1, len(question.options) + 1):
            first_opt = int(((q == k) & (o == m)).nonzero()[0])
            assert int(pos[first_opt]) == s + len(question.header)


def test_readouts_point_at_decision_tokens(encoder: RequestEncoder) -> None:
    tok = ByteTokenizer()
    batch = collate([encode(encoder, sample_request())], max_seq_len=768)
    flat = batch.input_ids.flatten()
    assert set(flat[batch.index["noul"]].tolist()) == {tok.token_id("[Q_NOUL]")}
    choice = flat[batch.index["choice"][batch.option_mask["choice"]]]
    assert set(choice.tolist()) == {tok.option_id}
    score = flat[batch.index["score"][batch.option_mask["score"]]]
    assert set(score.tolist()) == {tok.level_id}
    levels = batch.level_ids.flatten()[batch.index["score"][0]]
    assert levels.tolist() == [1, 2, 3, 4]


def test_packing_puts_several_documents_in_one_row(encoder: RequestEncoder) -> None:
    examples = [encode(encoder, sample_request(i)) for i in range(3)]
    batch = collate(examples, max_seq_len=sum(e.request.length for e in examples))
    assert batch.shape[0] == 1
    assert set(batch.doc_ids[0].tolist()) - {-1} == {0, 1, 2}
    assert batch.num_tokens == sum(e.request.length for e in examples)


def test_assign_rows_first_fit_decreasing() -> None:
    lengths = [5, 3, 4, 2, 6]
    rows = assign_rows(lengths, max_len=8)
    assert all(sum(lengths[i] for i in r) <= 8 for r in rows)
    assert sorted(i for r in rows for i in r) == [0, 1, 2, 3, 4]
    assert len(rows) == 3
    with pytest.raises(ValueError, match="exceeds"):
        assign_rows([9], max_len=8)


def test_state_is_truncated_to_fit_the_row() -> None:
    limits = PackingLimits(max_seq_len=128, max_state_tokens=96, max_question_tokens=32)
    enc = RequestEncoder(ByteTokenizer(), limits)
    request = DecisionRequest(
        state="x" * 1000, questions={"q": {"type": "noul", "instructions": "is it long?"}}
    )
    tokenized = enc.encode(request)
    assert len(tokenized.state) == 96
    assert tokenized.length <= 128


def test_augment_subsamples_but_keeps_target_mass() -> None:
    options = [[1, i] for i in range(10)]
    names = [f"o{i}" for i in range(10)]
    q = TokenizedQuestion("c", "choice", [0], options, names)
    s = TokenizedQuestion("s", "score", [0], [[2, 0], [2, 1], [2, 2]], ["0", "1", "2"])
    target = [0.0] * 10
    target[3], target[7] = 0.75, 0.25
    example = Example(TokenizedRequest([9], [q, s]), {"c": target, "s": [0.2, 0.3, 0.5]})
    out = augment(example, random.Random(0), max_options=4)
    assert out.targets is not None
    choice = next(x for x in out.request.questions if x.name == "c")
    score = next(x for x in out.request.questions if x.name == "s")
    assert len(choice.options) == 4
    kept = dict(zip(choice.option_names, out.targets["c"], strict=True))
    assert kept["o3"] == pytest.approx(0.75) and kept["o7"] == pytest.approx(0.25)
    assert score.option_names == ["0", "1", "2"]
    assert out.targets["s"] == [0.2, 0.3, 0.5]
