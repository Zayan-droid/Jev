import torch

from decider.masks import decision_mask, sliding_window

# Two documents in one row:
#   doc 0: state(3) | q1 header(2) opt1(2) opt2(2) | q2 header(4)
#   doc 1: state(2) | q1 header(1)
#   padding(2)
DOC = [0] * 13 + [1] * 3 + [-1] * 2
QUESTION = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 0, 0]
OPTION = [0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _mask(cross_option: bool) -> torch.Tensor:
    t = lambda xs: torch.tensor([xs])  # noqa: E731
    return decision_mask(t(DOC), t(QUESTION), t(OPTION), cross_option)[0, 0]


def _idx(doc: int, q: int, o: int | None = None) -> list[int]:
    return [
        i
        for i in range(len(DOC))
        if DOC[i] == doc and QUESTION[i] == q and (o is None or OPTION[i] == o)
    ]


def test_state_attends_only_to_state() -> None:
    m = _mask(cross_option=True)
    for i in _idx(0, 0):
        allowed = set(torch.nonzero(m[i]).flatten().tolist())
        assert allowed == set(_idx(0, 0))


def test_questions_are_isolated_from_each_other() -> None:
    m = _mask(cross_option=True)
    for i in _idx(0, 1):
        for j in _idx(0, 2):
            assert not m[i, j] and not m[j, i]


def test_everyone_sees_the_state_of_their_own_document() -> None:
    m = _mask(cross_option=False)
    for i in _idx(0, 1) + _idx(0, 2):
        assert all(m[i, j] for j in _idx(0, 0))
    for i in _idx(1, 1):
        assert all(m[i, j] for j in _idx(1, 0))
        assert not any(m[i, j] for j in _idx(0, 0))


def test_header_does_not_see_options_and_options_see_header() -> None:
    m = _mask(cross_option=True)
    header, opt1 = _idx(0, 1, 0), _idx(0, 1, 1)
    for h in header:
        assert not any(m[h, j] for j in opt1)
    for i in opt1:
        assert all(m[i, h] for h in header)


def test_cross_option_flag_controls_sibling_attention() -> None:
    opt1, opt2 = _idx(0, 1, 1), _idx(0, 1, 2)
    on, off = _mask(cross_option=True), _mask(cross_option=False)
    assert all(on[i, j] for i in opt1 for j in opt2)
    assert not any(off[i, j] for i in opt1 for j in opt2)


def test_no_row_is_fully_masked_and_padding_only_sees_itself() -> None:
    m = _mask(cross_option=True)
    assert bool(m.diagonal().all())
    for i in [16, 17]:
        assert torch.nonzero(m[i]).flatten().tolist() == [i]


def test_sliding_window_uses_position_distance() -> None:
    # Two questions that both restart at position 3, far apart in token index.
    pos = torch.tensor([[0, 1, 2, 3, 4, 3, 4]])
    q = torch.tensor([[0, 0, 0, 1, 1, 2, 2]])
    zeros = torch.zeros_like(q)
    base = decision_mask(zeros, q, zeros, cross_option=True)
    local = sliding_window(base, pos, window=1)[0, 0]
    # Both questions see exactly the same state tokens (position 2 only).
    assert local[3].tolist()[:3] == [False, False, True]
    assert local[5].tolist()[:3] == [False, False, True]
