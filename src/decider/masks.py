"""Attention masks derived from segment labels.

Query token ``i`` may attend to key token ``j`` iff both belong to the same document and

1. ``j`` is a state token (everyone reads the state), or
2. ``i`` and ``j`` are in the same question and ``j`` is part of its header, or
3. ``i`` and ``j`` are in the same option of the same question, or
4. ``cross_option`` is on and ``i``, ``j`` are both option tokens of the same question,

and, additionally, state tokens only attend to state tokens. That last rule makes the state's
keys and values independent of the questions, which is what allows caching the encoded state.

Every token also attends to itself, so no query row is ever fully masked (a fully masked row
makes SDPA return NaN, which would then leak into gradients through padding).
"""

from __future__ import annotations

import torch


def decision_mask(
    doc_ids: torch.Tensor,
    question_ids: torch.Tensor,
    option_ids: torch.Tensor,
    cross_option: bool,
    query_slice: slice | None = None,
) -> torch.Tensor:
    """Boolean mask of shape ``(B, 1, Lq, Lk)``; ``True`` means "may attend".

    ``query_slice`` restricts the query axis (used when the state is served from a cache and
    only question tokens are run).
    """
    sl = query_slice or slice(None)
    doc_q, doc_k = doc_ids[:, sl, None], doc_ids[:, None, :]
    q_q, q_k = question_ids[:, sl, None], question_ids[:, None, :]
    o_q, o_k = option_ids[:, sl, None], option_ids[:, None, :]

    same_question = q_q == q_k
    key_is_state = q_k == 0
    allowed = same_question & ((o_k == 0) | (o_q == o_k))
    if cross_option:
        allowed |= same_question & (o_q > 0) & (o_k > 0)
    allowed |= key_is_state
    allowed &= (q_q != 0) | key_is_state
    allowed &= (doc_q == doc_k) & (doc_q >= 0)

    length_q, length_k = allowed.shape[1], allowed.shape[2]
    q_pos = torch.arange(length_k, device=doc_ids.device)[sl]
    k_pos = torch.arange(length_k, device=doc_ids.device)
    eye = q_pos[:, None] == k_pos[None, :]
    assert eye.shape == (length_q, length_k)
    allowed |= eye
    return allowed.unsqueeze(1)


def sliding_window(
    mask: torch.Tensor,
    position_ids: torch.Tensor,
    window: int,
    query_slice: slice | None = None,
) -> torch.Tensor:
    """Restrict ``mask`` to keys within ``window`` positions of the query.

    Distance is measured in *position ids*, not token indices. With tree positions a question
    sitting at the end of a long packed row is still "next to" the state, exactly as if it were
    the only question, so local attention preserves question independence.
    """
    sl = query_slice or slice(None)
    distance = (position_ids[:, sl, None] - position_ids[:, None, :]).abs()
    return mask & (distance <= window).unsqueeze(1)
