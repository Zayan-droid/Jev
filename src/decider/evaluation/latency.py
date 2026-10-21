"""Latency and memory benchmarks for the full pass and the cached-state pass."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from decider.model.decider import DeciderModel, StateCache
from decider.packing import (
    Batch,
    Example,
    PackingLimits,
    RequestEncoder,
    TokenizedRequest,
    collate,
)
from decider.schema import DecisionRequest
from decider.tokenization import Tokenizer

_WORDS = [
    "the",
    "parcel",
    "arrived",
    "late",
    "and",
    "the",
    "customer",
    "asked",
    "for",
    "a",
    "refund",
    "on",
    "the",
    "damaged",
    "order",
]


def synthetic_request(tokenizer: Tokenizer, state_tokens: int, n_questions: int) -> DecisionRequest:
    """A request whose state is roughly ``state_tokens`` long with ``n_questions`` questions
    cycling through the three types."""
    text, words = "", 0
    while len(tokenizer.encode(text)) < state_tokens:
        chunk = " ".join(_WORDS[(words + i) % len(_WORDS)] for i in range(32))
        text = f"{text} {chunk}".strip()
        words += 32
    questions: dict[str, Any] = {}
    for i in range(n_questions):
        kind = ("noul", "choice", "score")[i % 3]
        if kind == "noul":
            questions[f"q{i}"] = {"type": "noul", "instructions": "Is a refund required?"}
        elif kind == "choice":
            questions[f"q{i}"] = {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {t: f"the {t} team" for t in ("billing", "shipping", "support")},
            }
        else:
            questions[f"q{i}"] = {
                "type": "score",
                "instructions": "How urgent is this?",
                "levels": ["low", "medium", "high", "critical"],
            }
    return DecisionRequest(state=text, questions=questions)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn: Any, device: torch.device, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return times


def _measure(
    model: DeciderModel,
    tokenized: TokenizedRequest,
    batch: Batch,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    holder: dict[str, StateCache] = {}

    def full() -> None:
        with autocast:
            model(batch)

    def encode_state() -> None:
        with autocast:
            holder["cache"] = model.encode_state(tokenized.state)

    def cached() -> None:
        with autocast:
            model.forward_cached(holder["cache"], batch)

    t_full = _time(full, device, repeats, warmup)
    t_state = _time(encode_state, device, repeats, warmup)
    t_cached = _time(cached, device, repeats, warmup)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    return {
        "full_p50_ms": float(np.percentile(t_full, 50)),
        "full_p95_ms": float(np.percentile(t_full, 95)),
        "state_encode_p50_ms": float(np.percentile(t_state, 50)),
        "cached_p50_ms": float(np.percentile(t_cached, 50)),
        "cached_p95_ms": float(np.percentile(t_cached, 95)),
        "peak_memory_mb": peak,
    }


@torch.inference_mode()
def benchmark(
    model: DeciderModel,
    tokenizer: Tokenizer,
    limits: PackingLimits,
    device: str | torch.device,
    state_tokens: Sequence[int] = (128, 512),
    question_counts: Sequence[int] = (1, 8, 32),
    repeats: int = 20,
    warmup: int = 3,
) -> list[dict[str, Any]]:
    device = torch.device(device)
    model = model.to(device).eval()
    encoder = RequestEncoder(tokenizer, limits)
    rows = []
    for s in state_tokens:
        if s > limits.max_state_tokens:
            continue
        for n in question_counts:
            try:
                tokenized = encoder.encode(synthetic_request(tokenizer, s, n))
            except ValueError:
                continue  # does not fit in max_seq_len
            batch = collate([Example(tokenized)], limits.max_seq_len, pack=False).to(device)
            row = _measure(model, tokenized, batch, device, repeats, warmup)
            rows.append(
                {
                    "state_tokens": len(tokenized.state),
                    "questions": n,
                    "total_tokens": tokenized.length,
                    **row,
                    "device": str(device),
                }
            )
    return rows
