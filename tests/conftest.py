from __future__ import annotations

import random

import pytest
import torch

from decider.model import DeciderModel, EncoderConfig, ModelConfig, build_model
from decider.packing import Example, PackingLimits, RequestEncoder
from decider.schema import DecisionRequest
from decider.tokenization import ByteTokenizer

LIMITS = PackingLimits(max_seq_len=768, max_state_tokens=256, max_question_tokens=96)


def tiny_config(local: bool = False, cross_option_layers: int = 1) -> ModelConfig:
    encoder = EncoderConfig(
        d_model=64,
        n_layers=3,
        n_heads=4,
        d_ff=128,
        global_every=2 if local else 1,
        local_window=6,
    )
    return ModelConfig(encoder=encoder, cross_option_layers=cross_option_layers, head_dropout=0.0)


def make_model(local: bool = False, cross_option_layers: int = 1, seed: int = 0) -> DeciderModel:
    torch.manual_seed(seed)
    model, _ = build_model(tiny_config(local, cross_option_layers))
    # The heads' last layer is zero-initialised (uniform outputs); randomise it so tests
    # compare informative numbers rather than constants.
    for head in model.heads.heads.values():
        torch.nn.init.normal_(head.out.weight, std=1.0)
    return model.eval()


@pytest.fixture(params=[False, True], ids=["global", "sliding"])
def model(request: pytest.FixtureRequest) -> DeciderModel:
    return make_model(local=request.param)


@pytest.fixture
def encoder() -> RequestEncoder:
    return RequestEncoder(ByteTokenizer(), LIMITS)


def sample_request(seed: int = 0) -> DecisionRequest:
    rng = random.Random(seed)
    colors = ["red", "green", "blue", "amber", "violet"]
    rng.shuffle(colors)
    return DecisionRequest(
        state={"ticket": f"order {rng.randint(100, 999)} arrived damaged", "priority": "high"},
        questions={
            "refund": {"type": "noul", "instructions": "Should `ticket` get a refund?"},
            "color": {
                "type": "choice",
                "instructions": "Which colour fits the mood of `ticket`?",
                "criteria": {c: f"the mood is {c}" for c in colors},
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is `ticket`?",
                "levels": ["trivial", "minor", "major", "critical"],
            },
            "escalate": {"type": "noul", "instructions": "Does this need a manager?"},
        },
    )


def encode(encoder: RequestEncoder, request: DecisionRequest) -> Example:
    return Example(encoder.encode(request))
