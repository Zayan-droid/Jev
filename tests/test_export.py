"""The ONNX graph reproduces the PyTorch model for every batch shape the server produces."""


import pytest
import torch

from decider.export import OnnxDecider, export_onnx, max_abs_difference
from decider.packing import Example, PackingLimits, RequestEncoder, collate
from decider.schema import DecisionRequest
from decider.tokenization import ByteTokenizer
from tests.conftest import make_model

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

LIMITS = PackingLimits(max_seq_len=512, max_state_tokens=256)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    model = make_model(local=True)  # sliding-window layers exercise both mask kinds
    path = export_onnx(model, ByteTokenizer(), LIMITS, tmp_path_factory.mktemp("onnx") / "m.onnx")
    return model, OnnxDecider(path, model.log_temperature)


def request(i: int, kinds: str = "ncs", options: int = 3) -> DecisionRequest:
    questions = {}
    if "n" in kinds:
        questions["n"] = {"type": "noul", "instructions": f"Is item {i} late?"}
    if "c" in kinds:
        questions["c"] = {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {f"t{j}": f"team {j}" for j in range(options)},
        }
    if "s" in kinds:
        questions["s"] = {
            "type": "score",
            "instructions": "How urgent?",
            "levels": ["lo", "mid", "hi"],
        }
    return DecisionRequest(state={"ticket": f"order {i} " + "x" * (10 * i)}, questions=questions)


@pytest.mark.parametrize(
    ("requests", "pack"),
    [
        ([request(0)], False),
        ([request(i, options=2 + i) for i in range(4)], True),  # packed rows, ragged options
        ([request(0, kinds="n")], False),  # no choice or score questions: empty index tensors
        ([request(1, kinds="c", options=7), request(2, kinds="s")], True),
    ],
    ids=["single", "packed-ragged", "noul-only", "mixed"],
)
def test_onnx_matches_pytorch(exported: tuple, requests: list, pack: bool) -> None:
    model, runner = exported
    encoder = RequestEncoder(ByteTokenizer(), LIMITS)
    batch = collate([Example(encoder.encode(r)) for r in requests], LIMITS.max_seq_len, pack=pack)
    with torch.no_grad():
        reference = model(batch)
    logits = runner(batch)
    assert max_abs_difference(reference, logits) < 1e-4
    probs = runner.probabilities(logits)
    if probs["choice"].numel():
        torch.testing.assert_close(probs["choice"].sum(-1), torch.ones(probs["choice"].shape[0]))
