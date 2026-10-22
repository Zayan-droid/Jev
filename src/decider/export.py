"""ONNX export of the decision model.

The exported graph takes the packed batch as plain tensors (token ids, segment labels, readout
indices) and returns raw logits per question type; the attention masks and tree-position
rotary embeddings are computed inside the graph, so every structural guarantee carries over.
Rows, sequence length, question counts and option counts are all dynamic. Temperatures stay
outside the graph (they live in the bundle config), exactly as in the PyTorch model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from decider.model.decider import DeciderModel, to_probabilities
from decider.packing import Batch, Example, PackingLimits, RequestEncoder, collate
from decider.schema import DecisionRequest
from decider.tokenization import Tokenizer

log = logging.getLogger(__name__)

INPUT_NAMES = (
    "input_ids",
    "position_ids",
    "level_ids",
    "doc_ids",
    "question_ids",
    "option_ids",
    "noul_index",
    "choice_index",
    "choice_mask",
    "score_index",
    "score_mask",
)
OUTPUT_NAMES = ("noul", "choice", "score")


class _TensorInterface(nn.Module):
    """The model behind a flat tensor signature, as ONNX requires."""

    def __init__(self, model: DeciderModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        level_ids: torch.Tensor,
        doc_ids: torch.Tensor,
        question_ids: torch.Tensor,
        option_ids: torch.Tensor,
        noul_index: torch.Tensor,
        choice_index: torch.Tensor,
        choice_mask: torch.Tensor,
        score_index: torch.Tensor,
        score_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = Batch(
            input_ids=input_ids,
            position_ids=position_ids,
            doc_ids=doc_ids,
            question_ids=question_ids,
            option_ids=option_ids,
            level_ids=level_ids,
            index={"noul": noul_index, "choice": choice_index, "score": score_index},
            option_mask={"choice": choice_mask, "score": score_mask},
            targets=None,
            readouts={},
            num_tokens=0,
        )
        logits = self.model(batch)
        return logits["noul"], logits["choice"], logits["score"]


def batch_tensors(batch: Batch) -> tuple[torch.Tensor, ...]:
    return (
        batch.input_ids,
        batch.position_ids,
        batch.level_ids,
        batch.doc_ids,
        batch.question_ids,
        batch.option_ids,
        batch.index["noul"],
        batch.index["choice"],
        batch.option_mask["choice"],
        batch.index["score"],
        batch.option_mask["score"],
    )


def example_batch(tokenizer: Tokenizer, limits: PackingLimits) -> Batch:
    """Two requests with every question type: the shapes the exporter traces with."""
    questions: dict[str, Any] = {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "payments",
                "returns": "damaged items",
                "other": "anything else",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "levels": ["low", "medium", "high"],
        },
    }
    encoder = RequestEncoder(tokenizer, limits)
    requests = [
        DecisionRequest(state={"ticket": f"order {i} arrived broken"}, questions=questions)
        for i in range(2)
    ]
    examples = [Example(t) for t in encoder.encode_many(requests)]
    return collate(examples, limits.max_seq_len, pack=False, pad_id=tokenizer.pad_id)


def export_onnx(
    model: DeciderModel, tokenizer: Tokenizer, limits: PackingLimits, path: Path
) -> Path:
    model = model.to("cpu").eval()
    interface = _TensorInterface(model).eval()
    rows, length = torch.export.Dim("rows"), torch.export.Dim("length")
    noul = torch.export.Dim("noul_questions")
    choice, choice_k = torch.export.Dim("choice_questions"), torch.export.Dim("choice_options")
    score, score_k = torch.export.Dim("score_questions"), torch.export.Dim("score_levels")
    seq = {0: rows, 1: length}
    dynamic_shapes = (
        *(seq,) * 6,
        {0: noul},
        {0: choice, 1: choice_k},
        {0: choice, 1: choice_k},
        {0: score, 1: score_k},
        {0: score, 1: score_k},
    )
    batch = example_batch(tokenizer, limits)
    start = time.perf_counter()
    with torch.no_grad():
        program = torch.onnx.export(
            interface,
            batch_tensors(batch),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            verbose=False,
        )
    if program is None:  # only the legacy TorchScript exporter returns None
        raise RuntimeError("torch.onnx.export returned no program")
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))
    log.info("exported %s in %.1f s", path, time.perf_counter() - start)
    return path


class OnnxDecider:
    """Runs an exported graph on a :class:`~decider.packing.Batch` with ONNX Runtime."""

    def __init__(self, path: Path | str, log_temperature: torch.Tensor | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.log_temperature = log_temperature

    def __call__(self, batch: Batch) -> dict[str, torch.Tensor]:
        feeds = {
            name: t.cpu().numpy() for name, t in zip(INPUT_NAMES, batch_tensors(batch), strict=True)
        }
        outputs = self.session.run(list(OUTPUT_NAMES), feeds)
        return {
            name: torch.from_numpy(np.asarray(o))
            for name, o in zip(OUTPUT_NAMES, outputs, strict=True)
        }

    def probabilities(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return to_probabilities(logits, self.log_temperature)


def max_abs_difference(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Largest difference between two sets of logits, ignoring the -inf padding slots."""
    worst = 0.0
    for name in OUTPUT_NAMES:
        x, y = a[name].float(), b[name].float()
        if x.numel() == 0:
            continue
        finite = torch.isfinite(x) & torch.isfinite(y)
        if not torch.equal(torch.isfinite(x), torch.isfinite(y)):
            return float("inf")
        worst = max(worst, float((x[finite] - y[finite]).abs().max()))
    return worst
