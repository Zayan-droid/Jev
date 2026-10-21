"""Baselines scored by the same harness as the model.

* ``nli`` - a zero-shot NLI cross-encoder (option C in the design): one forward pass per
  (state, question, option); ``log p(entailment)`` is the option's score.
* ``embedding`` - a sentence-embedding bi-encoder (option D): cosine similarity between the
  state (+ question) and each option.

Both are calibrated on the ``calib`` split like the model: temperature scaling for
``choice``/``score``; Platt scaling (affine) for ``noul``, because raw cosine similarities or
entailment log-odds are not centred at zero.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from decider.data.io import iter_records, split_path
from decider.evaluation import plots
from decider.evaluation.calibration import fit_temperature
from decider.evaluation.metrics import QuestionResult
from decider.evaluation.report import (
    EvalConfig,
    _json_default,
    _series,
    confidence_histograms,
    render_markdown,
    summarize_results,
)
from decider.schema import ChoiceQuestion, NoulQuestion, Record, ScoreQuestion, serialize_state

log = logging.getLogger(__name__)

NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _hypotheses(question: Any) -> list[str]:
    if isinstance(question, NoulQuestion):
        return [f'The answer to the question "{question.instructions}" is yes.']
    if isinstance(question, ChoiceQuestion):
        options = [f"{k}: {v}" for k, v in question.criteria.items()]
    else:
        assert isinstance(question, ScoreQuestion)
        options = list(question.levels)
    return [f"{question.instructions} The answer is: {o}." for o in options]


class NLIScorer:
    def __init__(self, device: torch.device, model_id: str = NLI_MODEL, batch_size: int = 32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = next(i for name, i in labels.items() if name.startswith("entail"))

    @torch.inference_mode()
    def log_p_entail(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                truncation="only_first",
                max_length=384,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.autocast(
                self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                logits = self.model(**enc).logits.float()
            out.append(torch.log_softmax(logits, -1)[:, self.entail].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, record: Record) -> dict[str, np.ndarray]:
        premise = serialize_state(record.state)
        pairs = [(name, h) for name, q in record.questions.items() for h in _hypotheses(q)]
        logp = self.log_p_entail([premise] * len(pairs), [h for _, h in pairs])
        scores: dict[str, list[float]] = {}
        for (name, _), lp in zip(pairs, logp, strict=True):
            scores.setdefault(name, []).append(float(lp))
        out = {}
        for name, values in scores.items():
            if record.questions[name].type == "noul":
                p = float(np.exp(values[0]))
                out[name] = np.array([np.log(p + 1e-9) - np.log(1 - p + 1e-9)])
            else:
                out[name] = np.array(values)
        return out


class EmbeddingScorer:
    def __init__(self, device: torch.device, model_id: str = EMBEDDING_MODEL, batch_size: int = 64):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0)

    def score(self, record: Record) -> dict[str, np.ndarray]:
        state = serialize_state(record.state)
        out = {}
        for name, q in record.questions.items():
            if isinstance(q, NoulQuestion):
                e = self.embed([state, q.instructions])
                out[name] = np.array([float(e[0] @ e[1])])
            else:
                options = (
                    [f"{k}: {v}" for k, v in q.criteria.items()]
                    if isinstance(q, ChoiceQuestion)
                    else list(q.levels)
                )
                e = self.embed([f"{state} {q.instructions}", *options])
                out[name] = (e[1:] @ e[0]).numpy().astype(np.float64)
        return out


_META_KEYS = ("injected", "negation_pair")


def _results(records: Sequence[Record], scorer: Any, split: str) -> list[QuestionResult]:
    results = []
    for i, record in enumerate(records):
        if i and i % 500 == 0:
            log.info("%s: %d/%d records", split, i, len(records))
        for name, z in scorer.score(record).items():
            q = record.questions[name]
            target = record.targets[name]
            if isinstance(q, NoulQuestion):
                assert isinstance(target, int | float)
                names: tuple[str, ...] = ()
                t = np.array([float(target)])
            elif isinstance(q, ChoiceQuestion):
                assert isinstance(target, dict)
                names = tuple(q.criteria)
                t = np.array([target.get(k, 0.0) for k in names])
            else:
                assert isinstance(target, list)
                names = tuple(str(i) for i in range(len(q.levels)))
                t = np.array(target, dtype=np.float64)
            meta = {"id": record.id}
            meta.update({k: v for k, v in record.meta.items() if k in _META_KEYS})
            results.append(
                QuestionResult(
                    record.id,
                    name,
                    q.type,
                    record.source,
                    record.family,
                    names,
                    z.astype(np.float64),
                    t,
                    meta,
                )
            )
    return results


def _fit_platt(results: Sequence[QuestionResult]) -> tuple[float, float]:
    z = torch.tensor([r.logits[0] for r in results], dtype=torch.float64)
    y = torch.tensor([r.target[0] for r in results], dtype=torch.float64)
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        params[0] = 1.0
    optimizer = torch.optim.LBFGS([params], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(params[0] * z + params[1], y)
        loss.backward()
        return loss

    optimizer.step(closure)
    a, b = params.detach().tolist()
    return float(a), float(b)


def _apply(results: Sequence[QuestionResult], platt: tuple[float, float]) -> None:
    a, b = platt
    for r in results:
        if r.type == "noul":
            r.logits = a * r.logits + b


def _sample(data_dir: Path, split: str, n: int | None, seed: int) -> list[Record]:
    path = split_path(data_dir, split)
    if not path.exists():
        return []
    records = list(iter_records(path))
    if n is not None and len(records) > n:
        records = random.Random(seed).sample(records, n)
    return records


def evaluate_baseline(
    name: str, data_dir: Path, cfg: EvalConfig, out_dir: Path, device: str = "auto"
) -> dict[str, Any]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    scorer: Any = NLIScorer(dev) if name == "nli" else EmbeddingScorer(dev)
    limit = cfg.max_examples or 2000

    calib = _results(_sample(data_dir, cfg.calibration_split, limit, cfg.seed), scorer, "calib")
    platt = _fit_platt([r for r in calib if r.type == "noul"])
    _apply(calib, platt)
    temperatures = {"noul": 1.0}
    for t in ("choice", "score"):
        temperatures[t] = fit_temperature([r for r in calib if r.type == t])
    log.info("platt=%s temperatures=%s", platt, temperatures)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "baseline": name,
        "model": NLI_MODEL if name == "nli" else EMBEDDING_MODEL,
        "temperatures": temperatures,
        "platt": {"a": platt[0], "b": platt[1]},
        "splits": {},
    }
    for split in cfg.splits:
        records = _sample(data_dir, split, limit, cfg.seed)
        if not records:
            continue
        results = _results(records, scorer, split)
        _apply(results, platt)
        report["splits"][split] = {
            "examples": len(records),
            "questions": len(results),
            "uncalibrated": summarize_results(results, None, cfg.n_bins, 0),
            "calibrated": summarize_results(results, temperatures, cfg.n_bins, cfg.bootstrap),
            "confidence_histogram": confidence_histograms(results, temperatures),
        }
        fig = plots.reliability_figure(_series(results, temperatures), f"{name}: {split}")
        fig.savefig(out_dir / "figures" / f"reliability_{split}_calibrated.png", dpi=120)
        plots.close(fig)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    (out_dir / "report.md").write_text(render_markdown(report))
    return report
