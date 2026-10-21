"""Step-based training loop with evaluation, checkpointing and telemetry.

``Trainer.fit`` runs the whole lifecycle of one experiment:

1. build model + data, resume from ``last.pt`` when the run directory already has one;
2. train with bf16 autocast, gradient accumulation and clipping, logging throughput, GPU and
   health metrics;
3. evaluate on ``val`` periodically (log loss, ECE, reliability diagram), keep the best
   checkpoint by validation log loss and stop early when it stops improving;
4. restore the best weights, fit temperatures on ``calib``, run the full evaluation harness;
5. export a model bundle and log everything to MLflow.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from decider.bundle import save_bundle
from decider.data.dataset import EncodedSplit, MixtureSampler, TrainBatches
from decider.data.io import load_manifest
from decider.evaluation import plots
from decider.evaluation.collect import collect_results
from decider.evaluation.metrics import grouped, score_results, summarize_scored
from decider.evaluation.report import EvalConfig, evaluate, sample_examples
from decider.inference import Predictor
from decider.model import DeciderModel, ModelConfig, build_model
from decider.packing import PackingLimits, RequestEncoder
from decider.training.checkpoint import CheckpointManager, rng_state, set_rng_state
from decider.training.health import SpikeDetector, UpdateRatioTracker, check_finite
from decider.training.losses import LossWeights, decision_loss
from decider.training.optim import build_optimizer, warmup_cosine
from decider.training.telemetry import Telemetry, TelemetryConfig
from decider.utils.gpu import GpuMonitor
from decider.utils.provenance import environment_info, git_info

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    max_steps: int = 2000
    max_batch_tokens: int = 16_384
    grad_accum: int = 1
    lr: float = 5e-4
    head_lr_multiplier: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.98)
    grad_clip: float = 1.0
    brier_weight: float = 0.25
    rps_weight: float = 0.5
    precision: str = "bf16"
    mixture_alpha: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)
    max_options: int | None = 24
    eval_every: int = 500
    eval_examples: int = 2000
    eval_batch_tokens: int = 32_768
    checkpoint_every: int = 500
    keep_checkpoints: int = 2
    early_stopping_patience: int | None = None
    log_every: int = 10
    update_ratio_every: int = 100
    seed: int = 0
    device: str = "auto"
    compile: bool = False


@dataclass
class RunConfig:
    name: str
    run_dir: str
    data_dir: str
    model: ModelConfig
    packing: PackingLimits
    train: TrainConfig
    telemetry: TelemetryConfig
    evaluation: EvalConfig
    register: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        train = dict(d["train"])
        if "betas" in train:
            train["betas"] = tuple(train["betas"])
        return cls(
            name=d["name"],
            run_dir=d["run_dir"],
            data_dir=d["data_dir"],
            model=ModelConfig.from_dict(d["model"]),
            packing=PackingLimits(**d["packing"]),
            train=TrainConfig(**train),
            telemetry=TelemetryConfig(**d.get("telemetry", {})),
            evaluation=EvalConfig(**d.get("evaluation", {})),
            register=dict(d.get("register", {})),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class _Averager:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, values: dict[str, float]) -> None:
        for k, v in values.items():
            self.sums[k] += v
            self.counts[k] += 1

    def pop(self) -> dict[str, float]:
        out = {k: self.sums[k] / self.counts[k] for k in self.sums}
        self.sums.clear()
        self.counts.clear()
        return out


class _Throughput:
    """Tokens, examples and data-loading time since the last read."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start = time.perf_counter()
        self.tokens = self.examples = 0
        self.data_time = 0.0

    def read(self, steps: int) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        out = {
            "perf/tokens_per_s": self.tokens / elapsed,
            "perf/examples_per_s": self.examples / elapsed,
            "perf/step_time_s": elapsed / max(1, steps),
            "perf/data_time_frac": self.data_time / elapsed,
        }
        self.reset()
        return out


class Trainer:
    def __init__(self, cfg: RunConfig, config_dict: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.config_dict = config_dict or dataclasses.asdict(cfg)
        self.run_dir = Path(cfg.run_dir)
        self.data_dir = Path(cfg.data_dir)
        self.device = resolve_device(cfg.train.device)
        self.amp = cfg.train.precision == "bf16" and self.device.type == "cuda"
        self.loss_weights = LossWeights(cfg.train.brier_weight, cfg.train.rps_weight)

    # ------------------------------------------------------------------ setup

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    def _predictor(self, model: DeciderModel) -> Predictor:
        return Predictor(
            model,
            self.tokenizer,
            self.cfg.packing,
            self.device,
            max_batch_tokens=self.cfg.train.eval_batch_tokens,
        )

    def fit(self) -> dict[str, Any]:
        cfg, tc = self.cfg, self.cfg.train
        seed_everything(tc.seed)
        if not (self.data_dir / "manifest.json").exists():
            raise FileNotFoundError(
                f"no dataset at {self.data_dir}; build it first with `decider-build-data`"
            )
        manifest = load_manifest(self.data_dir)

        model, self.tokenizer = build_model(cfg.model)
        model.to(self.device)
        encoder = RequestEncoder(self.tokenizer, cfg.packing)
        train = EncodedSplit.load(self.data_dir, "train", encoder)
        val = EncodedSplit.load(self.data_dir, "val", encoder)
        val_examples = sample_examples(val, tc.eval_examples, tc.seed)
        if not len(train):
            raise ValueError("the training split is empty")

        optimizer = build_optimizer(model, tc.lr, tc.weight_decay, tc.head_lr_multiplier, tc.betas)
        scheduler = warmup_cosine(optimizer, tc.warmup_steps, tc.max_steps, tc.min_lr_ratio)
        sampler = MixtureSampler(train.sources, tc.mixture_alpha, tc.seed, tc.source_weights)
        batches = TrainBatches(
            train,
            sampler,
            cfg.packing,
            tc.max_batch_tokens,
            self.tokenizer.pad_id,
            seed=tc.seed,
            max_options=tc.max_options,
        )
        ckpt = CheckpointManager(self.run_dir / "checkpoints", tc.keep_checkpoints)

        step, best, bad_evals, examples_seen, mlflow_run_id = 0, math.inf, 0, 0, None
        if ckpt.last.exists():
            state = ckpt.load(map_location=self.device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            batches.load_state_dict(state["batches"])
            set_rng_state(state["rng"])
            step, best, bad_evals = state["step"], state["best"], state["bad_evals"]
            examples_seen = state["examples_seen"]
            mlflow_run_id = state.get("mlflow_run_id")
            log.info("resumed from step %d", step)

        train_model: Any = torch.compile(model, dynamic=True) if tc.compile else model

        telemetry = Telemetry(cfg.telemetry, self.run_dir, cfg.name)
        tags = {
            **git_info(),
            **{f"env.{k}": v for k, v in environment_info().items()},
            "data.name": manifest["name"],
            "data.hash": manifest["data_hash"],
            "model.parameters": model.num_parameters(),
        }
        telemetry.start(self.config_dict, tags, resume_run_id=mlflow_run_id)
        if mlflow_run_id is None:
            telemetry.log_dict(manifest, "data/manifest.json")
            telemetry.log_dict(self.config_dict, "config/resolved.json")
            telemetry.log_dict(self._model_summary(model, train, sampler), "model/summary.json")

        def snapshot() -> dict[str, Any]:
            progress = {
                "step": step,
                "best": best,
                "bad_evals": bad_evals,
                "examples_seen": examples_seen,
                "mlflow_run_id": telemetry.run_id,
            }
            return self._state(model, optimizer, scheduler, batches, progress)

        spikes = SpikeDetector()
        ratios = UpdateRatioTracker.for_model(model)
        gpu = GpuMonitor()
        averager = _Averager()
        meter = _Throughput()
        status = "FAILED"
        try:
            while step < tc.max_steps:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                step_parts: dict[str, float] = {}
                for _ in range(tc.grad_accum):
                    t0 = time.perf_counter()
                    batch, stats = next(batches)
                    batch = batch.to(self.device)
                    meter.data_time += time.perf_counter() - t0
                    with self._autocast():
                        logits = train_model(batch)
                    assert batch.targets is not None
                    loss, parts = decision_loss(
                        logits, batch.targets, batch.option_mask, self.loss_weights
                    )
                    check_finite(loss, step)
                    (loss / tc.grad_accum).backward()
                    meter.tokens += stats.tokens
                    meter.examples += stats.examples
                    examples_seen += stats.examples
                    step_parts["loss"] = step_parts.get("loss", 0.0) + loss.item() / tc.grad_accum
                    for k, v in parts.items():
                        step_parts[k] = step_parts.get(k, 0.0) + v.item() / tc.grad_accum
                    step_parts["packing_efficiency"] = stats.packing_efficiency

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                track_ratio = tc.update_ratio_every and (step + 1) % tc.update_ratio_every == 0
                if track_ratio:
                    ratios.snapshot(model)
                optimizer.step()
                scheduler.step()
                step += 1
                ratio_metrics = (
                    {f"health/update_ratio/{k}": v for k, v in ratios.ratios(model).items()}
                    if track_ratio
                    else {}
                )

                if spikes.update(step_parts["loss"]):
                    log.warning("loss spike at step %d: %.4f", step, step_parts["loss"])
                averager.add({**step_parts, "grad_norm": float(grad_norm)})

                if step % tc.log_every == 0 or step == tc.max_steps:
                    avg = averager.pop()
                    efficiency = avg.pop("packing_efficiency", 0.0)
                    metrics: dict[str, float] = {f"train/{k}": v for k, v in avg.items()}
                    metrics.update(meter.read(tc.log_every))
                    metrics.update(
                        {
                            "train/lr": float(scheduler.get_last_lr()[0]),
                            "perf/packing_efficiency": efficiency,
                            "health/loss_spikes": spikes.spikes,
                            "health/loss_ema": spikes.mean,
                            "progress/epoch": examples_seen / len(train),
                            **{f"gpu/{k}": v for k, v in gpu.sample().items()},
                            **ratio_metrics,
                        }
                    )
                    telemetry.log_metrics(metrics, step)
                    self._print_progress(step, metrics)
                elif ratio_metrics:
                    telemetry.log_metrics(ratio_metrics, step)

                if step % tc.eval_every == 0 or step == tc.max_steps:
                    val_loss = self._validate(model, val_examples, telemetry, step)
                    if val_loss < best - 1e-4:
                        best, bad_evals = val_loss, 0
                        ckpt.save_best(model.state_dict(), step, val_loss)
                        telemetry.log_metrics({"val/best_log_loss": best}, step)
                    else:
                        bad_evals += 1
                    patience = tc.early_stopping_patience
                    if patience is not None and bad_evals >= patience:
                        log.info(
                            "early stopping at step %d (no improvement in %d evals)",
                            step,
                            bad_evals,
                        )
                        break
                    meter.reset()  # keep evaluation time out of the throughput numbers

                if step % tc.checkpoint_every == 0:
                    ckpt.save(snapshot(), step)
                    meter.reset()

            if step % tc.checkpoint_every:
                ckpt.save(snapshot(), step)
            result = self._finish(model, ckpt, telemetry, manifest)
            status = "FINISHED"
            return result
        except KeyboardInterrupt:
            status = "KILLED"
            log.warning("interrupted at step %d; saving a resumable checkpoint", step)
            ckpt.save(snapshot(), step)
            raise
        finally:
            telemetry.finish(status)

    # ------------------------------------------------------------------ helpers

    def _state(
        self,
        model: DeciderModel,
        optimizer: Any,
        scheduler: Any,
        batches: TrainBatches,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batches": batches.state_dict(),
            "rng": rng_state(),
            "config": self.config_dict,
            **progress,
        }

    def _model_summary(
        self, model: DeciderModel, train: EncodedSplit, sampler: MixtureSampler
    ) -> dict[str, Any]:
        per_module: dict[str, int] = defaultdict(int)
        for name, p in model.named_parameters():
            per_module[".".join(name.split(".")[:2])] += p.numel()
        return {
            "parameters": model.num_parameters(),
            "per_module": dict(per_module),
            "config": model.cfg.to_dict(),
            "train_examples": len(train),
            "train_tokens": int(train.lengths.sum()),
            "mixture": sampler.mixture(),
        }

    @torch.no_grad()
    def _validate(
        self, model: DeciderModel, examples: list, telemetry: Telemetry, step: int
    ) -> float:
        model.eval()
        predictor = self._predictor(model)
        predictor.calibrated = False
        results = collect_results(predictor, examples, "val")
        scored = score_results(results)
        overall = summarize_scored(scored)
        metrics = {f"val/{k}": v for k, v in overall.items() if k != "n"}
        for t, m in grouped(results, scored, lambda r: r.type).items():
            for k in ("accuracy", "log_loss", "ece", "brier"):
                metrics[f"val_{t}/{k}"] = m[k]
        for fam, m in grouped(results, scored, lambda r: r.family).items():
            metrics[f"val_family/{fam}/accuracy"] = m["accuracy"]
            metrics[f"val_family/{fam}/ece"] = m["ece"]
        telemetry.log_metrics(metrics, step)

        types = np.array(scored.types)
        series = {
            t: (scored.confidence[types == t], scored.correct[types == t])
            for t in ("noul", "choice", "score")
        }
        fig = plots.reliability_figure(series, f"validation, step {step}")
        telemetry.log_figure(fig, f"reliability/val_step_{step:07d}.png")
        plots.close(fig)
        log.info(
            "step %d | val log_loss %.4f | acc %.4f | ece %.4f",
            step,
            overall["log_loss"],
            overall["accuracy"],
            overall["ece"],
        )
        return float(overall["log_loss"])

    def _print_progress(self, step: int, m: dict[str, float]) -> None:
        log.info(
            "step %d/%d | loss %.4f | lr %.2e | grad %.2f | %.0f tok/s | gpu %.2f GB",
            step,
            self.cfg.train.max_steps,
            m.get("train/loss", float("nan")),
            m.get("train/lr", float("nan")),
            m.get("train/grad_norm", float("nan")),
            m.get("perf/tokens_per_s", 0.0),
            m.get("gpu/gpu_mem_peak_gb", 0.0),
        )

    def _finish(
        self,
        model: DeciderModel,
        ckpt: CheckpointManager,
        telemetry: Telemetry,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if ckpt.best.exists():
            best = ckpt.load(ckpt.best, map_location=self.device)
            model.load_state_dict(best["model"])
            log.info(
                "restored best checkpoint (step %d, val log loss %.4f)",
                best["step"],
                best["metric"],
            )
        model.eval()
        eval_dir = self.run_dir / "eval"
        report = evaluate(self._predictor(model), self.data_dir, self.cfg.evaluation, eval_dir)
        final: dict[str, float] = {}
        for split, entry in report["splits"].items():
            for k, v in entry["calibrated"]["overall"].items():
                if k != "n":
                    final[f"{split}/{k}"] = v
            for t, m in entry["calibrated"]["by_type"].items():
                final[f"{split}_{t}/accuracy"] = m["accuracy"]
                final[f"{split}_{t}/ece"] = m["ece"]
            for key in ("permutation", "independence"):
                if entry.get(key, {}).get("n"):
                    final[f"{split}/{key}_max_abs_diff"] = entry[key]["max_abs_diff"]
        final.update({f"temperature/{k}": v for k, v in report["temperatures"].items()})
        telemetry.log_metrics(final, self.cfg.train.max_steps)
        telemetry.log_artifact(eval_dir, "eval")

        bundle_dir = self.run_dir / "bundle"
        metadata = {
            "version": f"{self.cfg.name}-{git_info()['git_sha'][:7]}",
            "run_name": self.cfg.name,
            "mlflow_run_id": telemetry.run_id,
            "data": {"name": manifest["name"], "hash": manifest["data_hash"]},
            "code": git_info(),
            "summary": final,
        }
        save_bundle(bundle_dir, model, self.tokenizer, self.cfg.packing, metadata)
        reference = self._reference_confidence(report)
        (bundle_dir / "reference.json").write_text(json.dumps(reference, indent=2))
        telemetry.log_artifact(bundle_dir, "bundle")
        log.info("bundle written to %s", bundle_dir)

        if self.cfg.register.get("enabled") and telemetry.run_id:
            from decider.registry import register_bundle

            version, passed = register_bundle(
                telemetry.run_id,
                self.cfg.register.get("model_name", "decider"),
                report,
                gate=self.cfg.register.get("gate", {}),
            )
            telemetry.set_tags({"registry.version": version, "gate.passed": passed})
        return {"report": report, "bundle": str(bundle_dir), "run_id": telemetry.run_id}

    @staticmethod
    def _reference_confidence(report: dict[str, Any]) -> dict[str, Any]:
        """Calibrated confidence histograms on held-in test data: the baseline the serving
        monitor compares live traffic against (see :mod:`decider.serving.monitoring`)."""
        splits = report["splits"]
        source = "test" if "test" in splits else next(iter(splits), None)
        histograms = splits[source]["confidence_histogram"] if source else {}
        return {"split": source, "histograms": histograms, "temperatures": report["temperatures"]}
