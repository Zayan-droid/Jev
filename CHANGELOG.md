# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the released model
(`docs/MODEL_CARD.md`).

## [Unreleased]

### Added
- Synthetic template set v2 (counting, range and comparison questions in every training family,
  randomised level wording) and dataset `mix-v1`; template set v1 stays frozen and rebuilds
  `mix-v0` byte for byte.
- Dynamic micro-batching in the server (`--max-batch-size`, `--max-wait-ms`): concurrent
  requests share one packed forward pass; batched answers equal single-request answers.
- `decider-label --sample-from` builds the hand-labelled gold set from a stratified sample, with
  labels and model answers hidden.
- ONNX export (`decider-export`, `export` extra): the masks and tree positions are computed inside
  the graph, all shapes are dynamic, and logits match PyTorch to ~1e-7 (tested on packed and
  empty batches).
- Paired model comparison (`decider-compare`) with a record-level bootstrap.
- Question familiarity detector (`decider-familiarity`): Mahalanobis distance of readout
  features to the calibration questions, fitted at training time and stored in the bundle. The
  server returns flagged questions (`unfamiliar`), counts them (`decider_unfamiliar_total`),
  queues them for labelling and alerts when their share stays above 20 %.
- Load testing (`decider-loadtest`) and an experiment log (`docs/EXPERIMENTS.md`) with the
  v0.1 / v0.2 comparison, the familiarity study and serving throughput.
- Model v0.2 (`mix-v1`), registered as `challenger`: +1.1 points in distribution, −1.3 points
  held out (paired, both significant).
- Request size limits (100k-character states, 2k-character instructions, options and levels).
- Offline tests for ModernBERT loading (a tiny checkpoint saved by `transformers`) and for the
  Hugging Face tokenizer path; CI smoke-tests the serving container with a named log volume.

### Changed
- The toy experiment uses the ModernBERT tokenizer: 91.0 % test accuracy in 30 minutes, where the
  byte tokenizer was at ~58 % validation accuracy halfway through.
- Tokenized splits are stored as one int32 array per example (18 instead of ~32 bytes per
  token on mix-v1) and tokenized in streaming chunks, so loading a split never holds the whole
  split as Python lists.

### Fixed
- ModernBERT configs written by `transformers` 5 (`layer_types`, `rope_parameters`) were read
  with default values.
- A failed prediction-log write returned HTTP 500; log writes are now best-effort, counted and
  alerted on. The serving image creates its log directories for the non-root user.
- Six dashboard issues found under live traffic (missing zero series, stacked status lines,
  idle ratios shown as 0 %, mixed axis scales, coarse latency buckets, unreadable legends).

## [0.1.0] - 2026-10-22

First model: ModernBERT-base fine-tuned on `mix-v0` (test accuracy noul 92.5 %, choice 90.9 %,
score 77.4 %; ECE 0.006-0.025 after temperature scaling).

### Added
- Packed encoder with segment masks, tree position ids and per-type readout heads; ModernBERT
  weights load into the same module with exact hidden-state parity.
- State KV cache, proper-scoring-rule training objective, temperature scaling per question type.
- Dataset builders for six public datasets and programmatic task families, with deduplication,
  state-level leakage removal and SHA-256 manifests; DVC pipeline.
- Trainer with bit-exact resume, MLflow tracking, a Prometheus exporter and training-health
  monitors; evaluation harness with calibration, selective prediction, zero-shot, robustness
  and latency reports; zero-shot NLI and embedding baselines.
- Release gate, MLflow model registry with `challenger` / `champion` aliases.
- FastAPI server with Prometheus metrics, confidence-drift monitoring, PII-redacted prediction
  log and review queue; Python client; Docker image; Prometheus/Grafana stack with alert rules.

### Fixed
- Cross-option attention from the first layer made all options of a question identical;
  sibling attention is limited to the last layers.
- Sliding-window attention measured distance in token index, which broke question independence
  in long packed rows; it now uses position ids.
- Robustness checks sampled a single source; structural invariance is measured in fp32 on a
  random sample, bf16 noise reported separately.
- MLflow rejected release-gate tag names containing brackets.

[Unreleased]: https://github.com/Zayan-droid/Jev/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zayan-droid/Jev/releases/tag/v0.1.0
