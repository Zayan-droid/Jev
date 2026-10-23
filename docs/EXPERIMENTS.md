# Experiment log

One entry per experiment: what it was for, how it was run, what came out, what it changed.
Hardware throughout: one RTX 3050 Laptop GPU (6 GB), bf16. Run directories are local
(`runs/`); metrics, configs and reports are in the MLflow store.

## E1. Model v0.1: ModernBERT-base on `mix-v0`

**Question.** Does the packed design work with a pretrained encoder at all, and how calibrated
is it out of the box?

**Setup.** `experiment=modernbert-mix` at commit `501a502`; 8k-token batches, encoder LR 5e-5,
head LR 1e-3, 500 warmup steps, evaluation every 1k steps, early stopping on validation log loss.

**Result.** Validation log loss was best at step 3,000 (~1.1 epochs) and rose afterwards while
accuracy kept creeping up: the model became overconfident (validation ECE 0.012 → 0.044).
Early stopping ended the run at step 7,000 and the step-3,000 weights were released. Test:
noul 92.5 %, choice 90.9 %, score 77.4 %; ECE 0.006 / 0.014 / 0.025 after temperature
scaling (T ≈ 1.2-1.3). Held out: SciQ 86 %, CLINC150 70 %, but held-out counting templates
failed (`lookup.health` 4.6 %, `arith.difference` 29.9 %), confidently (score ECE 0.57).

**Decisions.** Select checkpoints on log loss, never accuracy. Held-out template failure
became the main target (E3, E4). Details: [MODEL_CARD.md](MODEL_CARD.md).

## E2. Toy model: byte tokenizer vs BPE

**Question.** The from-scratch `tiny` model (5M encoder parameters) learned the programmatic
tasks very slowly with the byte tokenizer. Is the tokenizer the bottleneck?

**Setup.** `experiment=toy` on `toy-v1`, 16k-token batches, LR 1e-3, identical except
`model.tokenizer`.

| tokenizer | val accuracy, step 1,000 | step 3,000 | test accuracy, step 6,000 | test ECE |
|---|---|---|---|---|
| bytes | 52.8 % | 57.9 % | (stopped) | |
| ModernBERT BPE | 69.5 % | 82.8 % | 91.0 % | 0.005 |

**Result.** BPE is decisively better. With bytes, every word has to be assembled from
characters before a question can be matched against the state, and the same token budget holds
fewer examples per batch. The small model still generalises poorly to held-out templates
(30.3 % accuracy, ECE 0.42), far below the pretrained model.

**Decision.** The toy experiment uses the BPE tokenizer. The byte tokenizer stays for the CI
smoke run, where it keeps the pipeline download-free.

## E3. Template diversity: `mix-v1` (model v0.2)

**Question.** Is the held-out template failure a data problem? v1 training data had two kinds
of `score` question, always worded the same way, so a model could learn level *positions*
instead of reading level descriptions.

**Hypothesis.** Adding counting, range and comparison questions to every training family,
with randomised level wording (template set v2), improves accuracy and calibration on the
*unchanged* held-out templates without hurting the public datasets.

**Setup.** `experiment=modernbert-mix-v1`: the E1 recipe on `mix-v1`, which differs from
`mix-v0` only in the synthetic template set (same sources, sizes and seed). Both models are
compared on the questions both datasets share (14,335 test records with 19,749 questions;
5,986 held-out records with 7,004 questions) with a record-level paired bootstrap
(`decider-compare`). Validation log loss was best at step 5,000 (v0.1: 3,000).

**Result.** Accuracy, v0.1 → v0.2, on the shared questions (`*`: the 95 % interval of the paired
difference excludes zero):

| split | group | n | v0.1 | v0.2 | difference | 95 % CI |
|---|---|---|---|---|---|---|
| test | all | 19,749 | 86.00 | 87.10 | +1.10 * | [+0.71, +1.52] |
| test | noul / choice / score | | | | +0.88 * / +1.19 * / +1.53 * | |
| test | intent routing (Banking77) | 3,056 | 88.02 | 91.16 | +3.14 * | [+2.09, +4.12] |
| heldout | all | 7,004 | 60.32 | 59.01 | −1.31 * | [−2.23, −0.40] |
| heldout | score | 1,486 | 17.29 | 23.55 | +6.26 * | [+3.43, +9.09] |
| heldout | choice | 4,307 | 73.69 | 69.28 | −4.41 * | [−5.36, −3.46] |
| heldout | lookup (counting templates) | 740 | 4.46 | 23.78 | +19.32 * | [+15.95, +22.84] |
| heldout | arithmetic | 746 | 30.03 | 23.32 | −6.70 * | [−10.99, −2.41] |
| heldout | intent routing (CLINC150) | 2,000 | 70.00 | 65.90 | −4.10 * | [−5.40, −2.85] |
| heldout | inventory / dates | 750 / 1,768 | | | −4.00 * / −3.45 * | |
| heldout | science QA (SciQ) | 1,000 | 86.00 | 84.80 | −1.20 | [−2.90, +0.50] |

Calibration on held-out `score` questions improved (ECE 0.57 → 0.47) but stays far from
usable. In distribution, calibration stays good but moves (test ECE noul 0.006 → 0.014,
choice 0.014 → 0.014, score 0.025 → 0.013).

The hypothesis holds only in part. The targeted failure moved: counting `lookup` questions
went from 4.5 % to 23.8 %, and held-out `score` questions gained 6 points. Everything else
held out got worse by 3-7 points, including CLINC150 and inventory questions that the change
did not touch, so the overall held-out difference is negative and significant. One
explanation, not tested here, is the longer fit (best step 5,000 vs 3,000): checkpoints are
selected on in-distribution validation loss, which rewards specialising to the training
templates.

**Decisions.** Neither model dominates, so the choice is a product decision: v0.2 is better on
in-distribution questions (where automation would be gated) and v0.1 transfers better. v0.2
passed the release gate and is registered as `challenger`; `champion` stays unassigned until
an owner promotes one. The README and model card keep describing v0.1. Next: an early-stopping
signal that sees shifted templates (a validation split built from templates disjoint from both
training and `heldout`), so selection stops rewarding specialisation.

## E4. Question familiarity

**Question.** Can the model tell when a question is unlike anything it was calibrated on?

**Setup.** Mahalanobis distance of each question's readout features to a per-type Gaussian
(shrunk covariance) fitted on 70 % of the calibration records; the threshold is the 95th
percentile on the other 30 %, so about 5 % of in-distribution questions are flagged by design
(`decider-familiarity`). Measured on up to 6,000 records per split: the flagged share, AUROC for
separating test from held-out questions (*shift*), and AUROC for ranking wrong answers above
right ones (*errors*) against the obvious baseline, low confidence.

**Result.**

| model | type | flagged, test | flagged, heldout | AUROC shift | AUROC errors on heldout: familiarity / confidence |
|---|---|---|---|---|---|
| v0.1 | noul | 3.5 % | 4.5 % | 0.868 | 0.535 / 0.436 |
| v0.1 | choice | 4.7 % | 58.1 % | 0.959 | 0.464 / 0.813 |
| v0.1 | score | 5.4 % | 100.0 % | 1.000 | 0.713 / 0.555 |
| v0.2 | noul | 6.5 % | 47.6 % | 0.914 | 0.371 / 0.488 |
| v0.2 | choice | 4.5 % | 53.6 % | 0.932 | 0.441 / 0.794 |
| v0.2 | score | 5.4 % | 81.7 % | 0.977 | 0.434 / 0.465 |

* The score detects *shift* well (AUROC 0.87-1.00): held-out `score` questions, where the
  model is right 17-24 % of the time, are almost all flagged.
* It does not detect *errors*: within a split, flagged questions are about as accurate as the
  rest, and its error AUROC (0.37-0.71) is below that of low confidence in distribution (0.83-0.92
  on test). On held-out `noul` and `score` questions, confidence is no better than chance
  (0.44-0.56): the model is confidently wrong there, and nothing computed from one answer
  catches it.
* Two bugs found on the way. Thresholding on the points the Gaussian was fitted on flagged
  8-14 % of test questions instead of 5 % (in 768 dimensions, in-sample distances are biased
  low). Splitting calib into fit and threshold parts by the dataset's own id hash put every
  public-source record on one side, because dataset splits are assigned by that hash; the
  split hash is now salted, and a test pins both.
* Serving reproduces the offline numbers: replaying test traffic through the v0.2 server
  flagged 4.3-5.3 % of answers per type, held-out traffic 42 / 50 / 78 % (noul / choice / score).

**Decisions.** The detector ships in every bundle (training fits it next to the temperatures)
as a *population* signal: flagged questions are returned in the response, counted in
`decider_unfamiliar_total`, shown on the serving dashboard, alerted on above 20 %, and queued for
labelling with the low-confidence ones. It is not a per-answer correctness score, and
confidence remains what automation is gated on.

## E5. Value of pretraining (planned)

`experiment=small-mix`: a 34M-parameter model (8 layers, width 384, ModernBERT's layout and
tokenizer) trained from scratch on `mix-v0` with the E1 recipe, to put a number on what
ModernBERT's pretraining contributes.

## E6. Serving throughput: dynamic batching and ONNX

**Question.** What do dynamic batching (`--max-batch-size`) and the ONNX export buy on this
hardware?

**Setup.** v0.2 bundle. Load test: `decider-loadtest`, 500 requests sampled from `mix-v1` test,
server (bf16, RTX 3050 laptop) and client on the same machine, end-to-end HTTP latency. ONNX:
`decider-export`, fp32 on an i5-13420H CPU (8 cores), onnxruntime 1.31 vs PyTorch 2.11, p50 of
10 runs per shape.

**Result.** Dynamic batching (up to 16 requests, 5 ms window) vs one request per pass:

| concurrency | requests/s | p50 ms | p95 ms | batched: requests/s | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 1 | 19.9 | 48.0 | 75.1 | 16.0 | 60.1 | 92.8 |
| 4 | 25.6 | 152.1 | 207.9 | 32.7 | 114.6 | 210.1 |
| 16 | 25.1 | 623.7 | 753.2 | 42.6 | 379.4 | 468.4 |

ONNX export: max |logit difference| to PyTorch 1.0e-5; CPU latency p50:

| state tokens | questions | PyTorch | onnxruntime |
|---|---|---|---|
| 32 | 1 | 108.8 ms | 39.2 ms |
| 128 | 4 | 365.4 ms | 159.0 ms |
| 384 | 8 | 731.5 ms | 638.6 ms |

**Decisions.** Without batching the GPU saturates at about 25 requests/s and latency grows
linearly with concurrency; batching raises throughput 1.7x at concurrency 16 and cuts p50 by
40 %, but costs 12 ms at concurrency 1 (the wait window, and no state cache). Batching stays
off by default and is the setting for concurrent traffic. For CPU-only serving, onnxruntime is
2-3x faster on short requests and about equal on long ones.
