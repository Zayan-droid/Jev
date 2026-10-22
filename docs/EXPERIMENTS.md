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
compared on the questions both datasets share (14,335 test, 5,986 held-out records) with a
record-level paired bootstrap (`decider-compare`).

**Result.** Training in progress.

## E4. Question familiarity

**Question.** Can the model tell when a question is unlike anything it was calibrated on?

**Setup.** Mahalanobis distance of each question's readout features to a per-type Gaussian
fitted on the calibration split (`decider-familiarity`); questions beyond the 95th percentile
are flagged. Measured: AUROC for separating test from held-out questions, and accuracy of
flagged vs unflagged questions.

**Result.** Pending (runs after E3 on both models).

## E5. Value of pretraining (planned)

`experiment=small-mix`: a 34M-parameter model (8 layers, width 384, ModernBERT's layout and
tokenizer) trained from scratch on `mix-v0` with the E1 recipe, to put a number on what
ModernBERT's pretraining contributes.
