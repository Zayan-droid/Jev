# Model card: decider v0.1 (ModernBERT-base, mix-v0)

## Model

| | |
|---|---|
| Version | `modernbert-mix-501a502` (MLflow registry: `decider` v2, release gate passed) |
| Architecture | ModernBERT-base encoder (150.8 M parameters) in `decider`'s own implementation, with segment-masked packing, tree position ids, level embeddings and three readout heads (`noul`, `choice`, `score`); sibling-option attention in the last 4 layers |
| Code | commit `501a502` (clean tree) |
| Data | `mix-v0`, data hash `330653cf66f74d77` (see [DATA.md](DATA.md)) |
| Training | bf16 autocast, gradient checkpointing, AdamW (encoder 5e-5, heads 1e-3), 500 warmup steps, cosine decay, 8,192-token batches; loss = log score + 0.25·Brier + 0.5·RPS (score) |
| Selection | best validation log loss over evaluations every 1,000 steps; early stopping (patience 4) ended the run at step 7,000; the released weights are from step 3,000 (~1.1 epochs) |
| Calibration | temperature per question type, fitted on `calib`: noul 1.207, choice 1.208, score 1.301 (the selected checkpoint was mildly overconfident) |
| Familiarity | per-type Gaussian over readout features, fitted on `calib` (`familiarity.npz`, added after release with `decider-familiarity --save`) |
| Hardware | one RTX 3050 Laptop GPU (6 GB), ~5.3k tokens/s, 3 h 4 min to step 7,000 |

## Intended use

Fast, calibrated answers to typed questions about short text or JSON states, as the first
stage of an automated decision pipeline: answer automatically when confident, route the rest
to a person or a larger model. Requests: one state (≤ 384 tokens are used), up to 32
questions, up to 255 options per question, 2-10 levels per score question.

Not intended for: open-ended generation, multi-step reasoning, decisions about people
(credit, hiring, legal, medical), or question types far from the training mix without
re-evaluation (see weaknesses).

## Evaluation

All numbers after temperature scaling. Full report: `eval/report.json` and `eval/report.md`
in the MLflow run; regenerate with `decider-eval --bundle <bundle> --data data/processed/mix-v0`.

### Test split (seen task families): 17,329 records, 27,441 questions

| type | n | accuracy (95 % CI) | log loss | ECE (95 % CI) | coverage at 95 % accuracy |
|---|---|---|---|---|---|
| noul | 13,480 | 92.5 % (92.1-92.9) | 0.171 | 0.006 (0.004-0.010) | 94.0 % |
| choice | 9,522 | 90.9 % (90.3-91.4) | 0.286 | 0.014 (0.010-0.019) | 90.2 % |
| score | 4,439 | 77.4 % (76.1-78.7) | 0.536 | 0.025 (0.019-0.038) | 53.7 % |

| source | accuracy | ECE |
|---|---|---|
| AG News (topic) | 94.2 % | 0.009 |
| Banking77 (77-way intent) | 88.0 % | 0.018 |
| MNLI (entailment) | 85.8 % | 0.039 |
| Yelp (1-5 stars + recommend) | 79.7 % | 0.026 |
| BoolQ (reading QA) | 77.0 % | 0.016 |
| programmatic (inventory, arithmetic, lookup) | 98.4 % | 0.003 |

Yelp star ratings are the hardest `score` task: adjacent star levels are genuinely ambiguous
(the RPS term rewards putting mass on neighbouring levels).

### Heldout split (zero-shot): 5,990 records, 7,008 questions

| source / family | accuracy | ECE |
|---|---|---|
| SciQ (unseen task family, 4 options) | 86.0 % | 0.111 |
| CLINC150 (unseen intent set, 20 options) | 70.0 % | 0.114 |
| `dates` (unseen programmatic family) | 55.8 % | 0.072 |
| held-out template `inv.most_common_color` | 96.0 % | 0.265 |
| held-out template `arith.difference` (score) | 29.9 % | 0.431 |
| held-out template `lookup.health` (score) | 4.6 % | 0.704 |

### Baselines (same 2,000-record samples)

| model | test noul / choice / score accuracy | test ECE | heldout noul / choice / score accuracy |
|---|---|---|---|
| decider | 92.7 / 91.2 / 77.5 | 0.012 | 66.7 / 73.9 / 18.5 |
| zero-shot NLI (`deberta-v3-base-zeroshot-v2.0`) | 79.3 / 42.7 / 40.7 | 0.050 | 54.6 / 52.4 / 34.6 |
| embedding similarity (`all-MiniLM-L6-v2`) | 58.3 / 48.1 / 25.3 | 0.064 | 60.4 / 62.8 / 15.2 |

### Structural and behavioural checks

| check | test | heldout |
|---|---|---|
| permutation: max change after shuffling options (fp32) | 3.5e-6 | 2.1e-6 |
| independence: max change after removing a question (fp32) | 3.2e-6 | 2.5e-6 |
| same two checks at bf16 (numerical noise of serving precision) | 0.009 / 0.012 | 0.014 / 0.016 |
| negation consistency, mean `|p(q) + p(¬q) − 1|` | 0.072 | 0.213 |
| prompt injection in the state: accuracy drop / false-yes increase | 0.25 pt / +1.4 pt | ≈0 / −12.2 pt |

### Latency and memory (RTX 3050 Laptop, bf16, one request)

| state tokens | questions | full pass p50 / p95 | cached state p50 |
|---|---|---|---|
| 130 | 1 | 17.8 / 23.5 ms | 18.7 ms |
| 384 | 8 | 27.6 / 29.3 ms | 17.8 ms |
| 384 | 32 | 48.9 / 50.4 ms | 30.5 ms |

Peak inference memory 715 MB.

### Release gate (`configs/gate/default.yaml`)

Passed: per-type ECE ≤ 0.05, permutation and independence ≤ 0.005 (fp32), injection false-yes
increase ≤ 0.10, p95 latency ≤ 250 ms.

## Weaknesses

1. **New question templates over familiar states.** Counting and range questions phrased as
   `score` in ways not seen in training fail almost completely (4.6 % and 29.9 %) and are
   confidently wrong. Calibration does not transfer: never automate a question type that has
   not been evaluated. The familiarity detector flags all of these questions (see below), but
   flagging is not fixing.
2. **Score questions** are the weakest type even in distribution (77 %); only about half of
   them can be automated at 95 % accuracy.
3. **Negation** is consistent in distribution (mean gap 0.07) but far less so on unseen
   families (0.21).
4. **Overconfidence with more training.** Beyond ~1 epoch, accuracy kept rising while log loss
   and ECE got worse; checkpoint selection on log loss and temperature scaling are required,
   not optional.
5. **Truncation.** States longer than 384 tokens are truncated from the end.
6. **Licences.** AG News, Yelp and SciQ are non-commercial; so is this model.
7. **No hand-labelled gold set yet**; every number here comes from converted public datasets
   and programmatic tasks.

## Knowing when the confidence is not trustworthy

The bundle's familiarity detector compares each question's readout features with those of the
calibration questions. About 5 % of in-distribution questions are flagged by design; on the
`heldout` split, 4.5 % of noul, 58 % of choice and 100 % of score questions are, and the score
separates test from held-out questions with AUROC 0.87 / 0.96 / 1.00. It does not predict
individual errors: within a split, flagged answers are about as accurate as the rest. Treat a
rising unfamiliar share (`decider_unfamiliar_total`) as a sign that traffic has moved away from
what this card measured. Details: [EXPERIMENTS.md](EXPERIMENTS.md#e4-question-familiarity).

## Successor

v0.2 (`modernbert-mix-v1-c6307c9`, registry v3, `challenger`) is trained on `mix-v1`. On the
same questions it is 1.1 points more accurate in distribution and 1.3 points less accurate on
`heldout` (both significant; counting templates improve from 4.5 % to 23.8 %, most other held-out
families lose 3-7 points). Neither dominates; which becomes `champion` is an owner's decision.
Comparison: [EXPERIMENTS.md](EXPERIMENTS.md#e3-template-diversity-mix-v1-model-v02).
