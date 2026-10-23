# Design

Status: v0.2 (revised after the first implementation pass). Section 9 lists what changed from
v0.1 and why.

## 1. Problem

Software that automates decisions (routing a ticket, flagging a transaction, grading an
answer) needs three things from a model:

1. **A valid answer every time**, of the type the caller asked for.
2. **A probability it can trust**, because automation is gated on confidence: answer
   automatically above a threshold, escalate below it.
3. **Low latency and cost**, because the call sits on the request path.

A generative LLM gives a *sample* of text. Validity can be forced with constrained decoding,
but the probability is a token probability, not an answer probability. It is split across
surface forms ("yes", "Yes", " yes") and is typically miscalibrated after instruction tuning.

`decider` is a **discriminative decision model**. It reads a state and a set of typed questions
and returns a **calibrated distribution per question in one forward pass**. It never generates
text.

| Type | Input | Output |
|---|---|---|
| `noul` | a yes/no proposition | `p(yes)` |
| `choice` | 2-255 unordered options, each with a description | one probability per option |
| `score` | 2-10 ordered levels, each with a description | a distribution over levels, `score = Σ pᵢ·i` |

Derived values use fixed formulas, never extra heads: `confidence = 1 - H(p)/log K`,
`choice = argmax p`.

### Goals, in priority order

1. **Calibrated.** Among answers given 80 %, about 80 % are right.
2. **Zero-shot over questions.** Works on question phrasings, options and tasks it has not seen.
3. **Independent questions.** Adding, removing or reordering questions never changes another
   question's answer.
4. **Order-invariant options.** Shuffling `choice` options permutes their probabilities exactly.
5. **Fast.** One forward pass, no decode loop, and the encoded state is reusable.

### Non-goals (v1)

Text generation and explanations, multi-step reasoning, dependent questions (orchestrate those
as sequential requests), images and audio.

The intended deployment is *System 1 + System 2*: the decision model is the fast gate on every
request; low-confidence or reasoning-heavy cases escalate to an LLM.

## 2. Architecture options

| Option | How it works | Strength | Weakness |
|---|---|---|---|
| **A. Packed encoder + segment mask** | `[state][Q1+opts][Q2+opts]…` in one sequence; a mask isolates questions; logits read at option tokens | one pass for all questions, bidirectional, small | less world knowledge than an LLM |
| B. Packed decoder LLM + mask | the same packing on a ~0.5-1B decoder with LoRA | knowledge, instruction following | causal attention, heavy for a 6 GB GPU |
| C. Cross-encoder per option | one pass per (state, question, option) | simple, strong | O(#options) passes, state re-encoded each time |
| D. Bi-encoder | embed state and options separately, dot product | fastest | weak on compositional questions |

**A is the main line.** B is a later bake-off: packing, masks, heads and losses carry over, only
the backbone changes. C and D are baselines (`decider-eval --baseline nli|embedding`).

## 3. Architecture

### 3.1 Packing

```
[CLS] state … [SEP] | [Q_CHOICE] instructions … [OPT] a: … [OPT] b: … | [Q_NOUL] instructions … | …
```

* JSON states are serialised canonically (sorted keys, compact separators), so equal states
  tokenize identically and hit the same cache entry.
* Five special tokens are added: `[Q_NOUL] [Q_CHOICE] [Q_SCORE] [OPT] [LVL]`. In a pretrained
  vocabulary they are initialised from the mean embedding of a short descriptive phrase.
* Several requests (documents) can share one row; first-fit-decreasing packing keeps padding
  low, and the mask keeps documents apart.

### 3.2 Attention mask

Every token carries `doc`, `q` (0 = state, k = question k) and `o` (0 = header, m = option m).
Query `i` may attend to key `j` iff `doc[i] == doc[j]` and

1. `q[j] == 0` (everyone reads the state), or
2. `q[i] == q[j]` and `o[j] == 0` (a question reads its own header), or
3. `q[i] == q[j]` and `o[i] == o[j]` (an option reads itself), or
4. in the last `cross_option_layers` layers only: `q[i] == q[j]` and both are option tokens;

and additionally `q[i] == 0 ⇒ q[j] == 0`: **the state never attends to questions**. Every token
also attends to itself, so no query row is ever fully masked (fully masked rows make SDPA return
NaN).

Consequences:

* **Independence (goal 3)** holds by construction: question blocks cannot see each other.
* **State caching (goal 5)**: the state's keys and values at every layer do not depend on the
  questions. `encode_state` runs the state once; `forward_cached` runs only question tokens
  against the cached keys/values. The server keeps a byte-budgeted LRU of encoded states.

### 3.3 Tree positions

* State tokens: positions `0 … S-1`.
* **Every question restarts at `S`**: questions are parallel branches, not a sequence.
* **Every option of a question restarts right after that question's header**: the model cannot
  tell which option came first. With rule 4 being a set operation, `choice` is exactly
  permutation-equivariant (measured: max |Δp| = 0 in fp32).
* `score` levels are ordered, so each `[LVL]` token adds a learned level embedding (zero-init,
  so a pretrained backbone computes its original function at step 0).

Position-dependent mechanisms must use these ids. RoPE does; so does the **sliding window**:
ModernBERT's local layers attend within ±64 *positions*, and measuring that distance in token
index instead would make later questions in a long packed row lose sight of the state.

### 3.4 Backbone

One encoder implementation (`decider.model.encoder`) covers both configurations:

| | from scratch (`tiny`, `small`) | pretrained (`modernbert-base`) |
|---|---|---|
| norm | RMSNorm / LayerNorm | LayerNorm without bias |
| MLP | SwiGLU / GeGLU | GeGLU |
| attention | global, or global every 3rd layer + ±64 sliding window | global every 3rd layer + ±64 sliding window |
| positions | RoPE θ=10k | RoPE θ=160k global / 10k local |

ModernBERT-base weights load into this module; `tests/test_pretrained_parity.py` checks the
hidden states against the `transformers` reference implementation (max abs difference 0.0 on an
input longer than the sliding window). Using our own module rather than the
reference implementation is what makes the custom masks, tree positions, level embeddings and
the state KV cache possible.

### 3.5 Heads

| Type | Read from | Head | Output |
|---|---|---|---|
| `noul` | `[Q_NOUL]` | MLP → 1 logit | `σ(z/T)` |
| `choice` | each `[OPT]` | shared MLP → 1 logit | `softmax(z/T)` over the question's options |
| `score` | each `[LVL]` | shared MLP → 1 logit | `softmax(z/T)` over levels |

The last head layer is zero-initialised, so every distribution starts uniform. `T` is fitted
post hoc per question type on the `calib` split (temperature scaling); `T = 1` during training.

## 4. Objective

For every question: cross-entropy against the target distribution (log score) + `λ_b ·` Brier;
`score` questions add `λ_r ·` RPS (ranked probability score, the discrete CRPS), which also
rewards mass on nearby levels. All three are strictly proper scoring rules: the minimiser is the
true conditional distribution, so the objective itself asks for calibrated outputs. Targets are
distributions: ties in programmatic tasks become honest soft labels.

**Consistency losses are deferred.** On labelled negation or paraphrase pairs, the proper
scoring rule already enforces `p(q) + p(¬q) = 1`; a consistency term only adds information on
*unlabelled* augmented pairs, which arrive with the teacher-generated data (Phase 4).
Consistency is measured instead (§6).

## 5. Data

### 5.1 Record schema

One JSON object per line (pydantic-validated, `decider.schema.Record`): `state`, `questions`,
`targets` (a distribution per question), `source`, `family`, `split`, `label_kind`,
`label_source`, `gen_version`, `meta`. Stored as gzipped JSONL: option sets differ per record,
so a columnar format would need a JSON string column anyway.

### 5.2 Sources (datasets `mix-v0`, `mix-v1`)

| Source | Family | Types | Role |
|---|---|---|---|
| Banking77 | intent routing | choice (77 options) | train / test |
| AG News | topic | choice + noul probe | train / test |
| MNLI | entailment | choice or noul | train / test |
| BoolQ | reading QA | noul | train / test |
| Yelp reviews | rating | score (5) + noul | train / test |
| CLINC150 | intent routing | choice (20 of 151) | **held out** (unseen dataset) |
| SciQ | science QA | choice (4) | **held out** (unseen family) |
| programmatic | inventory, prose inventory, arithmetic, lookup | all | train / test |
| programmatic | dates | all | **held out** (unseen family) |

Programmatic families compute exact labels from a random world and target known weak spots:
counting, comparisons, field lookup and indirection, negation (`p(q)` and `p(¬q)` in one
request), and prompt injection inside the state (labels unaffected, so following the injection
is an error). Specific question templates are also held out.

Templates are versioned as *template sets*. v1 (used by `mix-v0`) had only two kinds of
`score` question, always worded the same way, and the v0.1 model failed on held-out counting
and range questions. v2 (`mix-v1`) adds counting, range and comparison questions to every
training family and varies how levels are written, so a model has to read the level
descriptions rather than learn their positions. A set never changes once a dataset depends on
it: v1 is pinned by a fingerprint test and still rebuilds `mix-v0` byte for byte.

### 5.3 Splits and hygiene

* `train` / `val` (model selection) / `calib` (temperatures) / `test` / `heldout` / `gold`.
  `calib` is separate from `val`, so calibration is not tuned on the data used to pick the
  checkpoint.
* Normalised exact-match deduplication within splits; any evaluation record whose *state*
  appears in an earlier split is dropped (no evaluating on a passage seen in training).
  `assert_no_leakage` runs on every build.
* Deterministic: rebuilding a dataset version is byte-identical; `manifest.json` records
  per-file SHA-256, counts, licenses, config and code version.
* Training samples sources with `p ∝ n^α` (α = 0.5) instead of a hard per-source cap, and
  subsamples large option sets (keep all positives + random negatives, ≤ 24) while evaluation
  always uses full sets.

## 6. Evaluation

| Area | Metrics |
|---|---|
| Quality | accuracy, macro-F1, per type / family / source; RPS and MAE for `score` |
| Calibration | log loss, Brier, ECE (15 bins), adaptive ECE, reliability diagrams; before and after temperature scaling |
| Uncertainty | 95 % bootstrap intervals for accuracy and ECE |
| Selective prediction | accuracy-coverage curve, AURC, coverage at 95 % accuracy |
| Zero-shot | all of the above on `heldout` |
| Structure | permutation and independence gaps measured on the trained model and real data |
| Behaviour | negation consistency `|p(q) + p(¬q) - 1|`; injection: accuracy drop and false-"yes" increase |
| Speed | p50/p95 for full and cached passes across state sizes and question counts; peak memory |

Every training run ends by fitting temperatures and running this harness; the report
(`report.json`, `report.md`, figures) is logged to MLflow and stored next to the bundle,
together with one row per question.

Model versions are compared on the questions both were evaluated on (`decider-compare`):
paired differences per type, family or source, with a 95 % bootstrap interval that resamples
whole records, because questions about the same state are correlated.

## 7. MLOps

| Concern | Choice |
|---|---|
| Environment | `uv` + lockfile; CPU and CUDA 12.8 PyTorch builds selected by extras |
| Config | Hydra config groups (`model/ data/ train/ gate/ experiment/`); the resolved config is logged with every run |
| Data versioning | deterministic builds + manifests; `dvc.yaml` pipeline with a local DVC remote |
| Tracking | MLflow: params, tags (git SHA, dirty flag, data hash, hardware), metrics, system metrics, artifacts |
| Training telemetry | MLflow + Prometheus exporter (Grafana training dashboard) + JSONL log + console |
| Training health | non-finite loss guard, EMA loss-spike detector, gradient norm, update/weight ratios, GPU memory/utilisation/power |
| Checkpointing | atomic writes; model, optimizer, scheduler, data-stream and RNG state; resume is bit-exact (tested) |
| Registry | MLflow model registry; versions carry gate results as tags; `challenger` on gate pass, `champion` by explicit promotion |
| Release gate | per-type ECE, accuracy floor, permutation/independence gaps, injection effect, p95 latency |
| Serving | FastAPI, typed request/response, size limits, state KV cache or dynamic micro-batching (packed rows), CPU Docker image with health checks |
| Monitoring | Prometheus metrics with bounded labels; confidence drift (PSI vs. training reference); low-confidence rate; share of unfamiliar questions (familiarity detector); alert rules; Grafana serving dashboard |
| Feedback loop | low-confidence and unfamiliar answers queued (PII-redacted) → `decider-label` → `local_sources` of the next data version |
| CI | ruff, mypy, pytest (incl. invariance, parity and resume tests), a full smoke pipeline (build → train → eval → serve), the serving container with a named log volume, Prometheus/Grafana provisioning |

## 8. Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0. Foundations | reproducible project, CI, tracking | done |
| 1. Core | masks, tree positions, heads, losses, cache; invariance tests; toy task | done |
| 2. Data v0 + harness | converters, held-out families, evaluation harness, baselines | done; gold set pending (hand labelling) |
| 3. Model v0.1 | ModernBERT backbone, temperature scaling | done |
| 4. Synthetic engine | broader programmatic templates (template set v2); teacher-generated states/questions with k-sample soft labels | templates done; teacher next |
| 5. Calibration & robustness | consistency losses on unlabelled pairs, per-family calibration analysis | planned |
| 6. Scale & bake-off | larger encoder vs decoder + LoRA; 8k states; two-stage retrieval for >255 options | planned |
| 7. Production | dynamic batching, ONNX/TensorRT export, shadow deployment | serving, monitoring, registry, dynamic batching and ONNX export (parity-tested) done; TensorRT and shadow deployment planned |

## 9. Changes from v0.1

Found while implementing and testing v0.1:

1. **Cross-option attention in every layer collapses `choice`.** With shared positions, all
   `[OPT]` tokens of a question have the same query (same token, same position) and, with
   sibling attention, the same set of visible keys, so they get identical hidden states and
   identical scores. Fixed by allowing sibling attention only in the last
   `cross_option_layers` layers: earlier layers build an option-specific representation first.
   Covered by `test_choice_is_permutation_equivariant`, which also asserts the test is not
   vacuous.
2. **Sliding windows must use position ids.** See §3.3. Without it, independence holds only in
   global layers.
3. **Headers do not attend to options** (rule 4 is restricted to option tokens), so a header's
   representation is independent of its options.
4. **No fully masked rows.** Self-attention is always allowed; padding attends only to itself.
5. **Per-question `[SEP]` dropped.** Blocks are delimited by the mask, not tokens.
6. **Temperature per type, not per family**: families are unknown at inference time.
   Per-family ECE is reported to show where one temperature is not enough.
7. **Separate `calib` split** from `val`.
8. **Mixture sampling (`n^α`) instead of a 15 % cap**, step-based rather than epoch-based.
9. **Consistency losses deferred** (§4) and **CRPS named precisely** (RPS for discrete levels).
10. **JSONL instead of Parquet** for records (§5.1).
11. **Exact near-duplicate handling instead of MinHash for v0**: normalised exact matching plus
    state-level leakage removal. MinHash becomes worthwhile with teacher-generated data.
12. **Pretrained backbone via our own module**, verified by a parity test, instead of patching a
    third-party implementation.

## 10. Open questions

* How many cross-option layers? (ablation: 0, 2, 4 on choice-heavy families)
* Encoder vs decoder backbone (Phase 6).
* Teacher choice and its terms of use; k-sample votes vs. logprobs for soft labels (Phase 4).
* Calibration targets per family: what ECE is good enough to automate?
