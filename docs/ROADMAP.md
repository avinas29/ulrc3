# Roadmap, self-critique, and what would falsify this

## Part 1 — Self-critique

Written after the benchmark and the ablation study, not before. These are the
weaknesses a reviewer should find, listed so they don't have to.

### 1.0 The ablation deflated one of our headline algorithms

`python -m bench.ablation --full --mode extreme` (full table in
[BENCHMARKS.md §6](BENCHMARKS.md)) produced three findings we did not expect,
and one of them is against us:

**Phantom Attention contributes ~0.1 points of compression and no measurable
quality.** It is one of the most prominent ideas in the design, it has the
nicest mathematics, and on this benchmark it is *doing almost nothing*. The
likely cause is architectural rather than algorithmic: the submodular coverage
objective — with obligations entering as high-weight pseudo-concepts — already
performs the ranking, and attention only reaches the objective through the
modular salience term at μ=0.55. It is largely **redundant with a better
mechanism in the same system**.

We then ran the decisive follow-up (`no_coverage`, and `no_coverage +
no_attention`) to separate "dominated" from "useless". **Both score identically
to the full system.** Attention adds nothing even with its competitor removed —
and the submodular objective itself adds only 0.6 points of ratio.

So the finding is larger than one algorithm: on this benchmark the *ranking*
half of the system is nearly inert, and the *constraint* half (protection
lattice + audit + repair) produces the result. That is a genuinely useful
negative result, and it points at where the next engineering effort should go —
not at a better scorer.

What we owe the reader, and have not done:
1. **Delete or demote Phantom Attention.** It costs 16% of runtime for 0.1
   points. Keeping it is defensible only until the extrinsic harness tests it on
   diffuse-answer corpora, where ranking should matter. If it fails there too,
   it should go. A reviewer is entitled to read its continued presence as
   sunk-cost attachment, and they would not be entirely wrong.
2. **Re-weight the design narrative.** Much of the effort here went into
   scoring; the evidence says verification carried the result.

**Closure, ordering and lexical edges also show zero effect** — but for these
the metric is the problem, not the mechanism: string-containment answerability
structurally cannot see whether a retained call kept its definition, or whether
evidence landed in a position the reader recovers. They are *unvalidated*, not
*disproven*, and that distinction is only worth anything if we say which one it
is. Validating them needs the extrinsic harness (§2.5).

**The audit-and-repair loop dominates everything.** Removing it *increases*
compression by 1.6 points and costs **36.9 points** of answerability. The most
valuable mechanism in the system is the one that checks the work and fixes it —
not any of the ranking machinery. That is the strongest evidence for the thesis
(compiler + verifier > statistical filter), and it is also a mild rebuke to the
amount of design effort spent on scoring relative to verification.

### 1.1 Deduplication is 64% of runtime

Two rounds of work took it from 5.0 s to 0.28 s on a 64k document, but it is
still the dominant cost. The remaining time is SimHash shingling — `O(words)`
per unit with a blake2b per shingle.

*Fix:* a rolling 64-bit hash over the shingle window (Rabin–Karp) removes the
per-shingle digest entirely; expected 3–4× on this stage. Not done because the
current number is already well inside the latency budget, and premature
optimisation of a correct implementation is how correctness gets lost.

### 1.2 The confidence estimator is fitted on our own benchmark

Worse than that: the weights in `ulrc3/confidence.json` are **hand-set priors,
not fitted at all**. An earlier draft of this document and of
`p100_confidence.py` claimed they came from logistic regression on the
development benchmark. They did not — fitting needs ground-truth
downstream-correctness labels, which require a model in the loop that this
repository does not have. The claim has been corrected everywhere.

What this means in practice: the *ordering* the estimator induces (more
compression ⇒ less confidence) is sound and is all the adaptive loop uses. The
absolute probability is uncalibrated and should not be read as "94% chance the
answer is unchanged".

*Fix:* a `bench/fit_confidence.py` fitting a real logistic regression against
extrinsic labels, plus Brier score and a reliability diagram. Neither the
fitter nor the labels exist yet.

### 1.3 Belief revision can retract the wrong thing

Correction attribution is lexical-overlap with a recency prior. A user who says
"actually, that reminds me…" is not correcting anything, but shares vocabulary
with the preceding turn. Threshold 0.45 makes this rare, and the recency window
protects the last 4 turns, but the failure is **silent and destructive** — the
one place in the engine where a heuristic can delete correct information rather
than merely fail to keep it.

*Mitigations shipped:* corrections cannot retract other corrections; retraction
requires an explicit cue (`actually`, `correction`, `instead`, `scratch that`,
`I meant`, …), not mere contradiction; retracted spans stay in the residual
store and are recoverable.
*Not shipped:* a `strict_revision=False` switch that down-weights instead of
forbidding. It should exist.

### 1.4 Extractive is a ceiling

On pure narrative we cannot beat abstractive summarisation on ratio. Three
paragraphs of story cannot become one sentence without generating text. We
accept this deliberately — the trade is auditability — but a reviewer should
know it is a real ceiling, not an implementation gap.

### 1.5 The LLMLingua baseline is a surrogate

We cannot run their proxy LM here. `llmlingua_style` reproduces the algorithm's
*shape* (segment exemption → coarse budget → token-level dropping) with a
model-free self-information score. This faithfully reproduces the *failure mode*
(token deletion inside retained spans destroys identifiers and numbers) and
imperfectly reproduces the *ranking quality*. The structural wins (code, logs,
mixed content, contradictions) are architectural and would survive a real proxy;
the plain-retrieval margin should be treated as indicative.

### 1.6 Synthetic benchmarks flatter everyone

Generated haystacks are more homogeneous than real corpora. That makes dedup
look better than it will be on genuinely diverse text — for *every* system in
the table, but it is still a caveat on the absolute numbers. The
external-dataset adapter (`bench/datasets.load_external`) exists exactly so this
can be checked; no data ships with the repo because a benchmark that downloads
silently is not reproducible.

### 1.7 Number retention on aggregate content is 43%

By design: log lines, table rows and JSON array elements are represented by
aggregates. But 43% is a number a reviewer will notice, and the honest reading
is "individual values inside dropped rows are summarised, not preserved". It is
listed in the non-guarantees.

### 1.8 The heuristic tokenizer drifts on adversarial text

Measured **6.7% mean / 11.2% worst case** against cl100k after recalibration
(it was 16.5%/26.5% before, and the docs claimed 3–5% without ever measuring).
Worse on dense CJK or heavy emoji, where BPE behaves very differently from the
pre-tokenizer model. Budgets are then mis-set. Mitigated by the final count
always using the real tokenizer when available, and by the inflation guard.
Pinned by `tests/test_tokenization_calibration.py`.

### 1.9 Region segmentation has no confidence threshold

Viterbi always commits to a label. A genuinely ambiguous region gets a
pipeline anyway. Type entropy is recorded and lowers confidence, but there is no
"fall back to prose when unsure" rule. There should be.

### 1.10 Fabricated numbers found and corrected

An audit of every "measured" claim found four that were never measured. All are
now either measured or corrected, but they were in the document, and a reviewer
who spot-checked would have found them before I did:

| claim as written | reality |
|---|---|
| "within **1.8%** of the LP upper bound" | never computed. Now measured: **35.4% mean** against a conservative bound (`bench/lp_bound.py`). |
| confidence weights "fitted by logistic regression" | **hand-set priors.** Fitting needs downstream-correctness labels that do not exist here. |
| edge conductances "fitted by grid search" | **hand-set priors.** |
| tighten pass gains "8–14% on prose" | measured **1.2% mean, 4.9% max** of input. |
| heuristic tokenizer "within ~3–5%" | measured **16.5% mean** — since recalibrated to **6.7% mean / 11.2% max**, and pinned by a test. |

Four files referenced in prose (`metrics/extrinsic.py`, `bench/fit_confidence.py`,
`bench/tune_edge_weights.py`, `ulrc3/onnx/encoder.py`) did not exist either; the
references are corrected and `ulrc3/confidence.json` now ships.

The lesson is not "documentation drifted". It is that writing a number is
cheaper than measuring one, and the only defence is to measure or delete.

### 1.11 What we did not build

- **ONNX embedding backend.** An `onnx` extra is declared in `pyproject.toml`
  but there is no code behind it: semantic similarity from a MiniLM-class encoder would
  improve dedup on paraphrase, at the cost of the zero-model property. It should
  be strictly optional, and it is not there yet.
- **Extrinsic evaluation.** Now **largely closed**. 27 instances across all 9
  suites against `gemini-flash-latest`; paired on the 12 complete instances,
  ULRC³ scored 81.2% vs 84.0% for full context on 87% fewer tokens, and beat
  truncation by 68.8 points (p=0.002). See [EXTRINSIC.md](EXTRINSIC.md).

  Two caveats keep it from being finished: n=12 paired is enough for the large
  truncation effect but not for the small ulrc3-vs-full one (the CI still admits
  an 8-point real loss), and it is one model at one temperature. Question
  packing (`--pack 6`) now yields ~100 measurements per 20-request day, so
  n=100+ across a few days is reachable without a paid key.

  **It also found the one place the intrinsic metric flattered us:** on `logs`,
  answerability says 100% but the model actually scored 67%. That is the
  documented aggregate carve-out showing up as real accuracy loss — precisely
  what an upper-bound metric cannot see, and the reason this experiment was
  worth running.

  The exercise was worth it for a second reason: it found **four bugs in the
  measurement harness** (truncated model replies scored as compressor failures,
  gold answers that asked the wrong question, HTTP errors averaged in as zeros,
  and a cache key that made a broken config look fixed). None were in the
  engine — but all four would have produced a confidently wrong result.
- **Streaming compression.** The optimiser needs the whole IR; the streaming
  endpoint emits *stage* events, not partial text. Honest, but a true streaming
  mode (compress a sliding window with a carried-over obligation set) is
  feasible and not built.

---

## Part 2 — Roadmap

### Near term (weeks)

| item | why |
|---|---|
| Rolling-hash SimHash | −3× on the dominant stage |
| `strict_revision` switch | make §1.3 a user choice |
| Prose fallback on low type confidence | fixes §1.9 |
| Reliability diagram + Brier score in the bench report | makes §1.2 auditable |
| Tree-sitter backend (optional extra) | exact parsing for 40+ languages; current brace scanner is correct but coarse for macros and templates |
| `--profile` flag on the CLI | per-pass attribution without editing code |

### Medium term (months)

**1. Learned rung selection.**
The ladder structure is the contribution; the *policy* for choosing rungs is
currently a hand-tuned objective. A small model trained on
(context, query, downstream-correctness) triples could set `λ`, `μ_s` and the
concept weights per request. Crucially this keeps every guarantee: the model
would choose *among verified renderings*, never generate text. This is the
single highest-value extension.

**2. Cross-request differential compression.**
The chunk cache already gives 1 067× on repeats. The next step is *session
deltas*: for an agent loop where the context mutates slightly each turn, emit
only the diff against the previously-sent context plus a stable reference. This
is Semantic Delta Encoding lifted from within-document to across-request.

**3. Provider prefix-cache co-design.**
We already emit frozen blocks first in canonical order. With a provider that
exposes cache-boundary hints, the engine could align unit boundaries to cache
blocks and report the achieved prefix-hit length — turning compression and KV
caching into one optimisation instead of two.

**4. Multilingual.**
Lexicons are English. The architecture is language-agnostic (IDF, graph
centrality, submodular coverage) but the deontic/hedge/small-talk patterns are
not. A language pack is ~200 lines of `lexicon.py` per language plus a
segmentation review for scriptio continua.

**5. Extrinsic harness.**
Wire `metrics/extrinsic` to a real model: answer accuracy on HotpotQA/GSM8K,
pass@1 on HumanEval/MBPP with the compressed repo context, faithfulness by
NLI-entailment. Answerability predicts this but does not replace it.

### Longer term (research)

**6. Learned IR rendering.**
Today the surface syntax is hand-designed and tokenizer-measured. It could be
*searched*: given a target model, find the marker vocabulary and layout that
maximises downstream accuracy per token. Early evidence (the 2× cost of Unicode
brackets) suggests the search space is worth exploring.

**7. Compression-aware retrieval.**
Retrieval currently happens before compression, so the compressor cleans up
after a retriever that optimised the wrong objective. Jointly optimising
"retrieve *and* compress to a budget" is a single submodular problem over the
corpus, and CASCADE already solves that shape.

**8. Formal verification of the pass pipeline.**
The protection lattice, closure invariant and repair termination all have
hand-written proofs in [MATH.md](MATH.md). They are small enough to mechanise
(Lean/Coq) over an abstract IR. A prompt compressor with a machine-checked
preservation theorem would be a genuinely new artefact.

---

## Part 3 — What would falsify this

The approach makes falsifiable claims. Here is what would break them:

1. **If a downstream model scores no better on ULRC³ output than on
   `selective_context` output at equal budget**, then answerability is not
   predictive and the intrinsic metric is wrong. *Test:* the extrinsic harness
   (§5 above). This is the most important open experiment.

2. **If the fidelity ladder gives no advantage over keep-or-drop at equal
   budget** — i.e. if an ablation with `L(u) = {drop, full}` matches the full
   ladder — then the central structural claim is wrong. *Test:* one config flag;
   the ablation should be in the bench harness and is not yet.

3. **If Phantom Attention ranks no better than TF-IDF centrality**, then the
   attention model is decoration. *Test:* ablate the graph to lexical edges only
   and compare answerability at matched budget.

4. **If the guarantees fail on real-world data** in a way the synthetic suites
   do not surface, the audit is measuring the wrong thing. *Test:* run
   `ulrc3 verify` over a large real corpus and count violations.

Each of these is a small experiment. None is run here. Saying so is part of the
work.
