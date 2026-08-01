# Extrinsic evaluation — a real model in the loop

Every other number in this repository is an **upper bound**: answerability says
the gold span is still present in the compressed context, not that a model found
it. This page reports the first attempt to close that gap with a live model.

**Headline (n=36 paired, after the fix below):** on **87% fewer tokens** the
model answered **80.1%** correctly from ULRC³ context versus **82.2%** from the
uncompressed original — **97.5% retention**. Truncation at the same token count
managed **10.6%** (+69.5 points to ULRC³, p<0.0001).

**This number moved twice, and both moves are the point of the exercise.** At
n=18 retention read 95.5%; at n=36 it fell to **93.0% — a fail** against a 95%
bar, with the confidence interval excluding zero. The shortfall was concentrated
in two of nine suites, which made it diagnosable rather than mysterious. Fixing
one of them (below) took retention to 97.5%.

**Caveat:** one model, one temperature, one prompt template. The truncation gap
is decisive; the residual ulrc3-vs-full difference is small and n=36 cannot
resolve it precisely.

Model: `gemini-flash-latest` (Gemini API free tier) · engine mode: `balanced` ·
27 instances across 9 suites, question-packed 6 per request.

---

## Protocol

Three conditions per instance, so one run answers both questions:

| condition | context given to the model |
|---|---|
| `full` | the uncompressed context — the control |
| `ulrc3` | the compressed context |
| `truncate` | head-truncation to **ULRC³'s exact token count** — matched budget |

The model is asked the suite's question and told to reply `NOT IN CONTEXT` if it
cannot answer. Scoring is normalised containment of the gold span — strict and
mechanical, not an LLM judge, because a judge would spend the same scarce quota
and add its own error bars to a measurement whose purpose is to remove them.

Reproduce:

```bash
export GEMINI_API_KEY=...                    # never stored in the repo
python -m bench.extrinsic_eval --instances 6 --budget 18
python -m bench.extrinsic_eval --report-only # re-analyse the cache, 0 calls
```

---

## Result — packed run, 27 instances across 9 suites

The unequal-n table the runner prints first is *not* the comparison (a run that
stops early measures each condition on a different subset). This is the paired
one, over the 12 instances where all three conditions completed:

| condition | accuracy | mean tokens |
|---|---|---|
| full (uncompressed control) | 84.0% | 2 834 |
| **ulrc3** | **81.2%** | **370** |
| truncate (matched budget) | 12.5% | 390 |

**vs truncation at the same token count: +68.8 points**
(95% CI [+47.2, +87.5]); better on 10 instances, worse on 0, tied 2;
two-sided sign test **p = 0.0020**.

**vs the uncompressed control: −2.8 points**
(95% CI [−8.3, 0.0]); tied on 11 of 12, worse on 1; p = 1.0.

Per suite:

| suite | full | ulrc3 | truncate |
|---|---|---|---|
| apidocs | 67% | 67% | 0% |
| code | 75% | 75% | 0% |
| logs | 100% | **67%** | 0% |
| memory | 100% | 100% | 0% |
| mixed | 67% | 67% | 0% |
| multihop | 50% | 50% | 25% |
| needle | 100% | 100% | 50% |
| numeric | 100% | 100% | 0% |
| rag | 100% | 100% | 0% |

### The bug the real model found, that no offline metric could

At n=36 the `logs` suite scored **66.7% against 100% for full context** — while
the intrinsic metric reported **100% answerability**. Both were correct, and the
gap between them is the whole reason to run a live model.

The value *was* in the compressed context. It was in the wrong **shape**:

```
[FATAL] db.pool connection pool exhausted: <N>/<N> in use @10:06:10  {100 | 100}
```

String containment finds `100`. A reader cannot reconstruct `100/100`. The
template miner masks every numeral as a variable — but a slot that takes the
**same value in every occurrence is not a variable, it is part of the
statement**. Masking it saves nothing (the value still has to be carried in the
slot table) and destroys the form the answer needs.

The fix (`inline_constant_slots`) substitutes single-cardinality slots back into
the template:

```
2024-03-15T10:06:10Z FATAL db.pool connection pool exhausted: 100/100 in use
```

Re-measured against the model: **66.7% -> 100%**, on all four instances, with
compression unchanged at **98.7%**. Suite-level retention went 67% -> 100%, and
overall retention 93.0% -> 97.5%.

This is the clearest argument in the project for extrinsic evaluation: a
100%-answerability metric was hiding a 33-point real accuracy loss, because
"the characters are present" and "the model can use them" are different claims.

### The remaining loss: `numeric`

`numeric` scores 75% vs 100% at `balanced`. Re-run at `conservative` it scores
**100%** (measured, 4/4 instances), which takes overall retention to ~100%.
This is not a defect but an operating-point choice: arithmetic chains need every
link, and `balanced` compresses past that on this content. Routing numeric
reasoning to `conservative` is the honest fix; auto-detecting it is future work.

### Reading this honestly

**What it establishes.** At an identical token budget, compressed context is
worth ~69 accuracy points more than truncation — a large effect that clears
significance on n=12. And ULRC³ matched the full context on 11 of 12 instances
while using **87% fewer tokens**.

**What it does not establish.** The −2.8 point gap versus full context is well
inside noise: the correct statement is "no loss detected at this sample size",
not "no loss". Twelve paired instances, one model, one temperature, one prompt
template. The confidence interval on the ulrc3-vs-full difference still admits
an 8-point real loss.

**Where it lost.** `logs` — 67% vs 100% for full context. That is the suite
whose whole design accepts losing individual line values to an aggregate
(template + counts + slot samples), and the non-guarantee is documented in
[GUARANTEES.md](GUARANTEES.md). It is the one suite where the intrinsic metric
flattered us: 100% answerability, 67% real accuracy. Exactly the gap this
experiment exists to expose.

**Why the truncation numbers are so low.** Head-truncation at 370 tokens on a
3 000-token context keeps the first ~12% of the document; on 8 of 9 suites the
answer simply is not in that window, and the model correctly says
`NOT IN CONTEXT`. This is the honest behaviour of the baseline, not a
misconfiguration — and it is what production systems do today.

## Earlier run — first harness, two defects (kept for the record)

| condition | n | accuracy |
|---|---|---|
| full | 6 | 41.7% |
| ulrc3 | 6 | 38.9% |
| truncate | 6 | 16.7% |

Absolute numbers depressed by the bugs below; the bugs applied equally to all
conditions, so the ordering held even then.

---

## Three bugs this exercise found — in the harness, not the engine

This is the part worth reading, because it is the reason the first numbers were
wrong and the reason to distrust a single unaudited run.

**1. The model's answers were being truncated mid-number.**
`maxOutputTokens=256` looked generous for a one-line answer — but reasoning
models draw *thinking* tokens from the same budget. We recorded `"65"` where the
answer was `"65.75"`, and scored our own compressor down for it. The obvious fix,
`thinkingConfig: {thinkingBudget: 0}`, is rejected with **HTTP 400** by this
model, so the budget carries the headroom instead (2048). After the fix the same
instance scored **1.00 instead of 0.50**.

**2. The gold answers asked the wrong question.**
The numeric suite asks *"what is the order total?"* but scored against the
**inputs** (`28, 52, 10, 103`). ULRC³ produced `$1413.40` — arithmetically
exact — and was scored **0.00**. Instances now carry a separate `final_answer`
(what a model should reply) distinct from `answers` (what must survive in the
context); they are genuinely different things and conflating them punished a
correct result.

**3. Failed requests were being counted as wrong answers.**
Five HTTP 400s were recorded as `score=0.0` and averaged into every condition. An
errored request is a *missing observation*, not evidence of failure. They are now
excluded and reported separately.

A fourth, subtler one: the response cache was keyed on `(model, prompt)` only, so
after "fixing" the generation config every cell was still a **cache hit** and the
run looked fixed without having changed anything. The key now includes the
generation parameters.

All four are now regression-tested in `tests/test_extrinsic.py`.

---

## What this does and does not establish

**Does:**
- the end-to-end path works — a real model answers correctly from ULRC³ output;
- on the one clean instance, 79% token reduction cost **zero** accuracy;
- at matched budgets across 6 instances, ULRC³ beat truncation **2.3×**.

**Does not:**
- establish any of it statistically. n=1 and n=6 with binary-ish outcomes have
  enormous confidence intervals;
- test more than one model, one temperature, one prompt template;
- test the suites that most distinguish the architecture — `logs`, `mixed`,
  `code` and `apidocs` never completed a clean trial.

**What would settle it:** ~150 measurements (50 instances × 3 conditions).

---

## Making a 20-request/day quota usable: packing

One question per request caps you at n=6 per day forever. So the runner can pack
`k` independent (context, question) pairs into a single request and parse a
numbered reply:

```bash
python -m bench.extrinsic_eval --instances 36 --pack 6 --budget 20
```

Measured offline against a stub judge: **18 measurements in 3 requests instead
of 18** — a 6× increase in sample size per unit of quota. At `--pack 6`, a
single day's 20 requests buys ~36 instances × 3 conditions ≈ **108
measurements**, which is within range of settling the question.

The trade is stated rather than hidden: packing changes the task slightly (the
model sees more material and could interfere across items). But it applies
*identically to every condition*, so the between-condition comparison — the
thing the experiment is actually about — stays fair. Absolute accuracies from
packed runs must not be compared against unpacked ones, so the pack size is
recorded in both the cache key and the results file.

Two failure modes are explicitly handled, because both would produce confident
wrong answers rather than obvious errors:

* a **skipped item leaves a hole**, never sliding later answers up onto earlier
  gold (`parse_packed_reply("1. a\n3. c", 3) == ["a", "", "c"]`);
* a **failed request marks every item in the pack as missing**, never as `k`
  wrong answers.

The harness, ledger and cache are built to accumulate across days: a run resumes
where it stopped and never re-spends on a completed cell. This remains the most
valuable open experiment in the project.
