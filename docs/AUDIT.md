# Independent engineering audit

Adversarial review of ULRC³ against its own claims. Every number below was
re-measured during the audit, not copied from other docs.

**Verdict: 2 of 5 requirements fully satisfied, 2 partially, 1 failed.**

The engineering is sound and unusually well tested. The *claims* are calibrated
to a synthetic benchmark that materially flatters the engine, and the headline
compression number does not survive contact with real text.

---

## Finding 1 — the benchmark pads with what the compressor deletes

`bench/data/` contains only a `.gitignore`. Every one of the 156 instances is
generated from seeded templates in `bench/datasets.py`. Haystacks are padded by
sampling `_FILLER`, a list of **8 sentences**.

Measured overlap with the engine's own `HEDGE` / `FILLER` lexicon: **6 of 8**.

Share of each benchmark document that is verbatim `_FILLER`:

| suite | total tokens | `_FILLER` tokens | share |
|---|---|---|---|
| needle | 4 615 | 2 713 | **58.8%** |
| numeric | 1 777 | 1 038 | **58.4%** |
| rag | 1 391 | 787 | **56.6%** |
| multihop | 872 | 355 | **40.7%** |

So roughly half of the corpus is eight exactly-duplicated sentences that the
engine's lexicon was written to remove. Deleting them is trivial (the dedup pass
alone catches exact repeats) and cannot cost accuracy, because no gold answer is
ever inside filler. Both headline metrics are inflated by construction.

Removing the injected filler and re-compressing the same corpus:

| suite | as-shipped | de-filled | delta |
|---|---|---|---|
| memory | 78.2% | 38.0% | **−40.1** |
| mixed | 59.3% | 30.4% | **−28.9** |
| numeric | 88.2% | 81.7% | −6.4 |
| needle | 79.1% | 75.0% | −4.1 |
| **mean** | **73.6%** | **68.6%** | **−5.0** |

The mean is only 5 points softer, but `memory` — the conversation-compression
showcase — loses **40 points**. Its result is mostly padding removal.

---

## Finding 2 — on real text, no mode reaches 70%

Held-out corpus the engine was never tuned on: 16 documents / 147 k tokens of
Python standard-library source, third-party package docs, and real JSON.

| mode | reduction on real text | ≥70%? |
|---|---|---|
| conservative | 45.3% | ✗ |
| balanced | **53.1%** | ✗ |
| aggressive | 59.7% | ✗ |
| extreme | **61.0%** | ✗ |

By category at `balanced`: stdlib source **50.7%**, this repo's own docs
**46.9%**, real JSON **62.6%**, package docs 70.2%.

**The >70% requirement is met on synthetic data and failed on real data at every
operating point**, including `extreme`.

### Why — the floor binds, not the optimiser

Share of tokens seeded at ANCHORED or above:

| corpus | ≥ ANCHORED |
|---|---|
| synthetic benchmark | **7%** |
| real stdlib source | **96%** |

Real text is **13.3× more obligation-dense**. Identifiers, numbers, paths and
URLs saturate the protection lattice, so the optimiser has almost nothing
elastic left to trade. This is the guarantee mechanism working as designed and
capping the ratio — the synthetic corpus hides it because half of it is filler
carrying no obligations.

This is the single most important result in the audit: **compression ratio and
preservation guarantees are in direct tension, and the benchmark was not
constructed to expose it.**

---

## Finding 3 — no clean run has ever measured ≥95% retention

The `82.9% vs 82.2%` headline is a **reconstruction**, not a run. Provenance:

* the clean paired run is n=36 (36 full / 36 ulrc3 / 36 truncate, `--pack 6`);
* it measured **76.4% vs 82.2% = 93.0% retention — a fail** against the 95% bar;
* two suites were then fixed and re-measured *individually* (logs 67→100 on n=4,
  numeric 75→100 on n=4) and those two values patched into the per-suite mean.

The fixes are real and were verified against the live model, so ~100% is a
reasonable *estimate*. But the defensible statement is:

> The best clean end-to-end measurement is **93.0% retention (n=36)**, which
> fails the 95% target. Post-fix suite-level re-measurements imply ~100%, but no
> single unpatched run has demonstrated it.

Further limits: one model, one temperature, one prompt template, `--pack 6`
(which changes the task), and scoring by normalised string containment rather
than a judge. And on **real** text retention is **entirely unmeasured** — the
held-out corpus has no gold labels.

What survives scrutiny is the *matched-budget* comparison: **+70 points over
head-truncation at identical token count, p<0.0001**. That effect is far too
large to be an artefact, and it is the claim worth making.

---

## Finding 4 — latency is net negative

Measured against the live API (n=3), then charged for compression time:

| context | full | LLM on compressed | + compression | net |
|---|---|---|---|---|
| 4 506 tok | 2.24 s | 2.65 s | +0.100 s | **0.81×** |
| 4 654 tok | 2.34 s | 2.34 s | +0.101 s | **0.96×** |
| 12 407 tok | 3.22 s | 2.55 s | +0.266 s | **1.14×** |
| **mean** | 2.60 s | 2.51 s | | **0.97×** |

**Including its own overhead, the engine is 3% slower than sending the raw
prompt.** The requirement to "improve inference latency" is **not satisfied**.

The cause is structural, not a tuning bug: wall-clock for a short answer is
dominated by decode and network round-trip; compression only shrinks prefill.
The gain appears only once prefill dominates (1.14× at 12 k tokens).

---

## Finding 5 — JSON is largely broken

JSON is an advertised target type. Measured:

| shape | reduction |
|---|---|
| flat object, 50 / 500 / 20 000 keys | **0.0%** |
| nested config object | **0.0%** |
| array of 500 **identical** records | **0.0%** |
| array of 500 varied records | 3.6% |

500 byte-identical records — perfect redundancy — are not deduplicated at all.
A 20 000-key object costs **4.7 s** and returns the input unchanged.

Key preservation is enforced (`enforce_json_keys`), so a flat object is
incompressible *by policy*; that is defensible. Returning it unchanged after
seconds of work, with no signal to the caller, is not. Schema induction only
pays on heterogeneous record arrays, which is a much narrower claim than the
docs make.

---

## Finding 6 — the competitive comparison is unrun

`llmlingua` is not installed and appears nowhere in `pyproject.toml`,
`bench/`, or the test suite. `docs/COMPETITIVE.md` is an **architectural
argument, not an experiment**. No head-to-head against LLMLingua,
LongLLMLingua, LLMLingua-2, or Selective Context has ever been executed.

The 7 baselines in `bench/` are all self-implemented (truncation, random,
stopword removal, extractive summary, …) — reasonable controls, but all weak,
and none is the state of the art the project positions itself against.

---

## Finding 7 — service hardening gaps

| issue | evidence |
|---|---|
| No authentication | no `Depends`, no key check in `server/app.py` |
| No request size limit | only `ULRC3_MAX_CONCURRENCY` semaphore (default 32) |
| No rate limiting | none present |
| Unbounded CPU per request | 20 k-key JSON = 4.7 s single-threaded |

A public deployment can be exhausted by a handful of large bodies. Acceptable
for a demo; not for the "production-grade" framing.

Clean on the other axis: **no `eval`, `exec`, `pickle`, `os.system`, or
`subprocess` anywhere in the engine.** `ast.parse` runs on untrusted input but
CPython bounds nesting depth itself.

---

## What holds up

Stated as plainly as the failures.

* **Zero crashes on 21 adversarial inputs** — 500 k-char lines, null bytes,
  combining marks, malformed JSON/Python, unclosed fences, latin-1 binary,
  300-deep nesting. No hangs, no exceptions, no pathological blowup.
* **280 tests, lint clean, zero required dependencies**, installs and runs in a
  bare venv. This is genuinely rare.
* **Throughput 59–74 k tokens/s** single-core; compression cost is ~1/11 000th
  of the API spend it saves.
* **Cost reduction is real** and net-positive by four orders of magnitude.
* **The extrinsic harness found two defects no offline metric could see** (log
  slot masking; dropped arithmetic operands). That is the strongest evidence in
  the repo that the evaluation is adversarial rather than decorative.
* **Negative results are documented rather than buried** — Phantom Attention
  measured at ~0.1 points, the filler-suppression A/B that regressed, the LP gap
  of 35.4%, and now this audit.

---

## Requirement scorecard

| # | Requirement | Required | Actual (audited) | Status |
|---|---|---|---|---|
| 1 | Semantic redundancy removal | yes | Works; but ~50% of benchmark redundancy was self-inserted, and identical JSON records are missed entirely | ⚠ Partial |
| 2 | >70% compression | >70% | **73.6% synthetic / 53.1% real** (61.0% at `extreme`) | ✗ **Fails on real data** |
| 3 | >95% reasoning retention | >95% | **93.0% best clean run**; ~100% post-fix but reconstructed; unmeasured on real text | ⚠ Partial |
| 4 | API cost reduction | yes | Real, linear in tokens, 10 979× net positive | ✓ Satisfied |
| 5 | Latency improvement | yes | **0.97× including overhead** — net slower | ✗ **Not satisfied** |

---

## The five experiments that would settle it

1. **Replace the synthetic corpus.** Real gold-labelled data — LongBench,
   Natural Questions, HotpotQA, SWE-bench issues. Until then no ratio or
   retention number generalises. *Highest value by a wide margin.*
2. **Run LLMLingua head-to-head** at matched token budgets on that corpus. The
   central novelty claim is currently unfalsified because it is untested.
3. **One clean n≥100 extrinsic run** with no patched cells, ≥2 models, unpacked,
   reporting a confidence interval on retention.
4. **Latency at scale** — 50 paired requests across 1 k/10 k/50 k/200 k token
   contexts, to find the crossover where prefill actually dominates.
5. **Fix JSON dedup** — identical records must collapse; add a no-op detector
   that returns early instead of charging seconds for 0% reduction.

---

# Remediation — what was fixed, measured before and after

The audit above is preserved unedited. This section records what changed in
response to it. Every number is re-measured, not projected.

## R1. Repair could promote a whole function body (root cause of low real-world ratio)

`_repair` bundled *all* of a unit's missing obligations into one key set, so a
single `constraint` obligation — one deontic word in a docstring — escalated the
whole unit to the exempt spending cap. The only rung covering all 56 missing
keys was `full`, so **2 205 tokens of unrelated function body** were re-admitted
to carry one clause.

Compounding it, the carrier alternative was rejected because sufficiency was
checked by comparing extractor *keys*, and a `constraint` key is a digest of the
clause *as delimited in its original unit*. Re-extracting the same sentence in
isolation yields a different span, a different digest, and a false negative — so
every constraint carrier was discarded and promotion was the only survivor.

**Fixed:** repair splits each unit into its exempt group and its ordinary group
and prices them separately; sufficiency now mirrors `audit`'s own three-way
presence test (`_carries`).

* CPython `timeit.py` 35.2% → **66.0%**, `functools.py` 59.6% → **66.6%**,
  `pdb.py` 58.1% → **64.2%**, `zipimport.py` 39.6% → **44.7%**
* verify-stage inflation 2.97x → 1.30–2.72x
* **Risk: low.** The absolute guarantee is unchanged — constraints, security and
  negations are still exempt from the budget cap; only their price changed.

## R2. The call-graph rung (`uses`)

`SymbolDef.refs` — the call graph — was collected by the parser and never
rendered. Without it the cheapest rung carrying a body identifier was `full`.

**Added:** a rung between `sig` and `stub` emitting `# uses: a, b, c` (~10
tokens). Rendered as a comment, so `ast.parse` is unaffected in every position a
signature can occupy. **Risk: low.**

## R3. JSON was structurally broken

Schema induction was written, tested, and **unreachable for a top-level array** —
the single most common API-response shape. Every root was exploded into one unit
per child; small children have no rung cheaper than themselves, so the ladder
collapsed to `{drop, full}`, the ANCHORED floor equalled the whole payload, the
optimiser got zero groups, and the inflation guard returned the input verbatim.

**Fixed:** decompose only when a child is large enough to deserve its own budget
decision; otherwise ladder the root.

| shape | before | after |
|---|---|---|
| flat object, 500 keys | 0.0% | **36.9%**, 500/500 keys kept |
| flat object, 20 000 keys | 0.0% (4 669 ms) | **39.9%** (1 224 ms) |
| array of 500 identical records | 0.0% | **99.1%** |
| array of 500 varied records | 3.6% | **99.2%** |

Two regressions were introduced and caught during this work, both worth
recording because both were *silent*:

* passing `compact_json(data)` as the root's top rung emitted JSON **truncated
  mid-record** — 211 of 500 keys — while the audit still reported integrity 1.0,
  because the rung's obligations were extracted from the truncated text itself.
  The top rung is now the verbatim source slice.
* abstracting *single* values to type names turned every configuration file into
  a useless list of field types (`"port":"int"`). Type abstraction only pays
  when there are many instances of that type, so `induce_schema` now keeps
  scalar literals unless it is unifying across array elements. A realistic
  service config now retains **every** value at 39.4% reduction.

## R4. Benchmark bias removed

Padding was 8 hedging sentences, **6 of which matched the engine's own
`HEDGE`/`FILLER` lexicon**, at 41–59% of every document. Replaced with 16
ordinary declarative domain distractors carrying real entities, numbers and
dates, matched by **0** engine rules.

Cross-suite ratio at `balanced` fell **73.6% → 68.0%**. That 5.6-point drop is
the measurement error the old pool was hiding, and the honest number is the
lower one.

## R5. A real-corpus harness (`bench/real_corpus.py`)

Compresses files nobody wrote for this project — CPython stdlib, installed
package docs, real JSON schemas. It reports ratio, throughput and every
label-free guarantee, and **deliberately reports no retention or accuracy
number**, because real files carry no gold answers.

It immediately found two broken guarantees the synthetic suite never exposed:

| guarantee | before | after |
|---|---|---|
| provenance (no invented tokens) | 93.9% | **100%** |
| syntax (emitted code re-parses) | 84.8% | **97.0%** |

* **Invented tokens** came from truncation slicing mid-word — `"video"` → `vid`,
  `"command"` → `comm`. Both `compact_json` and `_union` now cut on a boundary.
* **Syntax** had two causes. A genuine one: `def f(x):  # note` gained ` ...`
  *inside the comment*, leaving the function with no body — a `SyntaxError` that
  only surfaces when the next sibling dedents, and impossible to hit on
  generated code because generated code has no trailing comments. And a
  reporting one: markdown/RST fences tagged `python` that were never
  standalone-parseable were counted as our failure; the check now compares
  against the original and only reports a *regression*.

## R6. Service hardening

| gap | fix |
|---|---|
| no payload cap (20 k-key body = 4.7 s of CPU) | `ULRC3_MAX_INPUT_CHARS`, HTTP 413 |
| no request timeout | `ULRC3_TIMEOUT_S`, HTTP 504 |
| no authentication | optional `ULRC3_API_KEY`, HTTP 401; unset = open, for the demo |

Both guards are regression-tested in `tests/test_service.py`.

---

## Before vs after

| metric | before | after |
|---|---|---|
| **Real held-out text, balanced** | 53.1% | **57.2%** |
| Real held-out text, aggressive | 59.7% | **62.3%** |
| Real held-out text, extreme | 61.0% | **63.6%** |
| Real Python source | 50.7% | **59.1%** |
| Real JSON | 62.6% | *see note* |
| Synthetic, balanced | 73.6% (biased) | **68.0%** (de-biased) |
| Benchmark padding matching engine lexicon | 6/8 | **0/16** |
| JSON: flat object | 0.0% | 36.9% |
| JSON: repeated records | 0.0% | 99.1% |
| Provenance on real files | 93.9% | **100%** |
| Syntax on real files | 84.8% | **97.0%** |
| Adversarial no-ops | 2 | 1 |
| Worst-case latency (20 k-key JSON) | 4 669 ms | 1 224 ms |
| Tests | 280 | **282** |

*Note:* real-JSON category reduction reads lower after the fix (43.5%) than
before (62.6%) because the earlier figure was partly obtained by discarding
values that the config fix now correctly preserves. Lower and correct.

## Still unresolved

| issue | needs redesign? |
|---|---|
| One residual syntax failure (`_pydatetime.py`) — same trailing-comment class, one emission path still unpatched | No — a bug |
| `conservative` mode lost ~3 points on real text | No — needs floor retuning |
| Retention on real text **unmeasured** — no gold labels exist | No, but needs real labelled corpora (LongBench/HotpotQA) |
| No LLMLingua head-to-head; `llmlingua_style` is our own reimplementation | No — needs the dependency and a GPU box |
| Latency still net-negative below ~12 k tokens | **Yes** — prefill is a minority of wall-clock; no compressor can fix this |
| Flat JSON caps near 40% | **Yes** — keys are enforced obligations; the ceiling is the guarantee |
| No clean n≥100 extrinsic run | No — needs quota/time |
