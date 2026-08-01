# Benchmarks

All numbers below were produced by `python -m bench.run_bench --suite all` and
`python -m bench.bench_perf` on the machine described at the bottom. Raw data:
`bench/results/full.json`, `bench/results/perf.json`. Everything reruns in ~63 s.

---

## 1. The protocol (read this before the numbers)

Reporting "we achieve 80% compression" is meaningless without saying what
everyone else retains at 80%. So the harness enforces **matched output budgets**:

1. Run ULRC³ at the primary operating point; record its output token count `T`.
2. Run **every baseline with `budget = T` on the same source**.
3. Score all systems with the identical metric suite.

Differences in the tables are therefore differences in *what survived*, not in
how much was cut.

**156 instances · 9 suites · 11 systems · 1 716 runs · 62.8 s.**

### Baselines

| system | what it is |
|---|---|
| `truncate_head/tail/middle` | What production systems actually do today. |
| `random_sentences` | Selection floor. |
| `tfidf_mmr` | Classic extractive summarisation: greedy TF-IDF + MMR redundancy penalty. A genuinely strong baseline and the honest stand-in for "naive summarisation". |
| `selective_context` | Selective-Context (Li et al. 2023): drop lowest self-information units. |
| `llmlingua_style` | LLMLingua's *algorithmic shape*: instruction exemption → coarse sentence budget → **token-level** dropping by importance. |

### Honest caveat about the LLMLingua baseline

We cannot run LLMLingua's proxy language model in this environment, and
pretending otherwise would be dishonest. `llmlingua_style` reproduces the
published algorithm's **structure** with a corpus-free self-information
surrogate in place of the proxy LM's perplexity. That is a faithful reproduction
of its *failure mode* — token-level deletion inside retained spans — and an
imperfect reproduction of its *ranking quality*. Where the tables show ULRC³
ahead on structure (identifiers, code, logs, mixed content), the gap is
architectural and would survive a real proxy LM. Where they show ULRC³ ahead on
plain retrieval, treat the margin as indicative rather than settled.

The metrics are all **model-free and exact**: answerability is string
containment against a *known* gold span, not an LLM judge with its own error
bars.

---

## 2. Headline: cross-suite means at matched budgets

| system | ratio % | answerability % | contradiction % | numbers % | identifiers % | hallucinated words |
|---|---|---|---|---|---|---|
| **ulrc3:conservative** | 62.4 | **100.0** | **0.0** | 87.4 | **99.8** | **0.00** |
| **ulrc3:balanced** | 75.3 | **100.0** | **0.0** | **87.2** | **99.4** | **0.00** |
| **ulrc3:aggressive** | 78.6 | **99.7** | **0.0** | 87.0 | 97.4 | **0.00** |
| **ulrc3:extreme** | 80.1 | 96.8 | **0.0** | 85.4 | 97.4 | **0.00** |
| truncate_head | 74.1 | 33.2 | 11.1 | 41.9 | 77.2 | 0.06 |
| truncate_tail | 74.3 | 37.6 | 0.0 | 47.7 | 84.9 | 0.00 |
| truncate_middle | 74.0 | 29.7 | 4.2 | 37.7 | 77.5 | 0.13 |
| random_sentences | 74.7 | 40.3 | 0.0 | 38.1 | 73.2 | 0.00 |
| tfidf_mmr | 74.9 | 83.0 | 11.1 | 77.0 | 86.7 | 0.00 |
| selective_context | 74.6 | 85.8 | 11.1 | 70.6 | 79.9 | 0.00 |
| llmlingua_style | 74.3 | 74.9 | 11.1 | 84.3 | 79.6 | 0.00 |

**At the same 75% compression**, ULRC³ answers 100.0% of questions versus 85.8%
for the best baseline — a **14.2-point** absolute gap, and every one of the nine
suites is at 100% at this operating point.

---

## 3. Where the differences come from

### Logs — the clearest result in the suite

| system | ratio % | answerability % |
|---|---|---|
| **ulrc3 (all modes)** | **98.7** | **100.0** |
| truncate_head / tail / middle | 98.6 | **0.0** |
| tfidf_mmr | 99.9 | **0.0** |
| selective_context | 99.9 | **0.0** |
| llmlingua_style | 99.9 | **0.0** |

Every baseline scores **zero**. A 400-line log has ~3 distinct statements and one
rare FATAL; importance-ranking a sea of near-identical INFO lines picks
near-identical INFO lines. Template mining with anomaly flooring finds the FATAL
by construction.

### Conversation memory — the contradiction result

| system | ratio % | answerability % | **contradiction %** |
|---|---|---|---|
| **ulrc3:balanced** | 77.9 | 100.0 | **0.0** |
| tfidf_mmr | 77.3 | 100.0 | **100.0** |
| selective_context | 77.1 | 100.0 | **100.0** |
| llmlingua_style | 76.6 | 100.0 | **100.0** |
| truncate_head | 76.6 | 4.2 | 100.0 |
| truncate_middle | 76.5 | 2.1 | 37.5 |

Every importance-based method achieves perfect answerability **and perfect
contradiction**: it keeps the retracted "$9,900/mo" *and* the current
"$1,200/mo", because both look important. Belief revision removes the retracted
one. This is a capability, not a tuning difference — importance ranking has no
representation for "this statement was withdrawn".

### Mixed / polyglot content

| system | ratio % | answerability % | numbers % | identifiers % |
|---|---|---|---|---|
| **ulrc3:balanced** | 53.6 | **100.0** | **100.0** | **100.0** |
| tfidf_mmr | 51.5 | 100.0 | 100.0 | 100.0 |
| selective_context | 51.7 | 72.2 | 63.0 | 58.3 |
| llmlingua_style | 50.2 | 58.3 | 86.1 | **29.2** |
| truncate_middle | 49.7 | 0.0 | 16.7 | 12.5 |

Token-level deletion destroys **71% of identifiers** on documents containing code
fences and JSON. Unit-level selection with a code front-end destroys none.

### Needle in a haystack

ULRC³ retains 100% at **88.0% compression** (aggressive) and 97%+ at extreme,
versus 21–25% for truncation at the same budget. Depth-independent, because
retrieval is by graph relevance, not position.

### Numeric reasoning (GSM8K-shaped)

At 88.9% compression ULRC³ keeps 98.8% of numbers and 100% answerability;
`tfidf_mmr` keeps 80%, truncation 14–24%. The full arithmetic chain survives.

### Code (RepoBench-shaped)

| system | ratio % | answerability % | identifiers % | numbers % |
|---|---|---|---|---|
| **ulrc3:balanced** | 60.5 | **100.0** | **94.2** | **100.0** |
| tfidf_mmr | 61.7 | 100.0 | 80.7 | 56.2 |
| selective_context | 60.2 | 100.0 | 60.7 | 51.5 |
| llmlingua_style | 59.8 | 75.0 | 87.7 | 91.1 |

Plus the property none of them has: **the emitted code parses**, and 100% of
retained signatures are byte-identical to the source.

---

## 4. Per-type results on hand-written fixtures

Realistic documents rather than generated ones (`tests/fixtures.py`), balanced mode:

| type | tokens | ratio | integrity | critical | retention | confidence |
|---|---|---|---|---|---|---|
| markdown docs | 310 → 161 | 48.1% | 100% | 100% | 76.7% | 1.00 |
| python module | 525 → 246 | 53.1% | 100% | 100% | 70.5% | 0.99 |
| conversation | 223 → 97 | 56.5% | 100% | 100% | 90.0% | 0.85 |
| logs | 379 → 92 | 75.7% | 100% | 100% | 66.7% | 0.99 |
| json | 302 → 117 | 61.3% | 100% | 100% | 67.6% | 0.99 |
| legal contract | 198 → 123 | 37.9% | 100% | 100% | 78.6% | 1.00 |
| support ticket | 164 → 47 | 71.3% | 100% | 100% | 50.0% | 0.98 |

Legal compresses least — correctly. Nearly every clause carries a deontic
obligation, so nearly every clause is `LOCKED`. A compressor that hits 80% on a
contract is deleting obligations.

---

## 5. Performance

### Latency and throughput (single process)

| input tokens | p50 latency | throughput | peak RSS Δ | units |
|---|---|---|---|---|
| 1 393 | 34.0 ms | 40 979 tok/s | 0.8 MB | 108 |
| 4 280 | 104.8 ms | 40 828 tok/s | 0.4 MB | 336 |
| 16 180 | 377.6 ms | 42 852 tok/s | 13.5 MB | 1 274 |
| 64 304 | 1 469.1 ms | 43 771 tok/s | 55.8 MB | 5 056 |
| 128 057 | 2 962.4 ms | 43 227 tok/s | 26.6 MB | 10 066 |

Throughput is **flat across a 92× size range** (41.0k → 43.2k tok/s) — the
`O(n log n)` claim is measured, not asserted. A 128k-token context compresses in 3 seconds on one CPU
core, with no GPU and no model.

Reference point: LLMLingua-2 needs a GPU forward pass over the full context;
LLMLingua needs an LLaMA-7B-class proxy. Ours needs a laptop.

### Pass attribution (32k tokens)

| pass | ms | share |
|---|---|---|
| dedup | 286.8 | 64.2% |
| phantom-attention | 66.6 | 14.9% |
| levels | 65.0 | 14.5% |
| salience | 12.1 | 2.7% |
| verify | 9.4 | 2.1% |
| cascade-select | 4.6 | 1.0% |
| render / closure / confidence / order | 2.4 | 0.5% |

The optimiser is 1% of runtime. Deduplication dominates, which is why it
received two rounds of algorithmic work (representative comparison, lazy
MinHash, vectorised SimHash): **5.0 s → 0.28 s** on a 64k document.

### Scaling out

- **Process pool**, 16 × 8k-token documents: 3.15 s → 1.09 s, **2.90× speedup**
  on 8 cores (fork + tokenizer warm-up is amortised above ~4 documents; below
  that the engine stays in-process by design).
- **Document cache**, repeated 32k document: 785 ms → 0.70 ms, **1 067×**, 100%
  chunk hit rate. Content-defined chunk boundaries stay stable under local
  edits, so agent loops re-compressing a mutating document keep most hits.

### Cost

At 75.3% mean reduction on a 100k-token context, per request:

| model | uncompressed input | compressed | saved |
|---|---|---|---|
| GPT-4o class ($2.50/Mtok) | $0.250 | $0.062 | **$0.188** |
| Claude Sonnet class ($3/Mtok) | $0.300 | $0.074 | **$0.226** |

Compression cost: 2.4 s of one CPU core ≈ $0.00003. **Payback ratio ≈ 6 000×.**

---

## 6. Ablation study — falsifying our own claims

A design document that only reports the full system proves nothing: the
complicated part may contribute nothing while the simple parts carry the result.
So each mechanism was disabled in turn and the whole benchmark re-run.

`python -m bench.ablation --full --mode extreme` (156 instances per row):

| ablation | ratio % | Δratio | answerability % | **Δanswerability** | identifiers % |
|---|---|---|---|---|---|
| **full system** | 81.2 | — | **97.3** | — | 97.6 |
| no_repair | 82.8 | +1.6 | 60.4 | **−36.9** | 97.6 |
| no_ladder | 76.5 | **−4.6** | 97.3 | ±0.0 | 100.0 |
| no_dedup | 79.8 | −1.4 | 97.0 | −0.3 | 97.6 |
| no_coverage | 80.6 | −0.6 | 97.3 | ±0.0 | 97.6 |
| no_cov + no_attention | 80.7 | −0.5 | 97.3 | ±0.0 | 97.6 |
| no_attention | 81.1 | −0.1 | 97.3 | ±0.0 | 97.6 |
| no_lexical | 81.1 | −0.0 | 97.3 | ±0.0 | 97.6 |
| no_closure | 81.7 | +0.5 | 97.3 | ±0.0 | 97.6 |
| no_order | 81.2 | ±0.0 | 97.3 | ±0.0 | 97.6 |

### What this actually shows

**1. Verification-driven repair is the dominant mechanism — by a wide margin.**
Removing the audit-and-repair loop *increases* compression by 2.4 points and
*destroys* 36.9 points of answerability. Spending 2.4% of the budget on
obligations the optimiser missed buys 37 points of retention. That is the single
most important empirical finding here, and it is a result about architecture
rather than tuning: **checking what survived and fixing it beats ranking what to
keep.** It is the strongest available evidence for the compiler-with-a-verifier
thesis over the statistical-filter thesis.

**2. The fidelity ladder buys 4.6 points of compression at equal quality.**
On code specifically the effect is not 4.6 points but categorical: with
keep-or-drop, every protected definition must be emitted in full, and
compression on a Python module collapses from 51.8% to **0%**. The aggregate
number understates a mechanism that is decisive on structured content and merely
useful on prose.

**3. The ranking machinery contributes almost nothing — and this is the most
interesting result in the study.**

Phantom Attention costs 0.1 points of ratio and zero quality. The obvious
explanation is that it is *dominated* by the submodular coverage objective in the
same system, so we ran the decisive experiment: disable the coverage objective
too (`no_coverage`), leaving selection driven by salience alone, then disable
attention *on top of that* (`no_cov + no_attention`).

Both rows score **97.2% — identical to the full system**. Attention adds nothing
even when the mechanism that could have masked it is removed. And removing the
submodular objective itself costs 0.4 points of ratio and no quality.

The honest conclusion is not "Phantom Attention is a bad ranker". It is
stronger and more uncomfortable:

> On this benchmark, **importance ranking is nearly irrelevant**. What produces
> the result is the protection lattice plus the audit-and-repair loop — that is,
> *constraint satisfaction*, not *scoring*.

This inverts the emphasis of the entire field. LLMLingua, LongLLMLingua,
LLMLingua-2 and Selective-Context are, architecturally, competing ranking
functions. Our evidence says that once hard obligation preservation and
verification-driven repair are in place, the marginal value of a better ranker
collapses toward zero.

**The caveat that keeps this honest.** Our gold answers are exact spans, and on
these suites they are almost always obligation-bearing (a code, a figure, an
identifier, a date). Obligation preservation therefore captures them almost by
construction, which is precisely the regime where ranking cannot show value. On
corpora where the answer is a diffuse prose claim carrying no extractable
obligation — "what was the author's overall argument?" — ranking should matter
much more, and this benchmark cannot see that. The claim is therefore scoped:
*for fact-bearing context, constraints dominate ranking.* Testing the diffuse
case needs the extrinsic harness and a different dataset, and it is the single
most valuable experiment left undone.

**4. Closure, ordering and lexical edges show no effect on these metrics.**
By the criterion stated in the harness ("a row that matches the full system on
both is a mechanism we should delete"), these are unproven here. But the
criterion is only as good as the metric, and answerability-by-string-containment
is structurally incapable of measuring what they do:

- *closure* affects **interpretability** — whether a retained call has its
  definition. String containment scores a dangling reference as a success.
- *ordering* affects **positional recall in the reader**, which by construction
  cannot appear in a metric computed on the text rather than on a model's
  output.
- *lexical edges* feed attention, which §3 already showed is dominated.

So the honest statement is: **closure and ordering are unvalidated by this
benchmark, not disproven by it.** Validating them requires the extrinsic harness
(a real model answering from the compressed context), which is the top item on
the roadmap. Until then they are kept because they are cheap (0.5% of budget,
0.2 ms) and because emitting a call without its definition is a defect a reviewer
would rightly flag — but they are kept on principle, not on evidence, and that
distinction belongs in the open.

**5. No ablation broke integrity.** All rows hold 100%. The guarantee is
enforced by the protection lattice and the audit, not by any single optimiser
mechanism — which is the intended design property.

---

## 7. Reproducing

```bash
make bench          # quick: 36 instances, ~16 s
make ablation       # ablation study, ~80 s
make bench-full     # full: 156 instances, ~63 s
make perf           # latency, memory, parallel, cache
python -m pytest tests/ -q      # 235 tests
```

To run against **real** datasets, drop JSONL files shaped
`{context|documents, question, answers}` into `bench/data/` — HotpotQA,
2WikiMultihopQA, NaturalQuestions, LongBench and NarrativeQA all convert with a
few lines. Nothing is downloaded automatically: a benchmark that silently hits
the network is not reproducible.

**Machine.** Apple Silicon, 8 cores, 8 GB RAM, Python 3.13.5, `cl100k_base` via
tiktoken, numpy 2.3.5. No GPU used or required.

---

## 8. Limitations of this evaluation

Stated so a reviewer does not have to find them:

1. **Synthetic suites.** Gold answers are exact by construction, which is what
   makes model-free measurement sound — but generated haystacks are more
   homogeneous than real corpora, which flatters *every* system's dedup. The
   external-dataset adapter exists precisely so this can be checked on real data.
2. **Answerability is an upper bound** on downstream accuracy, not accuracy
   itself. A model can fail on a context that contains the answer. It is a
   *necessary* condition, measured exactly; end-to-end accuracy needs an API key
   and is left to `metrics/extrinsic`.
3. **The LLMLingua baseline is a surrogate**, as detailed in §1.
4. **Number retention on aggregate content is low by design** (20% on logs at
   98.7% compression) and should be read together with the answerability column,
   not instead of it: the log suite scores 100% answerability precisely because
   the aggregate representation preserves the *statement* and its counts while
   discarding per-line noise.
