# Competitive analysis

An honest comparison against the systems this work builds on — including where
they are better.

---

## 1. Capability matrix

| | LLMLingua | LongLLMLingua | LLMLingua-2 | Selective-Context | RECOMP / summarisers | **ULRC³** |
|---|---|---|---|---|---|---|
| Selection granularity | token | token | token | phrase | sentence / abstractive | **unit with fidelity ladder** |
| Needs a proxy LM | ✅ LLaMA-7B class | ✅ | ✅ (distilled BERT) | ✅ | ✅ | ❌ **none** |
| Needs a GPU | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ **CPU only** |
| Latency (100k tok) | seconds–minutes | seconds–minutes | ~seconds | seconds | seconds | **2.4 s, 1 core** |
| Reproducible across proxy swaps | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **deterministic** |
| Content-type aware | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **8 pipelines, per-region** |
| Output parses as code | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **verified** |
| Preservation guarantee | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **audited + repaired** |
| Zero-hallucination proof | ❌ | ❌ | ❌ | ✅ (extractive) | ❌ | ✅ **verified** |
| Contradiction removal | ❌ | ❌ | ❌ | ❌ | ⚠️ implicit | ✅ **belief revision** |
| Dependency closure | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Recovery of dropped spans | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **handles + store** |
| Self-reported confidence | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **calibrated + adaptive** |
| Query-aware | ❌ | ✅ | ⚠️ | ❌ | ✅ | ✅ |
| Structure preserved (JSON keys, signatures) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 2. Against LLMLingua (Jiang et al., EMNLP 2023)

**What it got right, and what we took.** Budget allocation before token
selection; segment exemption for instructions and questions; the insight that
compression should be *iterative* rather than one-shot. Our water-filling
allocator and frozen-segment typing are direct descendants.

**Where the design breaks.**

1. **Token-level deletion has no notion of correctness.** Deleting tokens
   *inside* retained spans is the mechanism, so `$18,432,105.44` can become
   `$18,432.44` and `customer_id` can become `customer`. Measured on our mixed
   suite, the LLMLingua-shaped baseline destroys **71% of identifiers**. There
   is no configuration that fixes this; it is what the algorithm does.
2. **Quality is a function of the proxy model.** Importance is
   `−log p_θ(t | t_<)`, so results move when `θ` moves — a real reproducibility
   problem, and a deployment coupling (you now own two models).
3. **Perplexity ≠ importance.** High-surprisal tokens are often *formatting
   noise*: unusual punctuation, foreign words, rare casing. Low-surprisal tokens
   are often the critical ones, because they are predictable *from the context
   the model no longer has*.
4. **Cost.** A 7B forward pass over the full context to decide what to delete
   from the context. On 100k tokens that is comparable to the inference you are
   trying to save.

**What we do instead.** Unit-granular multiple-choice selection (a stub is a
first-class option), model-free scoring via Phantom Attention, and an audit that
*checks* preservation rather than assuming it.

---

## 3. Against LongLLMLingua (Jiang et al., ACL 2024)

**What it got right.** Question-aware coarse-to-fine compression; document
reordering by relevance; contrastive perplexity to separate query-relevant from
generically-fluent text. All three ideas appear here.

**Where we differ.**

| | LongLLMLingua | ULRC³ |
|---|---|---|
| Contrastive relevance | `perplexity(x\|q) − perplexity(x)` — needs 2 LM passes | IDF-weighted query coverage − genericity penalty; 0 passes |
| Reordering | by document relevance | **edge-loaded** (best at head *and* tail, worst in the middle), and disabled for order-sensitive content |
| Budget split | document-level, relevance-proportional | **water-filling on value density with per-document floors** — proportional allocation gives a long irrelevant chunk a large budget for being long |
| Duplicate chunks | ranked independently | clustered, delta-encoded, collapsed |

The reordering difference matters more than it sounds. Sorting documents by
relevance puts rank-3 in the middle — the worst position. Edge loading puts
rank-3 near the tail. On order-sensitive content (code, logs, dialogue,
procedures) we disable reordering entirely, which LongLLMLingua does not: it
will happily reorder a stack trace.

---

## 4. Against LLMLingua-2 (Pan et al., ACL 2024 Findings)

**What it got right.** Distilling compression into a small bidirectional encoder
was the right engineering move: it made the method 3–6× faster and made token
selection *bidirectional* rather than causal. It is the strongest of the family.

**Where the design still binds.**

1. **It requires training data from GPT-4 distillation.** Adapting to a new
   domain (your logs, your codebase, your legal templates) means a data
   collection and distillation run. ULRC³ adapts by adding a ~150-line pipeline
   — the optimiser does not change.
2. **Still token-classification**, so the identifier/number/syntax failures of
   §2 persist, just with a better classifier.
3. **Still no guarantee.** A token classifier at 95% precision drops 1 token in
   20; if that token is a negation, the answer flips. Nothing detects it.
4. **Still model-shaped costs**: GPU or a slow CPU inference path, a model
   artefact to ship, version skew between compressor and target model.

**The honest concession.** On pure natural-language QA where all content types
are one type and no structure needs preserving, a well-distilled LLMLingua-2
is a strong system and may beat us on the frontier at very high ratios, because
it can drop *within* a sentence more aggressively than clause-level extraction.
Our answer is that this is the narrowest slice of real context, and that we win
on everything structured — and that we can *prove* what survived.

---

## 5. Against Selective-Context (Li et al., 2023)

Closest in spirit: extractive, self-information-based, no fine-tuning. It also
scores best of the baselines in our harness (85.8% answerability).

Differences:

- **Self-information from a proxy LM** (theirs) vs **within-context IDF +
  graph centrality** (ours). Ours needs no model and is reproducible.
- **No dependency notion**: it will keep a call and drop the definition.
- **No content typing**: a JSON payload is scored as prose.
- **No audit**: nothing checks that the numbers survived. Measured: 70.6%
  number retention at matched budget, vs 87.2%.

---

## 6. Against RAG-specific compressors (RECOMP, FILCO, xRAG)

These train a compressor per task, often abstractively.

- Abstractive compression **cannot** make the zero-hallucination claim. RECOMP's
  own paper reports factual drift.
- Task-specific training is a data-collection project per deployment.
- They handle retrieved passages, not the system prompt, tool schemas, code
  blocks or logs that make up most of a real agent's context window.

ULRC³ compresses the *whole* request, with per-part policies, and never
generates a token.

---

## 7. Against "context engineering" platforms

Vendor context managers (LangChain compressors, LlamaIndex postprocessors,
provider-side truncation) are mostly:

- similarity-threshold filters over chunks (no budget optimisation), or
- LLM-call summarisers (expensive, hallucinating, non-deterministic), or
- head/tail truncation with a token counter.

Our harness measures the last one directly: **30–40% answerability** at 74%
compression. The gap to 100% is the whole product.

---

## 8. Where the competition is genuinely better

Stated plainly, because a comparison without this section is a sales page:

1. **Abstractive summarisation compresses narrative further.** Three paragraphs
   of story into one sentence is beyond any extractive method. If your context
   is pure prose and you tolerate paraphrase, a summariser wins on ratio.
2. **LLMLingua-2 can cut inside a sentence** more finely than clause-level
   extraction. On pure NL QA at extreme ratios that is a real advantage.
3. **A trained model adapts to *semantic* domain quirks** that our lexical
   features miss — sarcasm, implicature, domain jargon whose importance is not
   orthographically visible.
4. **Our conversation belief revision can mis-fire.** A trained model would not
   confidently *delete* a statement on a lexical-overlap heuristic. This is our
   largest residual risk and it is bounded, not eliminated.

---

## 9. The one-paragraph argument

Existing prompt compressors are *statistical filters*: they estimate token
importance with a language model and hope the estimate is good. ULRC³ is a
*compiler*: it parses the context into a typed IR, optimises it under explicit
constraints, and verifies the result against a decidable specification. The
practical consequence is that it is the only system in this table that can be
put in front of a paid model without a human reading its output first — because
it tells you, per request, exactly what it preserved, exactly what it dropped,
how confident it is, and how to get the dropped part back.

And it does that at 42 000 tokens/second on a laptop CPU, with zero model
downloads.
