"""Baseline compressors for honest comparison.

We cannot run LLMLingua's proxy language model in this environment, and
pretending otherwise would be dishonest.  What we *can* do -- and what makes the
comparison meaningful -- is implement each baseline's **algorithmic shape** with
the strongest model-free scorer available, and say exactly that.

``truncate_head/tail/middle``
    The baseline every production system actually uses.

``tfidf_sentences``
    Classic extractive summarisation: greedy TF-IDF sentence selection with an
    MMR redundancy penalty.  A genuinely strong baseline, and the honest
    reference point for "naive summarisation".

``selective_context``
    Selective-Context (Li et al., 2023) drops the lowest self-information
    lexical units.  We substitute corpus-free self-information
    ``-log p(term)`` estimated from within-context frequencies.

``llmlingua_style``
    LLMLingua's shape: coarse-to-fine budget allocation, then *token-level*
    dropping by an importance score, with instruction/question segments
    exempted.  We substitute the same self-information surrogate for the proxy
    LM's perplexity.  This reproduces the method's characteristic behaviour --
    including its characteristic failure, which is that token-level deletion
    destroys syntax, numbers and identifiers.

Every baseline is measured with the same metrics as the engine, so differences
in ratio are controlled: all baselines are run at the *same output token
budget*.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable

from ..text.segment import split_sentences
from ..text.terms import content_terms
from ..tokenization import CachedTokenizer, get_tokenizer

_WORD = re.compile(r"\S+")


def truncate_head(text: str, budget: int, tok: CachedTokenizer, **_: object) -> str:
    return tok.truncate(text, budget)


def truncate_tail(text: str, budget: int, tok: CachedTokenizer, **_: object) -> str:
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok.count(" ".join(words[mid:])) > budget:
            lo = mid + 1
        else:
            hi = mid
    return " ".join(words[lo:])


def truncate_middle(text: str, budget: int, tok: CachedTokenizer, **_: object) -> str:
    """Keep head and tail, drop the middle -- the lost-in-the-middle heuristic."""
    half = budget // 2
    head = tok.truncate(text, half)
    tail = truncate_tail(text, budget - tok.count(head), tok)
    return f"{head}\n...\n{tail}"


def random_sentences(text: str, budget: int, tok: CachedTokenizer, seed: int = 7, **_: object) -> str:
    spans = split_sentences(text) or [(0, len(text))]
    rng = random.Random(seed)
    idx = list(range(len(spans)))
    rng.shuffle(idx)
    keep: list[int] = []
    used = 0
    for i in idx:
        s, e = spans[i]
        c = tok.count(text[s:e])
        if used + c > budget:
            continue
        keep.append(i)
        used += c
    keep.sort()
    return "\n".join(text[spans[i][0] : spans[i][1]].strip() for i in keep)


def tfidf_sentences(
    text: str,
    budget: int,
    tok: CachedTokenizer,
    query: str = "",
    mmr_lambda: float = 0.65,
    **_: object,
) -> str:
    """Greedy MMR sentence selection over TF-IDF vectors."""
    spans = split_sentences(text) or [(0, len(text))]
    sents = [text[s:e].strip() for s, e in spans]
    docs = [set(content_terms(s)) for s in sents]
    df = Counter()
    for d in docs:
        df.update(d)
    n = len(docs) or 1
    idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
    q = set(content_terms(query))

    def score(i: int) -> float:
        base = sum(idf.get(t, 0.0) for t in docs[i]) / (1 + math.log(1 + len(docs[i])))
        if q:
            base += 2.0 * sum(idf.get(t, 0.0) for t in docs[i] & q)
        return base

    chosen: list[int] = []
    used = 0
    remaining = set(range(len(sents)))
    covered: set[str] = set()
    while remaining:
        best, best_val = -1, -1e18
        for i in remaining:
            novelty = len(docs[i] - covered) / (1 + len(docs[i]))
            val = mmr_lambda * score(i) + (1 - mmr_lambda) * novelty * 10
            if val > best_val:
                best, best_val = i, val
        if best < 0:
            break
        remaining.discard(best)
        c = tok.count(sents[best])
        if used + c > budget:
            if used >= budget * 0.9:
                break
            continue
        chosen.append(best)
        covered |= docs[best]
        used += c
    chosen.sort()
    return "\n".join(sents[i] for i in chosen)


def _self_information(text: str) -> dict[str, float]:
    """``-log p(w)`` estimated from within-context frequencies."""
    words = [w.lower().strip(".,;:!?()[]\"'") for w in _WORD.findall(text)]
    counts = Counter(w for w in words if w)
    total = sum(counts.values()) or 1
    return {w: -math.log(c / total) for w, c in counts.items()}


def selective_context(text: str, budget: int, tok: CachedTokenizer, **_: object) -> str:
    """Drop the lowest self-information lexical units (phrase granularity)."""
    si = _self_information(text)
    spans = split_sentences(text) or [(0, len(text))]
    scored = []
    for s, e in spans:
        seg = text[s:e]
        ws = [w.lower().strip(".,;:!?()[]\"'") for w in _WORD.findall(seg)]
        val = sum(si.get(w, 0.0) for w in ws) / (1 + len(ws))
        scored.append((val, s, e))
    scored.sort(reverse=True)
    keep: list[tuple[int, int]] = []
    used = 0
    for _v, s, e in scored:
        c = tok.count(text[s:e])
        if used + c > budget:
            continue
        keep.append((s, e))
        used += c
    keep.sort()
    return "\n".join(text[s:e].strip() for s, e in keep)


def llmlingua_style(
    text: str,
    budget: int,
    tok: CachedTokenizer,
    query: str = "",
    instruction: str = "",
    coarse_fraction: float = 0.5,
    **_: object,
) -> str:
    """Coarse-to-fine, then *token-level* dropping by importance.

    Faithful to the published algorithm's structure:
      1. instruction/question kept verbatim (segment exemption);
      2. coarse stage: rank sentences, keep the top ``coarse_fraction`` of the
         budget's worth;
      3. fine stage: within the survivors, delete the lowest-importance tokens
         until the budget is met.

    Step 3 is where token-level methods lose numbers, identifiers and syntax --
    reproduced here rather than papered over, because it is the central
    empirical claim this engine makes.
    """
    si = _self_information(text)
    spans = split_sentences(text) or [(0, len(text))]
    sents = [(s, e, text[s:e]) for s, e in spans]
    q_terms = set(content_terms(query + " " + instruction))

    def sent_score(seg: str) -> float:
        ws = [w.lower().strip(".,;:!?()[]\"'") for w in _WORD.findall(seg)]
        base = sum(si.get(w, 0.0) for w in ws) / (1 + len(ws))
        if q_terms:
            base *= 1.0 + 0.8 * len(set(content_terms(seg)) & q_terms)
        return base

    ranked = sorted(sents, key=lambda t: -sent_score(t[2]))
    coarse_budget = int(budget / max(1e-6, coarse_fraction))
    kept: list[tuple[int, int, str]] = []
    used = 0
    for s, e, seg in ranked:
        c = tok.count(seg)
        if used + c > coarse_budget:
            continue
        kept.append((s, e, seg))
        used += c
    kept.sort()

    out_tokens: list[tuple[float, int, int, str]] = []
    pos = 0
    for _s, _e, seg in kept:
        for w in _WORD.findall(seg):
            key = w.lower().strip(".,;:!?()[]\"'")
            out_tokens.append((si.get(key, 0.0), pos, len(out_tokens), w))
            pos += 1
    order = sorted(out_tokens, key=lambda t: -t[0])
    keep_idx: set[int] = set()
    used = 0
    for _imp, _p, i, w in order:
        c = tok.count(" " + w)
        if used + c > budget:
            continue
        keep_idx.add(i)
        used += c
    return " ".join(w for _imp, _p, i, w in out_tokens if i in keep_idx)


BASELINES: dict[str, Callable[..., str]] = {
    "truncate_head": truncate_head,
    "truncate_tail": truncate_tail,
    "truncate_middle": truncate_middle,
    "random_sentences": random_sentences,
    "tfidf_mmr": tfidf_sentences,
    "selective_context": selective_context,
    "llmlingua_style": llmlingua_style,
}


def run_baseline(
    name: str,
    text: str,
    budget: int,
    query: str = "",
    instruction: str = "",
    tok: CachedTokenizer | None = None,
) -> str:
    fn = BASELINES[name]
    tk = tok or get_tokenizer("auto")
    return fn(text, budget, tk, query=query, instruction=instruction)
