"""Salience fusion + **Semantic Entropy** weighting.

Salience combines six signals that are individually cheap and jointly hard to
fool:

    s(u) = σ( w_a·att(u) + w_q·rel(u|q) + w_o·obl(u) + w_p·prot(u)
              + w_k·kind(u) + w_f·feat(u) − w_r·red(u) )

The interesting term is ``obl`` -- *semantic entropy density*.  For a unit with
concept weights ``c_t`` (IDF-normalised within this context) we define

        H(u) = − Σ_t p_t log p_t ,   p_t = c_t / Σ c

and the **information density**

        I(u) = (Σ_t c_t) · (1 + γ·H(u)) / tokens(u)^β

Rationale: a unit whose weight is concentrated on one rare term is a *fact*; a
unit whose weight is spread over many mid-frequency terms is *narration*.  Both
matter, but per token the fact is worth more, and β<1 keeps the preference from
degenerating into "always pick the shortest".

Unlike perplexity-based importance (LLMLingua), this is computed from the
context's own statistics, needs no model, and is *stable*: it does not change
when the proxy LM is swapped, which is a real reproducibility problem for
perplexity-based compressors.
"""

from __future__ import annotations

import math

from ..text.lexicon import FILLER, HEDGE
from ..text.terms import content_terms
from ..types import Protection, UnitKind
from .base import Pass, PassContext

KIND_PRIOR: dict[UnitKind, float] = {
    UnitKind.INSTRUCTION: 1.0,
    UnitKind.TOOL_SCHEMA: 1.0,
    UnitKind.HEADING: 0.55,
    UnitKind.CODE_DEF: 0.75,
    UnitKind.CODE_IMPORT: 0.5,
    UnitKind.CODE_STMT: 0.6,
    UnitKind.CODE_COMMENT: 0.2,
    UnitKind.CODE_DOCSTRING: 0.35,
    UnitKind.JSON_NODE: 0.8,
    UnitKind.SQL_STMT: 0.8,
    UnitKind.LOG_TEMPLATE: 0.5,
    UnitKind.LIST_ITEM: 0.6,
    UnitKind.TABLE_ROW: 0.5,
    UnitKind.TURN: 0.6,
    UnitKind.SENTENCE: 0.5,
    UnitKind.PARAGRAPH: 0.5,
    UnitKind.MEMORY: 0.9,
}

W_ATT = 1.00
W_QUERY = 1.35
W_INFO = 0.95
W_PROT = 0.70
W_KIND = 0.45
W_FEAT = 0.55
W_RED = 0.80
BETA = 0.35
GAMMA = 0.55


class SaliencePass(Pass):
    name = "salience"

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        q_terms = set(content_terms(cir.query)) if cir.query else set()
        raw: list[float] = []
        for u in cir.units:
            info = _information_density(u)
            rel = 0.0
            if q_terms and u.concepts:
                rel = sum(w for t, w in u.concepts.items() if t in q_terms)
                # contrastive penalty: units relevant to *everything* are generic
                rel -= ctx.cfg.contrastive_lambda * u.features.get("generic", 0.0)
            prot = float(u.protection) / float(Protection.FROZEN)
            kind = KIND_PRIOR.get(u.kind, 0.5)
            feat = (
                1.0 * u.features.get("definition", 0.0)
                + 1.2 * u.features.get("endpoint", 0.0)
                + 1.0 * u.features.get("api_param", 0.0)
                + 0.9 * u.features.get("anomaly", 0.0)
                + 0.8 * u.features.get("record_value", 0.0) / 6.0
                + ctx.cfg.recency_weight * u.features.get("recency", 0.0)
                - 0.8 * u.features.get("example", 0.0)
                - 1.0 * u.features.get("boilerplate", 0.0)
                - 0.7 * u.features.get("boilerplate_lead", 0.0)
                - 2.0 * u.features.get("superseded", 0.0)
            )
            red = u.features.get("duplicate", 0.0) + 0.5 * u.features.get("delta", 0.0)
            red += _filler_ratio(u.text) * 0.8

            z = (
                W_ATT * u.attention
                + W_QUERY * rel
                + W_INFO * info
                + W_PROT * prot
                + W_KIND * kind
                + W_FEAT * feat
                - W_RED * red
            )
            raw.append(z)

        if not raw:
            return
        lo, hi = min(raw), max(raw)
        rng = (hi - lo) or 1.0
        for u, z in zip(cir.units, raw):
            u.salience = (z - lo) / rng
        ctx.note("salience_range", (round(lo, 3), round(hi, 3)))

    def note(self, ctx: PassContext) -> str:
        lo, hi = ctx.scratch.get("salience_range", (0, 0))
        return f"z in [{lo}, {hi}]"


def _information_density(u) -> float:
    c = u.concepts
    if not c:
        return 0.0
    total = sum(c.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for w in c.values():
        p = w / total
        if p > 0:
            h -= p * math.log(p + 1e-12)
    hmax = math.log(len(c) + 1e-12) or 1.0
    hn = h / hmax if hmax > 0 else 0.0
    return (total * (1.0 + GAMMA * hn)) / max(1.0, u.tokens) ** BETA


def _filler_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    n = sum(1 for w in words if w.strip(".,;:!?").lower() in FILLER)
    n += 3 * len(HEDGE.findall(text))
    return min(1.0, n / max(6, len(words)))
