"""CIR construction: documents -> typed regions -> units -> graph.

Front-end orchestration.  Three things happen here that are easy to get wrong:

1. **Frozen roles bypass detection.**  A system prompt is not "prose that looks
   important"; it is frozen by contract.  Instruction isolation is therefore a
   *typing* decision made before any scoring, which is why no amount of
   adversarial document content can cause an instruction to be compressed.
2. **Within-context IDF.**  Term statistics are computed over the units of *this
   request*, not a static corpus, because specificity is relative to the
   context window we are compressing.
3. **Region dispatch.**  Every document is segmented before typing, so a
   markdown file with three code fences and a log excerpt gets three different
   pipelines, not one averaged compromise.
"""

from __future__ import annotations

import re

from ..config import Config
from ..detect.doctype import Region, detect, entropy, segment
from ..ir.obligations import ObligationExtractor
from ..ir.protection import propagate, seed
from ..pipelines.base import BuildContext, get_pipeline
from ..request import Request
from ..text.terms import TermStats, bigrams, content_terms, extract_entities
from ..tokenization import CachedTokenizer
from ..types import CIR, Document, EdgeKind, Protection, Symbol, UnitKind

FROZEN_ROLES = frozenset({"system", "tools", "query", "instruction"})


def build(req: Request, cfg: Config, tok: CachedTokenizer) -> CIR:
    cir = CIR()
    cir.query = req.query or ""
    docs = req.docs()
    order_sensitive = False
    ent_sum = 0.0

    for d in docs:
        doc = Document(
            doc_id=d.doc_id or f"doc{len(cir.docs)}",
            text=d.text,
            role=d.role,
            weight=max(0.05, d.score),
            meta=dict(d.meta),
        )
        if d.title:
            doc.meta["title"] = d.title
        cir.docs[doc.doc_id] = doc

        if d.role in FROZEN_ROLES:
            doc.doctype = d.doctype or "prose"
            _build_frozen(cir, doc, cfg, tok)
            continue

        forced = cfg.doctype_override or d.doctype
        label, dist = detect(d.text, hint=forced)
        doc.doctype = label
        doc.doctype_scores = dist
        ent_sum += entropy(dist)

        # A structural oracle (a real parser accepted the whole document)
        # returns a degenerate distribution; in that case region splitting is
        # not just unnecessary, it is harmful.
        oracle = len(dist) == 1 and next(iter(dist.values())) >= 0.999
        regions = (
            [Region(0, len(d.text), forced or label, None, 1.0)]
            if (forced or oracle)
            else segment(d.text)
        )
        ctx = _context(cir, cfg, tok, doc, regions[0])
        for r in regions:
            if r.end <= r.start:
                continue
            pipe = get_pipeline(r.label)
            order_sensitive = order_sensitive or pipe.order_sensitive
            ctx.region = r
            ctx.heading_stack = ctx.heading_stack if pipe.name in ("prose", "legal", "apidocs") else []
            pipe.build(ctx)

    _finalise(cir, cfg, tok)
    cir.meta["order_sensitive"] = order_sensitive
    cir.meta["type_entropy"] = ent_sum / max(1, len(docs))
    return cir


# --------------------------------------------------------------------------
def _context(cir: CIR, cfg: Config, tok: CachedTokenizer, doc: Document, region: Region) -> BuildContext:
    return BuildContext(
        cir=cir,
        cfg=cfg,
        tok=tok,
        doc=doc,
        region=region,
        ex_prose=ObligationExtractor(
            enforce_entities=cfg.protection.enforce_entities,
            enforce_identifiers=cfg.protection.enforce_identifiers,
            code_context=False,
        ),
        ex_code=ObligationExtractor(
            enforce_entities=False,
            enforce_identifiers=cfg.protection.enforce_identifiers,
            code_context=True,
        ),
    )


def _build_frozen(cir: CIR, doc: Document, cfg: Config, tok: CachedTokenizer) -> None:
    """Frozen roles become one unit per line: verbatim, unreorderable."""
    from ..text.segment import iter_lines

    seg = doc.role
    if seg == "instruction":
        seg = "instruction"
    ctx = _context(cir, cfg, tok, doc, Region(0, len(doc.text), "prose"))
    prot = Protection.FROZEN
    if seg == "instruction" and not cfg.protection.freeze_system:
        prot = Protection.LOCKED
    for s, e, line in iter_lines(doc.text):
        if not line.strip():
            continue
        kind = UnitKind.TOOL_SCHEMA if seg == "tools" else UnitKind.INSTRUCTION
        u = ctx.emit(kind, s, e, segment=seg, protection=prot)
        if u is not None:
            u.salience = 1.0


def _finalise(cir: CIR, cfg: Config, tok: CachedTokenizer) -> None:
    """Concepts, symbols, cross-unit edges, protection fixpoint."""
    stats = TermStats()
    per_unit_terms: list[list[str]] = []
    for u in cir.units:
        terms = content_terms(u.text, keep_numbers=True)
        if len(terms) > 4:
            terms = terms + bigrams(terms)[: len(terms)]
        per_unit_terms.append(terms)
        stats.add(terms)
    for u, terms in zip(cir.units, per_unit_terms):
        u.concepts = stats.weights(terms)
        # a unit whose terms are all ubiquitous within this context is generic
        if terms:
            u.features["generic"] = sum(
                1 for t in set(terms) if stats.df.get(t, 0) > 0.5 * stats.n_docs
            ) / len(set(terms))

    # symbols (entities) -> REFERS edges + interning candidates
    by_symbol: dict[str, list[int]] = {}
    for u in cir.units:
        if u.kind in (UnitKind.LOG_TEMPLATE,):
            continue
        ents = u.symbols | (extract_entities(u.text) if len(u.text) < 4000 else set())
        u.symbols = ents
        for e in ents:
            by_symbol.setdefault(e, []).append(u.uid)
    for name, uids in by_symbol.items():
        if len(name) < 3:
            continue
        sym = cir.symbols.get(name)
        if sym is None:
            sym = Symbol(name=name, kind="entity", count=len(uids), tokens=tok.count(name))
            cir.symbols[name] = sym
        sym.count = len(uids)
        sym.units = set(uids)
        if 2 <= len(uids) <= 24:
            for i in range(len(uids) - 1):
                cir.add_edge(uids[i], uids[i + 1], EdgeKind.REFERS, 0.6)

    seed(cir, cfg.protection)
    _anchor_reasoning_chain(cir)
    propagate(cir, cfg.protection)

    cir.meta["stats"] = stats


#: Queries that ask for a *derived* value rather than a stated one.  The answer
#: is not in the context; it must be computed from operands that are.
COMPUTATIONAL_QUERY = re.compile(
    r"\b(total|sum|subtotal|average|mean|median|how\s+(?:much|many)|calculate|"
    r"compute|difference|net|gross|combined|altogether|per\s+unit|overall\s+cost|"
    r"final\s+(?:price|amount|cost)|after\s+(?:discount|tax|fees?))\b",
    re.IGNORECASE,
)


def _anchor_reasoning_chain(cir: CIR) -> None:
    """Protect every operand when the question requires arithmetic.

    Normal retrieval tolerates losing one fact among many -- the answer is a
    span, and a near-miss still answers.  A computation does not: drop one of
    four operands and the result is *wrong*, not partial.  So when the query
    asks for a derived value, every number-bearing prose unit is anchored.

    This is the one place a live model scored our output below the uncompressed
    control (75% vs 100% on the numeric suite) while the intrinsic metric read
    98.8% number recall -- the operands were nearly all there, and "nearly" is
    worth nothing to an arithmetic chain.
    """
    if not cir.query or not COMPUTATIONAL_QUERY.search(cir.query):
        return
    anchored = 0
    for u in cir.units:
        if u.kind not in _OPERAND_KINDS or u.protection >= Protection.ANCHORED:
            continue
        if any(k.startswith("n:") for k in u.obligations):
            u.protection = Protection.ANCHORED
            u.features["operand"] = 1.0
            anchored += 1
    cir.meta["operands_anchored"] = anchored


#: Only prose carries an operand in this sense; a log line or table row is
#: covered by its aggregate and would explode the floor if anchored.
_OPERAND_KINDS = frozenset(
    {UnitKind.SENTENCE, UnitKind.CLAUSE, UnitKind.PARAGRAPH, UnitKind.LIST_ITEM}
)


def make_stats(cir: CIR) -> TermStats:
    return cir.meta.get("stats") or TermStats()
