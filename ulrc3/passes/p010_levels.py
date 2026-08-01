"""Fidelity-ladder construction + **Critical Span Protection**.

Pipelines that have a natural ladder (code, logs, JSON, chat) build their own.
Everything else gets one here: ``drop < tight < full``, where *tight* is an
**extractive** rewrite of the unit -- deletions only, never generation.

The deletions are constrained by the protected spans computed during obligation
extraction.  Concretely, a deletion candidate ``[a,b)`` is applied only if

        ∀ (s,e) ∈ protected(u) :  [a,b) ∩ [s,e) = ∅

so a filler-word sweep can never bite into a number, a negation, an API
parameter, an identifier or a deontic clause.  This is the difference between
"we hope the model kept the important tokens" and "the important tokens are
provably still there".

What gets deleted (all measured to be recoverable or vacuous):

* hedges and meta-commentary ("it is important to note that");
* intensifier adverbs ("very", "basically", "actually");
* parentheticals with no obligations and < 40 chars;
* trailing example clauses with no obligations;
* assistant-style boilerplate openers.

Complexity: O(len(text)) per unit with precompiled patterns.
"""

from __future__ import annotations

import re

from ..text.lexicon import ASSISTANT_BOILERPLATE, FILLER, HEDGE
from ..text.segment import split_sentences
from ..types import Level, Protection, Unit, UnitKind
from .base import Pass, PassContext

_PAREN = re.compile(r"\s*\((?:[^()]{1,60})\)")
#: A deletion candidate containing any of these is refused outright, even when
#: it clears the protected-span test.  Digits, operators, code spans and units
#: of measure are load-bearing far more often than the obligation extractor can
#: prove -- e.g. "(1 - 1/e)" in a bound, or "(see `--strict`)".
_LOAD_BEARING = re.compile(r"[\d`$%°€£¥]|[=<>+*/^~|]|\b(?:not|no|never|must|only|max|min)\b", re.IGNORECASE)
_TRAILING_EXAMPLE = re.compile(
    r"[,;]?\s+(?:for\s+(?:example|instance)|e\.g\.|such\s+as)\b[^.;]{0,120}", re.IGNORECASE
)
_FILLER_WORD = re.compile(
    r"\b(" + "|".join(sorted(FILLER, key=len, reverse=True)) + r")\b\s*", re.IGNORECASE
)
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?])")

#: Unit kinds whose text is structural and must never be edited in place.
NO_EDIT = frozenset(
    {
        UnitKind.CODE_DEF,
        UnitKind.CODE_IMPORT,
        UnitKind.CODE_STMT,
        UnitKind.JSON_NODE,
        UnitKind.SQL_STMT,
        UnitKind.LOG_TEMPLATE,
        UnitKind.LOG_RECORD,
        UnitKind.TABLE_ROW,
        UnitKind.TOOL_SCHEMA,
    }
)

_MIN_TOKENS_FOR_TIGHT = 12


class LevelPass(Pass):
    name = "levels"

    def __init__(self, allow_intra_unit: bool = True) -> None:
        self.allow_intra_unit = allow_intra_unit
        self._tightened = 0
        self._saved = 0
        self._carriers = 0

    def run(self, ctx: PassContext) -> None:
        self._tightened = 0
        self._saved = 0
        self._carriers = 0
        for u in ctx.cir.units:
            if u.levels:
                u.level = u.max_level()
                continue
            levels: list[Level] = [Level("drop", "", 0, 0.0, {}, set())]

            # The *carrier* rung: only the obligation-bearing clauses.  This is
            # what makes a hard preservation guarantee affordable -- a LOCKED
            # 267-token paragraph costs 30 tokens here while still carrying
            # every number, constraint and identifier it is responsible for.
            carrier = carrier_text(u)
            if carrier and u.protection < Protection.FROZEN:
                ct = ctx.tok.count(carrier) + 1
                if ct < u.tokens * 0.85:
                    levels.append(
                        Level(
                            name="carrier",
                            text=carrier,
                            tokens=ct,
                            fidelity=0.5,
                            obligations=set(u.obligations),
                        )
                    )
                    self._carriers += 1

            if (
                self.allow_intra_unit
                and u.kind not in NO_EDIT
                and u.protection < Protection.FROZEN
                and u.tokens >= _MIN_TOKENS_FOR_TIGHT
            ):
                tight = tighten(u)
                if tight and tight != u.text:
                    t = ctx.tok.count(tight) + 1
                    if t < u.tokens - 1:
                        levels.append(
                            Level(
                                name="tight",
                                text=tight,
                                tokens=t,
                                fidelity=0.9,
                                obligations=None,  # recomputed on demand
                            )
                        )
                        self._tightened += 1
                        self._saved += u.tokens - t
            full = u.text.strip()
            levels.append(Level(name="full", text=full, tokens=u.tokens, fidelity=1.0,
                                obligations=set(u.obligations)))
            levels = [levels[0]] + sorted(levels[1:], key=lambda lv: lv.tokens)
            u.levels = levels
            u.level = len(levels) - 1

        # verify obligation coverage of every generated `tight` level exactly
        ex = ctx.scratch.get("ex_prose")
        if ex is not None:
            for u in ctx.cir.units:
                for lv in u.levels:
                    if lv.name == "tight" and lv.obligations is None:
                        lv.obligations = {k for _c, k, _l, _s, _e in ex.extract(lv.text)}

        if ctx.cfg.ablated("no_ladder"):
            # Collapse every ladder to {drop, full}: the decision space of a
            # keep-or-drop compressor.  Used to measure what the ladder buys.
            for u in ctx.cir.units:
                if len(u.levels) > 2:
                    u.levels = [u.levels[0], u.levels[-1]]
                u.level = u.max_level()

        assign_tiers(ctx)

    def _unused(self) -> None:  # pragma: no cover
        return None

    def note(self, ctx: PassContext) -> str:
        return f"{self._tightened} tightened (-{self._saved} tok), {self._carriers} carriers"


#: Obligation key prefixes that are tier 1 wherever prose carries them: losing
#: one changes the *meaning* of the context, not its level of detail.
#: (c=constraint, s=security, n=negation clause digests; then value classes.)
ALWAYS_TIER1 = frozenset({"cc", "sc", "nc", "n", "d", "u", "m", "k", "t", "a", "v"})

#: Kinds that carry an aggregate representation of their values (log template
#: with slot samples, table profile with min/max/cardinality, JSON schema).
#: Enforcing every numeral of a 50k-line log as tier 1 is not a guarantee, it is
#: a refusal to compress -- these kinds are covered by their aggregate instead.
_AGGREGATE_KINDS = frozenset(
    {UnitKind.LOG_TEMPLATE, UnitKind.LOG_RECORD, UnitKind.TABLE_ROW, UnitKind.JSON_NODE}
)
_PROSE_KINDS = frozenset(
    {
        UnitKind.SENTENCE, UnitKind.CLAUSE, UnitKind.PARAGRAPH, UnitKind.LIST_ITEM,
        UnitKind.HEADING, UnitKind.TURN, UnitKind.MEMORY, UnitKind.INSTRUCTION,
        UnitKind.CODE_DOCSTRING, UnitKind.CODE_COMMENT, UnitKind.TOOL_SCHEMA,
    }
)


def assign_tiers(ctx: PassContext) -> None:
    """Mark repair priority.

    Tiers no longer decide *what is guaranteed* -- that is settled structurally
    at audit time by comparing the retained rungs against the output (see
    :func:`ulrc3.ir.obligations.audit`).  What remains here is priority: which
    obligations the repair budget should be spent on first when the budget
    cannot cover everything.
    """
    units = ctx.cir.units
    for key, ob in ctx.cir.obligations.items():
        prefix = key.split(":", 1)[0]
        owners = [units[uid] for uid in ob.units if uid < len(units)]
        # An obligation carried *only* by superseded turns is not a fact worth
        # spending budget on -- it is a retracted one.  Repairing it back in
        # reintroduces the contradiction the supersession DAG just removed.
        if owners and all(u.features.get("superseded") for u in owners):
            ob.tier = 2
            ob.weight = 0.01
            continue
        prose_owned = any(u.kind in _PROSE_KINDS for u in owners)
        if prefix in _CRITICAL_PREFIXES or prefix in ALWAYS_TIER1 and prose_owned:
            ob.tier = 1
        else:
            ob.tier = 2


_CRITICAL_PREFIXES = frozenset({"cc", "sc", "nc"})


def carrier_text(u: Unit, max_fraction: float = 0.8) -> str:
    """Minimal obligation-preserving rendering of a unit.

    Granularity is the **sentence**, not the clause.  Clause-level carriers are
    marginally cheaper but read as broken fragments ("2011 and is headquartered
    in Dublin, Ireland"), and a fragment that a model has to guess the subject
    of is a false economy.  Keeping whole sentences that contain at least one
    protected span still removes the connective narration around them, and the
    result provably contains every obligation the unit owns.
    """
    spans: list[tuple[int, int]] = u.meta.get("protected_spans") or []
    if not spans:
        return ""
    text = u.text
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return ""  # single sentence: the carrier *is* the unit
    keep: list[tuple[int, int]] = []
    for a, b in sentences:
        if any(s < b and a < e for s, e in spans):
            keep.append((a, b))
    if not keep:
        return ""
    total = sum(b - a for a, b in keep)
    if total >= len(text.strip()) * max_fraction:
        return ""
    return " ".join(text[a:b].strip() for a, b in keep).strip()


def tighten(u: Unit) -> str:
    """Extractive intra-unit compression respecting protected spans."""
    text = u.text
    protected: list[tuple[int, int]] = u.meta.get("protected_spans", [])

    cuts: list[tuple[int, int]] = []

    def propose(a: int, b: int) -> None:
        if b <= a:
            return
        for s, e in protected:
            if a < e and s < b:
                return
        if _LOAD_BEARING.search(text[a:b]):
            return
        cuts.append((a, b))

    m = ASSISTANT_BOILERPLATE.match(text)
    if m and m.end() < len(text) - 10:
        propose(m.start(), m.end())
    for m in HEDGE.finditer(text):
        propose(m.start(), m.end() + (1 if text[m.end() : m.end() + 1] == "," else 0))
    for m in _PAREN.finditer(text):
        propose(m.start(), m.end())
    for m in _TRAILING_EXAMPLE.finditer(text):
        propose(m.start(), m.end())
    for m in _FILLER_WORD.finditer(text):
        propose(m.start(), m.end())

    if not cuts:
        return text
    cuts.sort()
    out: list[str] = []
    pos = 0
    for a, b in cuts:
        if a < pos:
            continue
        out.append(text[pos:a])
        pos = b
    out.append(text[pos:])
    res = "".join(out)
    res = _MULTISPACE.sub(" ", res)
    res = _SPACE_PUNCT.sub(r"\1", res)
    res = re.sub(r"\s{2,}", " ", res).strip()
    # never emit something that lost a sentence terminator it used to have
    if text.rstrip().endswith((".", "!", "?")) and not res.endswith((".", "!", "?")):
        res += "."
    # Deleting a leading phrase ("As previously mentioned, ") leaves the
    # sentence starting mid-clause in lower case, which reads as a fragment.
    # Restoring the capital is the only *non-span* edit the engine makes; it
    # changes case only, never characters, and the provenance check is
    # case-insensitive by construction.
    if res and text[:1].isupper() and res[:1].islower():
        res = res[0].upper() + res[1:]
    return res
