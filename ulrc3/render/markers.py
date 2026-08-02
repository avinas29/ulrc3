"""Tokenizer-aware marker vocabulary.

A detail that matters more than it should: the structural markers of the IR are
paid for on every request, and their cost is tokenizer-dependent.  Measured on
``cl100k_base``:

    "[["  -> 1 token          "⟦"  -> 2 tokens
    "#"   -> 1 token          "⟪"  -> 2 tokens
    "§"   -> 1 token          "▁"  -> 2 tokens

Pretty Unicode brackets literally double the framing cost, and on a document
with 40 sections that is ~80 wasted tokens *per request, forever*.

So we do not hard-code a marker set.  We define candidate sets, measure each
against the active tokenizer once per process, and pick the cheapest set that
is still visually unambiguous.  The chosen set is recorded in the output header
so decoding is never ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tokenization import Tokenizer


@dataclass(frozen=True)
class MarkerSet:
    name: str
    ctx: str
    sym: str
    system: str
    task: str
    query: str
    doc: str
    facts: str
    dropped: str
    handle_open: str
    handle_close: str
    bullet: str

    def cost(self, tok: Tokenizer) -> int:
        return sum(
            tok.count(x)
            for x in (
                self.ctx, self.sym, self.system, self.task, self.query,
                self.doc, self.facts, self.dropped,
                self.handle_open + "12" + self.handle_close, self.bullet,
            )
        )


CANDIDATES: tuple[MarkerSet, ...] = (
    MarkerSet(
        name="ascii-hash",
        ctx="#CTX", sym="#SYM", system="#SYS", task="#TASK", query="#Q",
        doc="#D", facts="#FACT", dropped="#CUT",
        handle_open="~", handle_close="", bullet="-",
    ),
    MarkerSet(
        name="section",
        ctx="§CTX", sym="§SYM", system="§SYS", task="§TASK", query="§Q",
        doc="§D", facts="§FACT", dropped="§CUT",
        handle_open="§", handle_close="", bullet="-",
    ),
    MarkerSet(
        name="bracket",
        ctx="[CTX]", sym="[SYM]", system="[SYS]", task="[TASK]", query="[Q]",
        doc="[D", facts="[FACT]", dropped="[CUT]",
        handle_open="[", handle_close="]", bullet="-",
    ),
    MarkerSet(
        name="minimal",
        ctx="CTX", sym="SYM", system="SYS", task="TASK", query="Q",
        doc="D", facts="FACT", dropped="CUT",
        handle_open="", handle_close="", bullet="-",
    ),
)

_CACHE: dict[str, MarkerSet] = {}


def choose(tok: Tokenizer, style: str = "auto") -> MarkerSet:
    """Pick the cheapest marker set under the active tokenizer."""
    if style and style != "auto":
        for c in CANDIDATES:
            if c.name.startswith(style) or style in c.name:
                return c
    key = getattr(tok, "name", "?")
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    best = min(CANDIDATES, key=lambda c: (c.cost(tok), CANDIDATES.index(c)))
    _CACHE[key] = best
    return best


#: Every non-source token the renderer may emit.  The provenance verifier uses
#: this as the closed vocabulary: any *word* in the output that is neither a
#: source word nor in this set is a hallucination, and there must be none.
MARKER_VOCAB: frozenset[str] = frozenset(
    ["ctx", "sym", "sys", "task", "q", "d", "fact", "cut", "v1", "mode", "tok", "src", "keep", "order", "salience", "original", "drop", "tight", "full", "stub", "sig", "ref", "carrier", "uses", "schema", "sampled", "memory", "compact", "detail", "verbatim", "delta", "except", "eg", "rows", "num", "enum", "card", "min", "max", "sum", "n", "keys", "profile", "expand", "more", "object", "array", "int", "float", "bool", "null", "str", "long", "uuid", "datetime", "date", "email", "url", "true", "false", "none", "nan", "lossless", "conservative", "balanced", "aggressive", "extreme", "prose", "markdown", "code", "json", "yaml", "logs", "conversation", "legal", "apidocs", "sql", "table", "mixed", "pref", "decision", "task", "constraint", "correction", "fact", "chat", "u", "a", "trace", "debug", "info", "notice", "warn", "warning", "error", "err", "fatal", "critical", "severe", "panic"]
)
