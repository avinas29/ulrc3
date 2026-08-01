"""Obligation extraction -- the mechanism behind the "never removed" guarantee.

Most compressors express preservation as a *hope* ("the model scored these
tokens as important").  We express it as a **proof obligation**.

Definition.  An obligation ``o`` is an atomic, canonicalisable fact occurring in
the source with a decidable membership test on any candidate output.  Let
``E(x)`` be the extractor applied to text ``x``.  The engine enforces

        E_hard(source)  ⊆  E_hard(output)                       (surjection test)

Because ``E`` is the *same function* on both sides, the test is exact and
self-consistent -- there is no distribution shift between "what we protected"
and "what we check".  Violations are repaired (never ignored) by re-admitting
the minimal-cost unit that carries the missing obligation, and the repair loop
provably terminates because repairs only ever *add* units to a finite set.

Additionally each obligation records its character span inside its unit, which
gives the intra-unit compressor a set of untouchable regions ("critical span
protection").  A filler-word deletion can therefore never eat half of a number,
a negation, or an API parameter name.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..text.lexicon import (
    _NEAR_NUMBER,
    DEONTIC_QUANT,
    DEONTIC_STRONG,
    NEGATION,
    SECURITY,
    SMALL_TALK,
)
from ..text.terms import extract_entities
from ..types import Obligation, ObligationClass, Unit

# --------------------------------------------------------------------------
# Patterns.  Ordered by priority: earlier patterns win overlapping matches.
# --------------------------------------------------------------------------
P_URL = re.compile(r"\b(?:https?|ftp|s3|gs|file|ws{1,2})://[^\s<>\"'`\])]+|(?<![\w/])www\.[^\s<>\"'`\])]+")
P_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
P_PATH = re.compile(
    r"(?<![\w.])(?:/[A-Za-z0-9_.\-]+){2,}/?|(?<![\w])[A-Za-z0-9_.\-]+/(?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+\.[A-Za-z]{1,5}\b"
    r"|(?<![\w])[A-Za-z]:\\[^\s\"'<>|]+"
)
P_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*(?:-[A-Za-z0-9.]+)?\b")
P_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
P_HEXID = re.compile(r"\b(?:0x[0-9a-fA-F]{4,}|[0-9a-f]{16,64})\b")
P_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
    r"|\b(?:Q[1-4]|H[12])\s?(?:FY)?\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b"
)
P_MONEY = re.compile(r"[$€£¥₹]\s?\d[\d,]*(?:\.\d+)?\s?(?:[KMBT]|thousand|million|billion|trillion)?\b", re.IGNORECASE)
P_PERCENT = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\s?%")
P_NUM_UNIT = re.compile(
    r"[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?\s?"
    r"(?:ms|us|ns|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?|d|days?|weeks?|months?|years?"
    r"|[KMGTP]?[Bb](?:ps|/s)?|bytes?|bits?|[KMG]Hz|Hz|px|pt|em|rem|%|x|X|°[CF]|kg|g|mg|lb|oz|km|m|cm|mm|mi|ft|in"
    r"|rps|qps|rpm|tps|iops|req/s|USD|EUR|GBP|JPY|INR)\b"
)
P_NUMBER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w])")
P_JSON_KEY = re.compile(r'"([A-Za-z_$][A-Za-z0-9_$\-.]*)"\s*:')
P_IDENT = re.compile(
    r"\b(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+[A-Za-z_$][A-Za-z0-9_$]*\b"  # dotted
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"  # snake_case
    r"|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"  # camelCase
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"  # PascalCase
    r"|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"  # CONST_CASE
    r"|\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()"  # call site (code context only)
)
#: Prose variant: everything except the bare call-site rule.
P_IDENT_PROSE = re.compile(
    r"\b(?:[A-Za-z_$][A-Za-z0-9_$]*\.)+[A-Za-z_$][A-Za-z0-9_$]*\b"
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
    r"|\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b"
    r"|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
)
P_SQL_IDENT = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE|INDEX\s+ON)\s+([`\"\[]?[A-Za-z_][\w.$]*[`\"\]]?)", re.IGNORECASE
)
P_TOOL_PARAM = re.compile(r"(?:^|[\s,({])(--?[a-zA-Z][\w-]{1,40})\b|\b([a-z_][a-z0-9_]{2,40})\s*=\s*[^\s=]")

_WS = re.compile(r"\s+")
#: Comment and docstring spans in the languages we handle.
_CODE_COMMENT_SPAN = re.compile(
    r'"""[\s\S]*?"""'
    r"|'''[\s\S]*?'''"
    r"|/\*[\s\S]*?\*/"
    r"|^[ \t]*(?:#|//)[^\n]*"
    r"|(?<=\S)[ \t]+(?:#|//)[^\n]*",
    re.MULTILINE,
)


def canon(s: str) -> str:
    return _WS.sub(" ", s.strip().lower().replace(",", "")).strip(" .,;:")


def _digest(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


class ObligationExtractor:
    """Deterministic, ordered, span-aware extractor.

    Complexity: O(len(text)) per pattern, ~20 patterns -> effectively linear
    with a small constant.  Overlap resolution is a single left-to-right sweep
    over an interval list, so the whole extractor is O(n + k log k).
    """

    def __init__(
        self,
        enforce_entities: bool = False,
        enforce_identifiers: bool = True,
        max_constraint_len: int = 400,
        code_context: bool = False,
    ) -> None:
        self.enforce_entities = enforce_entities
        self.enforce_identifiers = enforce_identifiers
        self.max_constraint_len = max_constraint_len
        #: In prose, ``rate limited (see below)`` must not yield an identifier
        #: obligation for ``limited``.  The bare call-site pattern is therefore
        #: enabled only when the text is known to be code.
        self.code_context = code_context

    # -- public API ---------------------------------------------------
    def extract(self, text: str) -> list[tuple[ObligationClass, str, str, int, int]]:
        """Return ``(class, key, literal, start, end)`` tuples, non-overlapping
        for the literal classes and additive for the phrase classes."""
        found: list[tuple[int, int, ObligationClass, str]] = []

        def add(pat: re.Pattern, cls: ObligationClass, group: int = 0) -> None:
            for m in pat.finditer(text):
                g = group
                lit = m.group(g) if m.lastindex and g else m.group(0)
                if lit is None:
                    continue
                s, e = (m.start(g), m.end(g)) if (m.lastindex and g) else (m.start(), m.end())
                found.append((s, e, cls, lit))

        # priority order matters: a URL contains dots and digits that must not
        # be re-extracted as versions or numbers
        add(P_URL, ObligationClass.URL)
        add(P_EMAIL, ObligationClass.EMAIL)
        add(P_UUID, ObligationClass.IDENTIFIER)
        add(P_HEXID, ObligationClass.IDENTIFIER)
        add(P_DATE, ObligationClass.DATE)
        add(P_MONEY, ObligationClass.NUMBER)
        add(P_PERCENT, ObligationClass.NUMBER)
        add(P_NUM_UNIT, ObligationClass.NUMBER)
        add(P_VERSION, ObligationClass.VERSION)
        add(P_PATH, ObligationClass.PATH)
        add(P_NUMBER, ObligationClass.NUMBER)
        for m in P_JSON_KEY.finditer(text):
            found.append((m.start(1), m.end(1), ObligationClass.JSON_KEY, m.group(1)))
        for m in P_SQL_IDENT.finditer(text):
            found.append((m.start(1), m.end(1), ObligationClass.SQL_IDENT, m.group(1).strip('`"[]')))
        if self.enforce_identifiers:
            add(P_IDENT if self.code_context else P_IDENT_PROSE, ObligationClass.IDENTIFIER)
        for m in P_TOOL_PARAM.finditer(text):
            for gi in (1, 2):
                if m.group(gi):
                    found.append((m.start(gi), m.end(gi), ObligationClass.TOOL_PARAM, m.group(gi)))

        # resolve overlaps: longest match wins, ties broken by pattern priority
        found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        chosen: list[tuple[int, int, ObligationClass, str]] = []
        last_end = -1
        for s, e, cls, lit in found:
            if s < last_end:
                continue
            chosen.append((s, e, cls, lit))
            last_end = e

        out: list[tuple[ObligationClass, str, str, int, int]] = []
        for s, e, cls, lit in chosen:
            key = self._key(cls, lit)
            if key:
                out.append((cls, key, lit, s, e))

        # phrase-level obligations (may overlap literals -- that is intended:
        # the clause *and* the number inside it are both protected)
        out.extend(self._phrases(text))
        if self.enforce_entities:
            for ent in extract_entities(text):
                idx = text.find(ent)
                if idx >= 0:
                    out.append((ObligationClass.ENTITY, f"e:{canon(ent)}", ent, idx, idx + len(ent)))
        return out

    # -- helpers ------------------------------------------------------
    def _key(self, cls: ObligationClass, lit: str) -> str | None:
        c = canon(lit)
        if not c:
            return None
        if cls is ObligationClass.NUMBER:
            c = c.replace(" ", "")
            if c in {"0", "1", "2"} and len(lit) <= 1:
                return None  # bare small integers are noise, not facts
            return f"n:{c}"
        if cls is ObligationClass.DATE:
            return f"d:{c}"
        if cls is ObligationClass.URL:
            return f"u:{c.rstrip('/.')}"
        if cls is ObligationClass.EMAIL:
            return f"m:{c}"
        if cls is ObligationClass.PATH:
            return f"p:{c}"
        if cls is ObligationClass.VERSION:
            return f"v:{c}"
        if cls is ObligationClass.JSON_KEY:
            return f"k:{c}"
        if cls is ObligationClass.SQL_IDENT:
            return f"t:{c}"
        if cls is ObligationClass.TOOL_PARAM:
            if len(c) < 3:
                return None
            return f"a:{c}"
        if cls is ObligationClass.IDENTIFIER:
            if len(c) < 3 or c in _IDENT_STOP:
                return None
            return f"i:{c}"
        return f"x:{c}"

    def _prose_regions(self, text: str) -> list[tuple[int, int]]:
        """Comment and docstring spans -- the only prose inside code.

        Running deontic patterns over executable lines yields obligations like
        ``constraint: raise ValueError("amount must be positive")`` whose
        "clause" is a code fragment: unmatchable after body elision and
        unrepairable as prose.  Restricting phrase extraction to comments and
        string literals keeps the real docstring constraints and drops the
        artefacts.
        """
        spans: list[tuple[int, int]] = []
        for m in _CODE_COMMENT_SPAN.finditer(text):
            spans.append((m.start(), m.end()))
        return spans

    def _phrases(self, text: str) -> list[tuple[ObligationClass, str, str, int, int]]:
        out: list[tuple[ObligationClass, str, str, int, int]] = []
        regions = self._prose_regions(text) if self.code_context else None
        if regions is not None and not regions:
            return out
        # In code, `max`, `min`, `not`, `limit` and `never` are identifiers and
        # keywords, not deontic operators.  Phrase obligations exist to protect
        # *natural-language* constraints; code semantics are protected
        # structurally by AST-level selection instead.  Extracting them here
        # produced unsatisfiable tier-1 obligations like "constraint:kappa =
        # [max(1,".
        specs = (
            ((DEONTIC_STRONG, ObligationClass.CONSTRAINT, False),)
            if self.code_context
            else (
                (DEONTIC_STRONG, ObligationClass.CONSTRAINT, False),
                (DEONTIC_QUANT, ObligationClass.CONSTRAINT, True),
                (NEGATION, ObligationClass.NEGATION, False),
            )
        ) + ((SECURITY, ObligationClass.SECURITY, False),)
        for pat, cls, needs_number in specs:
            for m in pat.finditer(text):
                if regions is not None and not any(a <= m.start() < b for a, b in regions):
                    continue
                s, e = _clause_bounds(text, m.start(), m.end(), self.max_constraint_len)
                clause = text[s:e].strip()
                if len(clause) < 4:
                    continue
                if needs_number and not _NEAR_NUMBER.search(clause):
                    continue  # a quantitative cue with no quantity is prose
                if len(clause.split()) < 4 or SMALL_TALK.match(clause):
                    continue  # "No problem at all" is not a negation obligation
                out.append((cls, f"{cls.value[0]}c:{_digest(canon(clause))}", clause, s, e))
        return out


_IDENT_STOP = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "will", "not",
        "you", "are", "was", "were", "can", "may", "all", "any", "but", "its",
        "e.g", "i.e", "etc", "vs", "www", "com", "org", "net",
    }
)

_CLAUSE_STOP = re.compile(r"[.;!?\n]|(?:,\s+(?:and|but|or|which|while)\b)")


def _clause_bounds(text: str, s: int, e: int, max_len: int) -> tuple[int, int]:
    """Expand a cue match to its enclosing clause, bounded by `max_len`."""
    left = s
    lo = max(0, s - max_len)
    for m in _CLAUSE_STOP.finditer(text, lo, s):
        left = m.end()
    right = e
    hi = min(len(text), e + max_len)
    m = _CLAUSE_STOP.search(text, e, hi)
    right = m.start() if m else hi
    while left < len(text) and text[left] in " \t\r\n":
        left += 1
    return left, right


# --------------------------------------------------------------------------
# Unit-level API
# --------------------------------------------------------------------------
def annotate_unit(unit: Unit, ex: ObligationExtractor) -> list[Obligation]:
    """Attach obligations + protected spans (unit-local coordinates) to a unit."""
    obs: list[Obligation] = []
    spans: list[tuple[int, int]] = []
    for cls, key, lit, s, e in ex.extract(unit.text):
        weight = _WEIGHTS.get(cls, 1.0)
        obs.append(Obligation(key=key, cls=cls, literal=lit, weight=weight, units={unit.uid}))
        unit.obligations.add(key)
        spans.append((s, e))
    if spans:
        unit.meta["protected_spans"] = _merge_spans(spans)
    return obs


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans.sort()
    out: list[list[int]] = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


_WEIGHTS: dict[ObligationClass, float] = {
    ObligationClass.SECURITY: 3.0,
    ObligationClass.CONSTRAINT: 2.6,
    ObligationClass.NEGATION: 2.2,
    ObligationClass.NUMBER: 2.0,
    ObligationClass.DATE: 1.8,
    ObligationClass.TOOL_PARAM: 1.8,
    ObligationClass.JSON_KEY: 1.7,
    ObligationClass.SQL_IDENT: 1.7,
    ObligationClass.IDENTIFIER: 1.5,
    ObligationClass.URL: 1.5,
    ObligationClass.EMAIL: 1.5,
    ObligationClass.PATH: 1.3,
    ObligationClass.VERSION: 1.4,
    ObligationClass.ENTITY: 1.1,
}


#: Classes whose loss changes meaning rather than detail -- always repaired
#: first, wherever in the source they occur.
CRITICAL_CLASSES = frozenset(
    {ObligationClass.CONSTRAINT, ObligationClass.SECURITY, ObligationClass.NEGATION}
)


@dataclass
class AuditReport:
    """Three independent measurements; see :class:`ulrc3.types.Verification`."""

    integrity_missing: list[Obligation] = field(default_factory=list)
    critical_missing: list[Obligation] = field(default_factory=list)
    other_missing: list[Obligation] = field(default_factory=list)
    integrity_total: int = 0
    integrity_kept: int = 0
    critical_total: int = 0
    critical_kept: int = 0
    retention_total: int = 0
    retention_kept: int = 0

    def repair_queue(self, limit: int = 128) -> list[Obligation]:
        """Repair order: integrity first, then critical, then by weight."""
        rest = sorted(self.other_missing, key=lambda o: -o.weight)
        return (self.integrity_missing + self.critical_missing + rest)[:limit]


def audit(
    rendered: str,
    required: Iterable[Obligation],
    ex: ObligationExtractor,
    retained_keys: set[str] | None = None,
) -> AuditReport:
    """Surjection test.

    ``retained_keys`` is the set of obligation keys that the *selected renderings
    claim to carry*.  Their absence from the output is an internal inconsistency
    (an integrity violation), not a budget decision -- which is why the two are
    counted separately.

    Phrase obligations are checked by normalised substring containment (they are
    emitted verbatim when their unit is kept); literal obligations are checked
    against the extractor's own output on the rendered text, which makes the
    test symmetric and immune to formatting drift.
    """
    present_keys: set[str] = set()
    for _cls, key, _lit, _s, _e in ex.extract(rendered):
        present_keys.add(key)
    normalised = canon(rendered)
    retained = retained_keys or set()

    rep = AuditReport()
    for ob in required:
        if not ob.hard:
            continue
        present = ob.key in present_keys
        if not present and ob.cls in _PHRASE_CLASSES and canon(ob.literal) in normalised:
            present = True
        # last resort: literal containment (handles extractor overlap shadowing,
        # e.g. a number that got absorbed into a longer URL match downstream)
        if not present and ob.literal and canon(ob.literal) in normalised:
            present = True

        rep.retention_total += 1
        if present:
            rep.retention_kept += 1

        if ob.key in retained:
            rep.integrity_total += 1
            if present:
                rep.integrity_kept += 1
            else:
                rep.integrity_missing.append(ob)
                continue
        if ob.cls in CRITICAL_CLASSES:
            rep.critical_total += 1
            if present:
                rep.critical_kept += 1
            else:
                rep.critical_missing.append(ob)
        elif not present and ob.key not in retained:
            rep.other_missing.append(ob)
    return rep


_PHRASE_CLASSES = frozenset(
    {ObligationClass.CONSTRAINT, ObligationClass.NEGATION, ObligationClass.SECURITY}
)
