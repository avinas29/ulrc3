"""Core Context-IR (CIR) datatypes.

The whole engine is organised around one idea: a prompt is *source code*, and
compression is *compilation* to an optimised intermediate representation.

The IR is **addressable and lossless**.  Compression never destroys the source;
it selects a *rendering policy* per node.  That single design decision buys us
three properties that no perplexity-based compressor has:

1. provenance  -- every emitted character can be traced to a source span, so
   hallucination is impossible *by construction* (see `passes.p090_verify`);
2. recovery    -- dropped spans keep stable handles and can be re-expanded on
   demand by an agent (see `recovery.py`);
3. verifiability -- retention invariants are checkable properties of the IR,
   not vibes (see `ir/protection.py` and `ir/obligations.py`).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------
# Protection lattice
# --------------------------------------------------------------------------
class Protection(enum.IntEnum):
    """A total order used as a join-semilattice for taint propagation.

    Higher = more protected.  ``join(a, b) = max(a, b)``.  The lattice is finite
    and every transfer function in :mod:`ulrc3.ir.protection` is monotone, so the
    dataflow fixpoint provably terminates.
    """

    DROPPABLE = 0  #: may vanish entirely
    ELASTIC = 1  #: may be paraphrased-by-extraction, truncated, merged
    ANCHORED = 2  #: must survive in *some* form (possibly a stub or reference)
    LOCKED = 3  #: must survive carrying *all* of its obligations intact
    FROZEN = 4  #: byte-identical, never reordered, never merged

    @property
    def label(self) -> str:
        return self.name.lower()


def demote(p: Protection, steps: int = 1) -> Protection:
    """Protection inherited across a dependency edge decays by `steps` levels.

    A FROZEN system prompt that references symbol ``S`` does not make ``S``'s
    definition FROZEN -- it makes it LOCKED.  Transitive closure therefore has a
    natural horizon instead of protecting the whole document.
    """
    return Protection(max(int(Protection.DROPPABLE), int(p) - steps))


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
class UnitKind(enum.Enum):
    """Semantic granule kinds.  The unit -- not the token -- is the atom of
    selection.  Token-level dropping (LLMLingua) produces text no human can
    audit and no parser can validate; unit-level selection keeps the output
    grammatical, parseable and diff-able against the source."""

    SENTENCE = "sentence"
    CLAUSE = "clause"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"
    TABLE_ROW = "table_row"
    CODE_IMPORT = "code_import"
    CODE_DEF = "code_def"
    CODE_STMT = "code_stmt"
    CODE_COMMENT = "code_comment"
    CODE_DOCSTRING = "code_docstring"
    JSON_NODE = "json_node"
    LOG_RECORD = "log_record"
    LOG_TEMPLATE = "log_template"
    TURN = "turn"
    MEMORY = "memory"
    INSTRUCTION = "instruction"
    TOOL_SCHEMA = "tool_schema"
    SQL_STMT = "sql_stmt"
    FENCE = "fence"
    RAW = "raw"


class EdgeKind(enum.Enum):
    """Typed relations over units.

    ``REQUIRES`` is the only edge kind with *hard* semantics: the retained set
    must be closed under it (up to the declared horizon).  Everything else
    informs scoring.
    """

    REQUIRES = "requires"  #: dst is meaningless without src
    REFERS = "refers"  #: soft reference (shared entity, citation)
    DUPLICATE_OF = "duplicate_of"  #: near-identical content
    ELABORATES = "elaborates"  #: dst adds detail to src
    SUPERSEDES = "supersedes"  #: dst invalidates src (belief revision)
    CONTAINS = "contains"  #: hierarchy
    ANSWERS = "answers"  #: dst responds to src (QA / dialogue)
    ADJACENT = "adjacent"  #: positional neighbour


@dataclass(slots=True)
class Span:
    """Half-open byte-ish range into a source document (character offsets)."""

    doc_id: str
    start: int
    end: int

    def __len__(self) -> int:  # pragma: no cover - trivial
        return max(0, self.end - self.start)

    def overlaps(self, other: Span) -> bool:
        return self.doc_id == other.doc_id and self.start < other.end and other.start < self.end


@dataclass(slots=True)
class Level:
    """One rendering fidelity of a unit.

    Existing compressors make a binary decision per token: keep or drop.  We
    make a *multiple-choice* decision per unit -- drop / reference / stub /
    compressed / full -- each with its own token cost, its own concept coverage
    and its own obligation set.  Selection therefore becomes a Multiple-Choice
    Knapsack Problem, which strictly dominates keep-or-drop: for the same budget
    the optimiser can buy a 6-token signature stub instead of paying 180 tokens
    for a full function body or losing the symbol entirely.

    Every level is **extractive**: its text is composed only of source spans and
    a closed vocabulary of structural markers.  That is what makes the zero-
    hallucination guarantee checkable.
    """

    name: str
    text: str
    tokens: int
    fidelity: float = 1.0
    concepts: dict[str, float] | None = None
    obligations: set[str] | None = None


DROP = Level(name="drop", text="", tokens=0, fidelity=0.0)


@dataclass(slots=True)
class Unit:
    """A selectable semantic granule."""

    uid: int
    doc_id: str
    kind: UnitKind
    span: Span
    text: str
    order: int  #: original document order (stable sort key)
    depth: int = 0  #: hierarchy depth (heading level, AST depth, ...)
    parent: int | None = None
    protection: Protection = Protection.ELASTIC
    tokens: int = 0  #: rendering cost under the active tokenizer
    salience: float = 0.0  #: final importance in [0, 1]
    attention: float = 0.0  #: simulated attention mass (PPR stationary)
    coverage_gain: float = 0.0  #: marginal submodular gain when selected
    levels: list[Level] = field(default_factory=list)  #: index 0 is always DROP
    level: int = 0  #: chosen level index
    render: str | None = None  #: chosen surface form (None -> use `text`)
    obligations: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    concepts: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    segment: str = "body"  #: logical region (system/instruction/query/body/...)

    @property
    def surface(self) -> str:
        if self.render is not None:
            return self.render
        if self.levels and 0 <= self.level < len(self.levels):
            return self.levels[self.level].text
        return self.text

    @property
    def selected(self) -> bool:
        return self.level > 0

    @property
    def cost(self) -> int:
        """Token cost at the currently chosen level."""
        if self.levels and 0 <= self.level < len(self.levels):
            return self.levels[self.level].tokens
        return self.tokens if self.level > 0 else 0

    def max_level(self) -> int:
        return max(0, len(self.levels) - 1)

    def min_level(self) -> int:
        """Lowest level admissible under this unit's protection class."""
        if not self.levels:
            return 0
        if self.protection >= Protection.FROZEN:
            return self.max_level()
        if self.protection >= Protection.LOCKED:
            # lowest level that still carries every obligation of the unit
            for i, lv in enumerate(self.levels):
                if i == 0:
                    continue
                obs = lv.obligations if lv.obligations is not None else self.obligations
                if self.obligations <= (obs or set()):
                    return i
            return self.max_level()
        if self.protection >= Protection.ANCHORED:
            return 1
        return 0

    def set_level(self, idx: int) -> None:
        self.level = max(0, min(idx, self.max_level()))

    def is_protected(self, at_least: Protection = Protection.ANCHORED) -> bool:
        return self.protection >= at_least


@dataclass(slots=True)
class Edge:
    src: int
    dst: int
    kind: EdgeKind
    weight: float = 1.0


class ObligationClass(enum.Enum):
    """Classes of atomic facts whose survival is *audited*, not hoped for."""

    NUMBER = "number"
    DATE = "date"
    IDENTIFIER = "identifier"  #: code symbol / variable / function
    JSON_KEY = "json_key"
    SQL_IDENT = "sql_ident"
    URL = "url"
    PATH = "path"
    EMAIL = "email"
    VERSION = "version"
    ENTITY = "entity"
    CONSTRAINT = "constraint"  #: modal / deontic phrase
    NEGATION = "negation"
    UNIT = "unit"  #: measurement unit attached to a number
    TOOL_PARAM = "tool_param"
    SECURITY = "security"
    IMPERATIVE = "imperative"

    @property
    def hard(self) -> bool:
        """Hard classes are *enforced*: a missing one triggers repair."""
        return self in _HARD_OBLIGATIONS


_HARD_OBLIGATIONS = frozenset(
    {
        ObligationClass.NUMBER,
        ObligationClass.DATE,
        ObligationClass.IDENTIFIER,
        ObligationClass.JSON_KEY,
        ObligationClass.SQL_IDENT,
        ObligationClass.URL,
        ObligationClass.PATH,
        ObligationClass.EMAIL,
        ObligationClass.VERSION,
        ObligationClass.CONSTRAINT,
        ObligationClass.NEGATION,
        ObligationClass.TOOL_PARAM,
        ObligationClass.SECURITY,
    }
)


@dataclass(slots=True)
class Obligation:
    """An atomic must-preserve item with a canonical matcher.

    ``key`` is the canonical form used for the surjection test in the verifier;
    ``literal`` is the exact source string that must appear when the class is
    verbatim-enforced.

    **Tiers.**  A blanket promise that no identifier is ever removed is not a
    guarantee, it is a refusal to compress.  We therefore define the tier
    structurally rather than by wishful thinking:

        tier 1  the obligation is carried by the unit's *cheapest admissible
                rendering*, so it survives whenever the unit survives at all,
                at any compression ratio.  Signatures, JSON keys, table
                headers, constraints, log templates and every obligation of a
                FROZEN/LOCKED region land here.

        tier 2  the obligation only exists in a higher rung (a function body, a
                sampled record, a verbatim log line).  It is preserved when the
                budget allows, recovered into the FACT block when it is
                high-value, and *always accounted for* -- never silently lost.

    Tier-1 recall is the guarantee and must be 1.0; tier-2 recall is reported.
    """

    key: str
    cls: ObligationClass
    literal: str
    weight: float = 1.0
    units: set[int] = field(default_factory=set)
    verbatim: bool = True
    tier: int = 1

    @property
    def hard(self) -> bool:
        return self.cls.hard


@dataclass(slots=True)
class Symbol:
    """An interned entity / identifier with occurrence statistics."""

    name: str
    kind: str  #: entity | code | url | json_key | table | ...
    count: int = 0
    tokens: int = 0
    units: set[int] = field(default_factory=set)
    alias: str | None = None  #: assigned short alias, if interning pays off


@dataclass(slots=True)
class Document:
    """One input document with its detected type and segmentation."""

    doc_id: str
    text: str
    doctype: str = "prose"
    doctype_scores: dict[str, float] = field(default_factory=dict)
    role: str = "body"  #: system | instruction | query | tools | body | history
    meta: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  #: relevance prior (e.g. retriever score for RAG)


@dataclass
class CIR:
    """The Context Intermediate Representation: a typed, span-addressed DAG."""

    units: list[Unit] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    docs: dict[str, Document] = field(default_factory=dict)
    obligations: dict[str, Obligation] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    query: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------
    def add_unit(self, unit: Unit) -> Unit:
        unit.uid = len(self.units)
        self.units.append(unit)
        return unit

    def add_edge(self, src: int, dst: int, kind: EdgeKind, weight: float = 1.0) -> None:
        if src == dst or src < 0 or dst < 0:
            return
        self.edges.append(Edge(src, dst, kind, weight))

    def add_obligation(self, ob: Obligation) -> Obligation:
        cur = self.obligations.get(ob.key)
        if cur is None:
            self.obligations[ob.key] = ob
            return ob
        cur.units |= ob.units
        cur.weight = max(cur.weight, ob.weight)
        return cur

    # -- access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[Unit]:
        return iter(self.units)

    def unit(self, uid: int) -> Unit:
        return self.units[uid]

    def selected_units(self) -> list[Unit]:
        return [u for u in self.units if u.level > 0]

    def output_tokens(self) -> int:
        return sum(u.cost for u in self.units if u.level > 0)

    def out_edges(self) -> dict[int, list[Edge]]:
        idx: dict[int, list[Edge]] = {}
        for e in self.edges:
            idx.setdefault(e.src, []).append(e)
        return idx

    def in_edges(self) -> dict[int, list[Edge]]:
        idx: dict[int, list[Edge]] = {}
        for e in self.edges:
            idx.setdefault(e.dst, []).append(e)
        return idx

    def total_tokens(self) -> int:
        return sum(u.tokens for u in self.units)

    def units_of(self, doc_id: str) -> list[Unit]:
        return [u for u in self.units if u.doc_id == doc_id]


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class PassStat:
    name: str
    ms: float
    units_in: int
    units_out: int
    tokens_in: int
    tokens_out: int
    note: str = ""


@dataclass
class Verification:
    """Output of the audit stage -- the compressor's own proof obligations.

    Three distinct numbers, because conflating them is how compressors make
    promises they cannot keep:

    ``integrity``  Of everything the engine chose to *retain*, how much survived
                   intact?  **This must be 1.0 at every compression ratio.**  It
                   is the formal statement of "we never partially destroy what
                   we keep": no half-deleted number, no truncated URL, no
                   negation stripped out of a sentence we kept.  Token-level
                   compressors cannot make this claim -- deleting tokens inside
                   a retained span is precisely their mechanism.

    ``critical``   Constraints, security policies and negations *anywhere* in
                   the source.  Repair prioritises these because a dropped
                   "must not" is dangerous in a way a dropped adjective is not.
                   Target 1.0; shortfalls are enumerated in ``missing``.

    ``retention``  Every obligation in the source.  This one is *supposed* to
                   fall as the budget tightens -- it is the tradeoff dial, not a
                   guarantee.  Reporting it honestly is the point.
    """

    ok: bool = True
    provenance_ok: bool = True
    inflation_ok: bool = True
    frozen_ok: bool = True
    integrity_total: int = 0
    integrity_kept: int = 0
    critical_total: int = 0
    critical_kept: int = 0
    retention_total: int = 0
    retention_kept: int = 0
    missing: list[str] = field(default_factory=list)
    repairs: int = 0
    syntax_ok: bool = True
    syntax_notes: list[str] = field(default_factory=list)

    @property
    def integrity(self) -> float:
        if not self.integrity_total:
            return 1.0
        return self.integrity_kept / self.integrity_total

    @property
    def critical_recall(self) -> float:
        if not self.critical_total:
            return 1.0
        return self.critical_kept / self.critical_total

    @property
    def retention(self) -> float:
        if not self.retention_total:
            return 1.0
        return self.retention_kept / self.retention_total

    #: Backwards-compatible alias: the headline guarantee.
    @property
    def obligation_recall(self) -> float:
        return self.integrity


@dataclass
class CompressionResult:
    text: str
    tokens_in: int
    tokens_out: int
    doctypes: dict[str, str] = field(default_factory=dict)
    verification: Verification = field(default_factory=Verification)
    confidence: float = 1.0
    stats: list[PassStat] = field(default_factory=list)
    residuals: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        """Fraction of tokens removed."""
        if self.tokens_in <= 0:
            return 0.0
        return 1.0 - (self.tokens_out / self.tokens_in)

    @property
    def compression_rate(self) -> float:
        """Input/output token ratio (``4x`` style)."""
        return (self.tokens_in / self.tokens_out) if self.tokens_out else float("inf")

    def summary(self) -> str:
        v = self.verification
        return (
            f"{self.tokens_in} -> {self.tokens_out} tok "
            f"({self.ratio * 100:.1f}% removed, {self.compression_rate:.2f}x) | "
            f"integrity {v.integrity * 100:.1f}% | "
            f"critical {v.critical_recall * 100:.1f}% | "
            f"retention {v.retention * 100:.1f}% | "
            f"conf {self.confidence:.2f}"
        )


def iter_kinds(units: Iterable[Unit]) -> dict[str, int]:
    out: dict[str, int] = {}
    for u in units:
        out[u.kind.value] = out.get(u.kind.value, 0) + 1
    return out
