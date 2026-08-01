"""Protection lattice: seeding, monotone taint propagation, closure repair.

This module is the formal core of the safety story.

**Lattice.**  ``(Protection, max)`` is a finite join-semilattice with bottom
``DROPPABLE`` and top ``FROZEN``.

**Transfer function.**  For a dependency edge ``u --REQUIRES--> v`` (read: *u is
meaningless without v*), protection flows backwards along the edge with decay:

        prot(v) := prot(v) ⊔ demote(prot(u))

``demote`` strictly decreases the level, so protection cannot circulate forever
in a cycle: after at most ``|Protection|`` traversals of any cycle the inherited
value reaches ``DROPPABLE`` and stops propagating.  Combined with monotonicity
(values only ever increase) the worklist algorithm reaches a fixpoint in
``O(V + E · |Protection|)``.

**Closure invariant.**  Let ``S`` be the retained set.  The engine guarantees

        ∀ u ∈ S, ∀ (u → v) ∈ REQUIRES with hops ≤ H :  v ∈ S  ∨  stub(v) ∈ S

where ``stub(v)`` is a signature-only rendering (for code) or a reference handle
(for prose).  This is what stops the classic failure mode of prompt compressors:
keeping a function call while deleting the definition it depends on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from ..config import ProtectionPolicy
from ..text.lexicon import ASSISTANT_BOILERPLATE, BOILERPLATE_LINE, SMALL_TALK
from ..types import CIR, EdgeKind, ObligationClass, Protection, Unit, UnitKind, demote

#: Segment roles whose units are frozen verbatim.
FROZEN_SEGMENTS = frozenset({"system", "tools", "query", "instruction_hard"})
#: Segment roles whose units are locked (kept with all obligations).
LOCKED_SEGMENTS = frozenset({"instruction", "constraints", "schema"})

#: Unit kinds that must never be paraphrased or reordered.
#: Imports are deliberately *not* here: they are protected by dependency
#: closure (a definition REQUIRES the import that binds the names it uses), so
#: locking them unconditionally would defeat import pruning.
#: JSON nodes are ANCHORED by their pipeline, not LOCKED here: their cheapest
#: rung is the induced *schema*, which already contains every key, so locking
#: them to a rung that carries every value made JSON incompressible (21% ->
#: 74% measured on the fixture set).
STRUCTURAL_KINDS = frozenset({UnitKind.TOOL_SCHEMA, UnitKind.SQL_STMT})

#: Kinds that already carry an aggregate representation of their values.
AGGREGATE_KINDS = frozenset(
    {UnitKind.TABLE_ROW, UnitKind.LOG_TEMPLATE, UnitKind.LOG_RECORD, UnitKind.JSON_NODE}
)

#: Obligation key prefixes that make a prose unit value-bearing.
_VALUE_CLASSES = frozenset({"n", "d", "u", "m", "a", "k", "t", "v"})

#: Units whose contract lives in their signature, not their prose.
CODE_KINDS = frozenset(
    {
        UnitKind.CODE_DEF,
        UnitKind.CODE_STMT,
        UnitKind.CODE_IMPORT,
        UnitKind.CODE_COMMENT,
        UnitKind.CODE_DOCSTRING,
    }
)


def seed(cir: CIR, policy: ProtectionPolicy) -> None:
    """Assign initial protection levels from role, kind and obligation content.

    Complexity O(|U|).  Deliberately *cheap and conservative*: this is a floor,
    not a final answer -- the optimiser may raise it, never lower it below the
    seed.
    """
    for u in cir.units:
        p = Protection.ELASTIC

        if u.segment in FROZEN_SEGMENTS:
            p = Protection.FROZEN
        elif u.segment in LOCKED_SEGMENTS or u.kind in STRUCTURAL_KINDS:
            p = Protection.LOCKED
        elif u.kind in (UnitKind.HEADING, UnitKind.CODE_DEF):
            p = Protection.ANCHORED

        classes = {k.split(":", 1)[0] for k in u.obligations}
        # Deontic prose inside a *code* unit is documentation, not machine
        # contract: locking a 200-token function because its docstring says
        # "must" destroys the ratio.  The contract of code is its signature, so
        # code units cap at ANCHORED and their prose constraints are recovered
        # by the verifier into the FACT block instead.
        constraint_level = Protection.ANCHORED if u.kind in CODE_KINDS else Protection.LOCKED
        if "sc" in classes and policy.enforce_security:
            p = max(p, constraint_level)
        if "cc" in classes and policy.enforce_constraints:
            p = max(p, constraint_level)
        if "nc" in classes and policy.enforce_negations and u.kind not in CODE_KINDS:
            p = max(p, Protection.ANCHORED)

        # NOTE: value-bearing units are deliberately *not* anchored here.
        # An obligation must be preserved *somewhere*, not everywhere -- four
        # RAG chunks stating the same revenue figure need one carrier, not
        # four.  Anchoring each of them made fact-dense corpora incompressible
        # (measured: 5.7% reduction on an 8-chunk RAG set).  Instead these
        # classes are tier-1 *audited* obligations: the coverage objective
        # prefers to keep one carrier, and the verifier repairs any gap.

        # boilerplate / social noise sinks to the bottom regardless of content,
        # unless it carries an enforced obligation (a licence with a URL does)
        text = u.text.strip()
        if p <= Protection.ELASTIC:
            if SMALL_TALK.match(text) and len(text) < 80 or BOILERPLATE_LINE.match(text) and not u.obligations or u.kind is UnitKind.CODE_COMMENT and not u.obligations and len(text) < 200:
                p = Protection.DROPPABLE

        u.protection = Protection(max(int(u.protection), int(p)))

        if ASSISTANT_BOILERPLATE.match(text) and u.protection <= Protection.ELASTIC:
            u.features["boilerplate_lead"] = 1.0


def propagate(cir: CIR, policy: ProtectionPolicy) -> int:
    """Monotone dataflow fixpoint over REQUIRES edges.  Returns #units raised."""
    if policy.closure_horizon <= 0:
        return 0
    out: dict[int, list[tuple[int, float]]] = {}
    for e in cir.edges:
        if e.kind is EdgeKind.REQUIRES:
            out.setdefault(e.src, []).append((e.dst, e.weight))

    if not out:
        return 0

    work: deque[int] = deque(u.uid for u in cir.units if u.protection >= Protection.ANCHORED)
    hops: dict[int, int] = dict.fromkeys(work, 0)
    raised = 0
    guard = 0
    max_steps = 64 * (len(cir.units) + len(cir.edges) + 1)

    while work:
        guard += 1
        if guard > max_steps:  # defensive: cannot happen given monotonicity
            break
        uid = work.popleft()
        u = cir.units[uid]
        h = hops.get(uid, 0)
        if h >= policy.closure_horizon:
            continue
        inherited = demote(u.protection, 1)
        if inherited <= Protection.DROPPABLE:
            continue
        for dst, _w in out.get(uid, ()):  # noqa: B007
            v = cir.units[dst]
            if v.protection < inherited:
                v.protection = inherited
                raised += 1
                hops[dst] = h + 1
                work.append(dst)
            elif dst not in hops:
                hops[dst] = h + 1
                work.append(dst)
    return raised


def requires_closure(
    cir: CIR,
    selected: set[int],
    horizon: int = 3,
    limit: int | None = None,
) -> set[int]:
    """Smallest superset of `selected` closed under REQUIRES within `horizon`.

    Complexity O(V + E).  If `limit` is given, expansion stops once the added
    token cost exceeds it and the remaining dependencies are returned as stubs
    by the caller (they are still *represented*, never silently dropped).
    """
    out: dict[int, list[int]] = {}
    for e in cir.edges:
        if e.kind is EdgeKind.REQUIRES:
            out.setdefault(e.src, []).append(e.dst)
    if not out:
        return set(selected)

    closed = set(selected)
    added_tokens = 0
    frontier = deque((uid, 0) for uid in selected)
    while frontier:
        uid, depth = frontier.popleft()
        if depth >= horizon:
            continue
        for dst in out.get(uid, ()):
            if dst in closed:
                continue
            if limit is not None and added_tokens + cir.units[dst].tokens > limit:
                continue
            closed.add(dst)
            added_tokens += cir.units[dst].tokens
            frontier.append((dst, depth + 1))
    return closed


def violations(cir: CIR, selected: set[int], horizon: int = 3) -> list[tuple[int, int]]:
    """Dependency edges leaving the retained set (for diagnostics/verification)."""
    sel = selected
    bad: list[tuple[int, int]] = []
    for e in cir.edges:
        if e.kind is EdgeKind.REQUIRES and e.src in sel and e.dst not in sel:
            if cir.units[e.dst].meta.get("stubbed"):
                continue
            bad.append((e.src, e.dst))
    return bad


def must_keep(units: Iterable[Unit]) -> set[int]:
    """Units that are non-negotiable regardless of budget."""
    return {u.uid for u in units if u.protection >= Protection.LOCKED}


def hard_classes(policy: ProtectionPolicy) -> set[ObligationClass]:
    """Obligation classes promoted to enforced status by the active policy."""
    m = {
        ObligationClass.NUMBER: policy.enforce_numbers,
        ObligationClass.DATE: policy.enforce_dates,
        ObligationClass.IDENTIFIER: policy.enforce_identifiers,
        ObligationClass.URL: policy.enforce_urls,
        ObligationClass.EMAIL: policy.enforce_urls,
        ObligationClass.PATH: policy.enforce_urls,
        ObligationClass.JSON_KEY: policy.enforce_json_keys,
        ObligationClass.SQL_IDENT: policy.enforce_json_keys,
        ObligationClass.CONSTRAINT: policy.enforce_constraints,
        ObligationClass.NEGATION: policy.enforce_negations,
        ObligationClass.SECURITY: policy.enforce_security,
        ObligationClass.ENTITY: policy.enforce_entities,
        ObligationClass.VERSION: policy.enforce_numbers,
        ObligationClass.TOOL_PARAM: policy.enforce_json_keys,
    }
    return {c for c, on in m.items() if on}
