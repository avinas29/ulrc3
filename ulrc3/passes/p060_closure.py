"""Dependency-closure enforcement.

The optimiser maximises coverage; it has no idea that ``charge_customer()`` is
useless without the ``Invoice`` class, that a table row is meaningless without
its header, or that a clause referencing "Section 4.2" needs Section 4.2.

This pass restores the closure invariant

        ∀ u selected, ∀ (u → v) ∈ REQUIRES within horizon H : v selected

by promoting every violating dependency to its **cheapest non-empty rung** --
the signature stub for code, the heading for prose, the header row for tables.
Cost is therefore bounded by Σ c(v, 1) over violations, typically 2-4% of the
budget, and the result is a context in which every retained statement is
interpretable.

Termination: promotions are monotone (levels only increase) over a finite set,
so at most |U| promotions occur; the loop is bounded by ``H`` BFS layers.
"""

from __future__ import annotations

from ..ir.protection import requires_closure, violations
from ..types import EdgeKind, Protection, UnitKind
from .base import Pass, PassContext


class ClosurePass(Pass):
    name = "closure"

    def __init__(self) -> None:
        self._promoted = 0
        self._cost = 0
        self._orphans = 0

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        horizon = ctx.cfg.protection.closure_horizon
        self._promoted = 0
        self._cost = 0
        self._orphans = 0

        if ctx.cfg.ablated("no_closure"):
            return

        selected = {u.uid for u in cir.units if u.level > 0}
        closed = requires_closure(cir, selected, horizon=horizon)
        for uid in closed - selected:
            u = cir.units[uid]
            if not u.levels or len(u.levels) < 2:
                continue
            u.set_level(1)
            u.meta["stubbed"] = True
            self._promoted += 1
            self._cost += u.levels[1].tokens

        # duplicates must not outlive their canonical representative
        canon_of: dict[int, int] = {}
        for e in cir.edges:
            if e.kind is EdgeKind.DUPLICATE_OF:
                canon_of[e.src] = e.dst
        for src, dst in canon_of.items():
            u, c = cir.units[src], cir.units[dst]
            if u.level > 0 and c.level == 0 and len(c.levels) > 1:
                c.set_level(1)
                c.meta["stubbed"] = True
                self._promoted += 1
                self._cost += c.levels[1].tokens

        self._drop_orphan_headings(cir)

        remaining = violations(cir, {u.uid for u in cir.units if u.level > 0}, horizon)
        ctx.note("closure_violations", len(remaining))

    # -- orphans -------------------------------------------------------
    def _drop_orphan_headings(self, cir) -> None:
        """Remove headings whose entire section was dropped.

        Closure runs *downwards* (a kept sentence keeps its heading); this is
        the upward direction.  A heading with no surviving descendant is pure
        framing cost -- "## Overview" followed by nothing tells the model less
        than nothing, because it implies content that is not there.
        """
        children: dict[int, list[int]] = {}
        for u in cir.units:
            if u.parent is not None:
                children.setdefault(u.parent, []).append(u.uid)

        def has_live_descendant(uid: int, depth: int = 0) -> bool:
            if depth > 8:
                return False
            for c in children.get(uid, ()):  # noqa: B007
                cu = cir.units[c]
                if cu.level > 0 and cu.kind is not UnitKind.HEADING:
                    return True
                if has_live_descendant(c, depth + 1):
                    return True
            return False

        for u in cir.units:
            if (
                u.kind is UnitKind.HEADING
                and u.level > 0
                and u.protection < Protection.LOCKED
                and not u.obligations
                and not has_live_descendant(u.uid)
            ):
                u.set_level(0)
                self._orphans += 1

    def note(self, ctx: PassContext) -> str:
        return (
            f"+{self._promoted} stubs (+{self._cost} tok), "
            f"-{self._orphans} orphan headings, "
            f"{ctx.scratch.get('closure_violations', 0)} residual"
        )
