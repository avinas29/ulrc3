"""Context ordering: lost-in-the-middle mitigation by **edge loading**.

Liu et al. (2023) showed decoder LLMs recover information near the *start* and
*end* of a long context far better than from the middle -- accuracy can drop by
20+ points for mid-context evidence.  Compression makes this worse, not better:
after 80% reduction the surviving evidence is dense everywhere, so *every*
position matters.

Mitigation: sort retained groups by salience and lay them out alternately at the
head and the tail, so the least important material lands in the middle, which is
exactly the region the model reads worst.

        ranked:  g1 g2 g3 g4 g5 g6 g7
        layout:  g1 g3 g5 g7 g6 g4 g2
                 └── head ──┘ └─ tail ─┘

Crucially this is applied at **group** granularity (document, or heading
section), never inside a group, and never at all for order-sensitive content
(code, logs, conversations, legal clauses, procedures), where sequence *is*
semantics.  A ``[order=...]`` marker in the header tells the model what it is
looking at, and every unit keeps its provenance handle so citations resolve.
"""

from __future__ import annotations

from ..types import CIR, Unit
from .base import Pass, PassContext


class OrderPass(Pass):
    name = "order"

    def __init__(self, order_sensitive: bool = False) -> None:
        self.order_sensitive = order_sensitive
        self._mode = "original"

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        mode = ctx.cfg.render.order
        if ctx.cfg.ablated("no_order"):
            mode = "original"
        elif mode == "auto":
            mode = "original" if self.order_sensitive else "salience"
        self._mode = mode
        if mode == "original":
            ctx.note("order_mode", "original")
            for u in cir.units:
                u.meta["render_rank"] = (0, u.order)
            return

        groups = self._groups(cir)
        ranked = sorted(
            groups.items(),
            key=lambda kv: -_group_value(cir, kv[1]),
        )
        head: list[tuple[int, list[int]]] = []
        tail: list[tuple[int, list[int]]] = []
        for i, (_key, uids) in enumerate(ranked):
            (head if i % 2 == 0 else tail).append((i, uids))
        layout = [uids for _i, uids in head] + [uids for _i, uids in reversed(tail)]
        for rank, uids in enumerate(layout):
            for uid in uids:
                cir.units[uid].meta["render_rank"] = (rank, cir.units[uid].order)
        ctx.note("order_mode", "salience")

    def _groups(self, cir: CIR) -> dict[tuple[str, int], list[int]]:
        out: dict[tuple[str, int], list[int]] = {}
        for u in cir.units:
            if u.level <= 0:
                continue
            key = (u.doc_id, _section_of(cir, u))
            out.setdefault(key, []).append(u.uid)
        for uids in out.values():
            uids.sort(key=lambda uid: cir.units[uid].order)
        return out

    def note(self, ctx: PassContext) -> str:
        return f"order={self._mode}"


def _section_of(cir: CIR, u: Unit, max_depth: int = 12) -> int:
    """Top-most heading ancestor (the reorderable block)."""
    cur: int | None = u.uid
    last = u.uid
    for _ in range(max_depth):
        node = cir.units[cur] if cur is not None else None
        if node is None or node.parent is None:
            break
        last = node.parent
        cur = node.parent
    return last


def _group_value(cir: CIR, uids: list[int]) -> float:
    tot_tok = sum(cir.units[u].cost for u in uids) or 1
    return sum(cir.units[u].salience * cir.units[u].cost for u in uids) / tot_tok
