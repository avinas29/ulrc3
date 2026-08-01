"""Phantom Attention: model-free simulation of where an LLM would look.

See :mod:`ulrc3.ir.graph` for the derivation.  This pass assembles the
multigraph, builds the personalisation vector, and solves for the stationary
distribution.

Personalisation mass is placed on:

* **query units** (weight ``query_weight``) -- the task is the attractor;
* **instruction / system units** -- always attended;
* **positional priors** at the head and tail of the context, matching the
  measured U-shaped attention profile ("lost in the middle") -- note that we use
  it to *score*, and separately counteract it when *ordering* (p070).

Failure mode and mitigation: on a context with no query and no structure, PPR
degenerates towards a degree-centrality ranking, which favours long units.  We
therefore normalise attention by ``tokens^0.5`` before it enters salience, so
that the score measures *density* of attention, not total mass.
"""

from __future__ import annotations

import math

from ..ir.graph import EDGE_CONDUCTANCE, SparseGraph, build_lexical_edges, positional_kernel
from ..text.terms import content_terms
from ..types import EdgeKind, Protection
from .base import Pass, PassContext


class AttentionPass(Pass):
    name = "phantom-attention"

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        n = len(cir.units)
        if n == 0:
            return
        if n == 1:
            cir.units[0].attention = 1.0
            return

        if ctx.cfg.ablated("no_attention"):
            for u in cir.units:
                u.attention = 0.5
            return

        g = SparseGraph(n)
        g.extend(cir.edges, symmetric=True)

        concepts = [u.concepts for u in cir.units]
        lex = [] if ctx.cfg.ablated("no_lexical") else build_lexical_edges(
            concepts, knn=ctx.cfg.perf.knn
        )
        for i, j, w in lex:
            g.add(i, j, EDGE_CONDUCTANCE[EdgeKind.REFERS] * w)

        # adjacency chain with exponential decay (positional locality)
        by_doc: dict[str, list[int]] = {}
        for u in cir.units:
            by_doc.setdefault(u.doc_id, []).append(u.uid)
        for uids in by_doc.values():
            for idx, uid in enumerate(uids):
                for d in (1, 2, 3):
                    if idx + d < len(uids):
                        w = EDGE_CONDUCTANCE[EdgeKind.ADJACENT] * positional_kernel(d)
                        g.add(uid, uids[idx + d], w)
                        g.add(uids[idx + d], uid, w)

        p = self._personalisation(ctx)
        r = g.pagerank(
            p,
            alpha=ctx.cfg.perf.ppr_alpha,
            iterations=ctx.cfg.perf.ppr_iterations,
        )
        peak = max(r) or 1.0
        for u in cir.units:
            # density, not mass: divide by sqrt(cost) so long units do not win
            # by being long
            u.attention = (r[u.uid] / peak) / math.sqrt(max(1.0, u.tokens / 24.0))
        top = max((u.attention for u in cir.units), default=1.0) or 1.0
        for u in cir.units:
            u.attention /= top
        ctx.note("graph_nnz", g.nnz)

    # -- personalisation ----------------------------------------------
    def _personalisation(self, ctx: PassContext) -> list[float]:
        cir = ctx.cir
        n = len(cir.units)
        p = [0.0] * n
        q_terms = set(content_terms(cir.query)) if cir.query else set()
        qw = ctx.cfg.query_weight

        docs_len: dict[str, int] = {}
        for u in cir.units:
            docs_len[u.doc_id] = max(docs_len.get(u.doc_id, 0), u.order)

        for u in cir.units:
            base = 0.05
            if u.segment in ("query", "instruction", "system", "tools"):
                base += 2.0 * qw
            if u.protection >= Protection.LOCKED:
                base += 0.8
            elif u.protection >= Protection.ANCHORED:
                base += 0.3
            if q_terms and u.concepts:
                hit = sum(w for t, w in u.concepts.items() if t in q_terms)
                base += qw * hit * 3.0
            # U-shaped positional prior within each document
            span = max(1, docs_len.get(u.doc_id, 1))
            rel = u.order / span
            base += 0.25 * (math.exp(-rel * 4.0) + math.exp(-(1.0 - rel) * 4.0))
            base += ctx.cfg.recency_weight * rel * u.features.get("recency", 0.0)
            p[u.uid] = base
        return p

    def note(self, ctx: PassContext) -> str:
        return f"nnz={ctx.scratch.get('graph_nnz', 0)}"
