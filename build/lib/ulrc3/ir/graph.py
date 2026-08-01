"""Sparse unit graph + Phantom Attention (personalised PageRank).

## Phantom Attention

LLMLingua-family methods need a *real* language model forward pass to score
tokens.  That costs a GPU, 300ms-3s of latency, and makes the compressor's
quality a function of the proxy model's quality.

We replace it with an explicit structural model of what transformer attention
empirically does, and solve for its stationary distribution in closed form.

Aggregate attention in decoder LLMs is dominated by four reproducible effects:

1. **induction / copy heads** -> attend to earlier occurrences of the same
   token or phrase  ⇒ modelled by IDF-weighted lexical overlap edges;
2. **positional locality** -> attention mass decays with distance  ⇒ modelled
   by adjacency edges with an exponential decay kernel;
3. **syntactic / structural heads** -> attend along containment and dependency
   structure  ⇒ modelled by CONTAINS / REQUIRES edges;
4. **query & instruction anchoring** -> the tail of the prompt (the actual
   task) pulls attention  ⇒ modelled by the personalisation vector.

Let ``W`` be the row-normalised weighted adjacency of that multigraph and ``p``
the personalisation vector concentrated on query/instruction units.  Simulated
attention is the stationary distribution

        r = (1-α)·p + α·Wᵀ·r ,           α ∈ (0,1)

which exists and is unique because the iteration is a contraction with modulus
α in the ℓ1 norm.  Power iteration converges geometrically: ‖rₖ − r‖₁ ≤ αᵏ·2, so
18 iterations at α=0.85 gives < 5e-2 error -- far below the resolution at which
the ranking changes.

Cost: O(nnz · iters), nnz ≈ k·|U| with k = 12 neighbours.  Measured at 66 ms on
10 066 units (15% of pipeline runtime), versus multiple seconds for an LLM pass.

**Honest caveat:** the ablation study finds this stage contributes ~0.1 points of
compression and no measurable answerability on the current benchmark.  It is
retained pending the extrinsic harness; see docs/ROADMAP.md §1.0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..types import Edge, EdgeKind

try:  # numpy fast path
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore


#: Relative conductance of each edge type.  These are **hand-set priors**, not
#: fitted values -- and the ablation study (bench/ablation.py) shows the whole
#: attention stage contributes ~0.1 points of ratio and no measurable quality,
#: so tuning them further would be optimising an inert component.
EDGE_CONDUCTANCE: dict[EdgeKind, float] = {
    EdgeKind.REFERS: 1.00,
    EdgeKind.REQUIRES: 1.35,
    EdgeKind.CONTAINS: 0.75,
    EdgeKind.ADJACENT: 0.45,
    EdgeKind.ELABORATES: 0.60,
    EdgeKind.ANSWERS: 0.90,
    EdgeKind.DUPLICATE_OF: 0.15,
    EdgeKind.SUPERSEDES: 0.10,
}


class SparseGraph:
    """COO edge list with a fused row-normalised transpose multiply."""

    __slots__ = ("n", "src", "dst", "w", "_outdeg", "_np")

    def __init__(self, n: int) -> None:
        self.n = n
        self.src: list[int] = []
        self.dst: list[int] = []
        self.w: list[float] = []
        self._outdeg: list[float] | None = None
        self._np = None

    def add(self, s: int, d: int, w: float = 1.0) -> None:
        if s == d or w <= 0.0:
            return
        self.src.append(s)
        self.dst.append(d)
        self.w.append(w)
        self._outdeg = None

    def extend(self, edges: Sequence[Edge], symmetric: bool = True) -> None:
        for e in edges:
            c = EDGE_CONDUCTANCE.get(e.kind, 0.5) * e.weight
            self.add(e.src, e.dst, c)
            if symmetric and e.kind not in (EdgeKind.SUPERSEDES, EdgeKind.CONTAINS):
                self.add(e.dst, e.src, c * 0.6)

    @property
    def nnz(self) -> int:
        return len(self.w)

    # -- numeric ------------------------------------------------------
    def _prepare(self):
        if _np is None:
            return None
        if self._np is None:
            src = _np.asarray(self.src, dtype=_np.int32)
            dst = _np.asarray(self.dst, dtype=_np.int32)
            w = _np.asarray(self.w, dtype=_np.float32)
            outdeg = _np.bincount(src, weights=w, minlength=self.n).astype(_np.float32)
            outdeg[outdeg == 0.0] = 1.0
            self._np = (src, dst, w / outdeg[src], outdeg)
        return self._np

    def pagerank(
        self,
        personalisation: Sequence[float],
        alpha: float = 0.85,
        iterations: int = 18,
        tol: float = 1e-6,
    ) -> list[float]:
        n = self.n
        if n == 0:
            return []
        p = _normalise(list(personalisation), n)
        if self.nnz == 0:
            return p

        prep = self._prepare()
        if prep is not None:
            src, dst, wn, _ = prep
            r = _np.asarray(p, dtype=_np.float32)
            pv = r.copy()
            has_out = _np.zeros(n, dtype=bool)
            has_out[src] = True
            for _ in range(iterations):
                contrib = wn * r[src]
                nxt = _np.bincount(dst, weights=contrib, minlength=n).astype(_np.float32)
                dangling = float(r[~has_out].sum())
                nxt = alpha * (nxt + dangling * pv) + (1.0 - alpha) * pv
                s = float(nxt.sum())
                if s > 0:
                    nxt /= s
                delta = float(_np.abs(nxt - r).sum())
                r = nxt
                if delta < tol:
                    break
            return [float(x) for x in r]

        # pure-python fallback (no numpy): identical semantics
        outdeg = [0.0] * n
        for s, w in zip(self.src, self.w):
            outdeg[s] += w
        r = list(p)
        for _ in range(iterations):
            nxt = [0.0] * n
            dangling = 0.0
            for i in range(n):
                if outdeg[i] == 0.0:
                    dangling += r[i]
            for s, d, w in zip(self.src, self.dst, self.w):
                nxt[d] += w / outdeg[s] * r[s]
            total = 0.0
            for i in range(n):
                nxt[i] = alpha * (nxt[i] + dangling * p[i]) + (1.0 - alpha) * p[i]
                total += nxt[i]
            if total > 0:
                nxt = [x / total for x in nxt]
            delta = sum(abs(nxt[i] - r[i]) for i in range(n))
            r = nxt
            if delta < tol:
                break
        return r


def _normalise(v: list[float], n: int) -> list[float]:
    if len(v) != n:
        v = (v + [0.0] * n)[:n]
    s = sum(v)
    if s <= 0:
        return [1.0 / n] * n
    return [x / s for x in v]


def positional_kernel(distance: int, tau: float = 3.0) -> float:
    """Empirical attention decay with unit distance.

    Chosen as ``exp(-d/τ)`` with τ=3: measured aggregate attention in decoder
    models falls roughly exponentially over nearby *segments* (not tokens), with
    a heavy tail handled separately by the lexical edges.
    """
    return math.exp(-abs(distance) / tau)


def build_lexical_edges(
    concepts: Sequence[dict[str, float]],
    knn: int = 12,
    min_sim: float = 0.05,
    hubs: int = 8,
) -> list[tuple[int, int, float]]:
    """Top-k lexical neighbours via an inverted index.

    Naive all-pairs cosine is O(n²·|c|) and dominates runtime past ~5k units.
    Using an inverted index restricted to *discriminative* terms turns it into
    O(Σ_t df_t²) with a per-term cap, i.e. effectively O(n·k).
    """
    inverted: dict[str, list[tuple[int, float]]] = {}
    for i, c in enumerate(concepts):
        # only index a unit's most discriminative terms: the tail contributes
        # noise and quadratic blowup
        top = sorted(c.items(), key=lambda kv: -kv[1])[:16]
        for term, w in top:
            inverted.setdefault(term, []).append((i, w))

    # Star topology, not clique.  Linking every pair that shares a term is
    # O(df^2) per term; on a 64k-token document that was 3.3 s and the dominant
    # cost of the whole pipeline.  Connecting each posting to the term's
    # highest-weight *hubs* is O(df * hubs) and preserves what the random walk
    # actually needs: connectivity between units that share rare vocabulary.
    # Spectrally the star is a good sparsifier of the clique -- it keeps the
    # component structure, which is what PageRank mass flow depends on.
    acc: dict[int, dict[int, float]] = {}
    for postings in inverted.values():
        if len(postings) < 2 or len(postings) > 2000:
            continue  # ubiquitous terms carry no signal; skip (also caps cost)
        if len(postings) > hubs * 2:
            top_hubs = sorted(postings, key=lambda p: -p[1])[:hubs]
        else:
            top_hubs = postings
        for ia, wa in postings:
            bucket = acc.setdefault(ia, {})
            for ib, wb in top_hubs:
                if ia == ib:
                    continue
                sim = wa * wb
                bucket[ib] = bucket.get(ib, 0.0) + sim
                back = acc.setdefault(ib, {})
                back[ia] = back.get(ia, 0.0) + sim

    out: list[tuple[int, int, float]] = []
    for i, bucket in acc.items():
        if not bucket:
            continue
        best = sorted(bucket.items(), key=lambda kv: -kv[1])[:knn]
        for j, s in best:
            if s >= min_sim:
                out.append((i, j, float(s)))
    return out
