"""CASCADE -- Constraint-Aware Submodular Cascaded Allocation with Dependency
Enforcement.  The optimiser at the centre of the engine.

## The problem, stated properly

Every unit ``u`` offers a ladder of renderings ``L(u) = {0=drop, 1, ..., m_u}``
with token cost ``c(u,l)`` and concept coverage ``C(u,l)``.  Choose exactly one
level per unit to maximise information coverage subject to a token budget and to
the protection floor:

        maximise    F({(u, l_u)})
        subject to  Σ_u c(u, l_u) ≤ B
                    l_u ≥ minlevel(prot(u))              (protection)
                    S closed under REQUIRES               (dependency, p060)

This is a **Multiple-Choice Knapsack Problem** with a submodular objective --
strictly more expressive than the keep/drop knapsack every existing prompt
compressor solves, because it can buy a cheap faithful *stub* instead of
choosing between a 200-token body and nothing.

## The objective

        F(S) = Σ_{c ∈ concepts} w_c · (1 − e^{−λ · x_c(S)})
               + μ · Σ_u salience(u) · fidelity(u, l_u)

``x_c(S)`` is the accumulated coverage of concept ``c``.  ``F`` is monotone and
submodular (a nonneg. weighted sum of concave functions of modular set
functions).  Obligations enter as pseudo-concepts with high weight, so the
optimiser is *paid* to satisfy the audit rather than being punished afterwards.

## The algorithm

Greedy on the **benefit/cost ratio of a single upgrade step**, with CELF lazy
evaluation.  Each unit's next candidate is "go up one rung".  Because ``F`` is
submodular, gains are non-increasing, so a stale heap entry can only overestimate
-- the standard lazy-greedy correctness argument applies.

Guarantee: for monotone submodular maximisation under a knapsack constraint,
greedy-by-ratio combined with the best single element achieves
``(1 − 1/e)/2 ≈ 0.316`` of the optimum, and the partial-enumeration variant
reaches ``1 − 1/e``.  Measured against a *conservative* LP upper bound
(``bench/lp_bound.py``): mean gap 35.4%, median 32.4% over 156 instances.  The
bound over-estimates on two counts (fractional relaxation and empty-set gain
evaluation), so the true gap to the integral optimum is smaller -- but it is
well above the 1.8% an earlier draft of this docstring asserted without ever
computing it.

Complexity: O(M log M) where M = Σ_u |L(u)| is the number of rungs.  Measured at
~1% of total pipeline runtime (4.6 ms on 10 066 units).

## Budget allocation across documents

Before the greedy runs, the budget is split across documents/segments by
**water-filling**: maximise Σ_i v_i·log(1 + b_i/κ_i) s.t. Σ b_i = B gives the
closed form ``b_i = clamp(v_i/λ − κ_i, floor_i, cap_i)`` with λ found by
bisection.  This is what prevents the classic RAG failure where one long chunk
eats the entire budget: allocation is proportional to *value density*, with a
guaranteed floor per document so nothing is silently zeroed.
"""

from __future__ import annotations

import heapq
import math

from ..text.terms import content_terms
from ..types import Level, Unit, UnitKind
from .base import Pass, PassContext

#: Kinds whose values are represented by an aggregate rather than individually.
AGGREGATE_KINDS = frozenset(
    {UnitKind.LOG_TEMPLATE, UnitKind.LOG_RECORD, UnitKind.TABLE_ROW, UnitKind.JSON_NODE}
)

LAMBDA_SAT = 0.9  #: saturation rate of the concave coverage kernel
MU_SALIENCE = 0.55  #: weight of the modular salience term
OBLIGATION_WEIGHT = 3.5  #: pseudo-concept weight for a hard obligation
QUERY_BOOST = 2.5
#: Obligations that exist only inside aggregate content bid at a fraction of
#: fact weight -- see `_concept_weights`.
AGGREGATE_OBLIGATION_SCALE = 0.12


# --------------------------------------------------------------------------
# Water-filling
# --------------------------------------------------------------------------
def waterfill(
    values: list[float],
    floors: list[int],
    caps: list[int],
    budget: int,
    kappa: list[float] | None = None,
    iterations: int = 60,
) -> list[int]:
    """Concave budget allocation.  Returns integer budgets summing to ``budget``.

    Maximises ``Σ v_i log(1 + b_i/κ_i)`` subject to ``Σ b_i = B`` and box
    constraints.  ``b_i(λ) = clamp(v_i/λ − κ_i, floor_i, cap_i)`` is monotone
    decreasing in λ, so bisection converges linearly in the number of iterations
    (60 gives ~1e-18 relative precision on the multiplier).
    """
    n = len(values)
    if n == 0:
        return []
    if kappa is None:
        kappa = [max(1.0, c / 8.0) for c in caps]
    total_floor = sum(floors)
    if budget <= total_floor:
        return list(floors)
    if budget >= sum(caps):
        return list(caps)

    def alloc(lam: float) -> list[int]:
        out = []
        for v, f, c, k in zip(values, floors, caps, kappa):
            b = (v / lam) - k if lam > 0 else float(c)
            out.append(int(max(f, min(c, b))))
        return out

    lo, hi = 1e-9, max(1.0, max(values) * 1e6)
    for _ in range(iterations):
        mid = math.sqrt(lo * hi)
        s = sum(alloc(mid))
        if s > budget:
            lo = mid
        else:
            hi = mid
    out = alloc(hi)
    # distribute the rounding remainder to the highest-value groups
    rem = budget - sum(out)
    if rem > 0:
        order = sorted(range(n), key=lambda i: -values[i])
        for i in order:
            room = caps[i] - out[i]
            if room <= 0:
                continue
            take = min(room, rem)
            out[i] += take
            rem -= take
            if rem <= 0:
                break
    return out


# --------------------------------------------------------------------------
# Coverage machinery
# --------------------------------------------------------------------------
class Coverage:
    """Accumulator for the concave coverage objective."""

    __slots__ = ("weights", "x", "lam")

    def __init__(self, weights: dict[str, float], lam: float = LAMBDA_SAT) -> None:
        self.weights = weights
        self.x: dict[str, float] = {}
        self.lam = lam

    def gain(self, contrib: dict[str, float]) -> float:
        g = 0.0
        for c, d in contrib.items():
            w = self.weights.get(c)
            if not w or d <= 0:
                continue
            x = self.x.get(c, 0.0)
            g += w * (math.exp(-self.lam * x) - math.exp(-self.lam * (x + d)))
        return g

    def add(self, contrib: dict[str, float]) -> None:
        for c, d in contrib.items():
            if d > 0:
                self.x[c] = self.x.get(c, 0.0) + d

    def remove(self, contrib: dict[str, float]) -> None:
        for c, d in contrib.items():
            if d > 0:
                self.x[c] = max(0.0, self.x.get(c, 0.0) - d)


def level_contribution(u: Unit, lv: Level, stats) -> dict[str, float]:
    """Concept contribution of one rung, memoised on the Level."""
    if lv.concepts is not None:
        return lv.concepts
    if lv.name == "drop":
        lv.concepts = {}
        return lv.concepts
    if lv.name == "full" or lv.text == u.text:
        base = dict(u.concepts)
    else:
        base = stats.weights(content_terms(lv.text)) if stats is not None else dict(u.concepts)
        if not base:
            base = {k: v * lv.fidelity for k, v in u.concepts.items()}
    obs = lv.obligations if lv.obligations is not None else u.obligations
    for key in obs:
        base[f"@ob:{key}"] = 1.0
    lv.concepts = base
    return base


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------
class CascadeSelectPass(Pass):
    name = "cascade-select"

    def __init__(self) -> None:
        self._spent = 0
        self._upgrades = 0
        self._groups = 0
        self._no_coverage = False

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        self._no_coverage = ctx.cfg.ablated("no_coverage")
        if not cir.units:
            return
        stats = ctx.scratch.get("term_stats")
        weights = self._concept_weights(ctx, stats)

        # 1. floor everything at its protection minimum
        for u in cir.units:
            if not u.levels:
                continue
            u.set_level(u.min_level())
        base_cost = cir.output_tokens()
        budget = ctx.budget
        ctx.note("floor_tokens", base_cost)

        cov = Coverage(weights)
        for u in cir.units:
            if u.level > 0:
                cov.add(level_contribution(u, u.levels[u.level], stats))

        remaining = budget - base_cost
        if remaining <= 0:
            ctx.note("budget_exhausted_by_protection", True)
            self._spent = base_cost
            return

        # 2. split the remaining budget across documents by water-filling
        groups: dict[str, list[Unit]] = {}
        for u in cir.units:
            if u.levels and u.level < u.max_level():
                groups.setdefault(u.doc_id, []).append(u)
        if not groups:
            return
        keys = sorted(groups)
        values, floors, caps = [], [], []
        for k in keys:
            us = groups[k]
            doc = cir.docs.get(k)
            w = doc.weight if doc else 1.0
            values.append(max(1e-6, sum(u.salience for u in us) * w))
            floors.append(0)
            caps.append(sum(u.levels[-1].tokens - u.levels[u.level].tokens for u in us))
        alloc = waterfill(values, floors, caps, remaining)
        self._groups = len(keys)

        # 3. lazy-greedy upgrade inside each group
        spent_total = 0
        for k, b in zip(keys, alloc):
            spent_total += self._greedy(groups[k], cov, stats, b, ctx.cfg.effective_marginal_floor)

        # 4. spillover: whatever is left goes to a global round
        leftover = remaining - spent_total
        if leftover > 8:
            allu = [u for us in groups.values() for u in us if u.level < u.max_level()]
            spent_total += self._greedy(allu, cov, stats, leftover, ctx.cfg.effective_marginal_floor)

        self._spent = base_cost + spent_total
        ctx.note("selected_tokens", self._spent)

    # -- greedy --------------------------------------------------------
    def _greedy(self, units: list[Unit], cov: Coverage, stats, budget: int, floor: float = 0.0) -> int:
        """Lazy-greedy upgrades.

        ``floor`` implements the **marginal-utility stop**: the budget is a
        ceiling, not a quota.  Once the benefit/cost ratio of the best remaining
        upgrade falls below ``floor x`` the best ratio seen, we stop and return
        *under* budget rather than padding the context with material whose
        marginal value has collapsed.  On filler-heavy corpora this alone buys
        several points of compression at identical answerability -- and it is
        the behaviour a user actually wants from ``target_ratio``: "at most this
        many tokens", not "exactly this many".
        """
        if budget <= 0 or not units:
            return 0
        heap: list[tuple[float, int, int, int]] = []
        counter = 0
        best_ratio = 0.0
        # Greedy-by-ratio alone has a well-known pathology: a single item whose
        # cost exceeds the whole budget is skipped, and the budget then fills
        # with cheap low-value items.  The textbook remedy -- and the reason the
        # (1-1/e)/2 bound holds at all -- is to also consider the best single
        # element and take whichever is better.  Without it, a 28-token sentence
        # carrying the only URL in the document lost to four repetitions of
        # "Unrelated narration." at a 20-token budget.
        singleton: tuple[float, int, int, int] | None = None  # (gain, cost, uid, level)
        for u in units:
            item = self._candidate(u, cov, stats)
            if item is None:
                continue
            ratio, cost = item
            counter += 1
            heapq.heappush(heap, (-ratio, counter, u.uid, u.level))
        by_uid = {u.uid: u for u in units}
        spent = 0
        while heap and spent < budget:
            neg, _cnt, uid, at_level = heapq.heappop(heap)
            u = by_uid[uid]
            if u.level != at_level:
                continue  # stale entry: the unit moved
            item = self._candidate(u, cov, stats)
            if item is None:
                continue
            ratio, cost = item
            if cost > budget - spent:
                gain = ratio * cost
                if singleton is None or gain > singleton[0]:
                    if cost <= max(budget * 2, budget + 64):
                        singleton = (gain, cost, uid, u.level + 1)
                continue
            # CELF: if the recomputed ratio dropped materially, requeue
            if ratio < -neg * 0.999:
                counter += 1
                heapq.heappush(heap, (-ratio, counter, uid, u.level))
                continue
            best_ratio = max(best_ratio, ratio)
            if floor > 0.0 and best_ratio > 0.0 and ratio < floor * best_ratio:
                break  # marginal utility collapsed: stop under budget
            nxt = u.level + 1
            cov.remove(level_contribution(u, u.levels[u.level], stats))
            u.set_level(nxt)
            cov.add(level_contribution(u, u.levels[nxt], stats))
            spent += cost
            self._upgrades += 1
            if u.level < u.max_level():
                item2 = self._candidate(u, cov, stats)
                if item2 is not None:
                    counter += 1
                    heapq.heappush(heap, (-item2[0], counter, uid, u.level))

        # greedy vs. best single element
        if singleton is not None:
            gain, cost, uid, level = singleton
            achieved = sum(
                u.salience * (u.levels[u.level].fidelity if u.levels else 0.0)
                for u in units if u.level > 0
            ) * MU_SALIENCE + cov.gain({})
            if gain > max(achieved, 0.0):
                u = by_uid[uid]
                for other in units:
                    if other.uid != uid and other.level > other.min_level():
                        cov.remove(level_contribution(other, other.levels[other.level], stats))
                        other.set_level(other.min_level())
                        cov.add(level_contribution(other, other.levels[other.level], stats))
                cov.remove(level_contribution(u, u.levels[u.level], stats))
                u.set_level(level)
                cov.add(level_contribution(u, u.levels[u.level], stats))
                self._upgrades += 1
                return cost
        return spent

    def _candidate(self, u: Unit, cov: Coverage, stats) -> tuple[float, int] | None:
        if not u.levels or u.level >= u.max_level():
            return None
        cur = u.levels[u.level]
        nxt = u.levels[u.level + 1]
        cost = nxt.tokens - cur.tokens
        if cost <= 0:
            cost = 1
        if self._no_coverage:
            # Ablation: drive selection purely by the modular salience term, so
            # the ranking signal (attention, query relevance, information
            # density) is measured *without* the submodular coverage objective
            # that otherwise dominates it.
            gain = u.salience * max(0.05, nxt.fidelity - cur.fidelity)
            return (gain / cost, cost) if gain > 0 else None

        cur_c = level_contribution(u, cur, stats)
        nxt_c = level_contribution(u, nxt, stats)
        delta = {c: nxt_c.get(c, 0.0) - cur_c.get(c, 0.0) for c in nxt_c}
        gain = cov.gain({c: d for c, d in delta.items() if d > 0})
        gain += MU_SALIENCE * u.salience * max(0.0, nxt.fidelity - cur.fidelity)
        if gain <= 0:
            return None
        return gain / cost, cost

    # -- weights -------------------------------------------------------
    def _concept_weights(self, ctx: PassContext, stats) -> dict[str, float]:
        cir = ctx.cir
        q_terms = set(content_terms(cir.query)) if cir.query else set()
        w: dict[str, float] = {}
        for u in cir.units:
            for c, _val in u.concepts.items():
                if c not in w:
                    w[c] = stats.idf(c) if stats is not None else 1.0
        for t in q_terms:
            if t in w:
                w[t] *= QUERY_BOOST * ctx.cfg.query_weight

        # Obligations owned *only* by aggregate-kind units (log lines, table
        # rows, JSON array elements) do not bid at fact weight.  Each of 200
        # near-identical log lines carries its own `dur=17ms`, and at weight 3.5
        # those become 200 high-value pseudo-concepts that buy the verbatim rung
        # for a template whose whole point is that the lines are interchangeable.
        # This is the same carve-out the guarantees document already states:
        # aggregate content is represented by its aggregate.
        units = cir.units
        for key, ob in cir.obligations.items():
            if not ob.hard:
                continue
            owners = [units[uid] for uid in ob.units if uid < len(units)]
            aggregate_only = bool(owners) and all(u.kind in AGGREGATE_KINDS for u in owners)
            scale = AGGREGATE_OBLIGATION_SCALE if aggregate_only else 1.0
            w[f"@ob:{key}"] = OBLIGATION_WEIGHT * ob.weight * scale
        return w

    def note(self, ctx: PassContext) -> str:
        return f"{self._upgrades} upgrades over {self._groups} groups -> {self._spent} tok"
