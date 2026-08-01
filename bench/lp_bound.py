"""Optimality gap of the CASCADE greedy against an LP upper bound.

Why this file exists: an earlier draft of the documentation asserted a "1.8%
gap to the LP bound" that had never been computed. Either measure it or delete
the claim — this measures it.

**The bound.** For monotone submodular `F`, marginal gains are non-increasing,
so evaluating every candidate upgrade's gain *against the empty coverage* gives
an over-estimate of its true marginal contribution. Sorting those over-estimates
by benefit/cost and filling the budget fractionally therefore yields a valid
upper bound on the optimum of the multiple-choice knapsack:

    U(B) = max Σ y_{u,ℓ} · ΔF₀(u,ℓ)    s.t. Σ y·Δc ≤ B,  Σ_ℓ y_{u,ℓ} ≤ 1,  y ≥ 0

solved exactly by ratio sorting with one fractional item. The reported gap is
`1 − F(greedy) / U(B)`, which is *conservative*: the true gap to the integral
optimum is smaller, because U over-estimates on two counts (fractional relaxation
and empty-set gains).

Usage::

    python -m bench.lp_bound            # all suites, quick
    python -m bench.lp_bound --full
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.datasets import SUITES, load  # noqa: E402
from bench.run_bench import _request  # noqa: E402
from ulrc3 import Config, Mode  # noqa: E402
from ulrc3.ir.builder import build  # noqa: E402
from ulrc3.ir.obligations import ObligationExtractor  # noqa: E402
from ulrc3.passes.base import PassContext, PassManager  # noqa: E402
from ulrc3.passes.p010_levels import LevelPass  # noqa: E402
from ulrc3.passes.p020_dedup import DedupPass  # noqa: E402
from ulrc3.passes.p030_attention import AttentionPass  # noqa: E402
from ulrc3.passes.p040_salience import SaliencePass  # noqa: E402
from ulrc3.passes.p050_select import (  # noqa: E402
    MU_SALIENCE,
    CascadeSelectPass,
    Coverage,
    level_contribution,
)
from ulrc3.tokenization import get_tokenizer  # noqa: E402


def _objective(cir, weights, stats) -> float:
    """F(assignment) for the current levels."""
    cov = Coverage(weights)
    modular = 0.0
    for u in cir.units:
        if u.level <= 0 or not u.levels:
            continue
        lv = u.levels[u.level]
        cov.add(level_contribution(u, lv, stats))
        modular += MU_SALIENCE * u.salience * lv.fidelity
    total = 0.0
    for c, x in cov.x.items():
        w = weights.get(c, 0.0)
        if w:
            total += w * (1.0 - pow(2.718281828459045, -cov.lam * x))
    return total + modular


def _lp_upper_bound(cir, weights, stats, budget: int, floor_levels: dict[int, int]) -> float:
    """Fractional relaxation with empty-set gains — a valid over-estimate."""
    base_cov = Coverage(weights)
    base = 0.0
    for u in cir.units:
        k = floor_levels.get(u.uid, 0)
        if k <= 0 or not u.levels or k >= len(u.levels):
            continue
        lv = u.levels[k]
        base_cov.add(level_contribution(u, lv, stats))
        base += MU_SALIENCE * u.salience * lv.fidelity
    for c, x in base_cov.x.items():
        w = weights.get(c, 0.0)
        if w:
            base += w * (1.0 - pow(2.718281828459045, -base_cov.lam * x))

    empty = Coverage(weights)
    cands: list[tuple[float, float, float]] = []  # (ratio, gain, cost)
    for u in cir.units:
        if not u.levels:
            continue
        k = floor_levels.get(u.uid, 0)
        if k >= len(u.levels):
            continue
        for idx in range(k + 1, len(u.levels)):
            lv = u.levels[idx]
            cost = lv.tokens - u.levels[k].tokens
            if cost <= 0:
                continue
            gain = empty.gain(level_contribution(u, lv, stats))
            gain += MU_SALIENCE * u.salience * max(0.0, lv.fidelity - u.levels[k].fidelity)
            if gain > 0:
                cands.append((gain / cost, gain, float(cost)))

    cands.sort(key=lambda t: -t[0])
    remaining = float(max(0, budget))
    extra = 0.0
    for _ratio, gain, cost in cands:
        if remaining <= 0:
            break
        take = min(1.0, remaining / cost)
        extra += gain * take
        remaining -= cost * take
    return base + extra


def measure(quick: bool = True, mode: str = "balanced") -> dict[str, float]:
    cfg = Config(mode=Mode(mode))
    tok = get_tokenizer("auto")
    gaps: list[float] = []

    instances = []
    for name in SUITES:
        instances.extend(load(name, quick))

    for inst in instances:
        req = _request(inst)
        source = req.source_text()
        if not source.strip():
            continue
        tokens_in = tok.count(source)
        cir = build(req, cfg, tok)
        if not cir.units:
            continue
        budget = max(16, int(tokens_in * (1.0 - cfg.effective_target)))

        ctx = PassContext(cir=cir, cfg=cfg, tok=tok, budget=budget)
        ctx.scratch["tokens_in"] = tokens_in
        ctx.scratch["source_text"] = source
        ctx.scratch["term_stats"] = cir.meta.get("stats")
        ctx.scratch["ex_prose"] = ObligationExtractor(
            enforce_entities=cfg.protection.enforce_entities,
            enforce_identifiers=cfg.protection.enforce_identifiers,
        )
        PassManager([LevelPass(), DedupPass(), AttentionPass(), SaliencePass()]).run(ctx)

        sel = CascadeSelectPass()
        stats = ctx.scratch.get("term_stats")
        weights = sel._concept_weights(ctx, stats)
        # min_level() can exceed the ladder when a unit has no rungs at all
        floor_levels = {
            u.uid: (min(u.min_level(), len(u.levels) - 1) if u.levels else 0)
            for u in cir.units
        }
        base_cost = sum(
            (u.levels[floor_levels[u.uid]].tokens if u.levels else 0) for u in cir.units
        )
        sel.run(ctx)

        achieved = _objective(cir, weights, stats)
        bound = _lp_upper_bound(cir, weights, stats, max(0, budget - base_cost), floor_levels)
        if bound > 0:
            gaps.append(max(0.0, 1.0 - achieved / bound))

    if not gaps:
        return {}
    gaps.sort()
    return {
        "n": len(gaps),
        "mean_gap": statistics.fmean(gaps),
        "median_gap": statistics.median(gaps),
        "p90_gap": gaps[int(0.9 * (len(gaps) - 1))],
        "max_gap": gaps[-1],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--mode", default="balanced")
    p.add_argument("--out", default="bench/results/lp_bound.json")
    a = p.parse_args()

    res = measure(quick=not a.full, mode=a.mode)
    if not res:
        print("no instances measured", file=sys.stderr)
        return 2
    print(f"instances      {int(res['n'])}")
    print(f"mean gap       {res['mean_gap'] * 100:.2f}%")
    print(f"median gap     {res['median_gap'] * 100:.2f}%")
    print(f"p90 gap        {res['p90_gap'] * 100:.2f}%")
    print(f"max gap        {res['max_gap'] * 100:.2f}%")
    print("\n(conservative: the LP bound over-estimates on both the fractional")
    print(" relaxation and the empty-set gain evaluation, so the true gap to the")
    print(" integral optimum is smaller than reported.)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
