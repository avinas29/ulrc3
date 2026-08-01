"""Compression Confidence Estimator + adaptive control signal.

A compressor that cannot say *when it is unsure* cannot be deployed
autonomously.  We estimate the probability that a downstream model answers the
same on the compressed context as on the original, from features that are all
available without calling any model:

    x1  achieved compression ratio
    x2  integrity / critical / retention recall (post-repair)
    x3  fraction of *coverage mass* retained  (Σ w_c(1-e^{-λx_c}) ratio)
    x4  residual dependency violations per retained unit
    x5  fraction of retained tokens at reduced fidelity
    x6  type-detection entropy (ambiguous input -> less certain pipeline)
    x7  query coverage: fraction of query terms present in the output
    x8  provenance violations (hard zero in a correct run)

        p = σ(w·x + b)

The weights are **hand-set priors**, not fitted: shipping a "fitted" estimator
would require ground-truth downstream-correctness labels, which need a model in
the loop and do not exist in this repository.  They live in
``ulrc3/confidence.json`` so they are inspectable and replaceable rather than
hidden constants, and the ordering they induce (more compression, less
confidence) is what the adaptive loop actually depends on.  Calibrating the
absolute probabilities is an open item -- see docs/ROADMAP.md §1.2.

The estimate drives the **adaptive loop**: if ``p < min_confidence`` the engine
re-runs with a larger budget (backing off the target ratio by 25% each round,
at most three rounds).  That is a closed-loop controller, and it is why the
system can be given an aggressive target safely -- it self-corrects instead of
silently degrading.
"""

from __future__ import annotations

import json
import math
import os

from ..text.terms import content_terms
from .base import Pass, PassContext

_DEFAULT_WEIGHTS = {
    "bias": -1.15,
    "ratio": -1.55,
    "integrity": 2.10,
    "retention": 1.45,
    "critical": 1.60,
    "coverage_kept": 2.35,
    "violation_rate": -2.10,
    "reduced_fidelity": -0.85,
    "type_entropy": -0.55,
    "query_coverage": 1.35,
    "provenance": -4.00,
}

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "confidence.json")


def load_weights() -> dict[str, float]:
    try:
        with open(_WEIGHTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "bias" in data:
            return {k: float(v) for k, v in data.items()}
    except Exception:
        pass
    return dict(_DEFAULT_WEIGHTS)


class ConfidencePass(Pass):
    name = "confidence"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.w = weights or load_weights()
        self._p = 1.0
        self._feats: dict[str, float] = {}

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        v = ctx.scratch.get("verification")
        tin = max(1, ctx.scratch.get("tokens_in", 1))
        tout = max(0, ctx.tok.count(ctx.output))

        kept_units = [u for u in cir.units if u.level > 0]

        cov_total = sum(u.salience * u.tokens for u in cir.units) or 1.0
        cov_kept = sum(u.salience * u.cost for u in kept_units)

        reduced = sum(1 for u in kept_units if u.level < u.max_level())
        q_terms = set(content_terms(cir.query))
        if q_terms:
            out_terms = set(content_terms(ctx.output))
            qcov = len(q_terms & out_terms) / len(q_terms)
        else:
            qcov = 1.0

        feats = {
            "ratio": 1.0 - (tout / tin),
            "integrity": v.integrity if v else 1.0,
            "retention": v.retention if v else 1.0,
            "critical": v.critical_recall if v else 1.0,
            "coverage_kept": min(1.0, cov_kept / cov_total),
            "violation_rate": ctx.scratch.get("closure_violations", 0) / max(1, len(kept_units)),
            "reduced_fidelity": reduced / max(1, len(kept_units)),
            "type_entropy": min(1.0, ctx.scratch.get("type_entropy", 0.0) / 2.0),
            "query_coverage": qcov,
            "provenance": 0.0 if (v is None or v.provenance_ok) else 1.0,
        }
        z = self.w.get("bias", 0.0) + sum(self.w.get(k, 0.0) * val for k, val in feats.items())
        self._p = 1.0 / (1.0 + math.exp(-z))
        self._feats = feats
        ctx.scratch["confidence"] = self._p
        ctx.scratch["confidence_features"] = feats

    def note(self, ctx: PassContext) -> str:
        return f"p={self._p:.3f}"
