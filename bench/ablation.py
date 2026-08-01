"""Ablation study — falsifying our own architectural claims.

Each ablation disables exactly one mechanism and re-runs the full benchmark at
the same operating point. The delta is what that mechanism is worth.

This exists because a design document that only reports the full system proves
nothing: it is always possible that the complicated part contributes nothing and
the simple parts carry the result. These runs are the check.

| ablation | disables | claim under test |
|---|---|---|
| `no_ladder` | multi-rung renderings; every unit becomes keep-or-drop | the fidelity ladder (the central structural claim) |
| `no_attention` | Phantom Attention; salience uses a constant | model-free attention simulation beats no ranking |
| `no_lexical` | lexical edges only; graph keeps structure edges | induction-head modelling matters |
| `no_dedup` | semantic dedup + delta encoding | redundancy removal is load-bearing |
| `no_closure` | dependency closure repair | closure costs budget — does it buy quality? |
| `no_order` | edge-loaded ordering | lost-in-the-middle mitigation |
| `no_repair` | verifier repair loop | the audit actually changes the output |
| `no_coverage` | submodular coverage; selection uses salience alone | isolates the ranking signal from the objective that dominates it |
| `no_cov+no_attn` | both | *is Phantom Attention a good ranker, or a dominated one?* |

Usage::

    python -m bench.ablation                 # all suites, quick
    python -m bench.ablation --full
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.datasets import SUITES, load  # noqa: E402
from bench.run_bench import _request  # noqa: E402
from ulrc3 import Compressor, Config, Mode  # noqa: E402
from ulrc3.metrics.intrinsic import evaluate  # noqa: E402

ABLATIONS: list[tuple[str, frozenset[str]]] = [
    ("full system", frozenset()),
    ("no_ladder", frozenset({"no_ladder"})),
    ("no_attention", frozenset({"no_attention"})),
    ("no_lexical", frozenset({"no_lexical"})),
    ("no_dedup", frozenset({"no_dedup"})),
    ("no_closure", frozenset({"no_closure"})),
    ("no_order", frozenset({"no_order"})),
    ("no_repair", frozenset({"no_repair"})),
    ("no_coverage", frozenset({"no_coverage"})),
    ("no_cov+no_attn", frozenset({"no_coverage", "no_attention"})),
]


def run(quick: bool = True, mode: str = "balanced") -> dict[str, dict[str, float]]:
    instances = []
    for name in SUITES:
        instances.extend(load(name, quick))

    results: dict[str, dict[str, float]] = {}
    for label, ablate in ABLATIONS:
        cfg = Config(mode=Mode(mode), ablate=ablate)
        eng = Compressor(cfg)
        rows: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for inst in instances:
            res = eng.compress(_request(inst))
            m = evaluate(
                res, inst.source(), inst.query, inst.answers,
                inst.entities or None, 0.0, inst.distractors,
            )
            rows.append(m.as_dict())
        elapsed = time.perf_counter() - t0

        results[label] = {
            "ratio": statistics.fmean(r["ratio"] for r in rows),
            "answerability": statistics.fmean(r["answerability"] for r in rows),
            "contradiction": statistics.fmean(r["distractor_rate"] for r in rows),
            "numbers": statistics.fmean(r["number_recall"] for r in rows),
            "identifiers": statistics.fmean(r["identifier_recall"] for r in rows),
            "integrity": statistics.fmean(r["integrity"] for r in rows),
            "syntax_ok": statistics.fmean(1.0 if r["syntax_ok"] else 0.0 for r in rows),
            "latency_ms": statistics.fmean(r["latency_ms"] for r in rows),
            "wall_s": round(elapsed, 2),
            "n": len(rows),
        }
        print(
            f"  {label:<14s} ratio {results[label]['ratio'] * 100:5.1f}%  "
            f"ans {results[label]['answerability'] * 100:5.1f}%  "
            f"contra {results[label]['contradiction'] * 100:5.1f}%  "
            f"ident {results[label]['identifiers'] * 100:5.1f}%  "
            f"({elapsed:.1f}s)",
            file=sys.stderr,
        )
    return results


def report(results: dict[str, dict[str, float]]) -> str:
    base = results["full system"]
    lines = [
        "# Ablation study",
        "",
        f"All suites, balanced mode, n={int(base['n'])} instances per row.",
        "Δ columns are relative to the full system.",
        "",
        "| ablation | ratio % | Δratio | answerability % | Δans | contradiction % | "
        "identifiers % | Δident | integrity % | latency ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        d_ratio = (r["ratio"] - base["ratio"]) * 100
        d_ans = (r["answerability"] - base["answerability"]) * 100
        d_id = (r["identifiers"] - base["identifiers"]) * 100
        lines.append(
            f"| {label} | {r['ratio'] * 100:.1f} | {d_ratio:+.1f} | "
            f"{r['answerability'] * 100:.1f} | {d_ans:+.1f} | "
            f"{r['contradiction'] * 100:.1f} | "
            f"{r['identifiers'] * 100:.1f} | {d_id:+.1f} | "
            f"{r['integrity'] * 100:.1f} | {r['latency_ms']:.1f} |"
        )
    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append(
        "A mechanism earns its place if removing it *either* costs compression at "
        "equal quality *or* costs quality at equal compression. A row that matches "
        "the full system on both is a mechanism we should delete."
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--mode", default="balanced")
    p.add_argument("--out", default="bench/results/ablation.json")
    a = p.parse_args()

    print("running ablations:", file=sys.stderr)
    results = run(quick=not a.full, mode=a.mode)
    md = report(results)
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        with open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
