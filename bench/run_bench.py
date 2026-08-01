"""Benchmark harness.

Runs every suite against the engine (at several operating points) and against
every baseline **at matched output budgets**, so that quality differences are
not confounded by compression differences.

Protocol
--------
1. For an instance and a mode, run ULRC3; record its output token count ``T``.
2. Run every baseline with ``budget = T`` on the *same* source.
3. Score all systems with the same metric suite.
4. Aggregate by (suite, system), report mean ± stderr.

That protocol is the whole point: "we get 80% compression" is meaningless
without "at the same 80%, here is what everyone retains".

Usage::

    python -m bench.run_bench --suite all
    python -m bench.run_bench --suite rag --quick --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.datasets import SUITES, Instance, load, load_external  # noqa: E402
from ulrc3 import Compressor, Config, Mode, Request  # noqa: E402
from ulrc3.metrics.baselines import BASELINES, run_baseline  # noqa: E402
from ulrc3.metrics.intrinsic import evaluate  # noqa: E402
from ulrc3.tokenization import get_tokenizer  # noqa: E402
from ulrc3.types import CompressionResult, Verification  # noqa: E402

MODES = ["conservative", "balanced", "aggressive", "extreme"]
PRIMARY_MODE = "balanced"


def _peak_rss_mb() -> float:
    try:
        import resource

        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return val / (1024 * 1024) if sys.platform == "darwin" else val / 1024
    except Exception:
        return 0.0


def _request(inst: Instance) -> Request:
    return Request(
        text=inst.text,
        documents=list(inst.documents),
        query=inst.query,
        system=inst.system,
        instruction=inst.instruction,
        doctype=inst.doctype,
    )


def _baseline_result(text: str, tokens_in: int, tokens_out: int, latency_ms: float) -> CompressionResult:
    return CompressionResult(
        text=text,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        verification=Verification(),
        confidence=0.0,
        meta={"latency_ms": latency_ms, "provenance_violations": 0, "derived_numerals": 0},
    )


def run_instance(
    inst: Instance,
    engines: dict[str, Compressor],
    tok: Any,
    with_baselines: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = inst.source()
    body = "\n".join([p for p in ([inst.text] + inst.documents) if p])

    budget_for_baselines: int | None = None
    for mode, eng in engines.items():
        res = eng.compress(_request(inst))
        m = evaluate(res, source, inst.query, inst.answers, inst.entities or None, _peak_rss_mb(), inst.distractors)
        rows.append({"suite": inst.suite, "iid": inst.iid, "system": f"ulrc3:{mode}", **m.as_dict()})
        if mode == PRIMARY_MODE:
            budget_for_baselines = res.tokens_out

    if with_baselines and budget_for_baselines:
        tin = tok.count(source)
        for name in BASELINES:
            t0 = time.perf_counter()
            out = run_baseline(
                name, body, budget_for_baselines, query=inst.query,
                instruction=inst.instruction, tok=tok,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            # baselines never see the frozen segments as protected, so prepend
            # the query the way a real system would
            full_out = (inst.query + "\n" + out) if inst.query else out
            res = _baseline_result(full_out, tin, tok.count(full_out), dt)
            m = evaluate(res, source, inst.query, inst.answers, inst.entities or None, 0.0, inst.distractors)
            rows.append({"suite": inst.suite, "iid": inst.iid, "system": name, **m.as_dict()})
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["suite"], r["system"]), []).append(r)
    for (suite, system), items in groups.items():
        agg: dict[str, float] = {"n": len(items)}
        keys = [
            k for k in items[0]
            if k not in ("suite", "iid", "system") and isinstance(items[0][k], (int, float, bool))
        ]
        for k in keys:
            vals = [float(x[k]) for x in items if x.get(k) is not None]
            if not vals:
                continue
            agg[k] = statistics.fmean(vals)
            if len(vals) > 1:
                agg[f"{k}_se"] = statistics.stdev(vals) / math.sqrt(len(vals))
        out.setdefault(suite, {})[system] = agg
    return out


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}"


def report_markdown(agg: dict[str, dict[str, dict[str, float]]]) -> str:
    lines: list[str] = ["# ULRC3 benchmark report", ""]
    systems_order = [f"ulrc3:{m}" for m in MODES] + list(BASELINES)
    for suite in sorted(agg):
        lines.append(f"## {suite}")
        lines.append("")
        lines.append(
            "| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | "
            "integrity% | halluc. | latency ms |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for system in systems_order:
            a = agg[suite].get(system)
            if not a:
                continue
            lines.append(
                f"| {system} | {_fmt_pct(a.get('ratio', 0))} | "
                f"{_fmt_pct(a.get('answerability', 0))} | "
                f"{_fmt_pct(a.get('distractor_rate', 0))} | "
                f"{_fmt_pct(a.get('number_recall', 0))} | "
                f"{_fmt_pct(a.get('identifier_recall', 0))} | "
                f"{_fmt_pct(a.get('integrity', 0))} | "
                f"{a.get('hallucinated_words', 0):.2f} | "
                f"{a.get('latency_ms', 0):.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def summary_table(agg: dict[str, dict[str, dict[str, float]]]) -> str:
    """Cross-suite mean at the primary operating point vs every baseline."""
    systems = [f"ulrc3:{m}" for m in MODES] + list(BASELINES)
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for system in systems:
        vals = [agg[s][system] for s in agg if system in agg[s]]
        if not vals:
            continue
        rows.append(
            (
                system,
                statistics.fmean(v.get("ratio", 0.0) for v in vals),
                statistics.fmean(v.get("answerability", 0.0) for v in vals),
                statistics.fmean(v.get("number_recall", 0.0) for v in vals),
                statistics.fmean(v.get("identifier_recall", 0.0) for v in vals),
                statistics.fmean(v.get("hallucinated_words", 0.0) for v in vals),
                statistics.fmean(v.get("distractor_rate", 0.0) for v in vals),
            )
        )
    out = ["", "## Cross-suite means", "",
           "| system | ratio% | answerability% | contradiction% | numbers% | identifiers% | halluc. |",
           "|---|---|---|---|---|---|---|"]
    for name, ratio, ans, num, ident, hall, dist in rows:
        out.append(
            f"| {name} | {_fmt_pct(ratio)} | {_fmt_pct(ans)} | {_fmt_pct(dist)} | {_fmt_pct(num)} | "
            f"{_fmt_pct(ident)} | {hall:.2f} |"
        )
    return "\n".join(out)


def main(suite: str = "all", out: str = "", quick: bool = False, baselines: bool = True) -> int:
    tok = get_tokenizer("auto")
    engines = {m: Compressor(Config(mode=Mode(m))) for m in MODES}

    names = list(SUITES) if suite == "all" else [s for s in suite.split(",") if s in SUITES]
    instances: list[Instance] = []
    for nm in names:
        instances.extend(load(nm, quick))
    external = load_external()
    if external and suite in ("all", "external"):
        instances.extend(external)
    if not instances:
        print(f"no instances for suite={suite!r}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances, 1):
        rows.extend(run_instance(inst, engines, tok, with_baselines=baselines))
        if i % 10 == 0 or i == len(instances):
            print(f"  {i}/{len(instances)} instances", file=sys.stderr)
    elapsed = time.perf_counter() - t0

    agg = aggregate(rows)
    md = report_markdown(agg) + "\n" + summary_table(agg)
    print(md)
    print(f"\n<!-- {len(instances)} instances, {len(rows)} runs, {elapsed:.1f}s -->")

    if out:
        payload = {"aggregate": agg, "rows": rows, "elapsed_s": elapsed, "quick": quick}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        base, _ = os.path.splitext(out)
        with open(base + ".md", "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {out} and {base}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="all")
    p.add_argument("--out", default="")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    a = p.parse_args()
    raise SystemExit(main(a.suite, a.out, a.quick, not a.no_baselines))
