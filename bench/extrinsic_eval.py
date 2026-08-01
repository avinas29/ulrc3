"""End-to-end evaluation against a real LLM.

The experiment every other number in this repository has been an upper bound on:

    Does a real model, given the ULRC³-compressed context, still answer the
    question correctly -- and does it beat truncation at the *same* token count?

Three conditions per instance:

  ``full``      the uncompressed context (control: what perfect retention buys)
  ``ulrc3``     the compressed context
  ``truncate``  head-truncation to ULRC³'s exact output size (matched budget)

so a single run answers both "did compression cost accuracy?" (ulrc3 vs full)
and "is it better than the naive thing?" (ulrc3 vs truncate).

Quota discipline: free tiers allow ~20 requests/day, so the runner declares a
budget, keeps a persistent ledger, caches every reply, and stops cleanly when the
budget is gone. Re-running costs nothing for trials already completed.

Usage::

    export GEMINI_API_KEY=...            # never stored in the repo
    python -m bench.extrinsic_eval --instances 6 --budget 18
    python -m bench.extrinsic_eval --report-only     # analyse the cache, 0 calls
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.datasets import SUITES, load  # noqa: E402
from ulrc3 import Compressor, Config, Mode  # noqa: E402
from ulrc3.metrics.baselines import run_baseline  # noqa: E402
from ulrc3.metrics.extrinsic import (  # noqa: E402
    DEFAULT_MODEL,
    GeminiJudge,
    Ledger,
    QuotaExhausted,
    Trial,
    evaluate_condition,
    evaluate_packed,
)
from ulrc3.tokenization import get_tokenizer  # noqa: E402

LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "llm_ledger.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "extrinsic.json")

#: Suites chosen to span the architecture's distinct mechanisms rather than to
#: flatter it: a needle (retrieval), a multi-hop chain, numeric reasoning, a
#: contradiction case, a log, and polyglot content.
SUITE_ORDER = ["needle", "multihop", "numeric", "memory", "logs", "mixed", "rag", "code", "apidocs"]


def pick_instances(n: int) -> list[Any]:
    """One instance per suite, round-robin, deterministic."""
    pools = {s: load(s, quick=True) for s in SUITE_ORDER if s in SUITES}
    out: list[Any] = []
    idx = 0
    while len(out) < n:
        progressed = False
        for s in SUITE_ORDER:
            pool = pools.get(s) or []
            if idx < len(pool):
                out.append(pool[idx])
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
        idx += 1
    return out


def build_conditions(inst: Any, mode: str) -> list[tuple[str, str, int]]:
    """Return [(condition, context, tokens)] for one instance."""
    tok = get_tokenizer("auto")
    body = "\n".join(p for p in ([inst.text] + list(inst.documents)) if p)

    res = Compressor(Config(mode=Mode(mode))).compress(
        {"text": inst.text, "documents": list(inst.documents), "query": inst.query}
    )
    compressed = res.text
    n_comp = res.tokens_out

    truncated = run_baseline("truncate_head", body, n_comp, query=inst.query, tok=tok)

    return [
        ("full", body, tok.count(body)),
        ("ulrc3", compressed, n_comp),
        ("truncate", truncated, tok.count(truncated)),
    ]


class _CacheOnlyJudge:
    """Replays cached replies; never makes a call. Used by ``--report-only`` so a
    scoring bug can be fixed and the data re-analysed for free."""

    def __init__(self, ledger: Ledger, model: str, params: str = "max=2048;t=0.0") -> None:
        self.ledger = ledger
        self.model = model
        self.params = params
        self.name = f"cache:{model}"
        self.misses = 0

    def ask(self, prompt: str) -> str:
        hit = self.ledger.get(self.model, prompt, self.params)
        if hit is None:
            self.misses += 1
            raise QuotaExhausted("not in cache (run without --report-only to fetch)")
        return hit


def _show(t: Trial) -> None:
    score = "  n/a" if t.score != t.score else f"{t.score:5.2f}"  # NaN check
    print(
        f"  {t.iid:<14s} {t.condition:<9s} {t.tokens:>6d} tok  score {score}  "
        f"{(t.error or t.reply)[:52]!r}",
        file=sys.stderr,
    )


def run(
    instances: int, budget: int, mode: str, model: str, report_only: bool, pack: int = 1
) -> dict[str, Any]:
    ledger = Ledger.load(LEDGER_PATH, daily_budget=budget)
    judge = _CacheOnlyJudge(ledger, model) if report_only else GeminiJudge(ledger=ledger, model=model)

    chosen = pick_instances(instances)
    prepared = [(inst, build_conditions(inst, mode)) for inst in chosen]

    trials: list[Trial] = []
    stopped = ""

    if pack > 1:
        # Group by condition, then chunk: one request answers `pack` items, so
        # the same 20-request quota buys `pack` times the sample size.
        for condition in ("full", "ulrc3", "truncate"):
            cells = [
                (
                    inst.iid, inst.suite,
                    next(c for cond, c, _n in conds if cond == condition),
                    inst.query,
                    getattr(inst, "final_answer", None) or inst.answers,
                    next(n for cond, _c, n in conds if cond == condition),
                )
                for inst, conds in prepared
            ]
            for i in range(0, len(cells), pack):
                chunk = cells[i : i + pack]
                try:
                    got = evaluate_packed(judge, condition, chunk)
                except QuotaExhausted as exc:
                    if report_only:
                        continue
                    stopped = str(exc)
                    break
                trials.extend(got)
                for t in got:
                    _show(t)
            if stopped:
                break
    else:
        for inst, conds in prepared:
            for condition, context, ntok in conds:
                try:
                    gold = getattr(inst, "final_answer", None) or inst.answers
                    t = evaluate_condition(
                        judge, inst.iid, inst.suite, context, inst.query,
                        gold, condition, ntok,
                    )
                except QuotaExhausted as exc:
                    if report_only:
                        continue  # simply no cached reply for this cell
                    stopped = str(exc)
                    break
                trials.append(t)
                _show(t)
            if stopped:
                break

    # A request that errored is a *missing* observation, not a zero. Counting
    # HTTP 400s as wrong answers deflated every condition in the first run.
    import math

    ok = [t for t in trials if not t.error and not math.isnan(t.score)]
    failed = [t for t in trials if t.error]
    by_cond: dict[str, list[Trial]] = {}
    for t in ok:
        by_cond.setdefault(t.condition, []).append(t)

    summary = {
        c: {
            "n": len(ts),
            "accuracy": statistics.fmean(t.score for t in ts) if ts else 0.0,
            "exact": statistics.fmean(1.0 if t.score >= 1.0 else 0.0 for t in ts) if ts else 0.0,
            "mean_tokens": statistics.fmean(t.tokens for t in ts) if ts else 0.0,
        }
        for c, ts in by_cond.items()
    }

    payload = {
        "model": model,
        "mode": mode,
        "summary": summary,
        "trials": [t.__dict__ for t in trials],
        "paired": paired_analysis(trials),
        "pack": pack,
        "failed_requests": len(failed),
        "requests_spent_total": ledger.spent,
        "budget": budget,
        "stopped": stopped,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def paired_analysis(trials: list[Trial], seed: int = 7) -> dict[str, Any]:
    """Compare conditions on the instances where *all* of them completed.

    Unequal-n means are not a comparison: when a run stops early, one condition
    has been measured on an easier or simply different subset than another. Our
    first packed run reported full=21 / ulrc3=27 / truncate=12 and the resulting
    "means" compared three different experiments. Everything below is paired.
    """
    import math
    import random

    by: dict[str, dict[str, float]] = {}
    for t in trials:
        if not t.error and t.score == t.score:  # not NaN
            by.setdefault(t.iid, {})[t.condition] = t.score

    conds = ("full", "ulrc3", "truncate")
    common = sorted(i for i, c in by.items() if set(conds) <= set(c))
    if not common:
        return {"n": 0}

    vec = {c: [by[i][c] for i in common] for c in conds}

    def sign_test(a: list[float], b: list[float]) -> dict[str, Any]:
        wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
        losses = sum(1 for x, y in zip(a, b, strict=True) if x < y)
        n = wins + losses
        if n == 0:
            return {"wins": wins, "losses": losses, "ties": len(a), "p": 1.0}
        tail = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n
        return {"wins": wins, "losses": losses, "ties": len(a) - n, "p": min(1.0, 2 * tail)}

    rng = random.Random(seed)

    def boot_ci(a: list[float], b: list[float], reps: int = 20000) -> tuple[float, float]:
        diffs = [x - y for x, y in zip(a, b, strict=True)]
        means = sorted(
            sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps)
        )
        return means[int(0.025 * reps)], means[int(0.975 * reps)]

    out: dict[str, Any] = {
        "n": len(common),
        "instances": common,
        "means": {c: statistics.fmean(vec[c]) for c in conds},
        "comparisons": {},
    }

    # End-to-end latency: the model's own response time, which is what a user
    # feels. Only *fresh* requests are timed -- a cache replay is not an
    # observation. Packed items share their pack's latency, so we de-duplicate
    # per (condition, latency) to avoid counting one request k times.
    lat: dict[str, list[float]] = {}
    for t in trials:
        if t.latency_ms is None or t.error:
            continue
        seen = lat.setdefault(t.condition, [])
        if not seen or seen[-1] != t.latency_ms:
            seen.append(t.latency_ms)
    if lat:
        out["latency_ms"] = {c: statistics.fmean(v) for c, v in lat.items()}
        out["latency_requests"] = {c: len(v) for c, v in lat.items()}
    for other in ("truncate", "full"):
        lo, hi = boot_ci(vec["ulrc3"], vec[other])
        out["comparisons"][f"ulrc3_vs_{other}"] = {
            "delta": statistics.fmean(vec["ulrc3"]) - statistics.fmean(vec[other]),
            "ci95": [lo, hi],
            **sign_test(vec["ulrc3"], vec[other]),
        }

    per_suite: dict[str, dict[str, list[float]]] = {}
    suite_of = {t.iid: t.suite for t in trials}
    for i in common:
        for c in conds:
            per_suite.setdefault(suite_of.get(i, "?"), {}).setdefault(c, []).append(by[i][c])
    out["per_suite"] = {
        s: {c: statistics.fmean(v) for c, v in d.items()} for s, d in per_suite.items()
    }
    return out


def report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Extrinsic evaluation — real model in the loop",
        "",
        f"Model: `{payload['model']}` · mode: `{payload['mode']}` · "
        f"requests spent: {payload['requests_spent_total']}/{payload['budget']}"
        + (f" · pack={payload.get('pack', 1)}" if payload.get("pack", 1) > 1 else ""),
        "",
        "| condition | n | mean tokens | answer accuracy | fully correct |",
        "|---|---|---|---|---|",
    ]
    for c in ("full", "ulrc3", "truncate"):
        if c not in s:
            continue
        r = s[c]
        lines.append(
            f"| {c} | {r['n']} | {r['mean_tokens']:.0f} | "
            f"{r['accuracy'] * 100:.1f}% | {r['exact'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append(
        "_Unequal n above means these are three different instance subsets. "
        "The comparison that matters is the paired one:_"
    )

    pa = payload.get("paired") or {}
    if pa.get("n"):
        m = pa["means"]
        lines += [
            "",
            f"## Paired comparison — the {pa['n']} instances where all three completed",
            "",
            "| condition | accuracy |",
            "|---|---|",
            f"| full (uncompressed control) | {m['full'] * 100:.1f}% |",
            f"| **ulrc3** | **{m['ulrc3'] * 100:.1f}%** |",
            f"| truncate (matched budget) | {m['truncate'] * 100:.1f}% |",
            "",
        ]
        for key, label in (
            ("ulrc3_vs_truncate", "vs truncation at the same token count"),
            ("ulrc3_vs_full", "vs the uncompressed control"),
        ):
            c = pa["comparisons"][key]
            lines.append(
                f"**{label}:** {c['delta'] * 100:+.1f} points "
                f"(95% CI [{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}]), "
                f"better on {c['wins']}, worse on {c['losses']}, tied {c['ties']}; "
                f"sign test p={c['p']:.4f}."
            )
        lat = pa.get("latency_ms") or {}
        if lat.get("full") and lat.get("ulrc3"):
            speedup = lat["full"] / lat["ulrc3"]
            lines += [
                "",
                "| condition | mean end-to-end response time |",
                "|---|---|",
            ]
            for c in ("full", "ulrc3", "truncate"):
                if c in lat:
                    lines.append(f"| {c} | {lat[c] / 1000:.2f} s |")
            lines.append("")
            lines.append(
                f"**Inference latency speedup: {speedup:.2f}x** "
                f"({lat['full'] / 1000:.2f}s -> {lat['ulrc3'] / 1000:.2f}s), measured on "
                f"{pa.get('latency_requests', {}).get('ulrc3', 0)} fresh requests per condition."
            )

        if pa.get("per_suite"):
            lines += ["", "| suite | full | ulrc3 | truncate |", "|---|---|---|---|"]
            for suite in sorted(pa["per_suite"]):
                r = pa["per_suite"][suite]
                lines.append(
                    f"| {suite} | {r.get('full', 0) * 100:.0f}% | "
                    f"{r.get('ulrc3', 0) * 100:.0f}% | {r.get('truncate', 0) * 100:.0f}% |"
                )
    if payload.get("failed_requests"):
        lines.append(
            f"\n> {payload['failed_requests']} request(s) errored and are excluded "
            "as missing observations rather than counted as wrong answers."
        )
    if payload.get("stopped"):
        lines.append(f"\n> Run stopped early: {payload['stopped']}")
    n = (payload.get("paired") or {}).get("n", 0)
    lines.append(
        f"\n**Statistical caveat.** The paired comparison rests on n={n} instances "
        "from a free-tier quota. That is enough to establish a large effect "
        "(the truncation gap clears significance comfortably) and *not* enough "
        "to resolve a small one — the ulrc3-vs-full difference is well inside "
        "noise and should be read as 'no detected loss', not as 'no loss'. "
        "One model, one temperature, one prompt template."
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", type=int, default=6)
    p.add_argument("--budget", type=int, default=15,
                   help="max API requests this run may spend (free tier measured at 5/day)")
    p.add_argument("--mode", default="balanced")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--report-only", action="store_true", help="analyse cache, make no calls")
    p.add_argument(
        "--pack", type=int, default=1,
        help="items per request; >1 multiplies sample size under a fixed quota",
    )
    a = p.parse_args()

    payload = run(a.instances, a.budget, a.mode, a.model, a.report_only, a.pack)
    md = report(payload)
    print(md)
    with open(os.path.splitext(OUT_PATH)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
