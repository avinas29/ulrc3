"""Compression on **real, held-out text the engine was never tuned on**.

Why this exists
---------------
Every other benchmark in this repository is synthetic: instances are generated
from seeded templates in :mod:`bench.datasets`.  Synthetic data can only ever
establish that the engine behaves as designed on inputs its author imagined.  It
cannot establish that the compression ratio generalises, and an independent
reviewer is right to discount it.

This harness compresses files the engine has never seen and that nobody wrote
for it: the Python standard library, third-party package documentation, and real
JSON schemas shipped inside installed distributions.  Nothing here is generated.

What it can and cannot measure
------------------------------
It measures **compression ratio, throughput, and every preservation guarantee
that is checkable without labels** -- provenance (no invented tokens), integrity
(nothing retained was partially destroyed), inflation, and syntax validity of
emitted code and JSON.

It deliberately reports **no accuracy or retention number**, because real files
carry no gold answers.  Anyone quoting a retention figure from this file would
be inventing it.  Downstream accuracy is measured only in
:mod:`bench.extrinsic_eval`, against a live model.

Usage::

    python -m bench.real_corpus                 # default: 24 files, all modes
    python -m bench.real_corpus --limit 60 --mode balanced
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import sysconfig
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ulrc3 import Compressor, Config, Mode  # noqa: E402
from ulrc3.tokenization import get_tokenizer  # noqa: E402

MIN_TOK, MAX_TOK = 800, 40_000


def _read(path: str) -> str | None:
    try:
        return open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None


def collect(limit: int, seed: int = 0) -> list[tuple[str, str, str]]:
    """Return [(category, name, text)] of real files, deterministically sampled."""
    tok = get_tokenizer("auto")
    rng = random.Random(seed)
    stdlib = sysconfig.get_paths()["stdlib"]
    site = sysconfig.get_paths()["purelib"]

    pools: dict[str, list[str]] = {
        "python": sorted(glob.glob(os.path.join(stdlib, "*.py"))),
        "markdown": sorted(glob.glob(os.path.join(site, "**", "*.md"), recursive=True)),
        "rst": sorted(glob.glob(os.path.join(site, "**", "*.rst"), recursive=True)),
        "json": sorted(glob.glob(os.path.join(site, "**", "*.json"), recursive=True)),
    }
    out: list[tuple[str, str, str]] = []
    per = max(1, limit // max(1, len(pools)))
    for cat, paths in pools.items():
        rng.shuffle(paths)
        taken = 0
        for p in paths:
            if taken >= per:
                break
            text = _read(p)
            if text is None or not (MIN_TOK < tok.count(text) < MAX_TOK):
                continue
            out.append((cat, os.path.basename(p), text))
            taken += 1
    return out


def run(limit: int, modes: list[Mode]) -> dict:
    corpus = collect(limit)
    if not corpus:
        print("no real files found on this interpreter", file=sys.stderr)
        return {}
    tok = get_tokenizer("auto")
    total_in = sum(tok.count(t) for _c, _n, t in corpus)
    print(
        f"real held-out corpus: {len(corpus)} files, {total_in:,} tokens "
        f"({', '.join(sorted({c for c, _n, _t in corpus}))})\n"
    )

    results: dict = {"files": len(corpus), "tokens_in": total_in, "modes": {}}
    for mode in modes:
        eng = Compressor(Config(mode=mode))
        ratios: list[float] = []
        by_cat: dict[str, list[float]] = {}
        elapsed = 0.0
        prov_ok = integ_ok = infl_ok = syn_ok = 0
        for cat, _name, text in corpus:
            t0 = time.perf_counter()
            res = eng.compress(text)
            elapsed += time.perf_counter() - t0
            r = 1.0 - res.tokens_out / max(1, res.tokens_in)
            ratios.append(r)
            by_cat.setdefault(cat, []).append(r)
            v = res.verification
            prov_ok += bool(v.provenance_ok)
            integ_ok += v.integrity >= 1.0
            infl_ok += bool(v.inflation_ok)
            syn_ok += bool(v.syntax_ok)
        n = len(ratios)
        results["modes"][mode.value] = {
            "reduction_mean": statistics.fmean(ratios),
            "reduction_median": statistics.median(ratios),
            "by_category": {k: statistics.fmean(v) for k, v in by_cat.items()},
            "tokens_per_s": total_in / elapsed if elapsed else 0.0,
            "provenance_ok": prov_ok / n,
            "integrity_1_0": integ_ok / n,
            "inflation_ok": infl_ok / n,
            "syntax_ok": syn_ok / n,
        }
        m = results["modes"][mode.value]
        cats = "  ".join(f"{k}={v * 100:.1f}%" for k, v in sorted(m["by_category"].items()))
        print(
            f"  {mode.value:<13} mean {m['reduction_mean'] * 100:5.1f}%  "
            f"median {m['reduction_median'] * 100:5.1f}%  "
            f"{m['tokens_per_s']:,.0f} tok/s   {cats}"
        )

    print("\nGuarantees on real input (label-free, so these are checkable):")
    m = results["modes"][modes[0].value]
    for k in ("provenance_ok", "integrity_1_0", "inflation_ok", "syntax_ok"):
        print(f"  {k:<16} {m[k] * 100:5.1f}% of files")
    print(
        "\nNo retention or accuracy number is reported here: real files carry no\n"
        "gold answers.  Downstream accuracy lives in bench/extrinsic_eval.py."
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--mode", default="all")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    modes = (
        [Mode.CONSERVATIVE, Mode.BALANCED, Mode.AGGRESSIVE, Mode.EXTREME]
        if a.mode == "all"
        else [Mode(a.mode)]
    )
    res = run(a.limit, modes)
    if a.json and res:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
