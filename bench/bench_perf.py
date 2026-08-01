"""Performance profile: latency, throughput, memory, scaling, parallel speedup.

Reports the numbers a production reviewer asks for, measured on this machine:

* wall-clock latency vs input size (1k -> 200k tokens);
* per-pass attribution (where the time actually goes);
* tokens/second throughput, single process and with the process pool;
* peak RSS delta;
* cache hit speed-up on repeated documents.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random  # noqa: E402

from bench.datasets import _haystack  # noqa: E402
from ulrc3 import Compressor, Config, Mode  # noqa: E402
from ulrc3.cache import ChunkCache, DocumentCache  # noqa: E402
from ulrc3.tokenization import get_tokenizer  # noqa: E402


def synth(target_tokens: int, seed: int = 5) -> str:
    rng = random.Random(seed)
    tok = get_tokenizer("auto")
    parts: list[str] = []
    n = 0
    while n < target_tokens:
        block = "\n\n".join(_haystack(rng, 6))
        parts.append(block)
        n += tok.count(block)
    return "\n\n".join(parts)


def _rss_mb() -> float:
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except Exception:
        return 0.0


def latency_scaling(sizes=(1_000, 4_000, 16_000, 64_000, 128_000), repeats: int = 3) -> list[dict[str, Any]]:
    eng = Compressor(Config(mode=Mode.BALANCED))
    tok = get_tokenizer("auto")
    rows: list[dict[str, Any]] = []
    for size in sizes:
        doc = synth(size)
        n = tok.count(doc)
        gc.collect()
        before = _rss_mb()
        times = []
        res = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = eng.compress(doc)
            times.append((time.perf_counter() - t0) * 1000.0)
        after = _rss_mb()
        rows.append(
            {
                "target_tokens": size,
                "tokens_in": n,
                "tokens_out": res.tokens_out,
                "ratio": round(res.ratio, 4),
                "p50_ms": round(statistics.median(times), 2),
                "min_ms": round(min(times), 2),
                "tokens_per_s": round(n / (statistics.median(times) / 1000.0)),
                "rss_delta_mb": round(after - before, 1),
                "units": res.meta["units"],
            }
        )
        print(
            f"  {n:>7,} tok -> {res.tokens_out:>6,} tok  "
            f"p50 {rows[-1]['p50_ms']:>8.1f} ms  "
            f"{rows[-1]['tokens_per_s']:>9,} tok/s  "
            f"units {res.meta['units']:>5}  rss+{rows[-1]['rss_delta_mb']:.1f}MB",
            file=sys.stderr,
        )
    return rows


def pass_attribution(size: int = 32_000) -> list[dict[str, Any]]:
    eng = Compressor(Config(mode=Mode.BALANCED))
    res = eng.compress(synth(size))
    total = sum(s.ms for s in res.stats)
    out = []
    for s in res.stats:
        out.append({"pass": s.name, "ms": round(s.ms, 2), "share": round(s.ms / total, 4) if total else 0.0})
    for row in sorted(out, key=lambda r: -r["ms"]):
        print(f"  {row['pass']:<20s} {row['ms']:>8.2f} ms  {row['share'] * 100:>5.1f}%", file=sys.stderr)
    return out


def parallel_speedup(docs: int = 16, size: int = 8_000) -> dict[str, Any]:
    payloads = [synth(size, seed=i) for i in range(docs)]
    cfg = Config(mode=Mode.BALANCED)
    eng = Compressor(cfg)

    t0 = time.perf_counter()
    for p in payloads:
        eng.compress(p)
    seq = time.perf_counter() - t0

    cfg_par = Config(mode=Mode.BALANCED)
    cfg_par.perf.parallel_min_chars = 0
    cfg_par.perf.parallel_min_docs = 2
    t0 = time.perf_counter()
    Compressor(cfg_par).compress_many(payloads)
    par = time.perf_counter() - t0

    print(f"  sequential {seq:.2f}s   parallel {par:.2f}s   speedup {seq / par:.2f}x", file=sys.stderr)
    return {"docs": docs, "size": size, "sequential_s": round(seq, 3),
            "parallel_s": round(par, 3), "speedup": round(seq / par, 2)}


def cache_speedup(size: int = 32_000) -> dict[str, Any]:
    doc = synth(size)
    eng = Compressor(Config(mode=Mode.BALANCED))
    dc = DocumentCache(ChunkCache())

    def fn(t: str) -> str:
        return eng.compress(t).text

    t0 = time.perf_counter()
    dc.compress_cached(doc, fn, 4000)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    _out, hits = dc.compress_cached(doc, fn, 4000)
    warm = time.perf_counter() - t0
    print(f"  cold {cold * 1000:.0f} ms  warm {warm * 1000:.1f} ms  "
          f"hit-rate {hits:.2f}  speedup {cold / max(warm, 1e-6):.0f}x", file=sys.stderr)
    return {"cold_ms": round(cold * 1000, 1), "warm_ms": round(warm * 1000, 2),
            "hit_rate": hits, "speedup": round(cold / max(warm, 1e-6), 1)}


def main() -> int:
    print("latency scaling:", file=sys.stderr)
    scaling = latency_scaling()
    print("\npass attribution (32k tokens):", file=sys.stderr)
    passes = pass_attribution()
    print("\nparallel batch:", file=sys.stderr)
    par = parallel_speedup()
    print("\ndocument cache:", file=sys.stderr)
    cache = cache_speedup()
    payload = {"scaling": scaling, "passes": passes, "parallel": par, "cache": cache,
               "cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "perf.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out}", file=sys.stderr)
    print(json.dumps(payload["scaling"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
