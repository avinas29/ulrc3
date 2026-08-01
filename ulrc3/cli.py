"""Command-line interface.

    ulrc3 compress FILE [-q QUERY] [-m MODE] [--ratio R] [--budget N]
    ulrc3 inspect  FILE            # per-pass telemetry + IR statistics
    ulrc3 verify   FILE            # run the audit only, exit non-zero on failure
    ulrc3 bench    [--suite NAME]  # run the benchmark harness
    ulrc3 serve    [--port 8000]   # start the FastAPI server

Reads stdin when FILE is ``-``.  Machine-readable output with ``--json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .config import Config, Mode
from .engine import Compressor
from .request import Doc, Request
from .types import CompressionResult
from .version import __version__


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.preset(args.preset) if getattr(args, "preset", None) else Config()
    if getattr(args, "mode", None):
        cfg.mode = Mode(args.mode)
    if getattr(args, "ratio", None) is not None:
        cfg.target_ratio = args.ratio
    if getattr(args, "budget", None):
        cfg.budget_tokens = args.budget
    if getattr(args, "tokenizer", None):
        cfg.tokenizer = args.tokenizer
    if getattr(args, "doctype", None):
        cfg.doctype_override = args.doctype
    if getattr(args, "min_confidence", None) is not None:
        cfg.min_confidence = args.min_confidence
    if getattr(args, "no_verify", False):
        cfg.verify = False
    if getattr(args, "order", None):
        cfg.render.order = args.order
    return cfg


def _request(args: argparse.Namespace, text: str) -> Request:
    docs: list[Doc] = []
    for p in getattr(args, "doc", None) or []:
        docs.append(Doc(text=_read(p), doc_id=os.path.basename(p), title=os.path.basename(p)))
    return Request(
        text=text,
        query=getattr(args, "query", "") or "",
        system=_read(args.system) if getattr(args, "system", None) else "",
        instruction=getattr(args, "instruction", "") or "",
        documents=docs,
    )


def cmd_compress(args: argparse.Namespace) -> int:
    text = _read(args.file) if args.file else ""
    cfg = _build_config(args)
    eng = Compressor(cfg)
    res = eng.compress(_request(args, text))
    if args.json:
        print(json.dumps(_result_dict(res, include_text=True), indent=2))
    else:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(res.text)
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            sys.stdout.write(res.text)
        print(f"\n-- {res.summary()}", file=sys.stderr)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    text = _read(args.file)
    cfg = _build_config(args)
    res = Compressor(cfg).compress(_request(args, text))
    v = res.verification
    if args.json:
        print(json.dumps(_result_dict(res, include_text=False), indent=2))
        return 0
    w = 22
    print(f"ULRC3 {__version__}  tokenizer={res.meta['tokenizer']}")
    print(f"{'input':<{w}} {res.tokens_in} tok")
    print(f"{'output':<{w}} {res.tokens_out} tok  ({res.ratio*100:.1f}% removed, {res.compression_rate:.2f}x)")
    print(f"{'doctypes':<{w}} {res.doctypes}")
    print(f"{'units':<{w}} {res.meta['units_kept']}/{res.meta['units']}")
    print(f"{'latency':<{w}} {res.meta['latency_ms']} ms")
    print(f"{'integrity':<{w}} {v.integrity*100:.2f}%  ({v.integrity_kept}/{v.integrity_total})")
    print(f"{'critical recall':<{w}} {v.critical_recall*100:.2f}%  ({v.critical_kept}/{v.critical_total})")
    print(f"{'retention':<{w}} {v.retention*100:.2f}%  ({v.retention_kept}/{v.retention_total})")
    print(f"{'frozen intact':<{w}} {v.frozen_ok}")
    print(f"{'provenance':<{w}} {'clean' if v.provenance_ok else 'VIOLATIONS'} "
          f"({res.meta['provenance_violations']} bad, {res.meta['derived_numerals']} derived)")
    print(f"{'syntax':<{w}} {'ok' if v.syntax_ok else v.syntax_notes}")
    print(f"{'confidence':<{w}} {res.confidence:.3f}")
    print(f"{'repairs':<{w}} {v.repairs} (rounds={res.meta['repair_rounds']})")
    print("\npasses:")
    print(f"  {'name':<20s} {'ms':>8s} {'units':>13s} {'tokens':>15s}  note")
    for s in res.stats:
        print(
            f"  {s.name:<20s} {s.ms:8.2f} {s.units_in:6d}->{s.units_out:<6d} "
            f"{s.tokens_in:7d}->{s.tokens_out:<7d}  {s.note}"
        )
    if v.missing:
        print("\nmissing obligations:")
        for m in v.missing[:12]:
            print(f"  - {m}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    text = _read(args.file)
    cfg = _build_config(args)
    res = Compressor(cfg).compress(_request(args, text))
    v = res.verification
    checks = {
        "integrity": v.integrity >= 1.0,
        "critical_recall": v.critical_recall >= 1.0,
        "provenance": v.provenance_ok,
        "frozen_verbatim": v.frozen_ok,
        "no_inflation": v.inflation_ok,
        "syntax": v.syntax_ok,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(checks.values()):
        print(f"\nmissing: {v.missing[:8]}", file=sys.stderr)
        return 1
    print(f"\nall checks passed  ({res.ratio*100:.1f}% removed)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from bench.run_bench import main as bench_main  # type: ignore

    return bench_main(suite=args.suite, out=args.out, quick=args.quick)


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception:
        print("uvicorn is required: pip install 'ulrc3[server]'", file=sys.stderr)
        return 2
    os.environ.setdefault("ULRC3_MODE", args.mode or "balanced")
    uvicorn.run(
        "ulrc3.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0


def _result_dict(res: CompressionResult, include_text: bool) -> dict:
    v = res.verification
    d = {
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
        "ratio": round(res.ratio, 4),
        "compression_rate": round(res.compression_rate, 3),
        "confidence": round(res.confidence, 4),
        "doctypes": res.doctypes,
        "verification": {
            "ok": v.ok,
            "integrity": round(v.integrity, 4),
            "critical_recall": round(v.critical_recall, 4),
            "retention": round(v.retention, 4),
            "frozen_ok": v.frozen_ok,
            "provenance_ok": v.provenance_ok,
            "inflation_ok": v.inflation_ok,
            "syntax_ok": v.syntax_ok,
            "repairs": v.repairs,
            "missing": v.missing[:16],
        },
        "meta": res.meta,
        "passes": [
            {
                "name": s.name, "ms": round(s.ms, 3),
                "units_in": s.units_in, "units_out": s.units_out,
                "tokens_in": s.tokens_in, "tokens_out": s.tokens_out, "note": s.note,
            }
            for s in res.stats
        ],
    }
    if include_text:
        d["text"] = res.text
    return d


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ulrc3", description="Ultra Low-Resource Context Compression Engine")
    p.add_argument("--version", action="version", version=f"ulrc3 {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("file", nargs="?", default="-", help="input file, or - for stdin")
        sp.add_argument("-q", "--query", help="query for query-aware compression")
        sp.add_argument("-m", "--mode", choices=[m.value for m in Mode])
        sp.add_argument("-p", "--preset", help="named preset: rag, agent, code, logs, legal, safe, max")
        sp.add_argument("-r", "--ratio", type=float, help="target fraction of tokens to remove")
        sp.add_argument("-b", "--budget", type=int, help="hard output token budget")
        sp.add_argument("--doc", action="append", help="additional document (repeatable)")
        sp.add_argument("--system", help="file containing the system prompt (frozen)")
        sp.add_argument("--instruction", help="instruction text (frozen)")
        sp.add_argument("--tokenizer", help="auto | cl100k | o200k | hf:<name> | heuristic")
        sp.add_argument("--doctype", help="force a pipeline")
        sp.add_argument("--order", choices=["auto", "original", "salience"])
        sp.add_argument("--min-confidence", type=float, dest="min_confidence")
        sp.add_argument("--no-verify", action="store_true")
        sp.add_argument("--json", action="store_true")

    c = sub.add_parser("compress", help="compress a document")
    common(c)
    c.add_argument("-o", "--out", help="write output to a file")
    c.set_defaults(func=cmd_compress)

    i = sub.add_parser("inspect", help="show pass telemetry and audit results")
    common(i)
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("verify", help="run the audit; exit 1 on any failure")
    common(v)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="run the benchmark harness")
    b.add_argument("--suite", default="all")
    b.add_argument("--out", default="")
    b.add_argument("--quick", action="store_true")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("serve", help="start the HTTP server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--mode", default="balanced")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # pragma: no cover
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
