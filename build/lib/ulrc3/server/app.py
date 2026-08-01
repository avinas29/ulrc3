"""FastAPI server.

Endpoints
---------
``POST /v1/compress``   compress a structured request
``POST /v1/compress/stream``  same, streamed as newline-delimited JSON events
``POST /v1/expand``     fault dropped spans back in (the recovery layer)
``POST /v1/estimate``   token count + projected ratios, without compressing
``GET  /v1/health``     liveness + cache/residual statistics
``GET  /v1/config``     effective configuration

Design notes
------------
* The engine is **stateless per call**, so a single ``Compressor`` is shared and
  requests run concurrently in the threadpool without locks.
* CPU-bound work is dispatched with ``run_in_threadpool`` so the event loop
  stays responsive; the tokenizer releases the GIL inside tiktoken's Rust core.
  For batches, ``Compressor.compress_many`` uses a process pool instead (2.9x
  measured on 8 cores).  The in-process/pool threshold is a configured default
  (``PerfPolicy.parallel_min_docs`` / ``parallel_min_chars``), not a measured
  crossover point.
* Residuals are stored per session so the recovery endpoint can serve them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from ..cache import DEFAULT_CACHE
from ..config import Config, Mode
from ..detect.doctype import detect
from ..engine import Compressor
from ..recovery import DEFAULT_STORE, Residual
from ..request import Request as UlrcRequest
from ..tokenization import get_tokenizer
from ..types import CompressionResult
from ..version import __version__
from .schemas import (
    CompressIn,
    CompressOut,
    EstimateIn,
    EstimateOut,
    ExpandIn,
    ExpandOut,
    HealthOut,
    PassOut,
    VerificationOut,
)

app = FastAPI(
    title="ULRC3 Context Compression Engine",
    version=__version__,
    description="Semantic-compiler context compression with verifiable preservation guarantees.",
)

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_STARTED = time.time()
_REQUESTS = 0
_BASE_CFG = Config(mode=Mode(os.environ.get("ULRC3_MODE", "balanced")))
_ENGINE = Compressor(_BASE_CFG)
_SEM = asyncio.Semaphore(int(os.environ.get("ULRC3_MAX_CONCURRENCY", "32")))


def _engine_for(body: CompressIn) -> Compressor:
    """Return a shared engine, or a per-request one when config differs."""
    if not (body.preset or body.min_confidence is not None or body.tokenizer):
        return _ENGINE
    cfg = Config.preset(body.preset) if body.preset else Config.from_dict(_BASE_CFG.to_dict())
    if body.min_confidence is not None:
        cfg.min_confidence = body.min_confidence
    if body.tokenizer:
        cfg.tokenizer = body.tokenizer
    cfg.keep_residuals = body.keep_residuals
    return Compressor(cfg)


def _to_out(res: CompressionResult, session: str) -> CompressOut:
    v = res.verification
    return CompressOut(
        text=res.text,
        tokens_in=res.tokens_in,
        tokens_out=res.tokens_out,
        ratio=round(res.ratio, 4),
        compression_rate=round(res.compression_rate, 3),
        confidence=round(res.confidence, 4),
        doctypes=res.doctypes,
        verification=VerificationOut(
            ok=v.ok,
            integrity=round(v.integrity, 4),
            critical_recall=round(v.critical_recall, 4),
            retention=round(v.retention, 4),
            frozen_ok=v.frozen_ok,
            provenance_ok=v.provenance_ok,
            inflation_ok=v.inflation_ok,
            syntax_ok=v.syntax_ok,
            repairs=v.repairs,
            missing=v.missing[:16],
        ),
        passes=[
            PassOut(
                name=s.name, ms=round(s.ms, 3), units_in=s.units_in, units_out=s.units_out,
                tokens_in=s.tokens_in, tokens_out=s.tokens_out, note=s.note,
            )
            for s in res.stats
        ],
        meta=res.meta,
        session=session,
        expandable=sorted(res.residuals.keys(), key=lambda x: int(x) if x.isdigit() else 0)[:64],
    )


def _run(body: CompressIn) -> tuple[CompressionResult, str]:
    eng = _engine_for(body)
    req = UlrcRequest(**body.to_request_kwargs())
    res = eng.compress(req)
    session = body.session or uuid.uuid4().hex[:16]
    if body.keep_residuals and res.residuals:
        DEFAULT_STORE.put(
            session,
            [Residual(handle=h, text=t, doc_id="") for h, t in res.residuals.items()],
        )
    return res, session


@app.get("/", include_in_schema=False)
async def demo() -> FileResponse:
    """The demo console.

    Served from the same process as the API so a reviewer needs one URL and no
    build step: paste text, see it compressed, see the audit that proves what
    survived.
    """
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.post("/v1/compress", response_model=CompressOut)
async def compress(body: CompressIn) -> CompressOut:
    global _REQUESTS
    if not (body.text or body.documents or body.messages or body.system or body.instruction):
        raise HTTPException(status_code=400, detail="empty request: nothing to compress")
    async with _SEM:
        res, session = await run_in_threadpool(_run, body)
    _REQUESTS += 1
    return _to_out(res, session)


@app.post("/v1/compress/stream")
async def compress_stream(body: CompressIn) -> StreamingResponse:
    """Newline-delimited JSON: progress events, then the final result.

    Compression is not incremental (the optimiser needs the whole IR), so the
    stream carries *stage* events rather than partial text.  That is honest and
    still useful: clients get progress and per-pass timings for long documents.
    """

    async def gen() -> AsyncIterator[bytes]:
        t0 = time.perf_counter()
        yield _event({"event": "start", "bytes": len(body.text or "")})
        async with _SEM:
            res, session = await run_in_threadpool(_run, body)
        for s in res.stats:
            yield _event(
                {
                    "event": "pass", "name": s.name, "ms": round(s.ms, 3),
                    "tokens_in": s.tokens_in, "tokens_out": s.tokens_out, "note": s.note,
                }
            )
        out = _to_out(res, session)
        yield _event({"event": "result", **json.loads(out.model_dump_json())})
        yield _event({"event": "done", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/v1/expand", response_model=ExpandOut)
async def expand(body: ExpandIn) -> ExpandOut:
    spans = DEFAULT_STORE.expand(body.session, body.handles)
    missing = [h for h in body.handles if h not in spans]
    return ExpandOut(spans=spans, missing=missing)


@app.post("/v1/estimate", response_model=EstimateOut)
async def estimate(body: EstimateIn) -> EstimateOut:
    text = body.text or "\n".join(body.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="empty request")
    tok = get_tokenizer(body.tokenizer or "auto")
    n = await run_in_threadpool(tok.count, text)
    label, scores = await run_in_threadpool(detect, text)
    projected = {
        m.value: int(n * (1.0 - _target(m)))
        for m in (Mode.CONSERVATIVE, Mode.BALANCED, Mode.AGGRESSIVE, Mode.EXTREME)
    }
    return EstimateOut(
        tokens=n,
        chars=len(text),
        doctype=label,
        doctype_scores={k: round(v, 4) for k, v in scores.items()},
        projected=projected,
    )


def _target(mode: Mode) -> float:
    from ..config import MODE_TARGETS

    return MODE_TARGETS[mode]


@app.get("/v1/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(
        status="ok",
        version=__version__,
        tokenizer=_ENGINE.tok.name,
        uptime_seconds=round(time.time() - _STARTED, 2),
        requests=_REQUESTS,
        cache=DEFAULT_CACHE.stats(),
        residuals=DEFAULT_STORE.stats(),
    )


@app.get("/v1/config")
async def config() -> JSONResponse:
    return JSONResponse(_BASE_CFG.to_dict())


@app.exception_handler(Exception)
async def unhandled(request: Any, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def _event(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
