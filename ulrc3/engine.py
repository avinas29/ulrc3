"""The engine: pass orchestration, repair loop, adaptive control, fallbacks.

Control flow::

    build IR ─► levels ─► dedup ─► phantom-attention ─► salience
             ─► CASCADE select ─► closure ─► order ─► render
             ─► verify ──(missing obligations)──► repair ─► re-render ─┐
             ─► confidence ──(p < target)──► raise budget ─► re-select ┘
             ─► inflation guard ─► result

Two feedback loops, both bounded and both monotone:

* the **repair loop** only adds carriers, so it terminates;
* the **adaptive loop** only raises the budget, so it terminates.

If anything is still wrong at the end (it should not be), the engine falls back
to the verbatim input and reports why.  Silent degradation is not an option in
a component that sits in front of a paid model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from .config import Config, Mode
from .ir.builder import build
from .ir.obligations import ObligationExtractor
from .passes.base import PassContext, PassManager
from .passes.p010_levels import LevelPass
from .passes.p020_dedup import DedupPass
from .passes.p030_attention import AttentionPass
from .passes.p040_salience import SaliencePass
from .passes.p050_select import CascadeSelectPass
from .passes.p060_closure import ClosurePass
from .passes.p070_order import OrderPass
from .passes.p080_render import RenderPass
from .passes.p090_verify import VerifyPass
from .passes.p100_confidence import ConfidencePass
from .request import Request
from .tokenization import get_tokenizer
from .types import CIR, CompressionResult, Verification

MAX_ADAPTIVE_ROUNDS = 3
MAX_REPAIR_ROUNDS = 4


class Compressor:
    """Thread-safe, stateless-per-call compression engine.

    One instance can serve concurrent requests: all mutable state lives in the
    per-call ``PassContext``/``CIR``.  The tokenizer is process-global and
    internally locked.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.tok = get_tokenizer(self.cfg.tokenizer, self.cfg.perf.cache_size)

    # -- public API ----------------------------------------------------
    def compress(self, payload: Any = None, **kwargs: Any) -> CompressionResult:
        t0 = time.perf_counter()
        req = Request.coerce(payload if payload is not None else kwargs.pop("request", ""), **kwargs)
        cfg = self._effective_config(req)

        source = req.source_text()
        tokens_in = self.tok.count(source)
        if not source.strip():
            return CompressionResult(text="", tokens_in=0, tokens_out=0)

        cir = build(req, cfg, self.tok)
        if not cir.units:
            return CompressionResult(
                text=source, tokens_in=tokens_in, tokens_out=tokens_in,
                meta={"reason": "no units"},
            )

        budget = self._budget(cfg, tokens_in)
        ctx = PassContext(cir=cir, cfg=cfg, tok=self.tok, budget=budget)
        ctx.scratch["tokens_in"] = tokens_in
        ctx.scratch["source_text"] = source
        ctx.scratch["term_stats"] = cir.meta.get("stats")
        ctx.scratch["type_entropy"] = cir.meta.get("type_entropy", 0.0)
        ctx.scratch["ex_prose"] = ObligationExtractor(
            enforce_entities=cfg.protection.enforce_entities,
            enforce_identifiers=cfg.protection.enforce_identifiers,
        )

        order_sensitive = bool(cir.meta.get("order_sensitive"))
        prep = PassManager(
            [
                LevelPass(allow_intra_unit=not order_sensitive or cfg.mode != Mode.LOSSLESS),
                DedupPass(),
                AttentionPass(),
                SaliencePass(),
            ]
        )
        prep.run(ctx)

        select = PassManager([CascadeSelectPass(), ClosurePass(), OrderPass(order_sensitive)])
        emit = PassManager([RenderPass(), VerifyPass(), ConfidencePass()])

        rounds = 0
        while True:
            select.run(ctx)
            emit.run(ctx)
            self._repair_loop(ctx, emit)
            conf = ctx.scratch.get("confidence", 1.0)
            if (
                cfg.min_confidence <= 0.0
                or conf >= cfg.min_confidence
                or rounds >= MAX_ADAPTIVE_ROUNDS
                or ctx.budget >= tokens_in
            ):
                break
            rounds += 1
            ctx.budget = min(tokens_in, int(ctx.budget * 1.35) + 32)
            ctx.scratch["adaptive_rounds"] = rounds

        result = self._finish(ctx, source, tokens_in, cir, t0)
        return result

    def compress_text(self, text: str, query: str = "", **kw: Any) -> str:
        return self.compress(Request(text=text, query=query), **kw).text

    def compress_many(
        self, payloads: Sequence[Any], workers: int = 0, **kwargs: Any
    ) -> list[CompressionResult]:
        """Batch API.  Uses processes when the workload justifies the fork cost."""
        from .parallel import map_compress

        return map_compress(self.cfg, list(payloads), workers=workers, **kwargs)

    # -- internals -----------------------------------------------------
    def _effective_config(self, req: Request) -> Config:
        cfg = self.cfg
        if req.mode or req.target_ratio is not None or req.budget_tokens is not None:
            cfg = Config.from_dict(self.cfg.to_dict())
            if req.mode:
                cfg.mode = Mode(req.mode)
            if req.target_ratio is not None:
                cfg.target_ratio = req.target_ratio
            if req.budget_tokens is not None:
                cfg.budget_tokens = req.budget_tokens
        return cfg

    def _budget(self, cfg: Config, tokens_in: int) -> int:
        if cfg.budget_tokens:
            return max(16, int(cfg.budget_tokens))
        return max(16, int(tokens_in * (1.0 - cfg.effective_target)))

    def _repair_loop(self, ctx: PassContext, emit: PassManager) -> None:
        rounds = 0
        while ctx.scratch.pop("needs_rerender", False) and rounds < ctx.cfg.protection.max_repair_rounds:
            rounds += 1
            emit.run(ctx)
        ctx.scratch["repair_rounds"] = rounds

    def _finish(
        self, ctx: PassContext, source: str, tokens_in: int, cir: CIR, t0: float
    ) -> CompressionResult:
        out = ctx.output
        tokens_out = self.tok.count(out)
        v: Verification = ctx.scratch.get("verification") or Verification()

        fallback_reason = ""
        if tokens_out >= tokens_in:
            # never make things worse
            out, tokens_out = source, tokens_in
            fallback_reason = "inflation"
            v.inflation_ok = False
        elif not v.provenance_ok:
            fallback_reason = "provenance"

        residuals: dict[str, str] = {}
        if ctx.cfg.keep_residuals:
            for u in cir.units:
                if u.level == 0 and u.text.strip():
                    residuals[str(u.uid)] = u.text

        res = CompressionResult(
            text=out,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            doctypes={d.doc_id: d.doctype for d in cir.docs.values()},
            verification=v,
            confidence=float(ctx.scratch.get("confidence", 1.0)),
            stats=ctx.stats,
            residuals=residuals,
            meta={
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "units": len(cir.units),
                "units_kept": sum(1 for u in cir.units if u.level > 0),
                "budget": ctx.budget,
                "floor_tokens": ctx.scratch.get("floor_tokens", 0),
                "adaptive_rounds": ctx.scratch.get("adaptive_rounds", 0),
                "repair_rounds": ctx.scratch.get("repair_rounds", 0),
                "confidence_features": ctx.scratch.get("confidence_features", {}),
                "tokenizer": self.tok.name,
                "fallback": fallback_reason,
                "mode": ctx.cfg.mode.value,
                "type_entropy": round(float(cir.meta.get("type_entropy", 0.0)), 3),
                "derived_numerals": ctx.scratch.get("derived_numerals", 0),
                "provenance_violations": ctx.scratch.get("provenance_violations", 0),
            },
        )
        return res


def compress(text: Any, **kwargs: Any) -> CompressionResult:
    """Module-level convenience wrapper."""
    cfg_kwargs = {k: kwargs.pop(k) for k in ("mode", "target_ratio", "budget_tokens") if k in kwargs}
    cfg = Config()
    if "mode" in cfg_kwargs and cfg_kwargs["mode"]:
        cfg.mode = Mode(cfg_kwargs["mode"])
    if cfg_kwargs.get("target_ratio") is not None:
        cfg.target_ratio = cfg_kwargs["target_ratio"]
    if cfg_kwargs.get("budget_tokens") is not None:
        cfg.budget_tokens = cfg_kwargs["budget_tokens"]
    return Compressor(cfg).compress(text, **kwargs)
