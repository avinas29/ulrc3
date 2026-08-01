"""Back-end: Context-IR -> surface bytecode.

The output is deliberately *not* prose.  It is a terse, self-describing program
that a frontier LLM decodes without any fine-tuning, because it is built from
three things models already read fluently: verbatim source text, markdown-ish
section markers, and a symbol table.

Layout::

    #CTX v1 mode=balanced 9840>1180tok keep=87/412 order=salience
    #SYM e1=Acme Corporation Limited; e2=customer_orders_v2
    #SYS   <- frozen system prompt, byte-identical
    #TASK  <- frozen instruction
    #Q     <- frozen query
    #D1 markdown "Billing API"
       ... retained units, original or salience order ...
    #FACT  <- minimal carriers recovered by the verifier
    #CUT 219u 6.1k  expand: 31,44,77

**Entity interning** (``#SYM``) is applied only when it pays for itself:
interning a name costs ``tok(alias) + tok(name) + 2`` once and saves
``count · (tok(name) − tok(alias))``.  We intern iff the difference is positive,
which on real corpora fires for long multiword organisation names and repeated
long paths -- and never for code identifiers, which are left byte-exact because
the model may have to emit them back.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..render.markers import MarkerSet, choose
from ..types import CIR, Protection, Unit, UnitKind
from .base import Pass, PassContext

_FROZEN_SEGMENTS = ("system", "tools", "instruction", "instruction_hard", "query")
_NO_INTERN_KINDS = frozenset(
    {
        UnitKind.CODE_DEF, UnitKind.CODE_IMPORT, UnitKind.CODE_STMT, UnitKind.CODE_COMMENT,
        UnitKind.JSON_NODE, UnitKind.SQL_STMT, UnitKind.LOG_TEMPLATE, UnitKind.TOOL_SCHEMA,
    }
)
_INDENT_SENSITIVE = frozenset(
    {
        UnitKind.CODE_DEF, UnitKind.CODE_STMT, UnitKind.CODE_COMMENT,
        UnitKind.CODE_DOCSTRING, UnitKind.CODE_IMPORT,
    }
)


class RenderPass(Pass):
    name = "render"

    def __init__(self) -> None:
        self._markers: MarkerSet | None = None
        self._interned = 0

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        m = choose(ctx.tok, ctx.cfg.render.marker_style)
        self._markers = m
        rp = ctx.cfg.render
        # On short inputs the framing (header + symbol table + drop notice)
        # costs more than it explains, so it is suppressed below 400 tokens.
        if ctx.scratch.get("tokens_in", 0) < 400:
            rp = replace(rp, emit_header=False, emit_dropped_notice=False)

        aliases = self._intern(ctx) if rp.emit_symbol_table else {}
        parts: list[str] = []

        # --- frozen prologue (order is fixed: system, tools, task, query) --
        for seg, marker in (
            ("system", m.system),
            ("tools", m.system),
            ("instruction_hard", m.task),
            ("instruction", m.task),
            ("query", m.query),
        ):
            body = self._segment_body(cir, seg, aliases, intern=False)
            if body:
                parts.append(f"{marker}\n{body}")

        # --- body documents ------------------------------------------------
        docs = self._body_docs(cir)
        multi = len(docs) > 1
        for i, (doc_id, units) in enumerate(docs, start=1):
            body = self._units_body(units, aliases, ctx)
            if not body:
                continue
            if multi:
                doc = cir.docs.get(doc_id)
                dt = doc.doctype if doc else ""
                title = (doc.meta.get("title") if doc else None) or ""
                head = f"{m.doc}{i} {dt}"
                if title:
                    head += f' "{title}"'
                parts.append(f"{head}\n{body}")
            else:
                parts.append(body)

        # --- recovered facts ------------------------------------------------
        facts = self._segment_body(cir, "facts", aliases, intern=False)
        if facts and ctx.cfg.render.emit_fact_block:
            parts.append(f"{m.facts}\n{facts}")

        header_lines: list[str] = []
        if rp.emit_symbol_table and aliases:
            table = "; ".join(f"{a}={n}" for n, a in aliases.items())
            header_lines.append(f"{m.sym} {table}")
        body_text = "\n".join(p for p in parts if p.strip())

        # --- dropped notice ---------------------------------------------
        tail = ""
        if rp.emit_dropped_notice:
            dropped = [u for u in cir.units if u.level == 0]
            if dropped:
                handles = ",".join(str(u.uid) for u in _top_dropped(dropped, 8))
                ktok = sum(u.tokens for u in dropped)
                tail = f"\n{m.dropped} {len(dropped)}u {_kfmt(ktok)}"
                if rp.emit_handles and handles:
                    tail += f" expand:{handles}"

        out = body_text + tail
        if rp.emit_header:
            kept = sum(1 for u in cir.units if u.level > 0)
            total = len(cir.units)
            tin = ctx.scratch.get("tokens_in", 0)
            tout = ctx.tok.count(out) + 12
            mode = ctx.cfg.mode.value
            order = ctx.scratch.get("order_mode", "original")
            head = (
                f"{self._markers.ctx} v1 {mode} {tin}>{tout}tok "
                f"keep={kept}/{total} order={order}"
            )
            out = head + ("\n" + "\n".join(header_lines) if header_lines else "") + "\n" + out
        elif header_lines:
            out = "\n".join(header_lines) + "\n" + out

        ctx.output = out.strip() + "\n"

    # -- helpers -------------------------------------------------------
    def _body_docs(self, cir: CIR) -> list[tuple[str, list[Unit]]]:
        buckets: dict[str, list[Unit]] = {}
        for u in cir.units:
            if u.level <= 0 or u.segment in _FROZEN_SEGMENTS or u.segment == "facts":
                continue
            buckets.setdefault(u.doc_id, []).append(u)
        order = sorted(
            buckets.items(),
            key=lambda kv: min(u.meta.get("render_rank", (0, u.order)) for u in kv[1]),
        )
        for _k, us in order:
            us.sort(key=lambda u: u.meta.get("render_rank", (0, u.order)))
        return order

    def _segment_body(self, cir: CIR, seg: str, aliases: dict[str, str], intern: bool) -> str:
        us = [u for u in cir.units if u.segment == seg and u.level > 0]
        if not us:
            return ""
        us.sort(key=lambda u: u.order)
        return "\n".join(self._text_of(u, aliases if intern else {}) for u in us).strip()

    def _units_body(self, units: list[Unit], aliases: dict[str, str], ctx: PassContext) -> str:
        lines: list[str] = []
        for u in units:
            txt = self._text_of(u, aliases)
            if not txt:
                continue
            if u.meta.get("is_class_header"):
                has_child = any(
                    v.parent == u.uid for v in ctx.cir.units if v.level > 0 and v.parent == u.uid
                )
                if has_child:
                    txt = re.sub(r"\n[ \t]*\.\.\.[ \t]*$", "", txt)
                elif not re.search(r"\.\.\.[ \t]*$", txt):
                    txt = txt.rstrip() + "\n    ..."
            lines.append(txt)
        return "\n".join(lines).strip()

    def _text_of(self, u: Unit, aliases: dict[str, str]) -> str:
        # Leading whitespace is *syntax* in code -- stripping it turns a method
        # into a module-level def and the block stops parsing.
        txt = u.surface.rstrip() if u.kind in _INDENT_SENSITIVE else u.surface.strip()
        if not txt:
            return ""
        if aliases and u.kind not in _NO_INTERN_KINDS and u.protection < Protection.FROZEN:
            for name, alias in aliases.items():
                if name in txt:
                    txt = txt.replace(name, alias)
        return txt

    # -- entity interning ---------------------------------------------
    def _intern(self, ctx: PassContext) -> dict[str, str]:
        cir = ctx.cir
        rp = ctx.cfg.render
        counts: dict[str, int] = {}
        for u in cir.units:
            if u.level <= 0 or u.kind in _NO_INTERN_KINDS or u.protection >= Protection.FROZEN:
                continue
            surface = u.surface
            for name in u.symbols:
                if " " not in name or len(name) < 8:
                    continue  # only multiword names: identifiers stay byte-exact
                c = surface.count(name)
                if c:
                    counts[name] = counts.get(name, 0) + c

        aliases: dict[str, str] = {}
        idx = 0
        for name, c in sorted(counts.items(), key=lambda kv: -kv[1] * len(kv[0])):
            if c < rp.intern_min_count:
                continue
            name_tok = ctx.tok.count(name)
            if name_tok < rp.intern_min_tokens:
                continue
            idx += 1
            alias = f"e{idx}"
            alias_tok = ctx.tok.count(alias)
            savings = c * (name_tok - alias_tok) - (alias_tok + name_tok + 2)
            if savings <= 0:
                idx -= 1
                continue
            aliases[name] = alias
        self._interned = len(aliases)
        return aliases

    def note(self, ctx: PassContext) -> str:
        return f"markers={self._markers.name if self._markers else '?'} interned={self._interned}"


def _top_dropped(units: list[Unit], n: int) -> list[Unit]:
    return sorted(units, key=lambda u: -u.salience)[:n]


def _kfmt(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)
