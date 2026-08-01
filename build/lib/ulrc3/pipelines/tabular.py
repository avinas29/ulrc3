"""Tabular (markdown/CSV/TSV) and SQL pipeline.

Tables compress as **schema + column profile + representative rows**:

* the header row is LOCKED (column names are the API of the data);
* a deterministic column profile (cardinality, range, modal values) is emitted
  once and carries the aggregate information that sampling would lose;
* individual rows compete for budget like any other unit, so the rows that
  matter for the query survive and the rest are summarised by the profile.

SQL keeps DDL verbatim (schemas are contracts), collapses repeated DML, and
protects every table/column identifier via the obligation system.
"""

from __future__ import annotations

import re

from ..text.segment import iter_lines
from ..types import EdgeKind, Level, Protection, UnitKind
from .base import BuildContext, Pipeline, register

_MD_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_MD_SEP = re.compile(r"^\s*\|?[-:| ]{5,}\|?\s*$")
_NUMERIC = re.compile(r"^[+-]?[\d,]*\.?\d+([eE][+-]?\d+)?$")
_MAX_PROFILE_VALUES = 6


@register("table")
class TablePipeline(Pipeline):
    name = "table"
    order_sensitive = False
    allow_intra_unit = False

    def build(self, ctx: BuildContext) -> None:
        base = ctx.region.start
        text = ctx.doc.text[base : ctx.region.end]
        lines = [(s, e, ln) for s, e, ln in iter_lines(text) if ln.strip()]
        if not lines:
            return
        delim = self._delimiter(lines)
        rows: list[tuple[int, int, list[str]]] = []
        for s, e, ln in lines:
            if _MD_SEP.match(ln):
                continue
            cells = self._cells(ln, delim)
            if cells:
                rows.append((s, e, cells))
        if len(rows) < 2:
            for s, e, _ln in lines:
                u = ctx.emit(UnitKind.TABLE_ROW, base + s, base + e)
                if u is not None:
                    _single_level(ctx, u)
            return

        hs, he, header = rows[0]
        hu = ctx.emit(UnitKind.TABLE_ROW, base + hs, base + he, protection=Protection.LOCKED,
                      meta={"table_header": True, "columns": header})
        if hu is None:
            return
        _single_level(ctx, hu)
        hu.symbols |= {c.strip() for c in header if c.strip()}

        body = rows[1:]
        profile = self._profile(header, body)
        pu = ctx.emit(
            UnitKind.TABLE_ROW,
            base + body[0][0],
            base + body[-1][1],
            text=profile,
            protection=Protection.ANCHORED,
            parent=hu.uid,
            meta={"table_profile": True, "derived": True, "rows": len(body)},
        )
        if pu is not None:
            _single_level(ctx, pu)

        for s, e, _cells in body:
            u = ctx.emit(UnitKind.TABLE_ROW, base + s, base + e, parent=hu.uid)
            if u is None:
                continue
            _single_level(ctx, u)
            ctx.cir.add_edge(u.uid, hu.uid, EdgeKind.REQUIRES, 1.0)

    # -- helpers ------------------------------------------------------
    def _delimiter(self, lines: list[tuple[int, int, str]]) -> str:
        sample = "\n".join(ln for _s, _e, ln in lines[:20])
        if sum(1 for _s, _e, ln in lines[:10] if _MD_ROW.match(ln)) >= 2:
            return "|"
        counts = {d: sample.count(d) for d in ("\t", ",", ";", "|")}
        return max(counts, key=lambda k: counts[k]) if any(counts.values()) else ","

    def _cells(self, line: str, delim: str) -> list[str]:
        if delim == "|":
            m = _MD_ROW.match(line)
            body = m.group(1) if m else line
            return [c.strip() for c in body.split("|")]
        return [c.strip().strip('"') for c in line.split(delim)]

    def _profile(self, header: list[str], body: list[tuple[int, int, list[str]]]) -> str:
        cols: list[list[str]] = [[] for _ in header]
        for _s, _e, cells in body:
            for i, c in enumerate(cells[: len(header)]):
                cols[i].append(c)
        parts: list[str] = [f"rows={len(body)}"]
        for name, values in zip(header, cols):
            vals = [v for v in values if v]
            if not vals:
                continue
            uniq = sorted(set(vals))
            if all(_NUMERIC.match(v.replace(",", "")) for v in vals):
                nums = [float(v.replace(",", "")) for v in vals]
                parts.append(f"{name}: num min={_fmt(min(nums))} max={_fmt(max(nums))} n={len(nums)}")
            elif len(uniq) <= _MAX_PROFILE_VALUES:
                parts.append(f"{name}: enum[{','.join(uniq)}]")
            else:
                parts.append(f"{name}: card={len(uniq)} eg={uniq[0]}")
        return "PROFILE " + "; ".join(parts)


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:g}"


@register("sql")
class SqlPipeline(Pipeline):
    name = "sql"
    order_sensitive = True
    allow_intra_unit = False

    _DDL = re.compile(r"^\s*(CREATE|ALTER|DROP|GRANT|COMMENT\s+ON)\b", re.IGNORECASE)
    _STMT_SPLIT = re.compile(r";\s*(?:\n|$)")

    def build(self, ctx: BuildContext) -> None:
        base = ctx.region.start
        text = ctx.doc.text[base : ctx.region.end]
        pos = 0
        seen_hash: dict[str, int] = {}
        for m in self._STMT_SPLIT.finditer(text):
            self._stmt(ctx, base, text, pos, m.end(), seen_hash)
            pos = m.end()
        if pos < len(text) and text[pos:].strip():
            self._stmt(ctx, base, text, pos, len(text), seen_hash)

    def _stmt(self, ctx: BuildContext, base: int, text: str, s: int, e: int, seen: dict[str, int]) -> None:
        body = text[s:e]
        if not body.strip():
            return
        ddl = bool(self._DDL.match(body.strip()))
        u = ctx.emit(
            UnitKind.SQL_STMT,
            base + s,
            base + e,
            code=True,
            protection=Protection.LOCKED if ddl else Protection.ELASTIC,
            meta={"ddl": ddl},
        )
        if u is None:
            return
        _single_level(ctx, u)
        norm = re.sub(r"\s+", " ", re.sub(r"'[^']*'|\b\d+\b", "?", body)).strip().lower()
        prev = seen.get(norm)
        if prev is not None:
            ctx.cir.add_edge(u.uid, prev, EdgeKind.DUPLICATE_OF, 1.0)
            u.meta["duplicate_of"] = prev
            if u.protection < Protection.LOCKED:
                u.protection = Protection.DROPPABLE
        else:
            seen[norm] = u.uid


def _single_level(ctx: BuildContext, unit) -> None:
    txt = unit.text.strip()
    unit.levels = [
        Level("drop", "", 0, 0.0, {}, set()),
        Level(
            "full",
            txt,
            ctx.tok.count(txt) + 1,
            1.0,
            None,
            {k for _c, k, _l, _s, _e in ctx.ex_prose.extract(txt)},
        ),
    ]
    unit.level = 1
    unit.tokens = unit.levels[1].tokens
