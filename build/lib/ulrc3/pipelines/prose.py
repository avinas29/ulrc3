"""Prose / markdown / legal / API-documentation front-end.

Unit granularity is the sentence, with three refinements that matter:

* **Heading scoping** -- every unit gets a REQUIRES edge to its enclosing
  heading, so retaining a fact automatically retains the section title that
  disambiguates it.  This alone fixes a large class of "compressed context is
  ambiguous" failures.
* **List/table awareness** -- enumerations are the highest-density factual
  regions in documentation and must not be sentence-split.
* **Definition capture** -- "X means Y", "X refers to Y", "X: Y" create a
  definitional dependency: any unit mentioning X requires the definition of X.
  This is what keeps legal and API text coherent under 80% compression.
"""

from __future__ import annotations

import re

from ..text.lexicon import EXAMPLE_LEAD
from ..text.segment import heading_level, is_list_item, iter_lines, split_blocks, split_sentences
from ..types import EdgeKind, Protection, UnitKind
from .base import BuildContext, Pipeline, register

_DEFINITION = re.compile(
    r"^\s*[\"'`]?(?P<term>[A-Z][\w .\-/]{1,60}?)[\"'`]?\s+"
    r"(?:means|refers?\s+to|is\s+defined\s+as|shall\s+mean|denotes|stands\s+for)\b",
    re.IGNORECASE,
)
_DEF_COLON = re.compile(r"^\s*[-*]?\s*[`\"']?(?P<term>[\w.\-]{2,40})[`\"']?\s*[:=]\s+(?P<body>\S.{4,})$")
_ENDPOINT = re.compile(r"^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/\S*)", re.IGNORECASE)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[-:| ]{5,}\|?\s*$")


@register("prose")
@register("markdown")
class ProsePipeline(Pipeline):
    name = "prose"
    order_sensitive = False
    allow_intra_unit = True
    #: Minimum characters before a paragraph is sentence-split.
    split_threshold = 0

    def build(self, ctx: BuildContext) -> None:
        text = ctx.doc.text
        base_start, base_end = ctx.region.start, ctx.region.end
        region_text = text[base_start:base_end]

        for b_start, b_end in split_blocks(region_text):
            s, e = base_start + b_start, base_start + b_end
            self._block(ctx, s, e)

    # -- internals ----------------------------------------------------
    def _block(self, ctx: BuildContext, s: int, e: int) -> None:
        raw = ctx.doc.text[s:e]
        lines = list(iter_lines(raw))

        # heading?
        if lines:
            lvl = heading_level(lines[0][2])
            if lvl and len(lines) == 1:
                self._heading(ctx, s + lines[0][0], s + lines[0][1], lvl)
                return
            if lvl and len(lines) > 1:
                self._heading(ctx, s + lines[0][0], s + lines[0][1], lvl)
                lines = lines[1:]

        if not lines:
            return

        # table?
        if sum(1 for _a, _b, ln in lines if _TABLE_LINE.match(ln)) >= max(2, len(lines) // 2):
            header = None
            for ls, le, ln in lines:
                if _TABLE_SEP.match(ln):
                    continue
                u = ctx.emit(UnitKind.TABLE_ROW, s + ls, s + le)
                if u is None:
                    continue
                if header is None:
                    header = u.uid
                    u.protection = max(u.protection, Protection.ANCHORED)
                    u.meta["table_header"] = True
                else:
                    ctx.cir.add_edge(u.uid, header, EdgeKind.REQUIRES, 1.0)
            return

        # list?
        list_lines = [t for t in lines if is_list_item(t[2])]
        if len(list_lines) >= max(2, int(len(lines) * 0.6)):
            for ls, le, ln in lines:
                u = ctx.emit(UnitKind.LIST_ITEM, s + ls, s + le)
                if u is not None:
                    self._mark(ctx, u, ln)
            return

        # paragraph -> sentences
        for ss, se in split_sentences(raw[lines[0][0] :]) or [(0, len(raw))]:
            off = lines[0][0]
            u = ctx.emit(UnitKind.SENTENCE, s + off + ss, s + off + se)
            if u is not None:
                self._mark(ctx, u, u.text)

    def _heading(self, ctx: BuildContext, s: int, e: int, level: int) -> None:
        while ctx.heading_stack:
            top = ctx.cir.units[ctx.heading_stack[-1]]
            if top.depth >= level:
                ctx.heading_stack.pop()
            else:
                break
        u = ctx.emit(
            UnitKind.HEADING,
            s,
            e,
            depth=level,
            parent=ctx.heading_stack[-1] if ctx.heading_stack else None,
            protection=Protection.ANCHORED,
        )
        if u is not None:
            ctx.heading_stack.append(u.uid)

    def _mark(self, ctx: BuildContext, unit, line: str) -> None:
        """Attach flavour-independent features used by the scorer."""
        m = _DEFINITION.match(line) or _DEF_COLON.match(line)
        if m:
            term = m.group("term").strip()
            if 2 <= len(term) <= 60:
                unit.meta["defines"] = term
                unit.symbols.add(term)
                unit.protection = max(unit.protection, Protection.ANCHORED)
                unit.features["definition"] = 1.0
        if EXAMPLE_LEAD.match(line):
            unit.features["example"] = 1.0
        if _ENDPOINT.match(line):
            unit.protection = max(unit.protection, Protection.LOCKED)
            unit.features["endpoint"] = 1.0


@register("legal")
class LegalPipeline(ProsePipeline):
    """Legal text: order-sensitive, clause-level, definition-heavy.

    Deviations from prose: numbered clauses become their own units (a clause is
    the citable atom), cross-references create REQUIRES edges, and the
    protection floor is one level higher because deleting a proviso changes the
    meaning of the obligation it modifies.
    """

    name = "legal"
    order_sensitive = True
    allow_intra_unit = False  #: never edit inside a legal clause

    _CLAUSE_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")
    _XREF = re.compile(r"\b(?:section|clause|article|paragraph|schedule|exhibit)\s+([\dIVXLC]+(?:\.\d+)*)\b", re.IGNORECASE)

    def build(self, ctx: BuildContext) -> None:
        super().build(ctx)
        # link cross-references: "subject to Section 4.2" REQUIRES clause 4.2
        by_number: dict[str, int] = {}
        for u in ctx.cir.units_of(ctx.doc.doc_id):
            m = self._CLAUSE_NUM.match(u.text)
            if m:
                by_number.setdefault(m.group(1), u.uid)
                u.meta["clause"] = m.group(1)
                u.protection = max(u.protection, Protection.ANCHORED)
        for u in ctx.cir.units_of(ctx.doc.doc_id):
            for m in self._XREF.finditer(u.text):
                tgt = by_number.get(m.group(1))
                if tgt is not None and tgt != u.uid:
                    ctx.cir.add_edge(u.uid, tgt, EdgeKind.REQUIRES, 1.0)


@register("apidocs")
class ApiDocsPipeline(ProsePipeline):
    """API reference: endpoints, parameters and status codes are LOCKED; the
    surrounding narrative is highly compressible because it is derivable."""

    name = "apidocs"
    order_sensitive = False

    _PARAM_ROW = re.compile(r"^\s*[-*|]?\s*[`\"']?([a-zA-Z_][\w.\[\]]*)[`\"']?\s*[|:(]\s*(\w+)")

    def _mark(self, ctx: BuildContext, unit, line: str) -> None:
        super()._mark(ctx, unit, line)
        if self._PARAM_ROW.match(line) or unit.features.get("endpoint"):
            unit.protection = max(unit.protection, Protection.LOCKED)
            unit.features["api_param"] = 1.0
        if re.search(r"\b[1-5]\d{2}\b\s*[-:( ]", line):
            unit.protection = max(unit.protection, Protection.ANCHORED)
