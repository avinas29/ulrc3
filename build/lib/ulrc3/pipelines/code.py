"""Code pipeline: compiler-grade context compression.

Selection granularity is the *definition*, and every definition has a fidelity
ladder:

    drop  <  signature  <  signature+doc  <  full body

A caller that survives compression therefore never loses the contract of the
callee: worst case it sees ``def charge(customer_id: str, cents: int) -> Receipt: ...``
instead of the body.  Types, parameter names, defaults, decorators and the
public API surface are structurally preserved -- they live in the *signature*,
which is the lowest non-empty rung of the ladder.

Guarantees enforced here (checked by the verifier, not assumed):

* imports required by retained code are retained (dependency closure);
* a retained method retains its class header (containment closure);
* duplicate implementations collapse to one, the rest become references;
* the emitted Python re-parses with ``ast.parse`` -- otherwise we roll back.
"""

from __future__ import annotations

import re

from ..codeir.generic import (
    LICENSE_HINT,
    detect_language,
    generic_stub,
    is_boilerplate_comment,
    parse_generic,
)
from ..codeir.pyast import (
    ModuleIR,
    SymbolDef,
    build_dependency_graph,
    header_text,
    parse_module,
    signature_stub,
)
from ..types import EdgeKind, Level, Protection, Unit, UnitKind
from .base import BuildContext, Pipeline, register

_CLASS_SPLIT_TOKENS = 220  #: above this, a class is decomposed into methods


@register("code")
class CodePipeline(Pipeline):
    name = "code"
    order_sensitive = True  #: code order is semantic (definition before use)
    allow_intra_unit = False  #: never delete tokens inside a code body

    def build(self, ctx: BuildContext) -> None:
        src_full = ctx.doc.text
        base = ctx.region.start
        src = src_full[base : ctx.region.end]
        lang = detect_language(src, ctx.region.lang or ctx.doc.meta.get("lang"))

        mod: ModuleIR
        if lang == "python":
            mod = parse_module(src)
            if not mod.parse_ok:
                mod = parse_generic(src, "python")
                lang = "python"
        elif lang == "unknown":
            return self._delegate_to_prose(ctx)
        else:
            mod = parse_generic(src, lang)

        # A code front-end that finds no code has misclaimed the region.  Line
        # heuristics happily label an API-reference table "code"; handing that
        # to the prose pipeline (which understands tables and headings) beats
        # emitting one opaque 150-token blob with no fidelity ladder.
        if not mod.symbols and not mod.imports and not _has_code_shape(src):
            return self._delegate_to_prose(ctx)

        ctx.doc.meta.setdefault("langs", set()).add(lang)
        sym_units: dict[str, int] = {}

        # ---------------- imports -------------------------------------
        import_units: list[tuple[Unit, set[str]]] = []
        for _text, s, e, names in mod.imports:
            u = ctx.emit(UnitKind.CODE_IMPORT, base + s, base + e, code=True,
                         protection=Protection.ELASTIC)
            if u is None:
                continue
            u.meta["import_names"] = names
            u.symbols |= names
            self._levels(ctx, u, [("full", u.text, 1.0)])
            import_units.append((u, names))

        # ---------------- definitions ---------------------------------
        tops = [s for s in mod.order() if s.parent is None]
        children: dict[str, list[SymbolDef]] = {}
        for s in mod.symbols.values():
            if s.parent:
                children.setdefault(s.parent, []).append(s)

        for sym in tops:
            kids = children.get(sym.qualname, [])
            uid = self._emit_symbol(ctx, base, src, mod, sym, kids, lang)
            if uid is None:
                continue
            sym_units[sym.qualname] = uid
            # Emit methods as separate units *only* when the class was actually
            # decomposed.  Otherwise the class unit already covers them, and a
            # class rendered at its `sig` rung would be followed by its own
            # methods -- `class C: ...` then an indented block, which does not
            # parse.  Caught by the syntax verifier on a real document.
            if not ctx.cir.units[uid].meta.get("is_class_header"):
                continue
            for child in sorted(kids, key=lambda c: c.start):
                cid = self._emit_symbol(ctx, base, src, mod, child, [], lang, parent_uid=uid)
                if cid is not None:
                    sym_units[child.qualname] = cid

        # ---------------- module-level statements ---------------------
        for s, e, side in mod.module_stmts:
            body = src[s:e].strip()
            is_doc = body.startswith(('"""', "'''", '"', "'")) and len(body) > 40
            u = ctx.emit(
                UnitKind.CODE_DOCSTRING if is_doc else UnitKind.CODE_STMT,
                base + s,
                base + e,
                code=not is_doc,
                protection=Protection.ELASTIC if is_doc else (
                    Protection.ANCHORED if side else Protection.ELASTIC
                ),
            )
            if u is None:
                continue
            if is_doc:
                # a module docstring is prose: give it a fidelity ladder
                # (first line / first paragraph / full) instead of paying for
                # 400 tokens of narrative at every ratio
                head = _doc_head(u.text)
                para = _doc_para(u.text)
                self._levels(
                    ctx, u,
                    [("sig", head, 0.3), ("stub", para, 0.6), ("full", u.text, 1.0)],
                    code=False,
                )
            else:
                self._levels(ctx, u, [("full", u.text, 1.0)])

        # ---------------- free-floating comments ----------------------
        covered = [(s.start, s.end) for s in mod.symbols.values()]
        for cs, ce, body in mod.comments:
            if any(a <= cs and ce <= b for a, b in covered):
                continue
            u = ctx.emit(UnitKind.CODE_COMMENT, base + cs, base + ce, code=True)
            if u is None:
                continue
            stripped = re.sub(r"[#/*\-=_~ \t]", "", body)
            if not stripped:
                # decorative rules ("# ----------") carry zero information
                u.protection = Protection.DROPPABLE
                u.features["boilerplate"] = 1.0
                u.salience = 0.0
            elif is_boilerplate_comment(body) or LICENSE_HINT.search(body):
                u.protection = Protection.DROPPABLE
                u.features["boilerplate"] = 1.0
            self._levels(ctx, u, [("full", u.text, 1.0)])

        # ---------------- dependency + duplicate edges ----------------
        self._link(ctx, mod, sym_units, import_units)

    # -- symbol emission ----------------------------------------------
    def _emit_symbol(
        self,
        ctx: BuildContext,
        base: int,
        src: str,
        mod: ModuleIR,
        sym: SymbolDef,
        kids: list[SymbolDef],
        lang: str,
        parent_uid: int | None = None,
    ) -> int | None:
        start, end = sym.start, sym.end
        split = bool(kids) and ctx.tok.count(src[start:end]) > _CLASS_SPLIT_TOKENS
        if split:
            first_child = min(k.start for k in kids)
            end = first_child
        u = ctx.emit(
            UnitKind.CODE_DEF,
            base + start,
            base + end,
            code=True,
            parent=parent_uid,
            protection=Protection.ELASTIC,
            meta={
                "qualname": sym.qualname,
                "symbol": sym.name,
                "kind": sym.kind,
                "signature": sym.signature,
                "complexity": sym.complexity,
                "body_hash": sym.body_hash,
                "exported": sym.exported,
                "lang": lang,
                "is_class_header": split,
            },
        )
        if u is None:
            return None
        u.symbols.add(sym.name)
        u.symbols.add(sym.qualname)
        if sym.exported:
            u.protection = max(u.protection, Protection.ANCHORED)

        doc_line = ""
        if sym.docstring:
            doc_line = sym.docstring.strip().splitlines()[0].strip().strip('"\'/*# ')
        stub = signature_stub(sym, src=src) if lang == "python" else generic_stub(sym, lang)
        if lang == "python":
            # `header` is a *verbatim source slice*, so it already carries its
            # own indentation -- re-indenting it stacked 4 more spaces onto
            # every method and the class body stopped parsing.
            header = header_text(sym, src)
            pad = _LEADING_WS.match(header).group(0)
            if kids and not split:
                # **API-surface rung.**  A class that is not decomposed must
                # still expose its methods: `class Ledger: ...` is parseable but
                # throws away the whole contract, and measured at 28% answer-
                # ability on the code suite.  Emitting the class header plus
                # every method's verbatim signature keeps the contract *and*
                # parses.
                sig_only = _api_surface(sym, kids, src, pad)
            elif split or "\n" in header:
                # A split class header must NOT be rendered as `class C: ...` --
                # the following (retained) methods would be an unexpected indent.
                sig_only = f"{header}\n{pad}    ..."
            else:
                sig_only = f"{header} ..."
            if doc_line:
                if kids and not split:
                    stub = _api_surface(sym, kids, src, pad, doc_line=doc_line)
                else:
                    stub = f'{header}\n{pad}    """{doc_line}"""\n{pad}    ...'
        else:
            sig_only = _first_line(stub) + " ..."
        levels: list[tuple[str, str, float]] = []
        if sym.kind == "assign":
            # a module constant has no meaningful stub: it *is* its value
            self._levels(ctx, u, [("full", u.text, 1.0)])
            return u.uid
        if lang == "python":
            levels.append(("sig", sig_only, 0.35))
            if doc_line:
                levels.append(("stub", stub, 0.6))
        else:
            levels.append(("sig", _indent_like(src, start, sig_only), 0.35))
            if doc_line:
                levels.append(("stub", _indent_like(src, start, stub), 0.6))
        if not split:
            levels.append(("full", u.text, 1.0))
        else:
            # a class header's "full" is the header plus its docstring; the
            # methods are separate units, composed back at render time
            levels.append(("full", u.text.rstrip() + "\n", 1.0))
        self._levels(ctx, u, levels)
        return u.uid

    def _levels(
        self,
        ctx: BuildContext,
        unit: Unit,
        specs: list[tuple[str, str, float]],
        code: bool = True,
    ) -> None:
        """Attach the fidelity ladder, computing cost and obligations per rung.

        ``code`` must match the extractor the *unit* was annotated with.  The
        audit compares the unit's obligation set against its chosen rung's; if
        the two are produced by different extractors the sets are incomparable,
        and a docstring's constraints become permanently unsatisfiable (measured
        on this repository's own ``engine.py``).
        """
        levels: list[Level] = [Level("drop", "", 0, 0.0, {}, set())]
        seen: set[str] = set()
        for name, text, fidelity in specs:
            text = text.rstrip()
            if not text or text in seen:
                continue
            seen.add(text)
            ex = ctx.ex_code if code else ctx.ex_prose
            obs = {k for _c, k, _l, _s, _e in ex.extract(text)}
            levels.append(
                Level(
                    name=name,
                    text=text,
                    tokens=ctx.tok.count(text) + 1,
                    fidelity=fidelity,
                    concepts=None,
                    obligations=obs,
                )
            )
        # monotone cost: a higher rung must not be cheaper than a lower one
        levels = [levels[0]] + sorted(levels[1:], key=lambda lv: lv.tokens)
        unit.levels = levels
        unit.level = len(levels) - 1
        unit.tokens = levels[-1].tokens

    # -- edges ---------------------------------------------------------
    def _link(
        self,
        ctx: BuildContext,
        mod: ModuleIR,
        sym_units: dict[str, int],
        import_units: list[tuple[Unit, set[str]]],
    ) -> None:
        graph = build_dependency_graph(mod)
        for qual, deps in graph.items():
            src_uid = sym_units.get(qual)
            if src_uid is None:
                continue
            for dep in deps:
                dst_uid = sym_units.get(dep)
                if dst_uid is not None and dst_uid != src_uid:
                    ctx.cir.add_edge(src_uid, dst_uid, EdgeKind.REQUIRES, 1.0)

        # import pruning as a *closure* problem, not a heuristic: a definition
        # requires exactly the imports whose bound names it references
        for qual, uid in sym_units.items():
            sym = mod.symbols.get(qual)
            if sym is None:
                continue
            refs = {r.split(".")[0] for r in sym.refs} | sym.refs
            for iu, names in import_units:
                if names & refs:
                    ctx.cir.add_edge(uid, iu.uid, EdgeKind.REQUIRES, 1.0)

        # duplicate implementations: keep one, reference the rest
        by_hash: dict[str, list[str]] = {}
        for qual, sym in mod.symbols.items():
            if sym.body_hash and sym.complexity >= 2:
                by_hash.setdefault(sym.body_hash, []).append(qual)
        for _h, quals in by_hash.items():
            if len(quals) < 2:
                continue
            quals.sort()
            canon_uid = sym_units.get(quals[0])
            if canon_uid is None:
                continue
            for other in quals[1:]:
                ouid = sym_units.get(other)
                if ouid is None:
                    continue
                ctx.cir.add_edge(ouid, canon_uid, EdgeKind.DUPLICATE_OF, 1.0)
                ou = ctx.cir.units[ouid]
                ou.meta["duplicate_of"] = quals[0]
                # cap the duplicate at signature level: its body is redundant
                if len(ou.levels) > 2:
                    ou.levels = ou.levels[:2]
                    ou.level = 1
                    ou.levels[1] = Level(
                        name="ref",
                        text=f"{ou.levels[1].text.rstrip().rstrip('.').rstrip()}  # == {quals[0]}",
                        tokens=ou.levels[1].tokens + 5,
                        fidelity=0.4,
                        obligations=ou.levels[1].obligations,
                    )

    # -- fallback ------------------------------------------------------
    def _delegate_to_prose(self, ctx: BuildContext) -> None:
        from .prose import ProsePipeline

        ProsePipeline().build(ctx)

    def _fallback_lines(self, ctx: BuildContext, base: int, src: str) -> None:
        """Unknown language: line groups, comments detected lexically."""
        from ..text.segment import iter_lines

        buf: list[tuple[int, int]] = []
        for s, e, line in iter_lines(src):
            if not line.strip():
                if buf:
                    self._flush(ctx, base, buf)
                    buf = []
                continue
            buf.append((s, e))
            if len(buf) >= 25:
                self._flush(ctx, base, buf)
                buf = []
        if buf:
            self._flush(ctx, base, buf)

    def _flush(self, ctx: BuildContext, base: int, buf: list[tuple[int, int]]) -> None:
        u = ctx.emit(UnitKind.CODE_STMT, base + buf[0][0], base + buf[-1][1], code=True)
        if u is not None:
            self._levels(ctx, u, [("full", u.text, 1.0)])


_CODE_SHAPE = re.compile(
    r"(?m)^\s*(?:def |class |function |func |fn |return\b|if \(|for \(|while \(|"
    r"import |from \w[\w.]* import |#include|package |use )|[;{}]\s*$|=>|::"
)


def _api_surface(
    sym: SymbolDef,
    kids: list[SymbolDef],
    src: str,
    pad: str,
    doc_line: str = "",
) -> str:
    """Class header + every method's verbatim signature, bodies elided.

    This is the code pipeline's most valuable rendering: it is typically 8-12%
    of the class's tokens and preserves 100% of its public contract -- names,
    parameters, defaults, type annotations and decorators, byte-identical.
    """
    lines = [header_text(sym, src)]
    if doc_line:
        lines.append(f'{pad}    """{doc_line}"""')
    body = 0
    for kid in sorted(kids, key=lambda k: k.start):
        head = header_text(kid, src).rstrip()
        if not head:
            continue
        lines.append(f"{head} ..." if "\n" not in head else f"{head}\n{pad}        ...")
        body += 1
    if body == 0:
        lines.append(f"{pad}    ...")
    return "\n".join(lines)


def _has_code_shape(src: str) -> bool:
    return len(_CODE_SHAPE.findall(src)) >= 3


def _first_line(s: str) -> str:
    for line in s.splitlines():
        if line.strip():
            return line.rstrip()
    return s.strip()


def _strip_quotes(s: str) -> str:
    s = s.strip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            s = s[3:]
            if s.endswith(q):
                s = s[:-3]
            return s.strip()
    return s.strip("\"'").strip()


def _doc_head(text: str) -> str:
    body = _strip_quotes(text)
    first = _first_line(body)
    return f'"""{first}"""' if first else ""


def _doc_para(text: str) -> str:
    body = _strip_quotes(text)
    para = body.split("\n\n", 1)[0].strip()
    return f'"""{para}"""' if para else ""


_LEADING_WS = re.compile(r"[ \t]*")


def _indent_like(src: str, pos: int, text: str) -> str:
    """Re-indent a generated stub to the column of the original definition.

    Subtle but load-bearing: a definition's span *starts at the beginning of its
    line* (so that decorators and indentation are inside the span), which means
    the indentation is at ``pos``, not before it.  Getting this wrong emits
    ``def method(...)`` at column 0 inside a class body and the result does not
    parse -- caught by the syntax verifier, but better not to produce it.
    """
    line_start = src.rfind("\n", 0, pos) + 1
    pad = _LEADING_WS.match(src, line_start).group(0)
    if not pad:
        return text
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.splitlines())
