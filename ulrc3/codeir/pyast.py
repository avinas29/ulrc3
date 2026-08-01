"""Python front-end: AST -> symbol table -> dependency graph -> program slice.

This is where "treat the prompt as source code" stops being a metaphor.

For code we do not score sentences; we run a miniature compiler:

1. **Parse** to an AST (stdlib ``ast``, zero dependencies, exact offsets).
2. **Symbol table**: every module-level and nested definition, its signature,
   its decorators, its exported name.
3. **Def-use / call graph**: which definition references which other symbol.
4. **Slice**: from the roots (entrypoints, symbols named in the query, exported
   API, classes with side effects) compute the backward reachable set.
5. **Elide**: definitions outside the slice degrade to *signature stubs* --
   ``def parse(text: str, *, strict: bool = False) -> Doc: ...`` -- which keeps
   the type contract, the parameter names and the call compatibility while
   deleting the body.  Nothing that a caller could depend on is lost.
6. **Merkle-hash** every function body so that duplicate implementations
   collapse to one plus references.

Every emitted Python region is re-parsed by the verifier; if it does not parse,
the change is rolled back.  Compression that produces uncompilable code is a bug
we can *detect*, and we do.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class SymbolDef:
    name: str
    qualname: str
    kind: str  #: function | class | assign | import | async_function
    lineno: int
    end_lineno: int
    col: int
    start: int  #: char offset
    end: int
    signature: str
    header_span: tuple[int, int] | None = None  #: verbatim `def ...:` slice
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    doc_span: tuple[int, int] | None = None
    body_hash: str = ""
    refs: set[str] = field(default_factory=set)
    exported: bool = True
    parent: str | None = None
    side_effect: bool = False
    complexity: int = 1


@dataclass
class ModuleIR:
    source: str
    symbols: dict[str, SymbolDef] = field(default_factory=dict)
    imports: list[tuple[str, int, int, set[str]]] = field(default_factory=list)
    module_stmts: list[tuple[int, int, bool]] = field(default_factory=list)
    comments: list[tuple[int, int, str]] = field(default_factory=list)
    parse_ok: bool = True
    error: str = ""

    def order(self) -> list[SymbolDef]:
        return sorted(self.symbols.values(), key=lambda s: s.start)


def line_offsets(src: str) -> list[int]:
    offs = [0]
    for line in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def _pos(offs: list[int], lineno: int, col: int) -> int:
    i = max(0, min(len(offs) - 1, lineno - 1))
    return offs[i] + col


class _Collector(ast.NodeVisitor):
    def __init__(self, src: str, offs: list[int]) -> None:
        self.src = src
        self.offs = offs
        self.mod = ModuleIR(source=src)
        self.scope: list[str] = []

    # -- helpers ------------------------------------------------------
    def _span(self, node: ast.AST) -> tuple[int, int]:
        start = _pos(self.offs, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))
        end_lineno = getattr(node, "end_lineno", None) or getattr(node, "lineno", 1)
        end_col = getattr(node, "end_col_offset", None) or 0
        end = _pos(self.offs, end_lineno, end_col)
        # include decorators in the span
        for d in getattr(node, "decorator_list", []) or []:
            ds = _pos(self.offs, d.lineno, d.col_offset)
            line_start = self.src.rfind("\n", 0, ds) + 1
            start = min(start, line_start)
        line_start = self.src.rfind("\n", 0, start) + 1
        if self.src[line_start:start].strip() == "":
            start = line_start
        return start, end

    def _sig(self, node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            args = _unparse_args(node.args)
            ret = f" -> {_safe_unparse(node.returns)}" if node.returns else ""
            return f"{prefix}{node.name}({args}){ret}"
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(_safe_unparse(b) for b in node.bases)
            kws = ", ".join(f"{k.arg}={_safe_unparse(k.value)}" for k in node.keywords if k.arg)
            inner = ", ".join(x for x in (bases, kws) if x)
            return f"class {node.name}({inner})" if inner else f"class {node.name}"
        return ""

    def _refs(self, node: ast.AST) -> set[str]:
        out: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                out.add(child.id)
            elif isinstance(child, ast.Attribute):
                base = child
                parts = []
                while isinstance(base, ast.Attribute):
                    parts.append(base.attr)
                    base = base.value
                if isinstance(base, ast.Name):
                    parts.append(base.id)
                    out.add(".".join(reversed(parts)))
                    out.add(base.id)
        return out

    # -- visitors -----------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self._toplevel(stmt)

    def _toplevel(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            s, e = self._span(stmt)
            names = set()
            for a in stmt.names:
                names.add((a.asname or a.name).split(".")[0])
            self.mod.imports.append((self._text(s, e), s, e, names))
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._definition(stmt)
            return
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            s, e = self._span(stmt)
            name = stmt.targets[0].id
            sym = SymbolDef(
                name=name,
                qualname=name,
                kind="assign",
                lineno=stmt.lineno,
                end_lineno=getattr(stmt, "end_lineno", stmt.lineno),
                col=stmt.col_offset,
                start=s,
                end=e,
                signature=self._text(s, e)[:200],
                refs=self._refs(stmt.value),
                side_effect=_has_call(stmt.value),
            )
            self.mod.symbols[name] = sym
            return
        s, e = self._span(stmt)
        self.mod.module_stmts.append((s, e, _is_side_effecting(stmt)))

    def _definition(self, node: ast.AST, parent: str | None = None) -> None:
        name = getattr(node, "name", "?")
        qual = f"{parent}.{name}" if parent else name
        s, e = self._span(node)
        doc = ast.get_docstring(node, clean=False)  # type: ignore[arg-type]
        doc_span = None
        body = getattr(node, "body", [])
        if doc and body and isinstance(body[0], ast.Expr):
            ds, de = self._span(body[0])
            doc_span = (ds, de)
        kind = (
            "class"
            if isinstance(node, ast.ClassDef)
            else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
        )
        sym = SymbolDef(
            name=name,
            qualname=qual,
            kind=kind,
            lineno=getattr(node, "lineno", 1),
            end_lineno=getattr(node, "end_lineno", 1),
            col=getattr(node, "col_offset", 0),
            start=s,
            end=e,
            signature=self._sig(node),
            decorators=[_safe_unparse(d) for d in getattr(node, "decorator_list", [])],
            docstring=doc,
            doc_span=doc_span,
            refs=self._refs(node),
            exported=not name.startswith("_"),
            parent=parent,
            complexity=_complexity(node),
        )
        sym.header_span = self._header_span(node, s, e)
        sym.body_hash = merkle_hash(node)
        self.mod.symbols[qual] = sym
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self._definition(child, qual)

    def _header_span(self, node: ast.AST, start: int, end: int) -> tuple[int, int] | None:
        """Byte range of the declaration header, up to and including its colon.

        Reconstructing the signature with ``ast.unparse`` is semantically exact
        but *not* byte-identical (``str='USD'`` vs ``str = "USD"``).  Since we
        promise signatures verbatim, we slice the source instead.
        """
        body = getattr(node, "body", None)
        if not body:
            return None
        body_start = _pos(self.offs, body[0].lineno, body[0].col_offset)
        colon = self.src.rfind(":", start, body_start)
        if colon < 0:
            return None
        return (start, colon + 1)

    def _text(self, s: int, e: int) -> str:
        return self.src[s:e]


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - exotic nodes
        return "..."


def _unparse_args(args: ast.arguments) -> str:
    parts: list[str] = []
    posonly = list(getattr(args, "posonlyargs", []) or [])
    normal = list(args.args)
    defaults = list(args.defaults)
    all_pos = posonly + normal
    pad = len(all_pos) - len(defaults)
    for i, a in enumerate(all_pos):
        s = a.arg
        if a.annotation is not None:
            s += f": {_safe_unparse(a.annotation)}"
        if i >= pad:
            s += f"={_safe_unparse(defaults[i - pad])}"
        parts.append(s)
        if posonly and i == len(posonly) - 1:
            parts.append("/")
    if args.vararg:
        v = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            v += f": {_safe_unparse(args.vararg.annotation)}"
        parts.append(v)
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        s = a.arg
        if a.annotation is not None:
            s += f": {_safe_unparse(a.annotation)}"
        if d is not None:
            s += f"={_safe_unparse(d)}"
        parts.append(s)
    if args.kwarg:
        k = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            k += f": {_safe_unparse(args.kwarg.annotation)}"
        parts.append(k)
    return ", ".join(parts)


def _has_call(node: ast.AST | None) -> bool:
    if node is None:
        return False
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def _is_side_effecting(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Expr, ast.If, ast.Try, ast.With, ast.For, ast.While)):
        return True
    return _has_call(stmt)


def _complexity(node: ast.AST) -> int:
    """Cyclomatic-ish complexity: a proxy for how much information a body holds."""
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
        elif isinstance(n, (ast.comprehension,)):
            c += 1
    return c


def merkle_hash(node: ast.AST) -> str:
    """Structural hash: identical logic with renamed locals hashes identically.

    Identifiers that are *bound locally* are alpha-renamed to positional slots
    before hashing, so ``def a(x): return x+1`` and ``def b(y): return y+1``
    collapse -- which is exactly the duplicate-implementation case we want to
    detect in repository-scale contexts.
    """
    parts: list[str] = []

    def walk(n: ast.AST, local: dict[str, str]) -> None:
        parts.append(type(n).__name__)
        if isinstance(n, ast.Name):
            parts.append(local.get(n.id, n.id))
            return
        if isinstance(n, ast.arg):
            parts.append(local.setdefault(n.arg, f"#{len(local)}"))
            return
        if isinstance(n, ast.Constant):
            parts.append(repr(type(n.value).__name__))
            parts.append(repr(n.value)[:32])
            return
        if isinstance(n, ast.Attribute):
            parts.append(n.attr)
        for child in ast.iter_child_nodes(n):
            walk(child, local)

    scope: dict[str, str] = {}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            scope[a.arg] = f"#{len(scope)}"
        for stmt in node.body:
            walk(stmt, scope)
    else:
        walk(node, scope)
    return hashlib.blake2b("|".join(parts).encode(), digest_size=12).hexdigest()


_COMMENT = re.compile(r"(?m)^([ \t]*)#(?!\!)(.*)$|(?<=\S)[ \t]+#(?!\!)(.*)$")


def parse_module(src: str) -> ModuleIR:
    offs = line_offsets(src)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        mod = ModuleIR(source=src, parse_ok=False, error=f"{exc.msg} (line {exc.lineno})")
        return mod
    c = _Collector(src, offs)
    c.visit(tree)
    mod = c.mod
    for m in _COMMENT.finditer(src):
        body = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        mod.comments.append((m.start(), m.end(), body.strip()))
    return mod


def build_dependency_graph(mod: ModuleIR) -> dict[str, set[str]]:
    """symbol -> set of symbols it references (intra-module resolution)."""
    names = {s.name: q for q, s in mod.symbols.items() if "." not in q}
    quals = set(mod.symbols)
    graph: dict[str, set[str]] = {}
    for qual, sym in mod.symbols.items():
        deps: set[str] = set()
        for r in sym.refs:
            if r == sym.name:
                continue
            if r in quals:
                deps.add(r)
            elif r in names and names[r] != qual:
                deps.add(names[r])
            else:
                head = r.split(".")[0]
                if head in names and names[head] != qual:
                    deps.add(names[head])
        if sym.parent:
            deps.add(sym.parent)
        graph[qual] = deps
    return graph


def slice_from(roots: Iterable[str], graph: dict[str, set[str]], max_depth: int = 6) -> set[str]:
    """Forward reachability from roots over the dependency graph (a *program
    slice*).  O(V+E)."""
    out: set[str] = set()
    frontier = [(r, 0) for r in roots if r in graph]
    while frontier:
        node, d = frontier.pop()
        if node in out or d > max_depth:
            continue
        out.add(node)
        for dep in graph.get(node, ()):  # noqa: B007
            if dep not in out:
                frontier.append((dep, d + 1))
    return out


def used_import_names(mod: ModuleIR, kept_spans: list[tuple[int, int]]) -> set[str]:
    """Names actually referenced by the retained code (for import pruning)."""
    kept_src = "".join(mod.source[s:e] for s, e in kept_spans)
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", kept_src))


def header_text(sym: SymbolDef, src: str) -> str:
    """Verbatim declaration header (decorators + signature + colon)."""
    if sym.header_span:
        return src[sym.header_span[0] : sym.header_span[1]].rstrip()
    head = sym.signature or f"{'class' if sym.kind == 'class' else 'def'} {sym.name}"
    decos = "".join(f"@{d}\n" for d in sym.decorators)
    return f"{decos}{head}:"


def signature_stub(sym: SymbolDef, keep_doc_line: bool = True, src: str = "") -> str:
    """Body-elided rendering that preserves the entire public contract."""
    lines: list[str] = []
    if src and sym.header_span:
        lines.append(header_text(sym, src))
    else:
        for d in sym.decorators:
            lines.append(f"@{d}")
        head = sym.signature or f"{'class' if sym.kind == 'class' else 'def'} {sym.name}"
        lines.append(f"{head}:")
    doc = ""
    if keep_doc_line and sym.docstring:
        first = sym.docstring.strip().splitlines()[0].strip().strip('"').strip("'")
        if first:
            doc = f'    """{first}"""'
    if doc:
        lines.append(doc)
    lines.append("    ...")
    return "\n".join(lines)
