"""Language-agnostic code front-end (JS/TS, Java, Go, Rust, C/C++, C#, PHP...).

A full parser per language is not shippable; a regex-only approach is wrong on
every non-trivial file.  We take the middle path that compilers themselves use
for fast passes: a **lexical scanner** that is string/comment/char-literal aware
and tracks brace depth exactly, plus per-language declaration patterns applied
*only at block headers*.  That gives correct block boundaries (the hard part)
with a per-language cost of one regex (the easy part).

Output is the same ``ModuleIR`` shape as the Python front-end, so the code
pipeline is language-independent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..text.lexicon import CODE_KEYWORDS
from .pyast import ModuleIR, SymbolDef

LANG_ALIASES = {
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "py": "python", "python3": "python",
    "rb": "ruby", "rs": "rust", "kt": "kotlin", "cs": "csharp",
    "c++": "cpp", "cc": "cpp", "hpp": "cpp", "h": "c",
    "yml": "yaml", "sh": "bash", "shell": "bash", "zsh": "bash",
}

#: Declaration headers.  Anchored at a block header line, so false positives in
#: expression position are impossible.
DECL_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "javascript": [
        ("function", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)")),
        ("class", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
        ("method", re.compile(r"^\s*(?:public|private|protected|static|async|get|set|\s)*([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?:\{)?\s*$")),
        ("interface", re.compile(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)")),
    ],
    "java": [
        ("class", re.compile(r"^\s*(?:public|private|protected|final|abstract|static|\s)*(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)")),
        ("method", re.compile(r"^\s*(?:public|private|protected|static|final|synchronized|abstract|native|default|\s)*[\w<>\[\],.\s?]+\s+([A-Za-z_$][\w$]*)\s*\([^;)]*\)\s*(?:throws [\w,.\s]+)?(?:\{)?\s*$")),
    ],
    "go": [
        ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)")),
        ("type", re.compile(r"^\s*type\s+([A-Za-z_][\w]*)\s+(?:struct|interface|func|map|\[)")),
    ],
    "rust": [
        ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+([A-Za-z_][\w]*)")),
        ("type", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+([A-Za-z_][\w]*)")),
        ("impl", re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?([A-Za-z_][\w]*)")),
    ],
    "c": [
        ("function", re.compile(r"^\s*(?:static\s+|inline\s+|extern\s+)*[\w*\s]+\s+\*?([A-Za-z_][\w]*)\s*\([^;]*\)\s*(?:\{)?\s*$")),
        ("type", re.compile(r"^\s*typedef\s+(?:struct|union|enum)?\s*[\w\s]*\{?")),
        ("struct", re.compile(r"^\s*(?:struct|union|enum)\s+([A-Za-z_][\w]*)")),
    ],
    "csharp": [
        ("class", re.compile(r"^\s*(?:public|private|protected|internal|sealed|static|abstract|partial|\s)*(?:class|interface|struct|record|enum)\s+([A-Za-z_][\w]*)")),
        ("method", re.compile(r"^\s*(?:public|private|protected|internal|static|virtual|override|async|\s)*[\w<>\[\],.\s?]+\s+([A-Za-z_][\w]*)\s*\([^;)]*\)\s*(?:\{)?\s*$")),
    ],
    "php": [
        ("function", re.compile(r"^\s*(?:public|private|protected|static|final|abstract|\s)*function\s+([A-Za-z_][\w]*)")),
        ("class", re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait)\s+([A-Za-z_][\w]*)")),
    ],
    "ruby": [
        ("function", re.compile(r"^\s*def\s+(?:self\.)?([A-Za-z_][\w!?]*)")),
        ("class", re.compile(r"^\s*(?:class|module)\s+([A-Za-z_][\w:]*)")),
    ],
}
DECL_PATTERNS["typescript"] = DECL_PATTERNS["javascript"]
DECL_PATTERNS["cpp"] = DECL_PATTERNS["c"] + DECL_PATTERNS["csharp"][:1]
DECL_PATTERNS["kotlin"] = [
    ("function", re.compile(r"^\s*(?:public|private|internal|protected|open|override|suspend|inline|\s)*fun\s+(?:<[^>]*>\s*)?([A-Za-z_][\w]*)")),
    ("class", re.compile(r"^\s*(?:public|private|internal|open|abstract|sealed|data|\s)*(?:class|object|interface)\s+([A-Za-z_][\w]*)")),
]

IMPORT_PATTERNS = re.compile(
    r"^\s*(?:import\s+[^\n;]+;?|from\s+['\"][^'\"]+['\"]\s+import[^\n]*|"
    r"#include\s*[<\"][^>\"]+[>\"]|"
    r"(?:const|let|var)\s+[\w{},\s*]+\s*=\s*require\([^)]*\);?|"
    r"use\s+[\w:{}, *]+;|package\s+[\w.]+;?|using\s+[\w.]+;|"
    r"require(?:_relative)?\s+['\"][^'\"]+['\"])\s*$",
    re.MULTILINE,
)

BLOCK_COMMENT = re.compile(r"/\*.*?\*/|<!--.*?-->|\"\"\".*?\"\"\"|'''.*?'''", re.DOTALL)
LINE_COMMENT = re.compile(r"(?m)^[ \t]*(?://|#(?!!)|--|;)[^\n]*$|(?<=\S)[ \t]+(?://|#(?!!))[^\n]*$")

LICENSE_HINT = re.compile(
    r"copyright|licen[cs]e|all rights reserved|spdx|apache|mit licen|gpl|redistribution|"
    r"@author|@since|@version|generated by|do not edit|eslint-disable|prettier-ignore|noqa",
    re.IGNORECASE,
)


def detect_language(text: str, hint: str | None = None) -> str:
    if hint:
        h = hint.lower().strip(". ")
        return LANG_ALIASES.get(h, h)
    scores: dict[str, float] = {}
    words = set(re.findall(r"[A-Za-z_#][\w]*", text[:20000]))
    for lang, kws in CODE_KEYWORDS.items():
        scores[lang] = float(len(words & kws))
    if re.search(r"^\s*(?:func\s+\w|package\s+main)", text, re.MULTILINE):
        scores["go"] = scores.get("go", 0) + 5
    if re.search(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|::<", text):
        scores["rust"] = scores.get("rust", 0) + 5
    if re.search(r"=>|\bconst\s+\w+\s*=|\bexport\s+(default|const|function)", text):
        scores["javascript"] = scores.get("javascript", 0) + 4
    if re.search(r":\s*(?:string|number|boolean)\b|\binterface\s+\w+\s*\{", text):
        scores["typescript"] = scores.get("typescript", 0) + 5
    if re.search(r"^\s*(?:public|private)\s+(?:static\s+)?(?:void|class)", text, re.MULTILINE):
        scores["java"] = scores.get("java", 0) + 5
    if re.search(r"^\s*def\s+\w+\s*\(|^\s*from\s+[\w.]+\s+import", text, re.MULTILINE):
        scores["python"] = scores.get("python", 0) + 6
    if not scores:
        return "unknown"
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "unknown"


@dataclass
class _Block:
    start: int
    end: int
    header_end: int
    depth: int
    header: str
    open_pos: int = 0


def scan_blocks(src: str, max_blocks: int = 20000) -> list[_Block]:
    """Brace-depth scan that respects strings, chars and comments.

    O(n) single pass.  Returns every ``{...}`` block with its header line.
    """
    blocks: list[_Block] = []
    stack: list[tuple[int, int, int]] = []  # (open_pos, depth, header_start)
    i = 0
    n = len(src)
    # `anchor` is the start of the current header: the position after the most
    # recent statement boundary.  Using the line start instead breaks on
    # single-line nesting (`class A { m(){...} }`), which is common in JS.
    anchor = 0
    while i < n and len(blocks) < max_blocks:
        c = src[i]
        if c == "\n":
            anchor = i + 1
            i += 1
            continue
        if c == ";":
            anchor = i + 1
            i += 1
            continue
        if c in "\"'`":
            q = c
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == q:
                    i += 1
                    break
                if q != "`" and src[i] == "\n":
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n:
            if src[i + 1] == "/":
                j = src.find("\n", i)
                i = n if j < 0 else j
                continue
            if src[i + 1] == "*":
                j = src.find("*/", i + 2)
                i = n if j < 0 else j + 2
                continue
        if c == "#" and (i == 0 or src[i - 1] == "\n"):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "{":
            stack.append((i, len(stack), anchor))
            anchor = i + 1
            i += 1
            continue
        if c == "}":
            if stack:
                open_pos, depth, hstart = stack.pop()
                blocks.append(
                    _Block(
                        start=hstart,
                        end=i + 1,
                        header_end=open_pos + 1,
                        depth=depth,
                        header=src[hstart:open_pos].strip(),
                        open_pos=open_pos,
                    )
                )
            anchor = i + 1
            i += 1
            continue
        i += 1
    blocks.sort(key=lambda b: (b.open_pos, -b.end))
    return blocks


def parse_generic(src: str, lang: str) -> ModuleIR:
    """Build a ModuleIR for a brace or indent language."""
    mod = ModuleIR(source=src)
    patterns = DECL_PATTERNS.get(lang, [])

    for m in IMPORT_PATTERNS.finditer(src):
        names = set(re.findall(r"[A-Za-z_][\w]*", m.group(0)))
        mod.imports.append((m.group(0).strip(), m.start(), m.end(), names))

    blocks = scan_blocks(src)
    claimed: list[tuple[int, int]] = []
    collected: list[SymbolDef] = []
    for b in blocks:
        header_line = b.header.splitlines()[-1] if b.header else ""
        name = None
        kind = "block"
        for k, pat in patterns:
            mm = pat.search(header_line) or (pat.search(b.header) if b.header else None)
            if mm:
                kind = k
                name = mm.group(1) if mm.lastindex else header_line.strip()[:40]
                break
        if not name:
            continue
        if b.depth > 1:
            continue
        body = src[b.start : b.end]
        sym = SymbolDef(
            name=name,
            qualname=name,
            kind="class" if kind in ("class", "interface", "type", "struct", "impl") else "function",
            lineno=src.count("\n", 0, b.start) + 1,
            end_lineno=src.count("\n", 0, b.end) + 1,
            col=0,
            start=b.start,
            end=b.end,
            signature=_clean_header(b.header),
            refs=set(re.findall(r"[A-Za-z_][\w]*", body)),
            exported=not name.startswith("_"),
            complexity=1 + len(re.findall(r"\b(if|for|while|switch|catch|case|match)\b", body)),
        )
        sym.body_hash = hashlib.blake2b(
            re.sub(r"\s+|\b[a-z_][a-z0-9_]{0,2}\b", " ", body).encode(), digest_size=12
        ).hexdigest()
        doc = _leading_comment(src, b.start)
        if doc:
            sym.docstring = doc[0]
            sym.doc_span = (doc[1], doc[2])
            sym.start = min(sym.start, doc[1])
        sym.col = b.open_pos  # reused as the containment anchor
        collected.append(sym)
        claimed.append((sym.start, sym.end))

    # containment -> parent/qualname.  Anchored on the *open brace* position so
    # that single-line nesting resolves correctly.
    collected.sort(key=lambda s: (s.col, -s.end))
    for i, sym in enumerate(collected):
        best: SymbolDef | None = None
        for j in range(i - 1, -1, -1):
            cand = collected[j]
            if cand.col < sym.col and sym.end <= cand.end:
                if best is None or (cand.end - cand.col) < (best.end - best.col):
                    best = cand
                break
        if best is not None and best is not sym:
            sym.parent = best.qualname
            sym.qualname = f"{best.qualname}.{sym.name}"
    for sym in collected:
        if sym.qualname not in mod.symbols:
            mod.symbols[sym.qualname] = sym

    for m in BLOCK_COMMENT.finditer(src):
        mod.comments.append((m.start(), m.end(), m.group(0)))
    for m in LINE_COMMENT.finditer(src):
        mod.comments.append((m.start(), m.end(), m.group(0)))

    covered = _merge(claimed + [(s, e) for _t, s, e, _n in mod.imports])
    pos = 0
    for s, e in covered:
        if s > pos and src[pos:s].strip():
            mod.module_stmts.append((pos, s, True))
        pos = max(pos, e)
    if pos < len(src) and src[pos:].strip():
        mod.module_stmts.append((pos, len(src), True))
    return mod


def _clean_header(h: str) -> str:
    line = h.splitlines()[-1] if h else ""
    return re.sub(r"\s+", " ", line).strip()


def _leading_comment(src: str, start: int) -> tuple[str, int, int] | None:
    """Doc comment immediately above a declaration."""
    upto = src.rfind("\n", 0, start)
    if upto <= 0:
        return None
    head = src[:upto]
    m = re.search(r"(/\*\*.*?\*/|(?:^[ \t]*(?://|#)[^\n]*\n?)+)[ \t]*\Z", head, re.DOTALL | re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip(), m.start(1), m.end(1)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = sorted(spans)
    out: list[list[int]] = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def generic_stub(sym: SymbolDef, lang: str) -> str:
    """Signature-preserving elision for brace languages."""
    header = sym.signature.rstrip("{ ").rstrip()
    if lang in ("python",):
        return f"{header}: ..."
    if lang in ("go", "rust", "javascript", "typescript", "java", "c", "cpp", "csharp", "kotlin", "php"):
        return f"{header} {{ /* ... */ }}"
    return f"{header} ..."


def is_boilerplate_comment(body: str) -> bool:
    return bool(LICENSE_HINT.search(body))
