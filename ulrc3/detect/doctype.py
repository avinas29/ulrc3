"""Document-type detection and heterogeneous-region segmentation.

Real prompts are *not* one type.  A support ticket contains prose, a JSON
payload, a stack trace and a log excerpt; a RAG context contains markdown docs
and code snippets.  Classifying the whole blob and applying one pipeline is the
single biggest source of quality loss in existing compressors.

We therefore do two things:

* **Per-line evidence**: 40 cheap orthographic/lexical features scored against
  11 content classes.  O(n) in characters, no model.
* **Viterbi smoothing**: labels form a first-order Markov chain with a switch
  penalty, so a single JSON-looking line inside prose does not flip the region.
  Decoding is O(L·|lines|·L) with L = 11 -- microseconds per KB.

The output is a list of typed regions, each dispatched to its own pipeline.
That is the "polyglot compiler front-end" of the architecture.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass

from ..text.lexicon import (
    APIDOC_MARKERS,
    CODE_KEYWORDS,
    LEGAL_MARKERS,
    LOG_LEVEL,
    ROLE_MARKER,
    TIMESTAMP,
)
from ..text.segment import heading_level, is_list_item, iter_lines

LABELS = (
    "prose",
    "markdown",
    "code",
    "json",
    "yaml",
    "logs",
    "conversation",
    "legal",
    "apidocs",
    "sql",
    "table",
)

#: Cost of switching region type between adjacent lines (log-domain).
SWITCH_PENALTY = 3.2
#: Extra stickiness for types with expensive misclassification.
STICKY = {"code": 1.4, "json": 1.8, "logs": 1.0, "table": 1.0}

_FENCE = re.compile(r"^\s*(```|~~~)\s*([A-Za-z0-9_+\-]*)")
_JSONISH = re.compile(r'^\s*[\[{]|^\s*"[^"]+"\s*:|^\s*[}\]],?\s*$')
_YAMLISH = re.compile(r"^\s*[A-Za-z_][\w.\-]*\s*:\s*(?:[^:]|$)|^\s*-\s+\S")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$|^\s*[\w\"'][^,\t]*(?:[,\t][^,\t]*){2,}\s*$")
_SQL_STMT = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH|GRANT|EXPLAIN)\b", re.IGNORECASE
)
_CODE_STRUCT = re.compile(
    r"^\s*(def|class|function|func|fn|impl|struct|interface|enum|type|const|let|var|public|"
    r"private|protected|static|async|export|import|from|package|use|#include|using|module|"
    r"return|if|for|while|switch|try|catch|except|finally|elif|else)\b"
)
_CODE_PUNCT = re.compile(r"[{}();]\s*$|=>|::|->|==|!=|\+=|&&|\|\||\bself\.|\bthis\.")
_STACKTRACE = re.compile(r"^\s*(at\s+[\w$.]+\(|File\s+\"|Traceback|\s+at\s|Caused by:|\w+Error:|\w+Exception)")
_URLLOG = re.compile(r'"(?:GET|POST|PUT|DELETE|HEAD|PATCH)\s+\S+\s+HTTP/\d\.\d"')
_KV_LOG = re.compile(r"\b\w+=(?:\"[^\"]*\"|\S+)(?:\s+\w+=)")

#: Sentinel key in a line's score dict marking an unconditional region
#: boundary.  Never a label -- filtered out before Viterbi.
_HARD_BREAK = "__break__"
#: Sentinel prefix marking a line whose label is *known exactly* (inside a
#: fence).  Viterbi is a smoother for ambiguous evidence; it must not be allowed
#: to override structure it cannot see.  Without this, the `code` label's
#: stickiness carried past a closing ``` and swallowed the following markdown.
_FORCED = "__forced__"


@dataclass
class Region:
    start: int
    end: int
    label: str
    lang: str | None = None
    confidence: float = 0.0

    def __len__(self) -> int:
        return self.end - self.start


def _line_scores(line: str, prev: str, idx: int) -> dict[str, float]:
    s: dict[str, float] = dict.fromkeys(LABELS, 0.0)
    t = line.strip()
    if not t:
        return s
    low = t.lower()
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t)
    wset = {w.lower() for w in words}
    nwords = len(words) or 1
    alpha = sum(c.isalpha() for c in t)
    punct = sum(1 for c in t if c in "{}[]()<>;:=|/\\*&^%$#@!~`+-")
    density = punct / max(1, len(t))
    indent = len(line) - len(line.lstrip())

    # --- prose baseline
    s["prose"] += 1.0
    if t.endswith((".", "!", "?", '."', '.”')) and nwords >= 6:
        s["prose"] += 1.6
    if alpha / max(1, len(t)) > 0.75 and density < 0.06:
        s["prose"] += 1.2

    # --- markdown
    if heading_level(line):
        s["markdown"] += 2.6
    if is_list_item(line):
        s["markdown"] += 1.4
    if re.search(r"\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|^\s*>\s|`[^`]+`", t):
        s["markdown"] += 1.5
    if re.match(r"^\s*\|?[-:| ]{6,}\|?\s*$", t):
        s["table"] += 2.5
        s["markdown"] += 1.0

    # --- code
    if _CODE_STRUCT.match(line):
        s["code"] += 2.4
    if _CODE_PUNCT.search(t):
        s["code"] += 1.7
    if density > 0.12:
        s["code"] += 1.5 * min(2.0, density * 8)
    if indent >= 4 and density > 0.04:
        s["code"] += 0.9
    kw_hits = 0
    for lang, kws in CODE_KEYWORDS.items():
        hits = len(wset & kws)
        if hits:
            kw_hits = max(kw_hits, hits)
            if lang == "sql":
                s["sql"] += 1.2 * hits
            else:
                s["code"] += 0.9 * hits
    if _STACKTRACE.match(line):
        s["logs"] += 2.0
        s["code"] += 0.5

    # --- json / yaml
    if _JSONISH.match(line):
        s["json"] += 2.4
    if t.count('"') >= 2 and ":" in t and (t.endswith(",") or t.endswith("{") or t.endswith("}")):
        s["json"] += 1.6
    if _YAMLISH.match(line) and '"' not in t and not t.endswith((",", "{", "[")):
        s["yaml"] += 1.3
    if re.match(r"^\s*---\s*$", t):
        s["yaml"] += 1.5

    # --- logs
    if TIMESTAMP.search(t[:40]):
        s["logs"] += 2.6
    if LOG_LEVEL.search(t[:60]):
        s["logs"] += 2.2
    if _URLLOG.search(t):
        s["logs"] += 3.0
    if _KV_LOG.search(t):
        s["logs"] += 1.4
    if re.match(r"^\s*\[?\d{4}-\d{2}-\d{2}", t):
        s["logs"] += 1.5

    # --- conversation
    if ROLE_MARKER.match(line):
        s["conversation"] += 3.2
    if re.match(r"^\s*(?:\*\*)?(?:user|assistant|system|human|ai)\b", low):
        s["conversation"] += 1.5

    # --- legal
    lm = len(LEGAL_MARKERS.findall(t))
    if lm:
        s["legal"] += 1.8 * lm
    if re.match(r"^\s*\d+(\.\d+)+\s+[A-Z]", t):
        s["legal"] += 1.0

    # --- api docs
    am = len(APIDOC_MARKERS.findall(t))
    if am:
        s["apidocs"] += 1.5 * am

    # --- sql
    if _SQL_STMT.match(line):
        s["sql"] += 3.0

    # --- table / csv
    if _TABLE_ROW.match(line) and nwords >= 3:
        s["table"] += 1.8
    if t.count("|") >= 2:
        s["table"] += 1.2
    if t.count("\t") >= 2:
        s["table"] += 1.5

    if prev and _JSONISH.match(prev) and _JSONISH.match(line):
        s["json"] += 0.8
    return s


def classify_lines(text: str) -> list[tuple[int, int, dict[str, float], str | None]]:
    """Score every line; fenced blocks are hard-assigned to their language.

    Fence delimiter lines are also marked as *hard breaks* (a 5th tuple slot is
    not used; see :data:`_HARD_BREAK` in the scores dict).  A closing ``` is
    exact structural information -- weighing it against a switch penalty let a
    code region swallow the markdown that followed it, which then failed the
    syntax verifier on any README with a Python fence.
    """
    out: list[tuple[int, int, dict[str, float], str | None]] = []
    fence: str | None = None
    fence_lang: str | None = None
    prev = ""
    for i, (s, e, line) in enumerate(iter_lines(text)):
        m = _FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
                fence_lang = (m.group(2) or "").lower() or None
            else:
                fence = None
                fence_lang = None
            scores = dict.fromkeys(LABELS, 0.0)
            scores["markdown"] = 5.0
            scores[_HARD_BREAK] = 1.0
            scores[_FORCED] = "markdown"
            # the *closing* delimiter must not carry the block's language
            out.append((s, e, scores, None if fence is None else fence_lang))
            prev = line
            continue
        if fence is not None:
            scores = dict.fromkeys(LABELS, 0.0)
            lbl = "code"
            if fence_lang in ("json",):
                lbl = "json"
            elif fence_lang in ("yaml", "yml"):
                lbl = "yaml"
            elif fence_lang in ("sql",):
                lbl = "sql"
            elif fence_lang in ("log", "logs", "text", "txt", "console", "shell", "bash", "sh"):
                lbl = "logs" if fence_lang.startswith("log") else "code"
            scores[lbl] = 6.0
            scores[_FORCED] = lbl
            out.append((s, e, scores, fence_lang))
            prev = line
            continue
        out.append((s, e, _line_scores(line, prev, i), None))
        prev = line
    return out


def _emit(scores: dict, label: str) -> float:
    v = scores.get(label, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def viterbi(scored: list[tuple[int, int, dict[str, float], str | None]]) -> list[str]:
    """First-order HMM decoding with a uniform switch penalty."""
    n = len(scored)
    if n == 0:
        return []
    L = list(LABELS)
    prev_cost = {lbl: -_emit(scored[0][2], lbl) for lbl in L}
    back: list[dict[str, str]] = []
    for i in range(1, n):
        emis = scored[i][2]  # sentinel key is not in LABELS, so it is ignored
        cur: dict[str, float] = {}
        bp: dict[str, str] = {}
        best_prev_lbl = min(prev_cost, key=lambda k: prev_cost[k])
        best_prev = prev_cost[best_prev_lbl]
        for lbl in L:
            stay = prev_cost[lbl]
            switch = best_prev + SWITCH_PENALTY + STICKY.get(best_prev_lbl, 0.0)
            if stay <= switch:
                cur[lbl] = stay - _emit(emis, lbl)
                bp[lbl] = lbl
            else:
                cur[lbl] = switch - _emit(emis, lbl)
                bp[lbl] = best_prev_lbl
        prev_cost = cur
        back.append(bp)
    path = [min(prev_cost, key=lambda k: prev_cost[k])]
    for i in range(len(back) - 1, -1, -1):
        path.append(back[i][path[-1]])
    path.reverse()
    return path


def segment(text: str, min_region: int = 1) -> list[Region]:
    """Split `text` into typed regions.

    Two-tier, and the order matters:

    1. **Fences are exact.**  A ```` ```python ```` block's extent and language
       are known, not inferred, so those regions are emitted directly.  Letting
       Viterbi decide them lets the `code` label's stickiness run past a closing
       fence and swallow the following markdown -- which then fails the syntax
       verifier on any README containing a Python example.
    2. **Everything else is inferred.**  Viterbi runs independently on each
       *unfenced* span, so a decision inside one span cannot leak across a fence
       into another.

    Complexity O(|lines| · |LABELS|), unchanged.
    """
    scored = classify_lines(text)
    if not scored:
        return [Region(0, len(text), "prose", confidence=1.0)]

    # -- tier 1: split the line stream at fence delimiters -------------
    blocks: list[tuple[int, int, str | None]] = []  # (first_line, last_line, forced)
    cur_start = 0
    cur_forced: str | None = _forced_label(scored[0][2])
    for i in range(1, len(scored)):
        f = _forced_label(scored[i][2])
        if f != cur_forced or scored[i][2].get(_HARD_BREAK):
            blocks.append((cur_start, i - 1, cur_forced))
            cur_start = i
            cur_forced = f
    blocks.append((cur_start, len(scored) - 1, cur_forced))

    # -- tier 2: infer labels inside each unfenced block ---------------
    regions: list[Region] = []
    for first, last, forced in blocks:
        window = scored[first : last + 1]
        if not window:
            continue
        if forced is not None:
            lang = next((ln for _s, _e, _sc, ln in window if ln), None)
            regions.append(Region(window[0][0], window[-1][1], forced, lang, 1.0))
            continue
        labels = viterbi(window)
        seg_start = window[0][0]
        cur = labels[0]
        acc = 0.0
        n = 0
        for j, (s_off, _e_off, sc, _ln) in enumerate(window):
            vals = [v for k, v in sc.items() if k in LABELS]
            top = max(vals) if vals else 0.0
            second = sorted(vals)[-2] if len(vals) > 1 else 0.0
            acc += (top - second) / (top + 1e-6) if top > 0 else 0.0
            n += 1
            if labels[j] != cur:
                regions.append(Region(seg_start, window[j - 1][1], cur, None, acc / max(1, n)))
                cur = labels[j]
                seg_start = s_off
                acc = 0.0
                n = 0
        regions.append(Region(seg_start, window[-1][1], cur, None, acc / max(1, n)))

    # -- merge adjacent same-label regions, but never across a fence ---
    merged: list[Region] = []
    fenced_starts = {r.start for r in regions if r.lang or r.confidence >= 1.0}
    for r in regions:
        if (
            merged
            and merged[-1].label == r.label
            and r.start not in fenced_starts
            and merged[-1].start not in fenced_starts
        ):
            merged[-1].end = r.end
            merged[-1].lang = merged[-1].lang or r.lang
        else:
            merged.append(r)
    if merged:
        merged[-1].end = len(text)
    return merged


def _forced_label(scores: dict) -> str | None:
    v = scores.get(_FORCED)
    return v if isinstance(v, str) else None


def detect(text: str, hint: str | None = None) -> tuple[str, dict[str, float]]:
    """Whole-document type with a normalised score distribution."""
    if hint:
        return hint, {hint: 1.0}
    stripped = text.strip()
    if not stripped:
        return "prose", {"prose": 1.0}

    # Exact structural oracles beat any heuristic.  If a real parser accepts
    # the whole document, the type is not a guess -- and we must NOT then split
    # it into regions, because a Python file with a long module docstring looks
    # like prose to every line-level heuristic while being, in fact, Python.
    if stripped[0] in "[{" and stripped[-1] in "]}":
        try:
            json.loads(stripped)
            return "json", {"json": 1.0}
        except Exception:
            pass
    if _looks_like_python(stripped):
        try:
            ast.parse(text)
            return "code", {"code": 1.0}
        except SyntaxError:
            pass

    regions = segment(text)
    weights: dict[str, float] = {}
    for r in regions:
        weights[r.label] = weights.get(r.label, 0.0) + len(r)
    total = sum(weights.values()) or 1.0
    dist = {k: v / total for k, v in weights.items()}
    label = max(dist, key=lambda k: dist[k])

    # A dialogue is a dialogue even when prose wins on volume: turn structure is
    # global, so it cannot be recovered region by region.  Code/logs/json are
    # deliberately NOT promoted this way -- they are handled per region, and
    # promoting them sent whole markdown documents through the code pipeline.
    if label == "prose" and dist.get("conversation", 0.0) >= 0.30:
        label = "conversation"
    if label == "prose" and dist.get("markdown", 0.0) >= 0.25:
        label = "markdown"
    # A short snippet with no code structure is prose, whatever the punctuation
    # density says.  One-sentence RAG chunks were being typed as code.
    if label == "code" and len(text) < 400 and not _CODE_STRUCT.search(text):
        label = "prose"
        dist = {"prose": 1.0}
    return label, dist


_PY_HINT = re.compile(r"(?m)^\s*(?:def |class |import |from \w[\w.]* import |@\w)")


def _looks_like_python(text: str) -> bool:
    """Cheap pre-filter so we only pay for ``ast.parse`` when it can succeed."""
    return len(_PY_HINT.findall(text[:200_000])) >= 2


def entropy(dist: dict[str, float]) -> float:
    """Type ambiguity -- high entropy triggers the mixed/polyglot pipeline."""
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return h
