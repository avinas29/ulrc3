"""Sentence / clause / block segmentation.

Deliberately rule-based and dependency-free: a 12MB neural sentence splitter is
not worth 0.3 F1 on a task whose errors are absorbed downstream (a mis-split
sentence is still a valid selectable unit).  The splitter is abbreviation-aware,
decimal-aware, URL-aware and code-fence-aware, which is where naive
``text.split('.')`` implementations actually lose information.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

ABBREVIATIONS = frozenset(
    ["mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "eg", "ie", "cf", "al", "inc", "ltd", "co", "corp", "dept", "univ", "approx", "fig", "figs", "no", "nos", "vol", "vols", "pp", "ch", "chap", "sec", "secs", "art", "arts", "para", "paras", "ref", "refs", "ed", "eds", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "u.s", "u.k", "e.g", "i.e", "a.m", "p.m", "ph.d", "m.d", "b.s", "m.s", "d.c"]
)

_SENT_END = re.compile(r"([.!?]+[\"')\]]*)(\s+|$)")
_URLISH = re.compile(r"(https?://|www\.|[\w.-]+@[\w.-]+)")
_NUMERIC_DOT = re.compile(r"\d\.\d")
_LIST_BULLET = re.compile(r"^\s*(?:[-*+•·]|\(?\d+[.)]|\(?[a-z][.)]|#{1,6}\s)", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of sentences.

    Complexity: O(n).  Offsets are preserved exactly so that every unit keeps a
    valid provenance span into the original document.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        if _is_false_break(text, m):
            continue
        seg = text[start:end]
        if seg.strip():
            spans.append((start, end))
        start = m.end()
    if start < n and text[start:].strip():
        spans.append((start, n))
    return spans


def _is_false_break(text: str, m: re.Match) -> bool:
    end = m.end(1)
    dot = m.group(1)
    if dot == "." :
        # decimals: 3.14
        if end < len(text) and _NUMERIC_DOT.search(text[max(0, end - 2) : end + 2] or ""):
            return True
        # abbreviations
        left = text[max(0, end - 20) : end - 1]
        word = re.split(r"[^\w.]", left)[-1].lower().strip(".")
        if word in ABBREVIATIONS:
            return True
        if len(word) == 1 and word.isalpha():  # initials "J. Smith"
            return True
        # urls / emails / file paths
        window = text[max(0, end - 40) : end + 10]
        if _URLISH.search(window):
            tail = text[end : end + 6]
            if tail[:1].isalnum() or tail[:1] in "/":
                return True
        # version numbers / ellipsis handled by the +? repetition
    nxt = text[m.end() : m.end() + 1]
    return bool(nxt and nxt.islower() and dot == ".")


def split_clauses(sentence: str, min_len: int = 40) -> list[tuple[int, int]]:
    """Split a long sentence at coordinating/subordinating boundaries.

    Used only when a sentence exceeds the elasticity threshold: dropping half a
    sentence is legitimate when the half is a parenthetical aside, and clause
    boundaries are where that is safe.
    """
    if len(sentence) < min_len * 2:
        return [(0, len(sentence))]
    marks = [m.start() for m in re.finditer(r"(?:;|,\s+(?:which|who|where|although|though|while|whereas|and|but|so)\b| -- | -- )", sentence)]
    if not marks:
        return [(0, len(sentence))]
    out: list[tuple[int, int]] = []
    prev = 0
    for pos in marks:
        if pos - prev >= min_len:
            out.append((prev, pos))
            prev = pos
    out.append((prev, len(sentence)))
    return out


def split_blocks(text: str) -> list[tuple[int, int]]:
    """Blank-line separated blocks, but never splitting a fenced code block."""
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    pos = 0
    start = 0
    in_fence = False
    buf_has_content = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            buf_has_content = True
            pos += len(line)
            continue
        if not in_fence and not line.strip():
            if buf_has_content:
                spans.append((start, pos))
            pos += len(line)
            start = pos
            buf_has_content = False
            continue
        buf_has_content = buf_has_content or bool(line.strip())
        pos += len(line)
    if buf_has_content and start < len(text):
        spans.append((start, len(text)))
    return spans


def iter_lines(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, line)`` with exact offsets (keepends stripped)."""
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        yield pos, pos + len(stripped), stripped
        pos += len(line)


def is_list_item(line: str) -> bool:
    return bool(_LIST_BULLET.match(line))


def heading_level(line: str) -> int:
    """Markdown / underline / numbered heading level; 0 if not a heading."""
    s = line.strip()
    if not s:
        return 0
    m = re.match(r"^(#{1,6})\s+\S", s)
    if m:
        return len(m.group(1))
    m = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+[A-Z]", s)
    if m and len(s) < 120:
        return min(6, m.group(1).count(".") + 1)
    if len(s) < 90 and s.isupper() and s[-1] not in ".!?":
        return 2
    if len(s) < 90 and re.match(r"^[A-Z][\w \-/&,']+:$", s):
        return 3
    return 0
