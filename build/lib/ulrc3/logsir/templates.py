"""Online log-template mining (a compact, deterministic Drain variant).

Logs are the highest-redundancy context type in existence: 50k lines routinely
carry ~40 distinct *statements*.  Summarising them destroys the one thing they
are for -- exact values and exact counts.  Templating preserves both:

    2024-03-15T10:22:01Z ERROR  payment.worker retry 3/5 for order 88213 (412ms)
    2024-03-15T10:22:04Z ERROR  payment.worker retry 4/5 for order 88213 (521ms)
        ->  ERROR payment.worker retry <N>/<N> for order <N> (<N>ms)   x2
            slots: order={88213}  latency_ms={412,521}   window 10:22:01..10:22:04

Complexity is O(lines) with a fixed-depth prefix tree: each line is masked, the
mask is looked up, and only same-length same-prefix candidates are compared.
No sorting, no all-pairs, no model.

**Anomaly preservation** is explicit: singleton templates and ERROR/FATAL groups
are floored at a fidelity that keeps a verbatim exemplar, because the rare line
is the reason anyone pasted the log into a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..text.lexicon import LOG_LEVEL, TIMESTAMP

_MASKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{12,}\b"), "<HEX>"),
    # Numbers carrying a unit must be masked *as a unit*, before the bare
    # numeral rule -- whose trailing `(?![\w>])` refuses to match `10` in
    # `10ms`.  Without this, `dur=10ms` and `dur=11ms` produce different masks,
    # get merged by the similarity fallback anyway, and the group then renders
    # with the *first* line's literal value as if it held for all occurrences.
    (re.compile(
        r"(?<![\w<])\d+(?:\.\d+)?"
        r"(?:ms|us|ns|s|m|h|d|kb|mb|gb|tb|[KMGTP]?B|%|x)\b"
    ), "<N>"),
    (re.compile(r"(?<![\w<])\d+(?:\.\d+)?(?![\w>])"), "<N>"),
    (re.compile(r"\"[^\"]{0,200}\""), "<STR>"),
    (re.compile(r"'[^']{0,200}'"), "<STR>"),
]
#: One combined alternation, applied in a single left-to-right pass.  Applying
#: each pattern separately groups the captured slots by *pattern type* rather
#: than by position, which makes it impossible to map slot i back to the i-th
#: placeholder in the mask -- and therefore impossible to inline a constant.
_COMBINED_MASK = re.compile(
    "|".join(f"(?P<m{i}>{pat.pattern})" for i, (pat, _tag) in enumerate(_MASKS))
)
_MASK_TAGS = [tag for _pat, tag in _MASKS]

_SEVERITY_ORDER = {
    "TRACE": 0, "DEBUG": 1, "INFO": 2, "INFORMATION": 2, "NOTICE": 2,
    "WARN": 3, "WARNING": 3, "ERROR": 4, "ERR": 4, "SEVERE": 4,
    "FATAL": 5, "CRITICAL": 5, "PANIC": 5,
}
_STACK_FRAME = re.compile(r"^\s+(?:at\s|File\s\"|\.{3})")


@dataclass
class LogLine:
    idx: int
    start: int
    end: int
    raw: str
    ts: str | None
    level: str | None
    body: str
    mask: str
    slots: list[str]


@dataclass
class Template:
    tid: int
    mask: str
    level: str | None
    count: int = 0
    first_idx: int = 0
    last_idx: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    exemplar: str = ""
    exemplar_span: tuple[int, int] = (0, 0)
    spans: list[tuple[int, int]] = field(default_factory=list)
    slot_values: list[set[str]] = field(default_factory=list)
    anomalous: bool = False

    @property
    def severity(self) -> int:
        return _SEVERITY_ORDER.get((self.level or "").upper(), 2)


def parse_line(idx: int, start: int, end: int, raw: str) -> LogLine:
    ts = None
    m = TIMESTAMP.search(raw[:64])
    if m:
        ts = m.group(0)
    lvl = None
    lm = LOG_LEVEL.search(raw[:120])
    if lm:
        lvl = lm.group(0).upper()
    body = raw
    if m:
        body = body.replace(m.group(0), "", 1)
    if lm:
        body = body.replace(lm.group(0), "", 1)
    body = body.strip(" \t[]|-:")
    # ONE left-to-right pass.  Applying each pattern in turn groups the captured
    # slots by *pattern type* rather than by position, so slot i no longer
    # corresponds to the i-th placeholder in the mask -- which silently
    # misassigns values when we inline constants (see inline_constant_slots).
    slots: list[str] = []

    def _rep(mo: re.Match) -> str:
        slots.append(mo.group(0))
        idx = next(i for i in range(len(_MASK_TAGS)) if mo.group(f"m{i}") is not None)
        return _MASK_TAGS[idx]

    mask = _COMBINED_MASK.sub(_rep, body)
    mask = re.sub(r"\s+", " ", mask).strip()
    return LogLine(idx, start, end, raw, ts, lvl, body, mask, slots)


class TemplateMiner:
    """Fixed-depth bucketed matcher: O(1) expected lookup per line."""

    def __init__(self, similarity: float = 0.72, max_templates: int = 4000) -> None:
        self.similarity = similarity
        self.max_templates = max_templates
        self.templates: list[Template] = []
        self._buckets: dict[tuple[int, str, str], list[int]] = {}
        self._exact: dict[str, int] = {}

    def add(self, line: LogLine) -> Template:
        exact = self._exact.get(line.mask)
        if exact is not None:
            return self._update(self.templates[exact], line)

        tokens = line.mask.split()
        key = (len(tokens), tokens[0][:12] if tokens else "", line.level or "")
        for tid in self._buckets.get(key, ()):  # noqa: B007
            t = self.templates[tid]
            if _token_sim(t.mask.split(), tokens) >= self.similarity:
                return self._update(t, line)

        if len(self.templates) >= self.max_templates:
            # degrade gracefully: bucket the overflow under a catch-all
            t = self.templates[-1]
            return self._update(t, line)

        t = Template(
            tid=len(self.templates),
            mask=line.mask,
            level=line.level,
            first_idx=line.idx,
            first_ts=line.ts,
            exemplar=line.raw,
            exemplar_span=(line.start, line.end),
            slot_values=[set() for _ in line.slots],
        )
        self.templates.append(t)
        self._exact[line.mask] = t.tid
        self._buckets.setdefault(key, []).append(t.tid)
        return self._update(t, line)

    def _update(self, t: Template, line: LogLine) -> Template:
        t.count += 1
        t.last_idx = line.idx
        t.last_ts = line.ts or t.last_ts
        if t.first_ts is None:
            t.first_ts = line.ts
        t.spans.append((line.start, line.end))
        while len(t.slot_values) < len(line.slots):
            t.slot_values.append(set())
        for i, v in enumerate(line.slots):
            if len(t.slot_values[i]) < 8:
                t.slot_values[i].add(v)
        return t

    def finish(self) -> list[Template]:
        for t in self.templates:
            t.anomalous = t.count == 1 or t.severity >= 4
        return sorted(self.templates, key=lambda x: x.first_idx)


def _token_sim(a: list[str], b: list[str]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


_PLACEHOLDER = re.compile(r"<(?:UUID|IP|HEX|N|STR)>")


def inline_constant_slots(t: Template) -> tuple[str, list[set[str]]]:
    """Substitute single-valued slots back into the mask.

    A slot that takes the *same* value in every occurrence is not a variable --
    it is part of the statement.  Masking it saves nothing (the value still has
    to be carried in the slot table) and destroys the form the reader needs:
    ``connection pool exhausted: <N>/<N> in use {100 | 100}`` instead of
    ``connection pool exhausted: 100/100 in use``.  Measured against a real
    model, that cost the whole answer -- the log suite scored 67% purely on
    this, with the value technically "preserved".

    Returns the rendered mask and the slots that genuinely vary.
    """
    parts: list[str] = []
    varying: list[set[str]] = []
    pos = 0
    for slot_i, m in enumerate(_PLACEHOLDER.finditer(t.mask)):
        parts.append(t.mask[pos : m.start()])
        vals = t.slot_values[slot_i] if slot_i < len(t.slot_values) else set()
        if len(vals) == 1:
            parts.append(next(iter(vals)))
        else:
            parts.append(m.group(0))
            varying.append(vals)
        pos = m.end()
    parts.append(t.mask[pos:])
    return "".join(parts), varying


def render_template(t: Template, with_slots: bool = True, with_exemplar: bool = False) -> str:
    """Compact group rendering.  All content words come from the source lines."""
    head = f"[{t.level}] " if t.level else ""
    mult = f" x{t.count}" if t.count > 1 else ""
    window = ""
    if t.first_ts and t.last_ts and t.first_ts != t.last_ts:
        window = f" @{_short_ts(t.first_ts)}..{_short_ts(t.last_ts)}"
    elif t.first_ts:
        window = f" @{_short_ts(t.first_ts)}"
    body, varying = inline_constant_slots(t)
    out = f"{head}{body}{mult}{window}"
    if with_slots:
        vals = [sorted(v) for v in varying if v and len(v) <= 8]
        shown = [",".join(v[:5]) for v in vals[:4] if v]
        if shown:
            out += "  {" + " | ".join(shown) + "}"
    if with_exemplar and t.count > 1:
        out += f"\n  eg: {t.exemplar.strip()}"
    return out


def _short_ts(ts: str) -> str:
    m = re.search(r"\d{2}:\d{2}:\d{2}", ts)
    return m.group(0) if m else ts[:19]


def mine(lines: list[tuple[int, int, str]], similarity: float = 0.72) -> list[Template]:
    """Entry point: ``[(start, end, raw)]`` -> ordered templates."""
    miner = TemplateMiner(similarity=similarity)
    pending_stack: Template | None = None
    for i, (s, e, raw) in enumerate(lines):
        if not raw.strip():
            continue
        if _STACK_FRAME.match(raw) and pending_stack is not None:
            # stack frames belong to the preceding error, not to a template
            pending_stack.spans.append((s, e))
            continue
        ll = parse_line(i, s, e, raw)
        t = miner.add(ll)
        pending_stack = t if t.severity >= 4 else None
    return miner.finish()
