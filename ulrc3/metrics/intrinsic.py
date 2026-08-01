"""Intrinsic metrics: measurable without calling any LLM.

Extrinsic metrics (answer accuracy, pass@1) need a model in the loop and are
**not implemented** -- see docs/ROADMAP.md, where wiring them up is the top item.
Everything here runs offline and deterministically, which is what makes the
benchmark reproducible on a laptop, but it also means every quality number in
this repository is an *upper bound* on downstream accuracy rather than downstream
accuracy itself.

The metric that matters most and is most often *not* reported by compression
papers is **answerability retention**: given a question whose gold answer is a
span of the source, does the compressed context still contain that span?  It is
an upper bound on downstream QA accuracy and it needs no model, so it can be
computed over thousands of instances in seconds.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..ir.obligations import ObligationExtractor, canon
from ..text.terms import content_terms
from ..types import CompressionResult

_WORD = re.compile(r"[A-Za-z0-9_]+")
_NUM = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


@dataclass
class IntrinsicMetrics:
    tokens_in: int = 0
    tokens_out: int = 0
    ratio: float = 0.0
    compression_rate: float = 0.0
    latency_ms: float = 0.0
    peak_rss_mb: float = 0.0
    integrity: float = 1.0
    critical_recall: float = 1.0
    retention: float = 1.0
    number_recall: float = 1.0
    entity_recall: float = 1.0
    url_recall: float = 1.0
    identifier_recall: float = 1.0
    answerability: float = 1.0
    distractor_rate: float = 0.0
    query_term_coverage: float = 1.0
    provenance_violations: int = 0
    hallucinated_words: int = 0
    frozen_ok: bool = True
    syntax_ok: bool = True
    code_parse_ok: bool | None = None
    confidence: float = 1.0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def _recall(source: str, output: str, pattern: re.Pattern) -> float:
    src = {canon(m.group(0)) for m in pattern.finditer(source)}
    src = {s for s in src if s}
    if not src:
        return 1.0
    out = {canon(m.group(0)) for m in pattern.finditer(output)}
    return len(src & out) / len(src)


def number_recall(source: str, output: str) -> float:
    return _recall(source, output, _NUM)


def entity_recall(source: str, output: str, entities: Iterable[str] | None = None) -> float:
    """Recall over supplied gold entities, or orthographic candidates."""
    if entities is None:
        from ..text.terms import extract_entities

        entities = extract_entities(source)
    gold = {canon(e) for e in entities if len(e) > 2}
    if not gold:
        return 1.0
    low = canon(output)
    return sum(1 for g in gold if g in low) / len(gold)


def answerability(output: str, answers: Sequence[str]) -> float:
    """Fraction of gold answer spans still present in the compressed context."""
    if not answers:
        return 1.0
    low = canon(output)
    hit = 0
    for a in answers:
        ca = canon(a)
        if not ca:
            hit += 1
            continue
        if ca in low:
            hit += 1
            continue
        # numeric answers may be reformatted by surrounding markup; compare the
        # numeral itself
        nums = _NUM.findall(a)
        if nums and all(canon(n) in low for n in nums):
            hit += 1
    return hit / len(answers)


def query_term_coverage(query: str, output: str) -> float:
    q = set(content_terms(query))
    if not q:
        return 1.0
    o = set(content_terms(output))
    return len(q & o) / len(q)


_CONTROL_LINE = re.compile(r"(?m)^\s*[#§\[]?(?:CTX|CUT|SYM|D\d+)\b.*$")


def hallucinated_words(source: str, output: str, allowed: set[str] | None = None) -> int:
    """Count output words absent from the source.

    Uses exactly the rule the engine's verifier uses, including the exclusion of
    generated control lines -- otherwise the metric would count the engine's own
    header as hallucination and the number would be meaningless.
    """
    from ..render.markers import MARKER_VOCAB

    allow = (allowed or set()) | set(MARKER_VOCAB)
    src = {w.lower() for w in _WORD.findall(source)}
    bad = 0
    for w in _WORD.findall(_CONTROL_LINE.sub("", output)):
        lw = w.lower()
        if lw in src or lw in allow or lw.isdigit() or re.fullmatch(r"e\d+|x\d+", lw):
            continue
        if any(p.lower() in src for p in re.split(r"[-_]", w) if p):
            continue
        bad += 1
    return bad


def distractor_rate(output: str, distractors: Sequence[str]) -> float:
    """Fraction of *known-wrong* values still present in the compressed context.

    This is the metric that separates a compressor from a filter.  When a
    dialogue contains "we want Enterprise" followed by "correction: we want
    Business", keeping both is not high recall -- it is a contradiction that
    will flip the downstream answer.  Selection-by-importance keeps both,
    because both look important.  Belief revision removes the superseded one.
    """
    if not distractors:
        return 0.0
    low = canon(output)
    return sum(1 for d in distractors if canon(d) and canon(d) in low) / len(distractors)


def code_parses(text: str) -> bool | None:
    """Try to parse emitted Python blocks; ``None`` when there are none."""
    blocks = _python_blocks(text)
    if not blocks:
        return None
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError:
            return False
    return True


_PY_BLOCK = re.compile(r"(?ms)^(?:def |class |import |from \w).*?(?=^\S|\Z)")


def _python_blocks(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if re.match(r"^(?:def |class |@|import |from \w[\w.]* import )", line) or cur and (line.startswith((" ", "\t")) or not line.strip()):
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return [b for b in out if b.strip()]


def json_keys_recall(source: str, output: str) -> float:
    try:
        data = json.loads(source)
    except Exception:
        return 1.0
    keys: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                keys.add(str(k))
                walk(v)
        elif isinstance(node, list):
            for v in node[:200]:
                walk(v)

    walk(data)
    if not keys:
        return 1.0
    low = output.lower()
    return sum(1 for k in keys if k.lower() in low) / len(keys)


def evaluate(
    result: CompressionResult,
    source: str,
    query: str = "",
    answers: Sequence[str] = (),
    entities: Iterable[str] | None = None,
    peak_rss_mb: float = 0.0,
    distractors: Sequence[str] = (),
) -> IntrinsicMetrics:
    v = result.verification
    ex = ObligationExtractor()
    ident_src = {k for c, k, _l, _s, _e in ex.extract(source) if k.startswith("i:")}
    ident_out = {k for c, k, _l, _s, _e in ex.extract(result.text) if k.startswith("i:")}
    url_src = {k for c, k, _l, _s, _e in ex.extract(source) if k.startswith(("u:", "m:"))}
    url_out = {k for c, k, _l, _s, _e in ex.extract(result.text) if k.startswith(("u:", "m:"))}

    return IntrinsicMetrics(
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        ratio=result.ratio,
        compression_rate=result.compression_rate,
        latency_ms=float(result.meta.get("latency_ms", 0.0)),
        peak_rss_mb=peak_rss_mb,
        integrity=v.integrity,
        critical_recall=v.critical_recall,
        retention=v.retention,
        number_recall=number_recall(source, result.text),
        entity_recall=entity_recall(source, result.text, entities),
        url_recall=(len(url_src & url_out) / len(url_src)) if url_src else 1.0,
        identifier_recall=(len(ident_src & ident_out) / len(ident_src)) if ident_src else 1.0,
        answerability=answerability(result.text, answers),
        distractor_rate=distractor_rate(result.text, distractors),
        query_term_coverage=query_term_coverage(query, result.text),
        provenance_violations=int(result.meta.get("provenance_violations", 0)),
        hallucinated_words=hallucinated_words(source, result.text),
        frozen_ok=v.frozen_ok,
        syntax_ok=v.syntax_ok,
        code_parse_ok=code_parses(result.text),
        confidence=result.confidence,
    )
