"""Conversation memory: typed records, belief revision, a supersession DAG.

Chat history is the worst-behaved context type: it is 60-80% redundant, it
contains corrections that *invalidate* earlier statements, and the naive fix
(drop the oldest turns) deletes exactly the durable facts the user stated first.

We model it as an append-only knowledge log with defeasible entries.

* Each turn yields typed **records**: FACT, PREFERENCE, DECISION, TASK,
  CONSTRAINT, CORRECTION, QUESTION, ANSWER.
* Records get a **subject key** (type + salient head term).  A later record with
  the same key ``supersedes`` the earlier one; corrections supersede their
  highest-overlap antecedent.
* The retained memory is the set of **non-superseded** records plus a verbatim
  recency window.  Superseded records are not merely down-weighted -- they are
  *wrong*, and keeping them costs tokens **and** accuracy.

This is what "ChatGPT-style memory" should be: not a summary, but a log with a
revision order.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..text.lexicon import DEONTIC, SMALL_TALK
from ..text.terms import content_terms, overlap_coefficient


class RecordType(enum.Enum):
    FACT = "fact"
    PREFERENCE = "pref"
    DECISION = "decision"
    TASK = "task"
    CONSTRAINT = "constraint"
    CORRECTION = "correction"
    QUESTION = "q"
    ANSWER = "a"
    SMALLTALK = "chat"


#: Ordered by precedence: the first pattern that matches wins.
CUES: list[tuple[RecordType, re.Pattern]] = [
    (RecordType.CORRECTION, re.compile(
        r"\b(actually|correction|i\s+meant|to\s+correct|that'?s\s+wrong|no,\s|not\s+quite|"
        r"scratch\s+that|ignore\s+(that|the\s+previous)|instead\s+of\s+that|let\s+me\s+rephrase|"
        r"i\s+misspoke|update:|revised:)\b", re.IGNORECASE)),
    (RecordType.PREFERENCE, re.compile(
        r"\b(i\s+(?:prefer|like|love|hate|dislike|want|need|usually|always|never)|"
        r"my\s+(?:preference|favou?rite|style|convention)|"
        r"(?:please\s+)?(?:always|never)\s+\w+|"
        r"i'?d\s+(?:rather|prefer)|"
        r"we\s+(?:prefer|use|want|need|require|standardi[sz]e))\b", re.IGNORECASE)),
    (RecordType.DECISION, re.compile(
        r"\b(let'?s\s+(?:go\s+with|use|do|pick|choose)|we(?:'ll|\s+will|\s+decided|\s+agreed|\s+chose)\s+"
        r"(?:to\s+)?\w+|decision:|final(?:ly|\s+answer)?:|going\s+with|settled\s+on|"
        r"approved|confirmed|sign(?:ed)?\s+off)\b", re.IGNORECASE)),
    (RecordType.TASK, re.compile(
        r"\b(todo|to-do|action\s+item|next\s+steps?|i\s+need\s+(?:to|you)|"
        r"can\s+you\s+\w+|could\s+you\s+\w+|please\s+\w+|"
        r"remind\s+me|follow[\s-]?up|deadline|by\s+(?:monday|tuesday|wednesday|thursday|friday|"
        r"tomorrow|next\s+week|eod|eow))\b", re.IGNORECASE)),
    (RecordType.CONSTRAINT, DEONTIC),
    (RecordType.FACT, re.compile(
        r"\b(?:my|our|the)\s+\w+\s+(?:is|are|was|were|has|have|costs?|equals?)\b|"
        r"\bi\s+(?:am|work|live|use|run|have|manage|own)\b|"
        r"\bwe\s+(?:are|use|run|have|support|deploy|host)\b|"
        r"\b\w+\s+(?:is|are)\s+(?:located|based|called|named|set\s+to|configured)\b", re.IGNORECASE)),
]

_QUESTION = re.compile(r"\?\s*$|^\s*(?:what|why|how|when|where|who|which|can|could|should|would|is|are|do|does|did)\b", re.IGNORECASE)


@dataclass
class Record:
    rtype: RecordType
    text: str
    turn: int
    role: str
    start: int
    end: int
    key: str = ""
    terms: set[str] = field(default_factory=set)
    superseded_by: int | None = None
    idx: int = -1

    @property
    def alive(self) -> bool:
        return self.superseded_by is None


def classify(sentence: str) -> RecordType:
    s = sentence.strip()
    if not s:
        return RecordType.SMALLTALK
    if SMALL_TALK.match(s) and len(s) < 80:
        return RecordType.SMALLTALK
    for rtype, pat in CUES:
        if pat.search(s):
            return rtype
    if _QUESTION.search(s):
        return RecordType.QUESTION
    return RecordType.ANSWER


def subject_key(rtype: RecordType, terms: Iterable[str]) -> str:
    """Coarse subject identity used for supersession.

    Two records collide when they are the same *kind* of statement about the
    same head term.  Deliberately coarse: over-merging loses a nuance, but
    under-merging keeps a contradiction, and contradictions are worse.
    """
    ts = [t for t in terms if len(t) > 2][:3]
    return f"{rtype.value}:{'|'.join(sorted(ts))}" if ts else f"{rtype.value}:~"


def extract_records(
    turns: list[tuple[int, str, str, int, int]],
    sentence_spans: dict[int, list[tuple[int, int]]] | None = None,
) -> list[Record]:
    """``turns`` = [(turn_index, role, text, start, end)] -> typed records."""
    from ..text.segment import split_sentences

    out: list[Record] = []
    for ti, role, text, start, _end in turns:
        spans = (sentence_spans or {}).get(ti) or split_sentences(text) or [(0, len(text))]
        for ss, se in spans:
            sent = text[ss:se].strip()
            if len(sent) < 3:
                continue
            rt = classify(sent)
            terms = set(content_terms(sent))
            r = Record(
                rtype=rt,
                text=sent,
                turn=ti,
                role=role,
                start=start + ss,
                end=start + se,
                terms=terms,
            )
            r.key = subject_key(rt, sorted(terms, key=lambda t: -len(t)))
            r.idx = len(out)
            out.append(r)
    return out


def revise(records: list[Record], overlap_threshold: float = 0.45) -> list[Record]:
    """Belief revision.  O(n) for key collisions + O(n·w) for corrections."""
    last_by_key: dict[str, int] = {}
    for r in records:
        if r.rtype in (RecordType.SMALLTALK, RecordType.QUESTION):
            continue
        prev = last_by_key.get(r.key)
        if prev is not None and r.rtype in (RecordType.FACT, RecordType.PREFERENCE, RecordType.DECISION):
            if overlap_coefficient(records[prev].terms, r.terms) >= 0.5:
                records[prev].superseded_by = r.idx
        last_by_key[r.key] = r.idx

    # Corrections invalidate their best-matching antecedent.  Search is over the
    # *whole* prior log with a recency prior, not a fixed window: in a 40-turn
    # dialogue the correction typically arrives 20+ turns after the statement it
    # retracts, and a 12-turn cutoff silently missed every one of them.
    for r in records:
        if r.rtype is not RecordType.CORRECTION:
            continue
        best: Record | None = None
        best_score = overlap_threshold
        for cand in records[: r.idx]:
            if not cand.alive or cand.rtype in (RecordType.SMALLTALK, RecordType.CORRECTION):
                continue
            overlap = overlap_coefficient(cand.terms, r.terms)
            if overlap <= 0.0:
                continue
            recency = 0.6 + 0.4 * math.exp(-(r.turn - cand.turn) / 40.0)
            score = overlap * recency
            if score >= best_score:
                best, best_score = cand, score
        if best is not None:
            best.superseded_by = r.idx
    return records


def render_record(r: Record, max_len: int = 220) -> str:
    """Compact ledger line.  Extractive: the text is the source sentence."""
    body = r.text.strip()
    if len(body) > max_len:
        body = body[:max_len].rsplit(" ", 1)[0]
    tag = r.rtype.value
    who = "u" if r.role.lower().startswith(("user", "human", "customer", "client")) else "a"
    return f"{tag}/{who}: {body}"


TYPE_WEIGHT: dict[RecordType, float] = {
    RecordType.CONSTRAINT: 3.0,
    RecordType.DECISION: 2.6,
    RecordType.PREFERENCE: 2.4,
    RecordType.TASK: 2.2,
    RecordType.FACT: 2.0,
    RecordType.CORRECTION: 2.0,
    RecordType.ANSWER: 1.0,
    RecordType.QUESTION: 0.8,
    RecordType.SMALLTALK: 0.05,
}
