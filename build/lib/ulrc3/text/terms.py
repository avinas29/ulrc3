"""Term extraction, IDF statistics, and lightweight entity recognition.

No NER model.  Instead: an orthographic + distributional recogniser that is
*recall-oriented* by design.  A false-positive entity costs a few tokens of
protection; a false negative costs a fact.  The asymmetry is not symmetric, so
neither is the detector.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .lexicon import STOPWORDS

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Entity surface patterns -----------------------------------------------
_PROPER_SEQ = re.compile(
    r"\b(?:[A-Z][\w&'’.-]*(?:\s+(?:of|de|the|and|für|van|von))?\s*){1,5}\b"
)
_ACRONYM = re.compile(r"\b[A-Z]{2,8}(?:-\d+)?\b")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b")
_QUOTED = re.compile(r"[\"'`]([^\"'`\n]{2,64})[\"'`]")

_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|\n\s*|[:;]\s+)")


def tokenize(text: str, lower: bool = True) -> list[str]:
    out = _TOKEN.findall(text)
    return [t.lower() for t in out] if lower else out


def content_terms(text: str, keep_numbers: bool = False) -> list[str]:
    """Content-bearing terms: stopword-free, camelCase-split, snake-split."""
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok[0].isdigit():
            if keep_numbers:
                out.append(tok)
            continue
        low = tok.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        out.append(low)
        if "_" in tok or _CAMEL.search(tok):
            for part in _CAMEL.sub(" ", tok.replace("_", " ")).split():
                p = part.lower()
                if len(p) > 2 and p not in STOPWORDS and p != low:
                    out.append(p)
    return out


def bigrams(terms: Sequence[str]) -> list[str]:
    return [f"{terms[i]}_{terms[i + 1]}" for i in range(len(terms) - 1)]


@dataclass
class TermStats:
    """Corpus-level document frequencies over units.

    IDF here is *within-context*: a term that appears in every retrieved chunk
    is uninformative **for this prompt** even if it is rare in English.  That is
    the correct notion of specificity for context compression and it is why we
    do not ship a static IDF table.
    """

    n_docs: int = 0
    df: Counter = field(default_factory=Counter)
    tf: Counter = field(default_factory=Counter)

    def add(self, terms: Iterable[str]) -> None:
        self.n_docs += 1
        uniq = set(terms)
        self.df.update(uniq)
        self.tf.update(uniq)

    def idf(self, term: str) -> float:
        # smoothed probabilistic IDF, floored at 0.05 so ubiquitous terms retain
        # a whisper of weight (they still disambiguate against *other* prompts)
        df = self.df.get(term, 0)
        if self.n_docs <= 1:
            return 1.0
        val = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return max(0.05, val)

    def weights(self, terms: Iterable[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in terms:
            out[t] = out.get(t, 0.0) + self.idf(t)
        # length normalisation keeps long units from dominating coverage
        norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
        return {k: v / norm for k, v in out.items()}

    def top(self, n: int = 20) -> list[tuple[str, float]]:
        return sorted(((t, self.idf(t)) for t in self.df), key=lambda x: -x[1])[:n]


def extract_entities(text: str, min_len: int = 2) -> set[str]:
    """Orthographic entity candidates.

    Covers: multiword proper nouns, acronyms, dotted/snake/camel identifiers,
    and short quoted literals (labels, enum values, filenames).
    """
    out: set[str] = set()

    # proper-noun sequences, excluding sentence-initial capitalisation artefacts
    starts = {m.end() for m in _SENTENCE_START.finditer(text)}
    starts.add(0)
    for m in _PROPER_SEQ.finditer(text):
        cand = m.group(0).strip(" .,;:")
        if len(cand) < min_len:
            continue
        words = cand.split()
        if m.start() in starts and len(words) == 1:
            continue  # "The" at sentence start is not an entity
        if len(words) == 1 and cand.lower() in STOPWORDS:
            continue
        if len(cand) > 80:
            continue
        out.add(cand)

    for m in _ACRONYM.finditer(text):
        if m.group(0) not in {"I", "A"}:
            out.add(m.group(0))
    for m in _IDENTIFIER.finditer(text):
        cand = m.group(0)
        if len(cand) >= 3 and cand.lower() not in STOPWORDS:
            out.add(cand)
    for m in _QUOTED.finditer(text):
        cand = m.group(1).strip()
        if 2 <= len(cand) <= 64 and not cand.isspace():
            out.add(cand)
    return out


def entity_kind(name: str) -> str:
    if re.match(r"^[A-Z]{2,8}(-\d+)?$", name):
        return "acronym"
    if "." in name or "_" in name or re.search(r"[a-z][A-Z]", name):
        return "identifier"
    if " " in name:
        return "entity"
    return "term"


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        w = b.get(k)
        if w is not None:
            dot += v * w
    return dot  # both sides are pre-normalised by TermStats.weights


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
