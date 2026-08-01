"""Semantic deduplication + **Semantic Delta Encoding**.

Redundancy in real contexts is rarely literal.  RAG returns four chunks that say
the same thing with different wording; a repo contains the same helper three
times; a support thread quotes the previous message.  Exact-match dedup catches
none of it, and embedding dedup needs a model.

We use two-stage LSH (SimHash bands -> MinHash confirmation), then, instead of
deleting the duplicates, we **encode their difference**:

    U17 = U4 except {timeout: 30s -> 5s, region: EU -> APAC}

The delta is computed on the *numeric-slotted normal form*, so paraphrases with
different parameters collapse to one statement plus a parameter table.  This is
strictly more faithful than deletion: the fact that a second, differing instance
existed is preserved, along with the values that differ -- which is precisely
the information a QA system needs and a summariser destroys.

Complexity: O(n) expected for clustering (bucketed), O(k·m) for delta extraction
inside clusters of size k with m tokens.
"""

from __future__ import annotations

import difflib
import re

from ..text.hashing import near_duplicate_clusters, normalize_for_hash
from ..types import EdgeKind, Level, Protection, Unit
from .base import Pass, PassContext

_MIN_TOKENS = 8
_WORD = re.compile(r"\S+")


class DedupPass(Pass):
    name = "dedup"

    def __init__(self, threshold: float = 0.82, max_units: int = 40000) -> None:
        self.threshold = threshold
        self.max_units = max_units
        self._clusters = 0
        self._saved = 0

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        self._clusters = 0
        self._saved = 0
        if ctx.cfg.ablated("no_dedup"):
            return
        cand = [
            u
            for u in cir.units
            if u.protection < Protection.FROZEN
            and u.tokens >= _MIN_TOKENS
            and u.kind.value not in ("log_template", "json_node")
        ]
        if len(cand) < 2 or len(cand) > self.max_units:
            return

        texts = [u.text for u in cand]
        clusters = near_duplicate_clusters(texts, threshold=self.threshold)
        for group in clusters:
            if len(group) < 2:
                continue
            units = [cand[i] for i in group]
            canon = max(units, key=lambda u: (len(u.obligations), u.tokens))
            self._clusters += 1
            for u in units:
                if u.uid == canon.uid:
                    continue
                cir.add_edge(u.uid, canon.uid, EdgeKind.DUPLICATE_OF, 1.0)
                u.meta["duplicate_of"] = canon.uid
                delta = self._delta(canon, u)
                if delta is None:
                    # Exact-enough duplicate: it may vanish, because its
                    # obligations are already carried by the canonical unit.
                    # This holds even for LOCKED units -- protection says "this
                    # content must survive", not "every copy of it must".
                    if u.protection < Protection.FROZEN and u.obligations <= canon.obligations:
                        u.protection = Protection.DROPPABLE
                        u.levels = u.levels[:1]
                        u.level = 0
                        cir.add_edge(u.uid, canon.uid, EdgeKind.REQUIRES, 1.0)
                    u.features["duplicate"] = 1.0
                    self._saved += u.tokens
                    continue
                text = f"= <{canon.uid}> except {delta}"
                tokens = ctx.tok.count(text) + 1
                if tokens < u.tokens * 0.7:
                    lvl = Level(
                        name="delta",
                        text=text,
                        tokens=tokens,
                        fidelity=0.7,
                        obligations={k for _c, k, _l, _s, _e in _extract(ctx, text)},
                    )
                    u.levels = [u.levels[0], lvl] + u.levels[1:]
                    u.level = min(u.level + 1, len(u.levels) - 1)
                    u.features["delta"] = 1.0
                    self._saved += max(0, u.tokens - tokens)
                    cir.add_edge(u.uid, canon.uid, EdgeKind.REQUIRES, 1.0)

    # -- delta ---------------------------------------------------------
    def _delta(self, canon: Unit, other: Unit) -> str | None:
        a = _WORD.findall(canon.text)
        b = _WORD.findall(other.text)
        if not a or not b:
            return None
        if normalize_for_hash(canon.text) == normalize_for_hash(other.text):
            # identical modulo numeric slots -> report only the differing slots
            pairs = _slot_diff(canon.text, other.text)
            return "{" + ", ".join(f"{x}->{y}" for x, y in pairs) + "}" if pairs else None
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
        if not ops or len(ops) > 6:
            return None
        parts: list[str] = []
        for tag, i1, i2, j1, j2 in ops:
            left = " ".join(a[i1:i2])[:60]
            right = " ".join(b[j1:j2])[:60]
            if tag == "replace":
                parts.append(f"{left}->{right}")
            elif tag == "delete":
                parts.append(f"-{left}")
            else:
                parts.append(f"+{right}")
        return "{" + ", ".join(parts) + "}"

    def note(self, ctx: PassContext) -> str:
        return f"{self._clusters} clusters, ~{self._saved} tok redundant"


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _slot_diff(a: str, b: str) -> list[tuple[str, str]]:
    na, nb = _NUM.findall(a), _NUM.findall(b)
    out: list[tuple[str, str]] = []
    for x, y in zip(na, nb):
        if x != y:
            out.append((x, y))
    return out[:6]


def _extract(ctx: PassContext, text: str):
    ex = ctx.scratch.get("ex_prose")
    return ex.extract(text) if ex is not None else []
