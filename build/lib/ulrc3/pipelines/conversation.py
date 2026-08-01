"""Conversation / chat-history pipeline.

Turn structure is recovered from role markers (or supplied directly by the API).
Each turn becomes a unit with the fidelity ladder

    drop  <  memory record  <  filler-stripped turn  <  verbatim turn

plus two global rules:

* a **verbatim recency window** (the last K turns are FROZEN) -- recency is the
  one prior we know the model needs and cannot recover;
* **supersession** -- turns whose every record has been invalidated by a later
  correction are capped at ``drop``, because keeping them is actively harmful.
"""

from __future__ import annotations

import re

from ..convo.memory import TYPE_WEIGHT, Record, RecordType, extract_records, render_record, revise
from ..text.lexicon import ASSISTANT_BOILERPLATE, EXAMPLE_LEAD, HEDGE, SMALL_TALK
from ..types import EdgeKind, Level, Protection, UnitKind
from .base import BuildContext, Pipeline, register

_ROLE_LINE = re.compile(
    r"^\s*(?:\*\*)?(user|human|assistant|ai|system|bot|agent|customer|support|client|rep|caller|operator|q|a)"
    r"(?:\*\*)?\s*[:>\]]\s?",
    re.IGNORECASE | re.MULTILINE,
)

#: Number of trailing turns kept verbatim.
RECENCY_WINDOW = 4

#: Record types worth writing to long-term memory.  Q/A exchanges and small
#: talk are reconstructible from the retained turns; facts, preferences,
#: decisions, tasks, constraints and corrections are not.
DURABLE = frozenset(
    {
        RecordType.FACT,
        RecordType.PREFERENCE,
        RecordType.DECISION,
        RecordType.TASK,
        RecordType.CONSTRAINT,
        RecordType.CORRECTION,
    }
)


@register("conversation")
class ConversationPipeline(Pipeline):
    name = "conversation"
    order_sensitive = True
    allow_intra_unit = True

    def build(self, ctx: BuildContext) -> None:
        base = ctx.region.start
        text = ctx.doc.text[base : ctx.region.end]
        turns = self._split_turns(text)
        if not turns:
            turns = [(0, ctx.doc.meta.get("role", "user"), text, 0, len(text))]

        records = revise(extract_records(turns))
        by_turn: dict[int, list[Record]] = {}
        for r in records:
            by_turn.setdefault(r.turn, []).append(r)

        n = len(turns)
        unit_by_turn: dict[int, int] = {}
        for ti, role, body, s, e in turns:
            recs = by_turn.get(ti, [])
            alive = [r for r in recs if r.alive and r.rtype is not RecordType.SMALLTALK]
            recent = ti >= n - RECENCY_WINDOW

            u = ctx.emit(
                UnitKind.TURN,
                base + s,
                base + e,
                depth=0,
                protection=Protection.LOCKED if recent else Protection.ELASTIC,
                meta={"role": role, "turn": ti, "n_records": len(alive)},
            )
            if u is None:
                continue
            unit_by_turn[ti] = u.uid
            u.features["recency"] = (ti + 1) / n
            u.features["record_value"] = sum(TYPE_WEIGHT.get(r.rtype, 1.0) for r in alive)
            if all((not r.alive) or r.rtype is RecordType.SMALLTALK for r in recs) and recs:
                u.protection = min(u.protection, Protection.DROPPABLE)
                u.features["superseded"] = 1.0

            self._ladder(ctx, u, body, role, alive, verbatim=recent)

            if ti > 0 and (ti - 1) in unit_by_turn:
                ctx.cir.add_edge(u.uid, unit_by_turn[ti - 1], EdgeKind.ANSWERS, 0.6)
                ctx.cir.add_edge(unit_by_turn[ti - 1], u.uid, EdgeKind.ADJACENT, 0.5)

        # A retracted premise retracts its answer.  "We want Enterprise" ->
        # "Enterprise costs $4200/mo" -> "correction: we want Business".  The
        # correction supersedes the request; the *price quoted for the retracted
        # request* is equally stale, and keeping it is a contradiction the
        # downstream model has no way to resolve.
        dead_turns = {
            ti for ti, uid in unit_by_turn.items()
            if ctx.cir.units[uid].features.get("superseded")
        }
        for ti in sorted(dead_turns):
            nxt = unit_by_turn.get(ti + 1)
            if nxt is None:
                continue
            v = ctx.cir.units[nxt]
            if v.protection >= Protection.LOCKED:
                continue  # inside the verbatim recency window
            if str(v.meta.get("role", "")).lower().startswith(("assistant", "ai", "bot", "agent", "support")):
                v.features["superseded"] = 1.0
                v.protection = Protection.DROPPABLE
                ctx.cir.add_edge(nxt, unit_by_turn[ti], EdgeKind.SUPERSEDES, 1.0)

        # A retracted statement is *forbidden*, not merely unattractive.  Left
        # as a penalty it still gets selected whenever budget allows, and the
        # compressed context then contains both "$5,135/mo" and "$2,307/mo"
        # with no way for the model to tell which is current.
        for uid in unit_by_turn.values():
            v = ctx.cir.units[uid]
            if v.features.get("superseded") and v.protection < Protection.LOCKED:
                v.levels = v.levels[:1] or [Level("drop", "", 0, 0.0, {}, set())]
                v.level = 0
                v.salience = 0.0

        # Low-information turns (pure hedging with no durable record and no
        # obligation) may never be emitted verbatim: their ceiling is `tight`.
        for uid in unit_by_turn.values():
            v = ctx.cir.units[uid]
            if v.protection >= Protection.LOCKED or v.features.get("superseded"):
                continue
            if v.features.get("record_value", 0.0) > 1.5 or v.obligations:
                continue
            if _hedge_ratio(v.text) >= 0.5:
                v.features["filler"] = 1.0
                v.protection = Protection.DROPPABLE
                if len(v.levels) > 2:
                    v.levels = v.levels[:-1]
                    v.level = min(v.level, len(v.levels) - 1)

        # supersession edges make the invalidation visible to the optimiser
        for r in records:
            if r.superseded_by is None:
                continue
            a = unit_by_turn.get(r.turn)
            b = unit_by_turn.get(records[r.superseded_by].turn)
            if a is not None and b is not None and a != b:
                ctx.cir.add_edge(b, a, EdgeKind.SUPERSEDES, 1.0)

    # -- helpers ------------------------------------------------------
    def _split_turns(self, text: str) -> list[tuple[int, str, str, int, int]]:
        marks = list(_ROLE_LINE.finditer(text))
        if len(marks) < 2:
            return []
        out: list[tuple[int, str, str, int, int]] = []
        for i, m in enumerate(marks):
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            out.append((len(out), m.group(1).lower(), body, start, start + len(body)))
        return out

    def _ladder(
        self,
        ctx: BuildContext,
        unit,
        body: str,
        role: str,
        alive: list[Record],
        verbatim: bool,
    ) -> None:
        levels: list[Level] = [Level("drop", "", 0, 0.0, {}, set())]

        # Only *durable* records earn a memory rung.  Giving every turn a cheap
        # ledger line lets pure filler ("Certainly!", "As previously
        # mentioned...") win the benefit/cost race against the one turn that
        # actually contains the decision -- measured: the correction turn was
        # dropped while eight filler turns survived.
        durable = [r for r in alive if r.rtype in DURABLE]
        if durable and not verbatim:
            mem = "\n".join(render_record(r) for r in durable)
            levels.append(
                Level(
                    name="memory",
                    text=mem,
                    tokens=ctx.tok.count(mem) + 1,
                    fidelity=0.55,
                    obligations={k for _c, k, _l, _s, _e in ctx.ex_prose.extract(mem)},
                )
            )

        stripped = _strip_boilerplate(body)
        if stripped and stripped != body:
            tagged = f"{role}: {stripped}"
            levels.append(
                Level(
                    name="tight",
                    text=tagged,
                    tokens=ctx.tok.count(tagged) + 1,
                    fidelity=0.85,
                    obligations={k for _c, k, _l, _s, _e in ctx.ex_prose.extract(tagged)},
                )
            )

        full = f"{role}: {body}"
        levels.append(
            Level(
                name="full",
                text=full,
                tokens=ctx.tok.count(full) + 1,
                fidelity=1.0,
                obligations={k for _c, k, _l, _s, _e in ctx.ex_prose.extract(full)},
            )
        )
        seen: set[str] = set()
        dedup = [levels[0]]
        for lv in levels[1:]:
            if lv.text in seen:
                continue
            seen.add(lv.text)
            dedup.append(lv)
        dedup = [dedup[0]] + sorted(dedup[1:], key=lambda lv: lv.tokens)
        unit.levels = dedup
        unit.level = len(dedup) - 1
        unit.tokens = dedup[-1].tokens


def _hedge_ratio(text: str) -> float:
    """Fraction of the turn's characters covered by hedge / meta phrases."""
    if not text:
        return 0.0
    covered = sum(m.end() - m.start() for m in HEDGE.finditer(text))
    covered += sum(m.end() - m.start() for m in ASSISTANT_BOILERPLATE.finditer(text))
    covered += sum(m.end() - m.start() for m in EXAMPLE_LEAD.finditer(text))
    return min(1.0, covered / max(1, len(text.strip())) * 3.0)


def _strip_boilerplate(body: str) -> str:
    """Remove LLM pleasantries and standalone social lines.  Extractive only."""
    out_lines: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if SMALL_TALK.match(s) and len(s) < 80:
            continue
        s2 = ASSISTANT_BOILERPLATE.sub("", s, count=1).strip()
        out_lines.append(s2 or s)
    return "\n".join(out_lines).strip()
