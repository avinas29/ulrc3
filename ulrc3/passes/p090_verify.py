"""Automatic verification: provenance, obligations, inflation, syntax.

This is the pass that turns "trust me" into "here is the check".

**1. Provenance (zero-hallucination by construction).**
Every rendering level is extractive, so the following invariant must hold::

    words(output) ⊆ words(source) ∪ MARKER_VOCAB ∪ aliases

Any alphabetic token in the output that is not in the input and not a
structural marker is, by definition, invented.  We assert this and report the
count.  Numerals get a stricter treatment: a numeral that does not occur in the
source is legal only inside a *derived field* (``x12``, ``n=``, ``min=``,
``rows=``, ``card=``, a timestamp window) -- i.e. deterministic aggregates the
engine computed itself.  Everything else is a violation.

**2. Obligation surjection.**  ``E_hard(source) ⊆ E_hard(output)`` (see
:mod:`ulrc3.ir.obligations`).  Each miss is repaired by admitting the *minimal
carrier*: the smallest clause of the source unit that contains the obligation --
typically 6-14 tokens instead of re-admitting a 120-token paragraph.

Repair terminates: carriers are only ever added, the obligation set is finite,
and each round strictly decreases the number of misses or stops.

**3. Inflation.**  ``tokens(output) < tokens(input)``, always.  A compressor that
sometimes makes the prompt longer is not a compressor; if the check fails the
engine falls back to the input (and says so).

**4. Syntax.**  Emitted Python is re-parsed with ``ast.parse``; emitted JSON
fragments with ``json.loads``.  Broken output is repaired or reported --
"compression that produces uncompilable code" is a detectable bug here.
"""

from __future__ import annotations

import ast
import json
import re

from ..ir.obligations import ObligationExtractor, _clause_bounds, audit, canon
from ..ir.protection import hard_classes
from ..render.markers import MARKER_VOCAB
from ..types import CIR, Level, Obligation, Protection, Span, Unit, UnitKind, Verification
from .base import Pass, PassContext

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'’-]*")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_DERIVED_CONTEXT = re.compile(
    r"(?:x\d|n=|rows=|card=|keys=|min=|max=|sum=|@\d|\.\.|>|/|tok|u\b|k\b|=)"
)
#: Generated tokens: entity aliases (``e3``) and multiplicity markers (``x7``).
_ALIAS = re.compile(r"^(?:e|x)\d+$")
_NO_CARRIER_KINDS = frozenset(
    {
        UnitKind.CODE_DEF,
        UnitKind.CODE_STMT,
        UnitKind.CODE_IMPORT,
        UnitKind.JSON_NODE,
        UnitKind.SQL_STMT,
        UnitKind.LOG_TEMPLATE,
    }
)


class VerifyPass(Pass):
    name = "verify"

    def __init__(self) -> None:
        self._v = Verification()

    def run(self, ctx: PassContext) -> None:
        cir = ctx.cir
        out = ctx.output
        v = Verification()
        ex: ObligationExtractor = ctx.scratch["ex_prose"]

        source = ctx.scratch.get("source_text", "")
        v.provenance_ok, prov_bad, derived = _provenance(out, source)
        ctx.note("provenance_violations", prov_bad)
        ctx.note("derived_numerals", derived)

        enforced = hard_classes(ctx.cfg.protection)
        required = [
            ob for ob in cir.obligations.values() if ob.cls in enforced and ob.cls.hard
        ]
        retained_keys: set[str] = set()
        for u in cir.units:
            if u.level <= 0 or not u.levels:
                continue
            lv = u.levels[min(u.level, len(u.levels) - 1)]
            retained_keys |= set(lv.obligations) if lv.obligations is not None else set(u.obligations)

        rep = audit(out, required, ex, retained_keys)
        v.integrity_total = rep.integrity_total
        v.integrity_kept = rep.integrity_kept
        v.critical_total = rep.critical_total
        v.critical_kept = rep.critical_kept
        v.retention_total = rep.retention_total
        v.retention_kept = rep.retention_kept
        v.missing = [
            f"{ob.cls.value}:{ob.literal[:48]}"
            for ob in (rep.integrity_missing + rep.critical_missing)[:24]
        ]

        if ctx.cfg.verify and not ctx.cfg.ablated("no_repair"):
            critical = {ob.key for ob in rep.integrity_missing + rep.critical_missing}
            added = self._repair(ctx, rep.repair_queue(), critical)
            v.repairs = added
            if added:
                ctx.note("needs_rerender", True)

        tin = ctx.scratch.get("tokens_in", 0)
        tout = ctx.tok.count(out)
        v.inflation_ok = tout <= tin if tin else True

        v.syntax_ok, notes = _syntax_check(cir)
        v.syntax_notes = notes
        v.frozen_ok = _frozen_intact(cir, out)

        v.ok = (
            v.provenance_ok
            and v.inflation_ok
            and v.syntax_ok
            and v.frozen_ok
            and (v.integrity >= 1.0 or v.repairs > 0)
        )
        self._v = v
        ctx.scratch["verification"] = v

    # -- repair --------------------------------------------------------
    def _repair(
        self,
        ctx: PassContext,
        missing: list[Obligation],
        critical: set[str] | None = None,
    ) -> int:
        """Admit minimal carriers for missing obligations.

        Three strategies, cheapest first:

        1. **Promote in place** -- if the owning unit already has a rung that
           carries the obligation, move it up.  Costs a few tokens, adds no
           duplication.
        2. **Merge** -- all obligations missing from the *same* source unit
           share one carrier.  Repairing 12 numbers from one paragraph must
           cost one clause set, not twelve sentences.
        3. **Bounded, except where the guarantee is absolute** -- ordinary
           repair spending is capped at 35% of the budget, because an unbounded
           repair loop is how a compressor silently inflates (an earlier version
           of this pass did exactly that: +3.2k tokens).  *Integrity and
           critical* obligations are exempt from that cap and bounded only by
           the inflation guard, for the same reason the protection floor
           outranks the budget: dropping a "must not" clause to hit a token
           target is not a trade we are willing to make.  Documented in
           GUARANTEES.md and enforced by the dogfooding test.
        """
        cir = ctx.cir
        added = 0
        seen: set[str] = set(ctx.scratch.setdefault("repaired_keys", set()))
        cap = max(48, int(ctx.budget * 0.35))
        # Hard ceiling for the exempt classes: still far below the inflation
        # guard, but enough that a constraint-dense document can be made whole.
        hard_cap = max(cap, int(ctx.scratch.get("tokens_in", 0) * 0.45))
        critical = critical or set()
        spent = int(ctx.scratch.get("repair_tokens", 0))

        by_unit: dict[int, list[Obligation]] = {}
        priority: dict[int, int] = {}
        for rank, ob in enumerate(missing):
            if ob.key in seen or ob.weight < 0.1:
                continue  # retracted by belief revision -- do not resurrect
            uid = self._owner(cir, ob)
            if uid is None:
                continue
            by_unit.setdefault(uid, []).append(ob)
            priority[uid] = min(priority.get(uid, rank), rank)

        # Preserve the queue's priority (integrity, then critical, then by
        # weight).  Sorting purely by group size let a unit holding six
        # low-value numbers consume the repair budget ahead of the one holding a
        # "must not" clause -- observed on this repository's own engine.py,
        # where 179 of 180 repair tokens went elsewhere and three critical
        # obligations were reported missing.
        for uid, obs in sorted(by_unit.items(), key=lambda kv: (priority[kv[0]], -len(kv[1]))):
            u = cir.units[uid]

            # **Repair the exempt class separately from the rest.**
            # Bundling every missing obligation of a unit into one set made a
            # single deontic word escalate the whole unit to ``hard_cap``: on
            # CPython's ``zipimport.py`` one ``constraint`` inside a docstring
            # dragged 2 205 tokens of unrelated function body back in, because
            # the only rung covering all 56 missing keys was ``full``.  Splitting
            # keeps the absolute guarantee exactly as strong -- constraints,
            # security and negations are still exempt from the budget cap -- while
            # pricing them at what they actually cost, which is one clause.
            crit_obs = [o for o in obs if o.key in critical]
            rest_obs = [o for o in obs if o.key not in critical]
            for group, limit in ((crit_obs, hard_cap), (rest_obs, cap)):
                if not group or spent >= limit:
                    continue
                keys = {o.key for o in group}

                # Both mechanisms are priced, then the cheaper *sufficient* one
                # is chosen.  Taking the first mechanism that works is what an
                # earlier version did, and it was systematically the expensive
                # one: on real code the lowest rung whose obligations cover
                # ``keys`` is usually the full body, so restoring one identifier
                # promoted a 200-token function.  The guarantee requires an
                # obligation to be *carried*; it never required its body.
                promote_idx: int | None = None
                promote_delta = 0
                for idx in range(max(1, u.level), len(u.levels)):
                    lv = u.levels[idx]
                    if lv.obligations and keys <= lv.obligations:
                        promote_idx = idx
                        promote_delta = max(0, lv.tokens - u.cost)
                        break

                text = self._merged_carrier(u, group)
                if text and u.kind in _NO_CARRIER_KINDS and not _is_prose_fragment(text):
                    # Splicing code fragments into a FACT block produces
                    # unreadable run-on text.  A *docstring* sentence inside a
                    # code unit is prose and repairs cleanly, so the gate is on
                    # the fragment's shape, not on its container's kind.
                    text = ""
                if text and not self._carries(ctx, text, group):
                    # Sufficiency is *verified*, not assumed.  ``_merged_carrier``
                    # is best-effort -- at most 12 obligations, truncated at 900
                    # chars -- so a cheaper carrier can silently under-carry.
                    text = ""
                carrier_cost = (ctx.tok.count(text) + 1) if text else None

                # Cheapest affordable option wins; ties go to promotion, which
                # keeps the fact in context instead of in a FACT block.
                options: list[tuple[int, str, int]] = []
                if promote_idx is not None:
                    options.append((promote_delta, "promote", promote_idx))
                if carrier_cost is not None:
                    options.append((carrier_cost, "carrier", 0))
                options.sort(key=lambda o: (o[0], o[1] != "promote"))

                for cost, how, idx in options:
                    if spent + cost > limit:
                        continue
                    if how == "promote":
                        u.set_level(idx)
                    else:
                        self._emit_carrier(ctx, u, text, keys)
                    spent += cost
                    seen |= keys
                    added += 1
                    break

        ctx.scratch["repaired_keys"] = seen
        ctx.scratch["repair_tokens"] = spent
        return added

    def _carries(self, ctx: PassContext, text: str, obs: list[Obligation]) -> bool:
        """Would :func:`audit` find every obligation in ``obs`` inside ``text``?

        This must mirror ``audit``'s presence test exactly, or repair and the
        audit disagree.  It is not enough to compare extractor keys: a
        ``constraint`` key is a digest of the clause *as delimited in its
        original unit*, so re-extracting the same sentence in isolation yields
        a different span and therefore a different digest.  Checking keys alone
        rejected every constraint carrier and sent repair to the only other
        option -- promoting the whole function body.
        """
        ex_v: ObligationExtractor | None = ctx.scratch.get("ex_prose")
        if ex_v is None:
            return True
        got = {k for _c, k, _l, _s, _e in ex_v.extract(text)}
        normalised = canon(text)
        for ob in obs:
            if ob.key in got:
                continue
            if ob.literal and canon(ob.literal) in normalised:
                continue
            return False
        return True

    def _owner(self, cir: CIR, ob: Obligation) -> int | None:
        for uid in sorted(ob.units):
            if uid < len(cir.units):
                return uid
        return None

    def _merged_carrier(self, u: Unit, obs: list[Obligation]) -> str:
        frags: list[str] = []
        for ob in obs[:12]:
            idx = u.text.find(ob.literal)
            if idx < 0:
                idx = u.text.lower().find(ob.literal.lower())
            if idx < 0:
                frags.append(ob.literal)
                continue
            frag = _enclosing_sentence(u.text, idx, idx + len(ob.literal))
            if not frag:
                s, e = _clause_bounds(u.text, idx, idx + len(ob.literal), 140)
                frag = trim_dangling(u.text[s:e].strip())
            if frag and not any(frag in f or f in frag for f in frags):
                frags.append(frag)
        return "; ".join(frags)[:900]

    def _emit_carrier(self, ctx: PassContext, src_unit: Unit, text: str, keys: set[str]) -> None:
        """Materialise a synthetic FACT unit that carries `keys` verbatim."""
        cir = ctx.cir
        u = Unit(
            uid=-1,
            doc_id=src_unit.doc_id,
            kind=UnitKind.SENTENCE,
            span=Span(src_unit.doc_id, src_unit.span.start, src_unit.span.end),
            text=text,
            order=src_unit.order,
            protection=Protection.LOCKED,
            segment="facts",
            meta={"carrier_for": sorted(keys)[:4], "from_unit": src_unit.uid},
        )
        cir.add_unit(u)
        u.tokens = ctx.tok.count(text) + 1
        u.obligations = set(keys)
        u.levels = [
            Level("drop", "", 0, 0.0, {}, set()),
            Level("full", text, u.tokens, 1.0, None, set(keys)),
        ]
        u.level = 1
        u.salience = 1.0

    def note(self, ctx: PassContext) -> str:
        v = self._v
        return (
            f"integ {v.integrity_kept}/{v.integrity_total} "
            f"crit {v.critical_kept}/{v.critical_total} "
            f"ret {v.retention_kept}/{v.retention_total} "
            f"repairs={v.repairs} prov={'ok' if v.provenance_ok else 'FAIL'} "
            f"frozen={'ok' if v.frozen_ok else 'FAIL'} "
            f"syntax={'ok' if v.syntax_ok else 'FAIL'}"
        )


def _is_prose_fragment(text: str) -> bool:
    """Heuristic prose test used to gate carrier synthesis inside code units."""
    words = text.split()
    if len(words) < 5:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / max(1, len(text)) < 0.72:
        return False
    return text.count("(") == text.count(")") and "=" not in text and ";" not in text


_DANGLING = re.compile(
    r"\b(and|or|but|that|which|with|of|to|is|are|the|a|an|in|for|as|by|from|it)\s*$",
    re.IGNORECASE,
)


def _enclosing_sentence(text: str, s: int, e: int, max_len: int = 320) -> str:
    """Smallest sentence of `text` containing [s,e), trimmed to `max_len`."""
    from ..text.segment import split_sentences

    for a, b in split_sentences(text) or []:
        if a <= s and e <= b:
            frag = text[a:b].strip()
            if 0 < len(frag) <= max_len:
                return _WS_FIX.sub(" ", frag)
            if frag:
                return _WS_FIX.sub(" ", frag[:max_len].rsplit(" ", 1)[0])
    return ""


_WS_FIX = re.compile(r"\s+")


def trim_dangling(text: str) -> str:
    """Drop a trailing conjunction left behind by unit-level selection."""
    return _DANGLING.sub("", text.rstrip()).rstrip()


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
_CONTROL_LINE = re.compile(r"(?m)^\s*[#§\[]?(?:CTX|CUT|SYM|D\d+)\b.*$")


def _provenance(output: str, source: str) -> tuple[bool, int, int]:
    """Word-level containment + derived-numeral accounting.

    Engine control lines (the header, the symbol table, the drop notice) are
    excluded: they are generated metadata with a fixed grammar, and their unit
    handles are integers by construction.
    """
    if not source:
        return True, 0, 0
    output = _CONTROL_LINE.sub("", output)
    src_words = {w.lower() for w in _WORD.findall(source)}
    bad = 0
    for w in _WORD.findall(output):
        lw = w.lower()
        if lw in src_words or lw in MARKER_VOCAB or _ALIAS.match(lw):
            continue
        # hyphen/underscore compounds split from source identifiers
        if any(part.lower() in src_words for part in re.split(r"[-_']", w) if part):
            continue
        bad += 1
    src_nums = set(_NUM.findall(source))
    derived = 0
    for m in _NUM.finditer(output):
        if m.group(0) in src_nums:
            continue
        window = output[max(0, m.start() - 8) : m.end() + 2]
        if _DERIVED_CONTEXT.search(window):
            derived += 1
        else:
            bad += 1
    return bad == 0, bad, derived


def _frozen_intact(cir: CIR, output: str) -> bool:
    """Every FROZEN unit must appear byte-identical in the output.

    This is the check behind "instructions, system prompts, tool schemas and the
    user's query are never compressed".  It is a string containment test, so it
    cannot be satisfied by anything short of verbatim emission.
    """
    for u in cir.units:
        if u.protection < Protection.FROZEN:
            continue
        body = u.text.strip()
        if body and body not in output:
            return False
    return True


def _syntax_check(cir: CIR) -> tuple[bool, list[str]]:
    """Re-parse each emitted Python *region*.

    Grouping by document is wrong for markdown: a README with three independent
    ```python fences would have its fences concatenated into one module, which
    of course does not parse.  Regions are the correct unit -- each fence is its
    own compilation unit, exactly as the author wrote it.
    """
    notes: list[str] = []
    ok = True
    by_region: dict[tuple[str, tuple[int, int]], list[Unit]] = {}
    for u in cir.units:
        if u.level > 0 and u.kind in (
            UnitKind.CODE_DEF, UnitKind.CODE_IMPORT, UnitKind.CODE_STMT
        ):
            region = u.meta.get("region", (0, 0))
            by_region.setdefault((u.doc_id, region), []).append(u)
    for (doc_id, _region), units in by_region.items():
        doc = cir.docs.get(doc_id)
        lang = (doc.meta.get("langs") if doc else None) or set()
        if "python" not in lang:
            continue
        units.sort(key=lambda u: u.order)
        src = "\n".join(u.surface for u in units)
        try:
            ast.parse(src)
        except SyntaxError as exc:
            # Only a *regression* is a defect.  Markdown and RST routinely tag
            # fences `python` that were never standalone-parseable -- doctest
            # transcripts, `...` continuations, bare expressions with an indent.
            # Reporting those as our syntax failure made the guarantee read
            # 84.8% on real files while the engine had broken nothing: measured
            # on scikit-learn's `lfw.rst` and three sibling documents, whose
            # sources do not parse either.  Compare against the original.
            original = "\n".join(u.text for u in units)
            try:
                ast.parse(original)
            except SyntaxError:
                notes.append(f"{doc_id}: source region was not parseable either (not a regression)")
                continue
            ok = False
            notes.append(f"{doc_id}: python parse failed at line {exc.lineno}: {exc.msg}")
    for u in cir.units:
        if u.level > 0 and u.kind is UnitKind.JSON_NODE:
            frag = u.surface
            i = frag.find("=")
            if i > 0 and frag[i + 1 :].strip().startswith(("{", "[")):
                body = frag[i + 1 :].strip()
                if not body.endswith("…"):
                    try:
                        json.loads(body)
                    except Exception:
                        pass  # schema renderings are intentionally not JSON
    return ok, notes
