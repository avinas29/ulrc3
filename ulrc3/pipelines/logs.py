"""Log pipeline: one unit per *template group*, not per line.

Fidelity ladder per group:

    drop  <  "template xN"  <  template + slot values + window  <  verbatim lines

Errors and singletons are floored above ``drop`` (anomalies are the payload),
and the group ordering follows first occurrence so the timeline survives.
"""

from __future__ import annotations

from ..logsir.templates import mine, render_template
from ..text.segment import iter_lines
from ..types import EdgeKind, Level, Protection, UnitKind
from .base import BuildContext, Pipeline, register

_MAX_VERBATIM_LINES = 12


@register("logs")
class LogPipeline(Pipeline):
    name = "logs"
    order_sensitive = True
    allow_intra_unit = False

    def build(self, ctx: BuildContext) -> None:
        base = ctx.region.start
        text = ctx.doc.text[base : ctx.region.end]
        lines = [(s, e, ln) for s, e, ln in iter_lines(text) if ln.strip()]
        if not lines:
            return
        templates = mine(lines)
        total = sum(t.count for t in templates)
        ctx.doc.meta["log_lines"] = total
        ctx.doc.meta["log_templates"] = len(templates)

        prev_uid = None
        for t in templates:
            s0, e0 = t.spans[0] if t.spans else (0, 0)
            u = ctx.emit(
                UnitKind.LOG_TEMPLATE,
                base + s0,
                base + min(e0, len(text)),
                text=text[s0:e0],
                protection=Protection.ELASTIC,
                meta={
                    "template": t.mask,
                    "count": t.count,
                    "level": t.level,
                    "severity": t.severity,
                    "anomalous": t.anomalous,
                },
            )
            if u is None:
                continue
            u.features["log_count"] = float(t.count)
            u.features["severity"] = float(t.severity)
            u.features["rarity"] = 1.0 / (1.0 + t.count)
            if t.severity >= 4 or t.count == 1:
                u.protection = max(u.protection, Protection.ANCHORED)
                u.features["anomaly"] = 1.0

            compact = render_template(t, with_slots=False)
            detailed = render_template(t, with_slots=True, with_exemplar=True)
            verbatim_spans = t.spans[:_MAX_VERBATIM_LINES]
            verbatim = "\n".join(text[a:b] for a, b in verbatim_spans)
            if len(t.spans) > _MAX_VERBATIM_LINES:
                verbatim += f"\n... x{t.count - _MAX_VERBATIM_LINES} more"

            specs = [("compact", compact, 0.4), ("detail", detailed, 0.75), ("verbatim", verbatim, 1.0)]
            levels: list[Level] = [Level("drop", "", 0, 0.0, {}, set())]
            seen: set[str] = set()
            for name, body, fid in specs:
                body = body.strip()
                if not body or body in seen:
                    continue
                seen.add(body)
                levels.append(
                    Level(
                        name=name,
                        text=body,
                        tokens=ctx.tok.count(body) + 1,
                        fidelity=fid,
                        obligations={k for _c, k, _l, _s, _e in ctx.ex_prose.extract(body)},
                    )
                )
            levels = [levels[0]] + sorted(levels[1:], key=lambda lv: lv.tokens)
            u.levels = levels
            u.level = len(levels) - 1
            u.tokens = levels[-1].tokens

            if prev_uid is not None:
                ctx.cir.add_edge(prev_uid, u.uid, EdgeKind.ADJACENT, 0.4)
            prev_uid = u.uid
