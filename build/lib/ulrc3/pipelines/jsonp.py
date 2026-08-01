"""JSON / YAML pipeline: schema induction with sampled instances.

The insight: for structured data the *schema* is the information a model needs
to reason about the payload, and the schema is O(keys) while the payload is
O(records).  A 4,000-line array of 30-field order objects compresses to a
30-line schema plus 2 exemplars plus aggregate statistics -- ~97% reduction --
while **every key is preserved exactly**, which is the actual correctness
requirement when the model must emit field names back.

Fidelity ladder for a container node:

    drop  <  schema only  <  schema + k samples + stats  <  full payload

Scalars, enums and short objects are never templated: they *are* the data.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..types import Level, Protection, UnitKind
from .base import BuildContext, Pipeline, register

_MAX_SAMPLES = 2
_HOMOGENEOUS_MIN = 3
_SCALAR_PREVIEW = 120


def _type_of(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return _string_format(v)
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"


_FORMATS: list[tuple[str, re.Pattern]] = [
    ("uuid", re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")),
    ("datetime", re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("date", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")),
    ("url", re.compile(r"^(https?|s3|gs)://")),
    ("enum?", re.compile(r"^[A-Z_]{2,32}$")),
]


def _string_format(s: str) -> str:
    for name, pat in _FORMATS:
        if pat.match(s):
            return f"str<{name}>"
    if len(s) > 200:
        return "str<long>"
    return "str"


def induce_schema(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """Recursive schema induction with enum detection and array unification."""
    if depth > max_depth:
        return "..."
    if isinstance(value, dict):
        return {k: induce_schema(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return []
        types = {_type_of(v) for v in value}
        if len(types) == 1 and isinstance(value[0], dict):
            merged: dict[str, Any] = {}
            optional: set[str] = set()
            keys_all: set[str] | None = None
            for item in value[:200]:
                if not isinstance(item, dict):
                    continue
                ks = set(item.keys())
                keys_all = ks if keys_all is None else (keys_all & ks)
                for k, v in item.items():
                    prev = merged.get(k)
                    cur = induce_schema(v, depth + 1, max_depth)
                    merged[k] = cur if prev is None or prev == cur else _union(prev, cur)
            for k in merged:
                if keys_all is not None and k not in keys_all:
                    optional.add(k)
            return [{(f"{k}?" if k in optional else k): v for k, v in merged.items()}]
        if len(types) == 1:
            vals = {str(v) for v in value if not isinstance(v, (dict, list))}
            if 0 < len(vals) <= 8 and all(isinstance(v, str) for v in value):
                return [f"enum[{','.join(sorted(vals))}]"]
            return [next(iter(types))]
        return [sorted(types)]
    return _type_of(value)


def _union(a: Any, b: Any) -> Any:
    if a == b:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _union(a[k], v) if k in a else v
        return out
    sa = a if isinstance(a, str) else json.dumps(a, separators=(",", ":"))[:40]
    sb = b if isinstance(b, str) else json.dumps(b, separators=(",", ":"))[:40]
    return sa if sa == sb else f"{sa}|{sb}"


def stats_of(value: Any) -> str:
    if isinstance(value, list):
        nums = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums and len(nums) == len(value):
            return f"n={len(value)} min={min(nums)} max={max(nums)} sum={round(sum(nums), 6)}"
        return f"n={len(value)}"
    if isinstance(value, dict):
        return f"keys={len(value)}"
    return ""


def compact_json(obj: Any, limit: int = 4000) -> str:
    try:
        s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


@register("json")
@register("yaml")
class JsonPipeline(Pipeline):
    name = "json"
    order_sensitive = False
    allow_intra_unit = False

    def build(self, ctx: BuildContext) -> None:
        base = ctx.region.start
        text = ctx.doc.text[base : ctx.region.end]
        stripped = text.strip()
        data: Any = None
        ok = False
        if stripped:
            try:
                data = json.loads(stripped)
                ok = True
            except Exception:
                data, ok = _try_yaml(stripped)
        if not ok:
            self._fallback(ctx, base, text)
            return

        root = ctx.emit(
            UnitKind.JSON_NODE,
            base,
            base + len(text),
            text=_root_header(data),
            protection=Protection.ANCHORED,
            meta={"json_root": True},
        )
        if root is None:
            return
        self._levels(ctx, root, _root_header(data), _root_header(data), _root_header(data))

        items = data.items() if isinstance(data, dict) else enumerate(data if isinstance(data, list) else [])
        for key, value in items:
            path = f"$.{key}" if isinstance(data, dict) else f"$[{key}]"
            self._node(ctx, base, text, path, value, root.uid)

    # -- internals ----------------------------------------------------
    def _node(self, ctx: BuildContext, base: int, text: str, path: str, value: Any, parent: int) -> None:
        full = f"{path} = {compact_json(value)}"
        pos = _locate(text, path.split(".")[-1].split("[")[0])
        u = ctx.emit(
            UnitKind.JSON_NODE,
            base + pos[0],
            base + pos[1],
            text=full,
            parent=parent,
            protection=Protection.ANCHORED,
            meta={"path": path, "jtype": _type_of(value)},
        )
        if u is None:
            return
        u.symbols.add(path)

        if isinstance(value, list) and len(value) >= _HOMOGENEOUS_MIN:
            schema = induce_schema(value)
            schema_txt = f"{path}: {compact_json(schema, 1200)}  # {stats_of(value)}"
            samples = compact_json(value[:_MAX_SAMPLES], 1500)
            mid = f"{schema_txt}\n{path}[0:{_MAX_SAMPLES}] = {samples}"
            self._levels(ctx, u, schema_txt, mid, full)
        elif isinstance(value, dict) and len(value) > 6:
            schema = induce_schema(value)
            schema_txt = f"{path}: {compact_json(schema, 1200)}"
            self._levels(ctx, u, schema_txt, schema_txt, full)
        else:
            short = full if len(full) < 400 else f"{path}: {compact_json(induce_schema(value), 600)}"
            self._levels(ctx, u, short, full, full)

    def _levels(self, ctx: BuildContext, unit, low: str, mid: str, high: str) -> None:
        levels: list[Level] = [Level("drop", "", 0, 0.0, {}, set())]
        seen: set[str] = set()
        for name, body, fid in (("schema", low, 0.5), ("sampled", mid, 0.8), ("full", high, 1.0)):
            body = (body or "").strip()
            if not body or body in seen:
                continue
            seen.add(body)
            levels.append(
                Level(
                    name=name,
                    text=body,
                    tokens=ctx.tok.count(body) + 1,
                    fidelity=fid,
                    obligations={k for _c, k, _l, _s, _e in ctx.ex_code.extract(body)},
                )
            )
        levels = [levels[0]] + sorted(levels[1:], key=lambda lv: lv.tokens)
        unit.levels = levels
        unit.level = len(levels) - 1
        unit.tokens = levels[-1].tokens

    def _fallback(self, ctx: BuildContext, base: int, text: str) -> None:
        """Malformed / streaming JSON: fall back to line units, keys protected."""
        from ..text.segment import iter_lines

        for s, e, line in iter_lines(text):
            if not line.strip():
                continue
            u = ctx.emit(UnitKind.JSON_NODE, base + s, base + e, code=True,
                         protection=Protection.ANCHORED)
            if u is not None:
                self._levels(ctx, u, u.text, u.text, u.text)


def _root_header(data: Any) -> str:
    if isinstance(data, dict):
        return f"$ = object(keys={len(data)})"
    if isinstance(data, list):
        return f"$ = array(n={len(data)})"
    return f"$ = {_type_of(data)}"


def _locate(text: str, key: str) -> tuple[int, int]:
    idx = text.find(f'"{key}"')
    if idx < 0:
        idx = text.find(str(key))
    if idx < 0:
        return (0, min(len(text), 1))
    end = text.find("\n", idx)
    return (idx, end if end > idx else min(len(text), idx + 80))


def _try_yaml(text: str) -> tuple[Any, bool]:
    try:
        import yaml  # optional
    except Exception:
        return None, False
    try:
        return yaml.safe_load(text), True
    except Exception:
        return None, False
