"""Pipeline interface and registry.

A *pipeline* is a content-type-specific front-end: it turns a region of raw text
into typed IR units plus the structural edges that make dependency closure
meaningful for that content type.  Everything after the front-end (scoring,
selection, budgeting, rendering, verification) is shared -- which is why adding
a new content type costs ~150 lines and zero changes to the optimiser.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Config
from ..detect.doctype import Region
from ..ir.obligations import ObligationExtractor, annotate_unit
from ..tokenization import CachedTokenizer
from ..types import CIR, Document, EdgeKind, Protection, Span, Unit, UnitKind


@dataclass
class BuildContext:
    """Shared state handed to every pipeline."""

    cir: CIR
    cfg: Config
    tok: CachedTokenizer
    doc: Document
    region: Region
    ex_prose: ObligationExtractor
    ex_code: ObligationExtractor
    heading_stack: list[int] = field(default_factory=list)
    counter: dict[str, int] = field(default_factory=dict)

    # -- emission -----------------------------------------------------
    def emit(
        self,
        kind: UnitKind,
        start: int,
        end: int,
        text: str | None = None,
        depth: int = 0,
        parent: int | None = None,
        segment: str | None = None,
        protection: Protection = Protection.ELASTIC,
        code: bool = False,
        meta: dict | None = None,
    ) -> Unit | None:
        """Create a unit with exact provenance and annotated obligations."""
        raw = self.doc.text[start:end] if text is None else text
        if not raw.strip():
            return None
        u = Unit(
            uid=-1,
            doc_id=self.doc.doc_id,
            kind=kind,
            span=Span(self.doc.doc_id, start, end),
            text=raw,
            order=len(self.cir.units),
            depth=depth,
            parent=parent if parent is not None else (self.heading_stack[-1] if self.heading_stack else None),
            segment=segment or self.doc.role,
            protection=protection,
            meta=meta or {},
        )
        u.meta.setdefault("region", (self.region.start, self.region.end))
        self.cir.add_unit(u)
        u.tokens = self.tok.count(u.text) + 1  # +1 for the joining newline
        ex = self.ex_code if code else self.ex_prose
        for ob in annotate_unit(u, ex):
            self.cir.add_obligation(ob)
        if u.parent is not None and u.parent != u.uid:
            # a unit is meaningless without its heading / enclosing scope
            self.cir.add_edge(u.uid, u.parent, EdgeKind.REQUIRES, 0.8)
            self.cir.add_edge(u.parent, u.uid, EdgeKind.CONTAINS, 1.0)
        return u

    def next_id(self, prefix: str) -> str:
        n = self.counter.get(prefix, 0) + 1
        self.counter[prefix] = n
        return f"{prefix}{n}"


class Pipeline(abc.ABC):
    """Content-type front-end."""

    name: str = "base"
    #: Whether unit order carries meaning (blocks salience reordering).
    order_sensitive: bool = False
    #: Whether intra-unit token surgery is permitted for this content type.
    allow_intra_unit: bool = True

    @abc.abstractmethod
    def build(self, ctx: BuildContext) -> None:
        """Populate ``ctx.cir`` with units and structural edges."""

    def post_select(self, cir: CIR, selected: set[int]) -> set[int]:
        """Hook: content-type-specific repair of the selected set."""
        return selected

    def render_unit(self, unit: Unit) -> str:
        return unit.surface


_REGISTRY: dict[str, Callable[[], Pipeline]] = {}


def register(label: str) -> Callable[[type[Pipeline]], type[Pipeline]]:
    def deco(cls: type[Pipeline]) -> type[Pipeline]:
        _REGISTRY[label] = cls
        return cls

    return deco


def get_pipeline(label: str) -> Pipeline:
    from . import code as _code  # noqa: F401  (registration side-effects)
    from . import conversation as _conv  # noqa: F401
    from . import jsonp as _json  # noqa: F401
    from . import logs as _logs  # noqa: F401
    from . import prose as _prose  # noqa: F401
    from . import tabular as _tab  # noqa: F401

    factory = _REGISTRY.get(label)
    if factory is None:
        factory = _REGISTRY["prose"]
    return factory()


def registry_labels() -> list[str]:
    from . import code as _code  # noqa: F401
    from . import conversation as _conv  # noqa: F401
    from . import jsonp as _json  # noqa: F401
    from . import logs as _logs  # noqa: F401
    from . import prose as _prose  # noqa: F401
    from . import tabular as _tab  # noqa: F401

    return sorted(_REGISTRY)
