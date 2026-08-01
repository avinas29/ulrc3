"""Request/response surface.

Structure is information.  A caller that tells us "this is the system prompt,
this is the user's query, these are retrieved chunks with these scores" gets a
materially better result than one that hands over a blob -- because instruction
isolation stops being inference and becomes a fact.

We accept both.  A blob is *structured by inference* (role markers, position,
imperative detection); structured input skips that step and its error bars.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Doc:
    """One context document."""

    text: str
    doc_id: str | None = None
    doctype: str | None = None  #: force a pipeline; None = auto-detect
    score: float = 1.0  #: retriever score / prior importance
    title: str | None = None
    role: str = "body"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    """A compression request.

    ``system``, ``tools``, ``instruction`` and ``query`` are **never compressed**
    -- they are frozen verbatim.  Everything else competes for the budget.
    """

    text: str = ""  #: raw blob (auto-structured)
    system: str = ""
    instruction: str = ""
    query: str = ""
    tools: Sequence[str | dict] = field(default_factory=list)
    documents: Sequence[str | Doc | dict] = field(default_factory=list)
    messages: Sequence[dict] = field(default_factory=list)
    target_ratio: float | None = None
    budget_tokens: int | None = None
    mode: str | None = None
    doctype: str | None = None

    # -- normalisation -------------------------------------------------
    def docs(self) -> list[Doc]:
        out: list[Doc] = []
        if self.system.strip():
            out.append(Doc(text=self.system, doc_id="system", role="system", doctype="prose"))
        for i, t in enumerate(self.tools):
            body = t if isinstance(t, str) else json.dumps(t, indent=1, ensure_ascii=False)
            out.append(Doc(text=body, doc_id=f"tool{i}", role="tools", doctype="json"))
        if self.instruction.strip():
            out.append(
                Doc(text=self.instruction, doc_id="instruction", role="instruction", doctype="prose")
            )
        if self.messages:
            body = "\n".join(
                f"{m.get('role', 'user')}: {_content(m)}" for m in self.messages
            )
            out.append(Doc(text=body, doc_id="history", role="body", doctype="conversation"))
        for i, d in enumerate(self.documents):
            if isinstance(d, Doc):
                doc = d
            elif isinstance(d, dict):
                doc = Doc(
                    text=d.get("text", ""),
                    doc_id=d.get("id") or d.get("doc_id"),
                    doctype=d.get("doctype") or d.get("type"),
                    score=float(d.get("score", 1.0)),
                    title=d.get("title"),
                    meta=d.get("meta", {}) or {},
                )
            else:
                doc = Doc(text=str(d))
            doc.doc_id = doc.doc_id or f"doc{i}"
            out.append(doc)
        if self.text.strip():
            out.append(Doc(text=self.text, doc_id="main", role="body", doctype=self.doctype))
        if self.query.strip():
            out.append(Doc(text=self.query, doc_id="query", role="query", doctype="prose"))
        return [d for d in out if d.text and d.text.strip()]

    def source_text(self) -> str:
        return "\n".join(d.text for d in self.docs())

    @classmethod
    def coerce(cls, obj: Any, **kw: Any) -> Request:
        if isinstance(obj, Request):
            for k, v in kw.items():
                if v is not None and hasattr(obj, k):
                    setattr(obj, k, v)
            return obj
        if isinstance(obj, str):
            return cls(text=obj, **{k: v for k, v in kw.items() if v is not None})
        if isinstance(obj, dict):
            merged = {**obj, **{k: v for k, v in kw.items() if v is not None}}
            fields = set(cls.__dataclass_fields__)
            return cls(**{k: v for k, v in merged.items() if k in fields})
        if isinstance(obj, Iterable):
            return cls(documents=list(obj), **{k: v for k, v in kw.items() if v is not None})
        raise TypeError(f"cannot coerce {type(obj)!r} into a Request")


def _content(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # Anthropic/OpenAI content blocks
        parts = []
        for b in c:
            if isinstance(b, dict):
                parts.append(b.get("text") or json.dumps(b, ensure_ascii=False))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(c)
