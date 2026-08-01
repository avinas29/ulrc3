"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocIn(BaseModel):
    text: str
    id: str | None = None
    doctype: str | None = None
    score: float = 1.0
    title: str | None = None


class CompressIn(BaseModel):
    text: str = ""
    system: str = ""
    instruction: str = ""
    query: str = ""
    tools: list[str | dict] = Field(default_factory=list)
    documents: list[str | DocIn] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    mode: str | None = None
    target_ratio: float | None = None
    budget_tokens: int | None = None
    preset: str | None = None
    doctype: str | None = None
    session: str | None = None
    keep_residuals: bool = True
    min_confidence: float | None = None
    tokenizer: str | None = None

    def to_request_kwargs(self) -> dict[str, Any]:
        docs: list[Any] = []
        for d in self.documents:
            if isinstance(d, DocIn):
                docs.append(
                    {
                        "text": d.text, "id": d.id, "doctype": d.doctype,
                        "score": d.score, "title": d.title,
                    }
                )
            else:
                docs.append(d)
        return {
            "text": self.text,
            "system": self.system,
            "instruction": self.instruction,
            "query": self.query,
            "tools": list(self.tools),
            "documents": docs,
            "messages": list(self.messages),
            "mode": self.mode,
            "target_ratio": self.target_ratio,
            "budget_tokens": self.budget_tokens,
            "doctype": self.doctype,
        }


class VerificationOut(BaseModel):
    ok: bool
    integrity: float
    critical_recall: float
    retention: float
    frozen_ok: bool
    provenance_ok: bool
    inflation_ok: bool
    syntax_ok: bool
    repairs: int
    missing: list[str] = Field(default_factory=list)


class PassOut(BaseModel):
    name: str
    ms: float
    units_in: int
    units_out: int
    tokens_in: int
    tokens_out: int
    note: str = ""


class CompressOut(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    ratio: float
    compression_rate: float
    confidence: float
    doctypes: dict[str, str] = Field(default_factory=dict)
    verification: VerificationOut
    passes: list[PassOut] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    session: str | None = None
    expandable: list[str] = Field(default_factory=list)


class ExpandIn(BaseModel):
    session: str
    handles: list[str]


class ExpandOut(BaseModel):
    spans: dict[str, str]
    missing: list[str] = Field(default_factory=list)


class EstimateIn(BaseModel):
    text: str = ""
    documents: list[str] = Field(default_factory=list)
    tokenizer: str | None = None


class EstimateOut(BaseModel):
    tokens: int
    chars: int
    doctype: str
    doctype_scores: dict[str, float] = Field(default_factory=dict)
    projected: dict[str, int] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: str
    version: str
    tokenizer: str
    uptime_seconds: float
    requests: int
    cache: dict[str, float] = Field(default_factory=dict)
    residuals: dict[str, float] = Field(default_factory=dict)
