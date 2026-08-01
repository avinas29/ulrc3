"""Recovery layer: compression as a *lossy but addressable* transformation.

Every dropped unit keeps a stable handle and its original text.  The compressed
prompt advertises the handles it dropped::

    #CUT 219u 6.1k expand:31,44,77

An agent that finds the compressed context insufficient can call back for the
original span instead of failing or hallucinating.  This turns a one-shot lossy
transform into a **two-level memory hierarchy**: the compressed context is L1,
the residual store is L2, and the model performs its own page-faults.

That is not available to any token-level compressor, because a token-level
compressor has no addressable units to fault back in.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .types import CompressionResult


@dataclass
class Residual:
    handle: str
    text: str
    doc_id: str
    created: float = field(default_factory=time.time)
    hits: int = 0


class ResidualStore:
    """Bounded LRU store of dropped spans, keyed by ``(session, handle)``.

    Thread-safe.  Memory is capped by *characters*, not entries, because one
    dropped appendix can outweigh ten thousand dropped sentences.
    """

    def __init__(self, max_chars: int = 32 * 1024 * 1024, ttl_seconds: float = 3600.0) -> None:
        self.max_chars = max_chars
        self.ttl = ttl_seconds
        self._data: OrderedDict[str, dict[str, Residual]] = OrderedDict()
        self._chars = 0
        self._lock = threading.Lock()

    # -- write ---------------------------------------------------------
    def put_result(self, session: str, result: CompressionResult) -> int:
        items = [
            Residual(handle=h, text=t, doc_id=result.doctypes and "" or "")
            for h, t in result.residuals.items()
        ]
        return self.put(session, items)

    def put(self, session: str, residuals: Iterable[Residual]) -> int:
        n = 0
        with self._lock:
            bucket = self._data.get(session)
            if bucket is None:
                bucket = {}
                self._data[session] = bucket
            for r in residuals:
                prev = bucket.get(r.handle)
                if prev is not None:
                    self._chars -= len(prev.text)
                bucket[r.handle] = r
                self._chars += len(r.text)
                n += 1
            self._data.move_to_end(session)
            self._evict_locked()
        return n

    # -- read ----------------------------------------------------------
    def get(self, session: str, handle: str) -> str | None:
        with self._lock:
            bucket = self._data.get(session)
            if not bucket:
                return None
            r = bucket.get(str(handle))
            if r is None:
                return None
            if self.ttl and (time.time() - r.created) > self.ttl:
                self._chars -= len(r.text)
                bucket.pop(str(handle), None)
                return None
            r.hits += 1
            self._data.move_to_end(session)
            return r.text

    def expand(self, session: str, handles: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for h in handles:
            v = self.get(session, str(h))
            if v is not None:
                out[str(h)] = v
        return out

    def drop(self, session: str) -> None:
        with self._lock:
            bucket = self._data.pop(session, None)
            if bucket:
                self._chars -= sum(len(r.text) for r in bucket.values())

    # -- maintenance ---------------------------------------------------
    def _evict_locked(self) -> None:
        now = time.time()
        if self.ttl:
            for session in list(self._data):
                bucket = self._data[session]
                stale = [h for h, r in bucket.items() if now - r.created > self.ttl]
                for h in stale:
                    self._chars -= len(bucket[h].text)
                    bucket.pop(h, None)
                if not bucket:
                    self._data.pop(session, None)
        while self._chars > self.max_chars and self._data:
            _session, bucket = self._data.popitem(last=False)
            self._chars -= sum(len(r.text) for r in bucket.values())

    def stats(self) -> dict[str, float]:
        with self._lock:
            return {
                "sessions": len(self._data),
                "handles": sum(len(b) for b in self._data.values()),
                "chars": self._chars,
                "utilisation": self._chars / self.max_chars if self.max_chars else 0.0,
            }


#: Process-wide default store used by the server and the CLI.
DEFAULT_STORE = ResidualStore()


def annotate_with_handles(result: CompressionResult) -> CompressionResult:
    """Attach a machine-readable expansion index to the result metadata."""
    result.meta["expandable"] = sorted(result.residuals.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    return result
