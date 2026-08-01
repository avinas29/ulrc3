"""Cross-request context cache + KV-prefix-friendly emission.

Two production concerns that pure algorithms miss:

**1. Repeated documents.**  In agent loops and multi-turn RAG the same document
is compressed dozens of times.  Content-defined chunking (see
:func:`ulrc3.text.hashing.content_chunks`) gives boundaries that are stable
under edits elsewhere in the file, so a chunk keeps its identity across
requests.  We cache the compressed rendering per (chunk hash, budget class,
query class) and reuse it -- a 40-90x latency win on cache hits.

**2. Prefix-cache alignment.**  Providers cache KV state by *exact prefix*.
Emitting frozen blocks first, in a canonical order, with byte-stable content,
maximises the length of the shared prefix across requests in a session -- so the
provider's own cache does the rest.  Concretely the emission order is

    header (stable) -> system -> tools -> instruction -> query -> body

with the volatile parts (body selection, drop notice) strictly *after* the
stable parts.  A one-token change in the body therefore cannot invalidate the
cached prefix covering the system prompt and tool schemas.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from .text.hashing import blake, content_chunks


@dataclass(frozen=True)
class CacheKey:
    chunk: str
    budget_class: int
    query_class: str
    mode: str

    def token(self) -> str:
        return f"{self.chunk}:{self.budget_class}:{self.query_class}:{self.mode}"


class ChunkCache:
    """LRU cache of compressed chunk renderings."""

    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> str | None:
        k = key.token()
        with self._lock:
            v = self._data.get(k)
            if v is None:
                self.misses += 1
                return None
            self._data.move_to_end(k)
            self.hits += 1
            return v

    def put(self, key: CacheKey, value: str) -> None:
        k = key.token()
        with self._lock:
            self._data[k] = value
            self._data.move_to_end(k)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float]:
        return {
            "entries": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


def budget_class(budget_tokens: int, buckets: int = 8) -> int:
    """Quantise the budget so near-identical budgets share cache entries.

    Exact-budget keying would make the cache useless (budgets vary by a token
    between requests); log-spaced buckets keep renderings interchangeable while
    bounding the fidelity error to one bucket.
    """
    if budget_tokens <= 0:
        return 0
    import math

    return max(0, min(buckets, int(math.log2(max(1, budget_tokens)) * buckets / 20)))


def query_class(query: str, width: int = 8) -> str:
    """Bucket queries by their content-term fingerprint.

    Query-aware renderings are only interchangeable between *similar* queries;
    hashing the sorted top terms gives a cheap equivalence class.
    """
    from .text.terms import content_terms

    terms = sorted(set(content_terms(query)))[:width]
    return blake("|".join(terms), 6) if terms else "0"


class DocumentCache:
    """Chunk-level memoisation wrapper around a compression function."""

    def __init__(self, cache: ChunkCache | None = None, min_chunk: int = 512) -> None:
        self.cache = cache or ChunkCache()
        self.min_chunk = min_chunk

    def compress_cached(
        self,
        text: str,
        compress_fn: Callable[[str], str],
        budget_tokens: int,
        query: str = "",
        mode: str = "balanced",
    ) -> tuple[str, float]:
        """Compress `text` chunk-wise, reusing previously compressed chunks.

        Returns ``(output, hit_rate)``.  Falls back to whole-document
        compression when the document is too small to chunk meaningfully.
        """
        if len(text) < self.min_chunk * 2:
            return compress_fn(text), 0.0
        chunks = content_chunks(text, avg=2048, min_size=self.min_chunk)
        bclass = budget_class(budget_tokens)
        qclass = query_class(query)
        out: list[str] = []
        hits = 0
        for start, end, digest in chunks:
            key = CacheKey(chunk=digest, budget_class=bclass, query_class=qclass, mode=mode)
            cached = self.cache.get(key)
            if cached is not None:
                out.append(cached)
                hits += 1
                continue
            rendered = compress_fn(text[start:end])
            self.cache.put(key, rendered)
            out.append(rendered)
        return "\n".join(out), (hits / len(chunks) if chunks else 0.0)


#: Process-wide default cache.
DEFAULT_CACHE = ChunkCache()
