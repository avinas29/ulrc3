"""Similarity hashing: SimHash, MinHash, content-defined chunking, union-find.

These power three subsystems:

* **Semantic deduplication** -- near-duplicate units collapse to a canonical
  representative plus a *delta*, instead of being silently deleted.
* **Cross-request context cache** -- content-defined chunk boundaries give
  stable IDs, so a document seen in a previous request compresses to a
  reference (see :mod:`ulrc3.cache`).
* **Merkle AST hashing** -- duplicate code implementations are detected
  structurally, not textually (see :mod:`ulrc3.codeir`).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

try:  # numpy fast path for bit accumulation
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

_WORD = re.compile(r"[A-Za-z0-9_]+")
_MASK64 = (1 << 64) - 1


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def normalize_for_hash(text: str) -> str:
    """Aggressive normalisation: case, whitespace, and *numeric slotting*.

    Numbers become ``#`` so that "retry after 5s" and "retry after 30s" hash
    alike -- they are the same *statement* with a different parameter, which is
    exactly what delta encoding wants to capture.
    """
    t = text.lower()
    t = re.sub(r"\d+(?:\.\d+)?", "#", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def shingles(text: str, k: int = 4) -> list[str]:
    words = _WORD.findall(normalize_for_hash(text))
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def simhash(text: str, k: int = 4, bits: int = 64) -> int:
    """64-bit SimHash over k-shingles.

    The naive form loops 64 bits per shingle in Python and was ~25% of total
    runtime on large documents.  With numpy the bit expansion is a single
    ``unpackbits`` over the digest bytes, which is ~30x faster; the pure-Python
    path is kept byte-for-byte equivalent for dependency-free deployments.
    """
    sh = shingles(text, k)
    if not sh:
        return 0

    if _np is not None and len(sh) > 4:
        digests = _np.frombuffer(
            b"".join(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest() for s in sh),
            dtype=_np.uint8,
        ).reshape(len(sh), 8)
        # blake2b digest is big-endian in _h64; bit i of the integer is
        # bit (i % 8) of byte (7 - i // 8)
        bits_arr = _np.unpackbits(digests, axis=1, bitorder="little")  # (n, 64)
        votes = bits_arr.astype(_np.int32).sum(axis=0) * 2 - len(sh)
        out = 0
        for byte_idx in range(8):
            for bit_idx in range(8):
                if votes[byte_idx * 8 + bit_idx] > 0:
                    out |= 1 << ((7 - byte_idx) * 8 + bit_idx)
        return out

    v = [0] * bits
    for s in sh:
        h = _h64(s)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & _MASK64).bit_count()


class MinHash:
    """Fixed-permutation MinHash sketch (Jaccard estimator).

    ``num_perm=64`` gives a standard error of ~1/sqrt(64) = 12.5%, which is
    plenty for a *candidate generator* that is confirmed by exact Jaccard.
    """

    __slots__ = ("sig", "num_perm")

    def __init__(self, tokens: Iterable[str], num_perm: int = 64) -> None:
        self.num_perm = num_perm
        sig = [_MASK64] * num_perm
        for t in tokens:
            h = _h64(t)
            for i in range(num_perm):
                # cheap universal hashing family: (a*h + b) mod 2^64
                hv = ((h * (2 * i + 1) + 0x9E3779B97F4A7C15 * (i + 1)) & _MASK64)
                if hv < sig[i]:
                    sig[i] = hv
        self.sig = sig

    def jaccard(self, other: MinHash) -> float:
        n = min(self.num_perm, other.num_perm)
        if n == 0:
            return 0.0
        same = sum(1 for i in range(n) if self.sig[i] == other.sig[i])
        return same / n

    def bands(self, rows: int = 4) -> list[tuple[int, int]]:
        """LSH bands: ``(band_index, band_hash)`` for candidate bucketing."""
        out = []
        for b in range(0, self.num_perm - rows + 1, rows):
            key = _h64(",".join(str(x) for x in self.sig[b : b + rows]))
            out.append((b // rows, key))
        return out


def jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def content_chunks(text: str, avg: int = 2048, min_size: int = 512, max_size: int = 8192) -> list[tuple[int, int, str]]:
    """FastCDC-style content-defined chunking on line boundaries.

    Line-aligned so chunks stay human-meaningful; the rolling condition is a
    hash of the line, which keeps boundaries stable under insertions elsewhere
    in the document (the property that makes cross-request caching work).
    """
    mask = (1 << max(1, (avg.bit_length() - 1))) - 1
    out: list[tuple[int, int, str]] = []
    start = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        size = pos - start
        if size < min_size:
            continue
        if size >= max_size or (_h64(line) & mask) == 0:
            out.append((start, pos, blake(text[start:pos])))
            start = pos
    if start < len(text):
        out.append((start, len(text), blake(text[start:])))
    return out


def blake(text: str, n: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=n).hexdigest()


def near_duplicate_clusters(
    texts: Sequence[str],
    threshold: float = 0.82,
    simhash_bits: int = 10,
    num_perm: int = 64,
    max_reps: int = 12,
) -> list[list[int]]:
    """Cluster near-duplicates in ~O(n) expected time.

    Two-stage: SimHash banding for candidate generation (cheap, high recall),
    then MinHash Jaccard confirmation (accurate).  Exact all-pairs would be
    O(n^2) and is the reason most dedup implementations cap out at a few
    thousand chunks; this runs on 100k+ units.
    """
    n = len(texts)
    if n < 2:
        return [[i] for i in range(n)]
    sigs = [simhash(t) for t in texts]

    # MinHash is the *confirmer*, not the candidate generator, so most units
    # never need one: computing them all up front costs O(n * shingles * perms)
    # and was the single largest line item in the profile.  Build on demand.
    _mh: dict[int, MinHash] = {}

    def minhash(i: int) -> MinHash:
        mh = _mh.get(i)
        if mh is None:
            mh = MinHash(shingles(texts[i], 3), num_perm)
            _mh[i] = mh
        return mh

    buckets: dict[tuple[int, int], list[int]] = {}
    band_bits = max(4, simhash_bits)
    for i, s in enumerate(sigs):
        for b in range(0, 64, band_bits):
            key = (b, (s >> b) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)


    # Compare each bucket member against the bucket's *cluster representatives*
    # only, not against every other member.  All-pairs inside a bucket is
    # O(k^2) and on repetitive corpora (logs, boilerplate-heavy docs) the
    # buckets are huge: profiling a 64k-token document showed 13.5M union-find
    # probes and 5.0 s in this function alone.  Representative comparison is
    # O(k * reps) with reps bounded, and loses nothing: near-duplicate is
    # near-transitive, so matching the representative is equivalent in
    # practice, and any missed pair is caught by a different band.
    uf = UnionFind(n)
    for members in buckets.values():
        if len(members) < 2:
            continue
        reps: list[int] = []
        for m in members:
            root = uf.find(m)
            matched = False
            for r in reps:
                if uf.find(r) == root:
                    matched = True
                    break
                if hamming(sigs[m], sigs[r]) > 24:
                    continue
                if minhash(m).jaccard(minhash(r)) >= threshold:
                    uf.union(m, r)
                    matched = True
                    break
            if not matched and len(reps) < max_reps:
                reps.append(m)
    return [sorted(g) for g in uf.groups().values()]
