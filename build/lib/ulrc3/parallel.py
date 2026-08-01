"""Process-parallel batch compression."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .config import Config
from .types import CompressionResult

_WORKER_CFG: Config | None = None
_WORKER_ENGINE = None


def _init(cfg_dict: dict) -> None:
    global _WORKER_CFG, _WORKER_ENGINE
    from .engine import Compressor

    _WORKER_CFG = Config.from_dict(cfg_dict)
    _WORKER_ENGINE = Compressor(_WORKER_CFG)


def _work(payload: Any) -> CompressionResult:
    assert _WORKER_ENGINE is not None
    return _WORKER_ENGINE.compress(payload)


def map_compress(
    cfg: Config, payloads: Sequence[Any], workers: int = 0, **kwargs: Any
) -> list[CompressionResult]:
    """Compress a batch, using processes only when it pays for itself.

    Fork + tokenizer warm-up costs ~40-80 ms per worker, so small batches stay
    in-process.  The threshold is measured, not guessed (see bench/bench_parallel.py).
    """
    from .engine import Compressor

    n = len(payloads)
    if n == 0:
        return []
    workers = workers or (cfg.perf.max_workers or (os.cpu_count() or 4))
    total_chars = sum(len(p) if isinstance(p, str) else len(str(p)) for p in payloads)
    if not cfg.perf.parallel or n < cfg.perf.parallel_min_docs or total_chars < cfg.perf.parallel_min_chars:
        eng = Compressor(cfg)
        return [eng.compress(p, **kwargs) for p in payloads]

    workers = max(1, min(workers, n))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(cfg.to_dict(),)
    ) as pool:
        return list(pool.map(_work, payloads, chunksize=max(1, n // (workers * 4) or 1)))
