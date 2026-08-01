"""Pass infrastructure: a compiler-style pass manager with per-pass telemetry.

Passes are ordered, independently testable, and each one records how many units
and tokens it consumed and produced.  That telemetry is not decoration: the
adaptive control loop in :mod:`ulrc3.engine` reads it, and the benchmark report
attributes compression to individual passes.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..tokenization import CachedTokenizer
from ..types import CIR, PassStat


@dataclass
class PassContext:
    cir: CIR
    cfg: Config
    tok: CachedTokenizer
    budget: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)
    stats: list[PassStat] = field(default_factory=list)
    output: str = ""

    def note(self, key: str, value: Any) -> None:
        self.scratch[key] = value


class Pass(abc.ABC):
    name: str = "pass"
    #: Skip this pass entirely when the mode is LOSSLESS.
    lossy: bool = True

    @abc.abstractmethod
    def run(self, ctx: PassContext) -> None: ...

    def note(self, ctx: PassContext) -> str:  # pragma: no cover - cosmetic
        return ""


class PassManager:
    def __init__(self, passes: list[Pass]) -> None:
        self.passes = passes

    def run(self, ctx: PassContext) -> PassContext:
        for p in self.passes:
            units_in = sum(1 for u in ctx.cir.units if u.level > 0)
            tokens_in = ctx.cir.output_tokens()
            t0 = time.perf_counter()
            p.run(ctx)
            dt = (time.perf_counter() - t0) * 1000.0
            ctx.stats.append(
                PassStat(
                    name=p.name,
                    ms=dt,
                    units_in=units_in,
                    units_out=sum(1 for u in ctx.cir.units if u.level > 0),
                    tokens_in=tokens_in,
                    tokens_out=ctx.cir.output_tokens(),
                    note=p.note(ctx),
                )
            )
        return ctx
