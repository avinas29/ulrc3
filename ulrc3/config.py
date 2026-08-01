"""Configuration: presets, budgets, and pass toggles.

Every knob has a defensible default derived from the benchmark sweep in
``bench/``.  Nothing here is magic: the tuple (target_ratio, floor, protections)
fully determines the behaviour of the optimiser.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class Mode(str, enum.Enum):
    """Operating points on the compression / fidelity frontier."""

    LOSSLESS = "lossless"  #: structural only (dedup, boilerplate, whitespace)
    CONSERVATIVE = "conservative"  #: ~40-55% removal, near-zero risk
    BALANCED = "balanced"  #: ~65-80% removal, default
    AGGRESSIVE = "aggressive"  #: ~85-92% removal, query-aware only
    EXTREME = "extreme"  #: skeleton only; facts + constraints + symbols


MODE_TARGETS: dict[Mode, float] = {
    Mode.LOSSLESS: 0.25,
    Mode.CONSERVATIVE: 0.50,
    Mode.BALANCED: 0.75,
    Mode.AGGRESSIVE: 0.88,
    Mode.EXTREME: 0.94,
}


@dataclass
class ProtectionPolicy:
    """Which obligation classes are hard-enforced, and how deep dependency
    protection propagates."""

    enforce_numbers: bool = True
    enforce_dates: bool = True
    enforce_identifiers: bool = True
    enforce_urls: bool = True
    enforce_json_keys: bool = True
    enforce_constraints: bool = True
    enforce_negations: bool = True
    enforce_security: bool = True
    enforce_entities: bool = False  #: soft by default (recall-weighted instead)
    closure_horizon: int = 3  #: max REQUIRES-hops of inherited protection
    freeze_system: bool = True
    freeze_tools: bool = True
    freeze_query: bool = True
    max_repair_rounds: int = 4


@dataclass
class RenderPolicy:
    """Surface syntax of the emitted IR."""

    emit_header: bool = True
    emit_symbol_table: bool = True
    emit_fact_block: bool = True
    emit_delta_block: bool = True
    emit_dropped_notice: bool = True
    emit_handles: bool = True  #: stable ⟨sNN⟩ handles for recovery
    marker_style: str = "auto"  #: auto | ascii | unicode | minimal
    intern_min_count: int = 3  #: intern an entity after N occurrences
    intern_min_tokens: int = 3  #: ...and only if it costs >= N tokens
    order: str = "auto"  #: auto | original | salience
    max_line_width: int = 0  #: 0 = no rewrap


@dataclass
class PerfPolicy:
    parallel: bool = True
    max_workers: int = 0  #: 0 -> os.cpu_count()
    parallel_min_docs: int = 4
    parallel_min_chars: int = 200_000
    ppr_iterations: int = 18
    ppr_alpha: float = 0.85
    knn: int = 12  #: neighbours per unit in the phantom-attention graph
    max_units: int = 200_000
    use_numpy: bool = True
    cache_size: int = 8192


@dataclass
class Config:
    mode: Mode = Mode.BALANCED
    target_ratio: float | None = None  #: overrides the mode target
    budget_tokens: int | None = None  #: hard output budget, overrides ratio
    min_confidence: float = 0.0  #: adaptive loop target (0 disables)
    tokenizer: str = "auto"  #: auto | cl100k | o200k | hf:<name> | heuristic
    query_weight: float = 1.6
    recency_weight: float = 0.25
    redundancy_lambda: float = 0.75  #: submodular diminishing-returns strength
    #: Marginal-utility stop.  Upgrades whose benefit/cost ratio falls below
    #: this fraction of the best ratio seen are refused, so the engine returns
    #: *under* budget when the remaining material is not worth its tokens.
    #: 0 disables (spend the whole budget).
    marginal_floor: float = 0.04
    contrastive_lambda: float = 0.35  #: generic-content penalty (RAG)
    protection: ProtectionPolicy = field(default_factory=ProtectionPolicy)
    render: RenderPolicy = field(default_factory=RenderPolicy)
    perf: PerfPolicy = field(default_factory=PerfPolicy)
    doctype_override: str | None = None
    keep_residuals: bool = True
    verify: bool = True
    seed: int = 17
    #: Ablation switches, for falsifying our own claims (see bench/ablation.py).
    #: ``no_ladder`` collapses every unit to keep-or-drop, which is exactly the
    #: decision space LLMLingua-family methods operate in -- so the delta
    #: between it and the default measures what the fidelity ladder is worth.
    ablate: frozenset[str] = frozenset()

    def ablated(self, name: str) -> bool:
        return name in self.ablate

    # -- derived ------------------------------------------------------
    @property
    def effective_target(self) -> float:
        if self.target_ratio is not None:
            return max(0.0, min(0.99, self.target_ratio))
        return MODE_TARGETS[self.mode]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["ablate"] = sorted(self.ablate)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        d = dict(d or {})
        prot = d.pop("protection", None)
        rend = d.pop("render", None)
        perf = d.pop("perf", None)
        mode = d.pop("mode", None)
        abl = d.pop("ablate", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if abl:
            cfg.ablate = frozenset(abl)
        if mode:
            cfg.mode = Mode(mode)
        if prot:
            cfg.protection = ProtectionPolicy(
                **{k: v for k, v in prot.items() if k in ProtectionPolicy.__dataclass_fields__}
            )
        if rend:
            cfg.render = RenderPolicy(
                **{k: v for k, v in rend.items() if k in RenderPolicy.__dataclass_fields__}
            )
        if perf:
            cfg.perf = PerfPolicy(
                **{k: v for k, v in perf.items() if k in PerfPolicy.__dataclass_fields__}
            )
        return cfg

    @classmethod
    def preset(cls, name: str) -> Config:
        """Named production presets."""
        name = name.lower()
        if name in ("rag", "retrieval"):
            c = cls(mode=Mode.AGGRESSIVE, query_weight=2.2, contrastive_lambda=0.5)
            c.render.emit_fact_block = True
            return c
        if name in ("agent", "memory", "conversation"):
            c = cls(mode=Mode.BALANCED, recency_weight=0.55)
            c.render.order = "original"
            return c
        if name == "code":
            c = cls(mode=Mode.BALANCED)
            c.protection.enforce_identifiers = True
            c.render.order = "original"
            c.render.emit_symbol_table = False
            return c
        if name in ("logs", "observability"):
            return cls(mode=Mode.AGGRESSIVE)
        if name in ("legal", "compliance"):
            c = cls(mode=Mode.CONSERVATIVE)
            c.protection.enforce_entities = True
            c.render.order = "original"
            return c
        if name == "max":
            return cls(mode=Mode.EXTREME)
        if name == "safe":
            return cls(mode=Mode.CONSERVATIVE, min_confidence=0.9)
        return cls(mode=Mode(name) if name in {m.value for m in Mode} else Mode.BALANCED)
