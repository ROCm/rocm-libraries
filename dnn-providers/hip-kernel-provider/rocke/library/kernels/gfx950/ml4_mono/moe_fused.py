# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Expert-major fused-MoE scaffold for ML4.

This follows the Qwen ``moe_fused`` shape: one expert-major body owns router,
gate/up, down and combine work, while m4/m5 are paired no-op externs.  The
host plan and CTA/line assignment are present now; the device body deliberately
stops at the staged barriers until the split ML4 stages pass the GPU bitwise
gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32

from .mono_layout import GRID, HIDDEN, INTER, TOP_K

STAGES = ("m3", "m4", "m5")
GU_UNITS = INTER // 16
DN_UNITS = HIDDEN // 16


@dataclass(frozen=True)
class FusedKnobs:
    shared_ctas: int = 32
    shared_base: int = 160
    iod_split: bool = True
    line_map: str = "bytes"
    combine: str = "task"
    mutate: str = ""

    def __post_init__(self):
        if not 1 <= self.shared_ctas <= 64:
            raise ValueError("shared_ctas must be in 1..64")
        if not 0 <= self.shared_base < GRID:
            raise ValueError("shared_base must be a CTA in the grid")
        if self.shared_base + self.shared_ctas > GRID:
            raise ValueError("shared expert group must fit without CTA wrap")
        if self.line_map not in ("bytes", "groups"):
            raise ValueError("line_map must be bytes or groups")
        if self.combine not in ("task", "shared", "last"):
            raise ValueError("combine must be task, shared or last")
        if self.mutate not in ("", "combine_swap"):
            raise ValueError("unknown fused mutant")

    def suffix(self) -> str:
        default = FusedKnobs()
        parts = []
        for name in ("shared_ctas", "shared_base", "iod_split", "line_map", "combine", "mutate"):
            value = getattr(self, name)
            if value != getattr(default, name):
                parts.append(f"{name}{value}".replace("_", ""))
        return "_".join(parts)


@dataclass(frozen=True)
class FusedPlan:
    rows: int
    experts: int = 128
    knobs: FusedKnobs = FusedKnobs()

    @property
    def routed_slots(self) -> tuple[int, int]:
        return TOP_K, min(self.rows * TOP_K, self.experts)

    @property
    def shared_range(self) -> range:
        return range(self.knobs.shared_base, self.knobs.shared_base + self.knobs.shared_ctas)

    @property
    def routed_ctas(self) -> int:
        return GRID - self.knobs.shared_ctas

    def line_slots(self, unique_experts: int) -> tuple[tuple[int, int], ...]:
        if not self.knobs.iod_split:
            return ((0, unique_experts),)
        first = (unique_experts * 4 + 3) // 8
        return ((0, first), (first, unique_experts))

    def assignment(self, unique_experts: int) -> dict[int, tuple[str, tuple[int, int], tuple[int, int]]]:
        """Host-visible expert-major CTA spans (gate/up, down)."""
        spans = self.line_slots(unique_experts)
        result = {}
        for bid in range(GRID):
            if bid in self.shared_range:
                q = bid - self.knobs.shared_base
                g0, g1 = q * GU_UNITS // self.knobs.shared_ctas, (q + 1) * GU_UNITS // self.knobs.shared_ctas
                d0, d1 = q * DN_UNITS // self.knobs.shared_ctas, (q + 1) * DN_UNITS // self.knobs.shared_ctas
                result[bid] = ("shared", (g0, g1), (d0, d1))
                continue
            line = (bid % 8) // 4 if self.knobs.iod_split else 0
            lo, hi = spans[line]
            n = max(1, self.routed_ctas // (2 if self.knobs.iod_split else 1))
            pos = bid % n
            slots = max(0, hi - lo)
            g0, g1 = pos * slots * GU_UNITS // n, (pos + 1) * slots * GU_UNITS // n
            d0, d1 = pos * slots * DN_UNITS // n, (pos + 1) * slots * DN_UNITS // n
            result[bid] = ("routed", (lo * GU_UNITS + g0, lo * GU_UNITS + g1), (lo * DN_UNITS + d0, lo * DN_UNITS + d1))
        return result

    def validate(self):
        if not 1 <= self.rows <= 4:
            raise ValueError("ML4 fused rows must be in 1..4")
        if not 16 <= self.experts <= 128 or self.experts % 8:
            raise ValueError("experts must be a multiple of 8 in 16..128")
        self.assignment(self.routed_slots[1])


class FusedCtx:
    """Minimal stage context; the real expert stream is the next gate."""

    def __init__(self, b, a, plan: FusedPlan):
        self.b, self.a, self.plan = b, a, plan
        self.lds = b.smem_alloc(I32, [1024], name_hint="ml4_fused_lds")

    def emit(self):
        # Keep the same persistent CTA shape and phase ordering as qwen's
        # fused body.  The barriers are intentional: they make the scaffold
        # linkable and give the later expert-major implementation fixed seams.
        self.b.smem_store_vN(self.lds, [self.b.const_i32(0)], self.a.task, 1)
        self.b.s_barrier_bare(fence="compiler")
        self.b.smem_store_vN(self.lds, [self.b.const_i32(1)], self.a.peer_slot, 1)
        self.b.s_barrier_bare(fence="compiler")
        self.b.smem_store_vN(self.lds, [self.b.const_i32(2)], self.a.layer, 1)


__all__ = ["DN_UNITS", "FusedCtx", "FusedKnobs", "FusedPlan", "GU_UNITS", "STAGES"]
