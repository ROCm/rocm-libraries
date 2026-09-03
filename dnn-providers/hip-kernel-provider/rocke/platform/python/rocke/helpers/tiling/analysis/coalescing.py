# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Coalescing model -- how a wave's per-instruction memory accesses land in cache lines. PURE calc (no
matplotlib): the shared backend for the ``/coalescing`` skill, agents, and the visualization renderer.

GENERIC by construction -- a function of a **distribution** (who touches what), the memory tensor's
**strides** (the real address of each coord, never an assumed row/col-major), the **direction** (load vs store,
which sets the transaction time order), and the **dtype**. It fixes on NO particular descriptor/layout.

It does NOT re-implement vectorization: the b128 cap + the actual vector-run pattern come from
:func:`vector_transactions`. This module adds only the **cross-lane cache-line fusion**: for each wave-wide
memory instruction (one vector per lane), how many distinct cache lines the lanes' addresses touch -> fused
(coalesced) vs scattered. Deterministic from the addresses; the correctness bar is the compiled ASM
(``llvm-objdump`` widths/counts) applied by the ``/coalescing`` skill.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..transforms import as_forward_map
from .vectorization import addr_fn_from_strides, vector_transactions

__all__ = ["Instruction", "CoalescingReport", "analyze_coalescing", "assert_asm_backed"]


@dataclass(frozen=True)
class Instruction:
    """One wave-wide memory instruction (the t-th vector of every lane). ``byte_addrs`` = sorted base byte
    addresses of every lane's vector; ``vw_elems`` = elements per lane's vector (b128-capped); ``lines`` = the
    distinct cache lines touched; ``min_lines`` = the fewest those bytes could occupy; ``fused`` = touches that
    minimum (fully coalesced). ``lane_vectors`` = the per-lane vectors ``(lane, base_byte, n_elems)`` this
    instruction issues, sorted by address -- retained so a renderer can colour by LANE identity (hue) without
    re-deriving the addresses. ``footprint`` = lines/min_lines = the cache-line WORKING SET this instruction
    burns relative to ideal (the eviction-pressure signal)."""

    tstep: int
    n_lanes: int
    vw_elems: int
    byte_addrs: tuple
    lines: tuple
    min_lines: int
    fused: bool
    lane_vectors: tuple = ()

    @property
    def footprint(self) -> float:
        return len(self.lines) / self.min_lines if self.min_lines else 1.0


@dataclass(frozen=True)
class CoalescingReport:
    direction: str            # "load" | "store"
    dims: tuple               # axis NAMES, per coord component -- carried so a render never guesses which is which
    strides: tuple            # element stride per axis (parallel to dims) -- the source of the addresses
    dtype_bits: int
    line_bytes: int           # ARCH cache-line size -- explicit, never assumed (differs per arch)
    per_instruction: tuple    # of Instruction, in issue order
    worst_lines: int          # max cache lines any single instruction touches
    best_lines: int
    fully_coalesced: bool     # every instruction fused
    footprint_ratio: float    # worst instruction's lines / its minimum -- the cache WORKING-SET / eviction cost

    @property
    def stride1_axis(self) -> str:
        """The contiguous axis (stride == 1, else the smallest stride) -- BY NAME, from the given dims."""
        i = self.strides.index(1) if 1 in self.strides else min(range(len(self.strides)),
                                                                 key=lambda k: self.strides[k])
        return self.dims[i]

    @property
    def ideal_vw_elems(self) -> int:
        """The per-lane vector width (elements) this LAYOUT+strides SUPPORT (b128-ideal). What codegen SHOULD be
        able to emit; the achieved width is a separate, ASM-observed fact -- see :meth:`reconcile`."""
        return self.per_instruction[0].vw_elems if self.per_instruction else 0

    def reconcile(self, achieved_vw_elems):
        """Compare the b128-IDEAL width this layout supports against the width the compiler ACTUALLY emitted
        (from ``llvm-objdump``). Returns ``(ok, note)``. A gap is NEVER silently reconciled away -- it is
        FLAGGED as a suspected bug, because an achieved < ideal width is exactly the signal that surfaced the
        C-store b64/b128 defect (the bug may live in the viz/model OR in the asm generation; either way the
        human must look). ``ok`` is False whenever achieved != ideal (over- OR under-shoot both suspicious)."""
        ideal = self.ideal_vw_elems
        if achieved_vw_elems == ideal:
            return True, f"achieved VW={achieved_vw_elems} == b128-ideal VW={ideal} (consistent)"
        rel = "under" if achieved_vw_elems < ideal else "OVER"
        return False, (f"DISCREPANCY: achieved VW={achieved_vw_elems} {rel}shoots b128-ideal VW={ideal} "
                       f"-- SUSPECTED BUG (viz/model OR asm generation); do not dismiss, investigate")

    def summary(self) -> str:
        axes = ", ".join(f"{d} stride {s}" for d, s in zip(self.dims, self.strides))
        return (f"{self.direction} [{axes}]: contiguous axis = {self.stride1_axis}; "
                f"{len(self.per_instruction)} instr, VW={self.ideal_vw_elems} elems (b128-ideal), "
                f"lines/instr {self.best_lines}..{self.worst_lines} ({self.line_bytes}B lines), "
                f"footprint {self.footprint_ratio:g}x, "
                f"{'FULLY COALESCED' if self.fully_coalesced else 'SCATTERED'}")


def analyze_coalescing(distribution, dims, strides, dtype_bits, *, direction="store", line_bytes):
    """GENERIC coalescing report for a ``distribution`` accessing a tensor.

    ``distribution`` is a ``WarpDistributionEncoding`` OR a forward map ``{(lane,reg)->coord}``. ``dims`` are the
    axis NAMES and ``strides`` the element stride of each (parallel tuples) -- pass BOTH so there is never an
    M/N mix-up: which axis is contiguous is decided by the strides, and every report/label uses the names.
    ``direction`` is ``"store"`` (registers -> memory; transactions ordered by address) or ``"load"`` (memory ->
    registers; ordered by fill/register). ``line_bytes`` is the arch cache-line size -- REQUIRED, never assumed
    (it differs per arch). Reuses :func:`vector_transactions` for the b128-IDEAL per-lane vector runs (what the
    LAYOUT+strides SUPPORT -- the achieved width is a codegen matter, reconciled by the ASM gate), then counts
    distinct ``line_bytes`` cache lines per wave-wide instruction (the fused/scattered + footprint story).
    """
    dims, strides = tuple(dims), tuple(int(s) for s in strides)
    if len(dims) != len(strides):
        raise ValueError(f"dims {dims} and strides {strides} must be parallel (one stride per named axis)")
    fwd = as_forward_map(distribution)                       # {(lane,reg) -> coord}
    addr = addr_fn_from_strides(strides)
    ebytes = max(1, dtype_bits // 8)
    order = "addr" if direction == "store" else "reg"        # store = memory order; load = register/fill order
    ts, _maxt = vector_transactions(fwd, lambda r, c: addr(r, c), dtype_bits, order_by=order, max_bits=128)

    by_inst: dict = {}
    for (lane, reg), t in ts.items():
        by_inst.setdefault(t, {}).setdefault(lane, []).append(fwd[(lane, reg)])
    insts = []
    for t in sorted(by_inst):
        lanes = by_inst[t]
        baddrs = sorted({addr(*c) * ebytes for coords in lanes.values() for c in coords})
        lines = sorted({b // line_bytes for b in baddrs})
        vw = max(len(c) for c in lanes.values())
        min_lines = -(-len(baddrs) * ebytes // line_bytes)   # ceil: fewest lines these bytes could occupy
        lane_vectors = tuple(sorted(
            (lane, min(addr(*c) for c in coords) * ebytes, len(coords))
            for lane, coords in lanes.items()))
        insts.append(Instruction(t, len(lanes), vw, tuple(baddrs), tuple(lines), min_lines,
                                 len(lines) <= min_lines, lane_vectors))
    worst = max((len(i.lines) for i in insts), default=0)
    best = min((len(i.lines) for i in insts), default=0)
    footprint = max((i.footprint for i in insts), default=1.0)
    return CoalescingReport(direction, dims, strides, dtype_bits, line_bytes, tuple(insts), worst, best,
                            all(i.fused for i in insts), footprint)


def assert_asm_backed(report, achieved_vw_elems):
    """HARD gate: the compiled ASM MUST back the model. Call with the per-lane vector width the kernel ACTUALLY
    emitted (from ``llvm-objdump``); RAISES if it disagrees with the b128-ideal width the layout supports. This
    is intentionally fatal -- an ideal-vs-achieved gap is never a warning to skim past: it is a bug in EITHER
    the viz/model OR the asm generation (that gap is exactly how the C-store b64/b128 defect was found), and a
    test standing on this report must FAIL until a human resolves it. Returns the (ok, note) on success."""
    ok, note = report.reconcile(achieved_vw_elems)
    if not ok:
        raise AssertionError(f"ASM does not back the coalescing model -- {note}")
    return ok, note
