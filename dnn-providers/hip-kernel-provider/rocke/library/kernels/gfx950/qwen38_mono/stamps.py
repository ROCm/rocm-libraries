# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-CTA stage stamps of rocKE layer kernels, in the ATOM mono format.

The same layout as ATOM's ``qwen4_exp_mono/kernels/stamps.py``
(``ATOM_Q38_STAMPS=1``), so the Qwen3.8 harness decoder
(``hackathon-prep/qwen38/harness/timeline.py``) reads rocKE and FlyDSL layer
kernels alike:

* ``POINTS`` int64 ``s_memrealtime`` ticks (100 MHz device clock) per CTA:
  point 0 = the CTA's kernel entry; per stage ``i`` of the build (its order)
  :func:`enter_ix` / :func:`exit_ix` (thread 0 entering / leaving the
  stage) and :func:`ready_ix` ``(i, w)`` (lane 0 of wave ``w`` leaving its
  last mailbox poll of the stage; 0 when the wave polled nothing);
* kept in LDS while the kernel runs (a global store per stamp would sit on
  every later ``s_waitcnt vmcnt``), then written to the scratch tail
  ``[LAYER_SLOTS][GRID][POINTS]`` at ``base`` bytes, indexed by the launch's
  layer id.

Emission: ``device_ops.memrealtime`` (inline asm), LDS stores / loads,
``sync_lds_only`` and one global store per point: both engines lower it.
"""

from __future__ import annotations

from rocke.core.ir import I32, I64, IRBuilder, Value
from rocke.helpers import device_ops as dops
from rocke.helpers.lds_arena import LdsEntry, LdsView

from . import irops as iro

__all__ = [
    "GRID",
    "LAYER_SLOTS",
    "LDS_ENTRY",
    "PER_STAGE",
    "POINTS",
    "WAVES",
    "Stamps",
    "enter_ix",
    "exit_ix",
    "ready_ix",
    "region_bytes",
]

GRID = 256
WAVES = 8
LAYER_SLOTS = 64
PER_STAGE = 2 + WAVES
POINTS = 160
MAX_STAGES = (POINTS - 1) // PER_STAGE  # 15, the QSA layer's
LDS_NAME = "stamps"
LDS_ENTRY = LdsEntry(LDS_NAME, 2 * POINTS)


def enter_ix(i: int) -> int:
    return 1 + i * PER_STAGE


def exit_ix(i: int) -> int:
    return 2 + i * PER_STAGE


def ready_ix(i: int, w: int = 0) -> int:
    return 3 + i * PER_STAGE + w


def region_bytes(grid: int = GRID) -> int:
    """Bytes of the scratch tail (all layer slots)."""
    return LAYER_SLOTS * grid * POINTS * 8


class Stamps:
    """Emitter of one CTA's stamps. ``lds`` holds :data:`LDS_ENTRY` (pinned:
    it lives across every LDS phase). All methods are called by every thread
    (each guards itself)."""

    def __init__(self, b: IRBuilder, lds: LdsView, stages: int, irops: bool = False):
        if stages > MAX_STAGES:
            raise ValueError(f"at most {MAX_STAGES} stamped stages, got {stages}")
        self.b = b
        self.lds = lds
        self.irops = irops  # clock reads and the flush pointer as IR ops (``irops``)
        self.stages = stages

    def _st(self, k: int | Value, v64: Value) -> None:
        b = self.b
        lo = b.trunc(v64, I32)
        hi = b.trunc(b.lshr(v64, b.const_i64(32)), I32)
        if isinstance(k, int):
            i0, i1 = 2 * k, 2 * k + 1
        else:
            i0 = b.mul(k, b.const_i32(2))
            i1 = b.add(i0, b.const_i32(1))
        self.lds.st_i32(b, LDS_NAME, i0, lo)
        self.lds.st_i32(b, LDS_NAME, i1, hi)

    def begin(self) -> None:
        """Zero this build's points, stamp point 0 (kernel entry)."""
        b = self.b
        zero = b.const_i64(0)
        with b.scf_if(dops.is_thread0(b)):
            for i in range(self.stages):
                self._st(enter_ix(i), zero)
                self._st(exit_ix(i), zero)
            self._st(0, iro.memrealtime(b, self.irops))
        # each wave zeroes its own ready points (in program order before them)
        with b.scf_if(b.cmp_eq(b.lane_id(), b.const_i32(0))):
            w = dops.wave_id(b)
            for i in range(self.stages):
                self._st(b.add(w, b.const_i32(ready_ix(i))), zero)

    def enter(self, i: int) -> None:
        with self.b.scf_if(dops.is_thread0(self.b)):
            self._st(enter_ix(i), iro.memrealtime(self.b, self.irops))

    def leave(self, i: int) -> None:
        with self.b.scf_if(dops.is_thread0(self.b)):
            self._st(exit_ix(i), iro.memrealtime(self.b, self.irops))

    def ready(self, i: int) -> None:
        """Lane 0 of the calling wave: its inputs of stage ``i`` arrived."""
        b = self.b
        with b.scf_if(b.cmp_eq(b.lane_id(), b.const_i32(0))):
            w = dops.wave_id(b)
            self._st(b.add(w, b.const_i32(ready_ix(i))), iro.memrealtime(b, self.irops))

    def flush(self, scratch: Value, base: int, layer: Value, bid: Value) -> None:
        """Kernel (or layer) end: an LDS barrier, then threads ``< POINTS``
        store the CTA's points to ``scratch + base`` (i64 address) at
        ``(layer * GRID + bid) * POINTS + tid``."""
        b = self.b
        b.sync_lds_only(fence="compiler")
        tid = b.thread_id_x()
        with b.scf_if(b.cmp_lt(tid, b.const_i32(POINTS))):
            lo = self.lds.ld_i32(b, LDS_NAME, b.mul(tid, b.const_i32(2)))
            hi = self.lds.ld_i32(
                b, LDS_NAME, b.add(b.mul(tid, b.const_i32(2)), b.const_i32(1))
            )
            v = b.lor(b.zext(lo, I64), b.shl(b.zext(hi, I64), b.const_i64(32)))
            ptr = iro.ptr(
                b, b.readfirstlane(b.add(scratch, b.const_i64(base))), self.irops, pointee=I64
            )
            idx = b.add(
                b.mul(b.add(b.mul(layer, b.const_i32(GRID)), bid), b.const_i32(POINTS)),
                tid,
            )
            b.global_store(ptr, idx, v, align=8)
