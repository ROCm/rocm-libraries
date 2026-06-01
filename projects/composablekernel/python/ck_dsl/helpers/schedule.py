# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduling policies.

The AMDGPU backend still performs final instruction scheduling, but CK-style
kernels often need explicit scheduler hints around MFMA / LDS / VMEM groups.
This module centralizes those hints so instance builders do not hard-code magic
mask constants.

The two CK Tile scheduling modes:

* **Intrawave** -- within one wave, interleave MFMA / DS_READ / VMEM groups
  via ``__builtin_amdgcn_sched_group_barrier`` so the AMDGPU post-RA
  scheduler keeps the MFMA pipe fed without stalling on ds_read latency.
  This is what ``compv3`` / ``compv4`` produce in the ``emit_hints`` path.

* **Interwave (ping-pong)** -- across waves in the same workgroup, alternate
  wave priorities with ``s_setprio(1)`` / ``s_setprio(0)`` bookending each
  MFMA group so waves that are in MFMA win the dispatch arbitration over
  waves issuing ``buffer_load`` / ``buffer_load_lds``. Pairs with a true
  double-buffered async DMA pipeline (see :class:`SoftwarePipeline`).
  This is the canonical CK Tile ``GemmPipelineScheduler::Interwave``
  pattern (see ``gemm_pipeline_ag_bg_cr_eight_waves_base.hpp``).

The two modes compose: ``mode='interwave'`` with ``emit_hints=True`` gives
both wave-level prio bookends and intrawave group barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..analysis.ir import LlvmIrStats
from ..core.ir import IRBuilder


# AMDGPU ``sched_group_barrier`` instruction-class masks, matching the
# bit layout that the AMDGPU backend recognises. The backend honours
# these by keeping each named instruction class together as a group
# during post-RA scheduling.
#
# Reference: ``__builtin_amdgcn_sched_group_barrier(mask, count, group)``.
VALU = 0x002  # vector ALU (v_add, v_mul, v_cvt, ...)
SALU = 0x004  # scalar ALU
MFMA = 0x008  # matrix-fused multiply-add
VMEM_READ = 0x020  # global / buffer load
VMEM_WRITE = 0x040
DS_READ = 0x100  # LDS load
DS_WRITE = 0x200  # LDS store
TRANS = 0x400  # transcendentals (v_exp_f32, v_log_f32, v_rcp_f32, ...)


@dataclass(frozen=True)
class SchedulePolicy:
    """Named scheduler hint policy for an MFMA hot loop.

    Attributes:
        name: human-readable tag for IR-stat checks and logging.
        emit_hints: enable intrawave ``sched_group_barrier`` emission
            inside the MFMA loop body.
        setprio_level: prologue priority (0..3). ``None`` skips.
        mode: ``'default'`` | ``'intrawave'`` | ``'interwave'``. Drives
            the ping-pong setprio bookends around each compute step.
        compute_high_prio / compute_low_prio: priorities used by the
            interwave ping-pong (default high=1, low=0; matches CK Tile).
    """

    name: str = "mem"
    emit_hints: bool = False
    setprio_level: Optional[int] = None
    mode: str = "default"
    compute_high_prio: int = 1
    compute_low_prio: int = 0

    @classmethod
    def for_pipeline(cls, pipeline: str) -> "SchedulePolicy":
        if pipeline == "mem":
            return cls(name="mem", emit_hints=False)
        if pipeline == "compv3":
            return cls(name="compv3", emit_hints=True, mode="intrawave")
        if pipeline == "compv4":
            return cls(
                name="compv4",
                emit_hints=True,
                setprio_level=1,
                mode="intrawave",
            )
        if pipeline == "async_dma":
            return cls(
                name="async_dma",
                emit_hints=True,
                setprio_level=1,
                mode="interwave",
            )
        if pipeline in ("interwave", "pingpong", "ping_pong"):
            return cls(
                name="interwave",
                emit_hints=True,
                setprio_level=1,
                mode="interwave",
            )
        if pipeline == "intrawave":
            return cls(
                name="intrawave",
                emit_hints=True,
                setprio_level=1,
                mode="intrawave",
            )
        raise ValueError(f"unknown schedule policy {pipeline!r}")

    def emit_prologue(self, b: IRBuilder) -> None:
        if self.setprio_level is not None:
            b.s_setprio(self.setprio_level)

    def emit_compute_prologue(self, b: IRBuilder) -> None:
        """Ping-pong wave-prio bookend: high prio at MFMA start.

        Only emitted for ``mode == 'interwave'``. Pairs with
        :meth:`emit_compute_epilogue` to bracket each ``compute`` step
        in a software-pipelined loop, so MFMA-heavy waves take dispatch
        priority over waves stalled on ``buffer_load`` / VMEM.
        """
        if self.mode == "interwave":
            b.s_setprio(self.compute_high_prio)

    def emit_compute_epilogue(self, b: IRBuilder) -> None:
        """Ping-pong wave-prio bookend: low prio after MFMA."""
        if self.mode == "interwave":
            b.s_setprio(self.compute_low_prio)

    def emit_after_mfma_step(
        self,
        b: IRBuilder,
        *,
        ds_read_count: int,
        mfma_count: int,
    ) -> None:
        """Emit a DS_READ group followed by an MFMA group hint.

        These ``sched_group_barrier`` calls force the AMDGPU post-RA
        scheduler to keep ds_reads ahead of MFMAs inside one wave's
        instruction stream — the intrawave half of CK Tile's
        scheduler. No-op when ``emit_hints=False``.
        """
        if not self.emit_hints:
            return
        b.sched_group_barrier(DS_READ, int(ds_read_count), 0)
        b.sched_group_barrier(MFMA, int(mfma_count), 0)

    def emit_mfma_valu_pairs(
        self,
        b: IRBuilder,
        *,
        pairs: int,
        valu_per_pair: int = 1,
        group: int = 0,
    ) -> None:
        """Emit ``pairs`` alternating ``(MFMA, VALU)`` group hints.

        Use inside an attention softmax / online-rescale loop where each
        MFMA is followed by a small fixed number of VALU ops (sub /
        mul / cmp) and the goal is for the post-RA scheduler to keep
        the MFMA pipe fed by interleaving VALU between each MFMA.
        """
        if not self.emit_hints:
            return
        for _ in range(int(pairs)):
            b.sched_group_barrier(MFMA, 1, group)
            b.sched_group_barrier(VALU, int(valu_per_pair), group)

    def emit_mfma_trans_pairs(
        self,
        b: IRBuilder,
        *,
        pairs: int,
        trans_per_pair: int = 1,
        group: int = 0,
    ) -> None:
        """Emit ``pairs`` alternating ``(MFMA, TRANS)`` group hints.

        Used in softmax-style loops where each MFMA is followed by an
        ``exp2`` / ``log2`` transcendental: the TRANS unit and the
        MFMA pipe are independent execution resources, so encouraging
        the scheduler to interleave them maximizes overlap.
        """
        if not self.emit_hints:
            return
        for _ in range(int(pairs)):
            b.sched_group_barrier(MFMA, 1, group)
            b.sched_group_barrier(TRANS, int(trans_per_pair), group)

    def emit_mfma_setprio_bookend(
        self,
        b: IRBuilder,
        emit_mfma_fn,
    ) -> None:
        """Wrap a single ``mfma`` emission in ``s_setprio(1)/(0)``.

        The fine-grained interwave ping-pong pattern: *every* MFMA is
        bracketed so the dispatcher gives MFMA-issuing waves max
        priority over waves stalled on ``buffer_load`` / VMEM. Caller
        passes a no-arg callable that emits the MFMA op via the IR
        builder; the bookends are emitted only when
        ``mode == 'interwave'``, otherwise the callable is invoked
        directly with no bracket.
        """
        if self.mode == "interwave":
            b.s_setprio(self.compute_high_prio)
            emit_mfma_fn()
            b.s_setprio(self.compute_low_prio)
        else:
            emit_mfma_fn()

    def assert_expected_ir(self, stats: LlvmIrStats) -> None:
        """Lightweight sanity check against lowered LLVM IR stats."""
        if self.emit_hints and stats.sched_group_barriers == 0:
            raise AssertionError(
                f"schedule policy {self.name} expected sched_group_barrier ops"
            )
