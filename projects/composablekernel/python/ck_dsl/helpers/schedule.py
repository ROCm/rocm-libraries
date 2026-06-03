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

from dataclasses import dataclass, field
from typing import Optional

from ..analysis.ir import LlvmIrStats
from ..core.ir import IRBuilder
from .atoms import MfmaAtom


# Element storage size in bytes, used by the ds_read2 16-byte heuristic and the
# ds_read issue-cycle pick. Mirrors ``sizeof(ADataType)`` in the CK schedulers.
_DTYPE_BYTES = {
    "f16": 2,
    "bf16": 2,
    "fp8e4m3": 1,
    "bf8e5m2": 1,
    "fp4": 1,  # nibble-packed; CK treats the packed storage element as 1 byte
    "fp6": 1,
    "f32": 4,
}


def _dtype_bytes(dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"no element byte-size for dtype {dtype!r}")
    return _DTYPE_BYTES[dtype]


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
class HotLoopInstList:
    """Per-iteration instruction counts for the XDLOPS GEMM hot loop.

    Pure-arithmetic port of CK's
    ``BlockwiseGemmXdlops_pipeline_hotloop_inst``
    (``ck/utility/blkgemmpipe_scheduler.hpp:20-107``) plus the ck_tile
    ``HotLoopScheduler`` derivations
    (``gemm_pipeline_ag_bg_cr_comp_v3.hpp:269-318``). Given the block tile
    geometry, the per-buffer vector / LDS widths, the operand dtypes and the
    :class:`~ck_dsl.helpers.atoms.MfmaAtom` timing, it computes every count the
    two-stage scheduler needs: A/B buffer-load, A/B LDS write/read, the C MFMA
    count, and the derived ds_read rates.

    All widths are in **elements** (e.g. ``a_buffer_load_width=8`` = an
    8-element global load, ``a_lds_read_width=8`` = AK1). ``a_repeat`` /
    ``b_repeat`` are MRepeat / NRepeat (how many XDL tiles one wave covers along
    M / N). The MFMA M/N/K and per-shape cycle come from ``atom``.

    Constructed via :meth:`from_geometry` which fills the derived fields.
    """

    # --- raw geometry inputs (mirrors the C++ template params) ---
    block_size: int = field()
    m_per_block: int = field()
    n_per_block: int = field()
    k_per_block: int = field()
    a_buffer_load_width: int = field()
    b_buffer_load_width: int = field()
    a_lds_write_width: int = field()
    b_lds_write_width: int = field()
    a_lds_read_width: int = field()
    b_lds_read_width: int = field()
    m_repeat: int = field()
    n_repeat: int = field()
    m_per_xdl: int = field()
    n_per_xdl: int = field()
    k_per_xdl: int = field()
    a_dtype_bytes: int = field()
    b_dtype_bytes: int = field()
    a_packed_size: int = field()
    b_packed_size: int = field()
    mfma_cycle: int = field()
    is_f4f6: bool = field()

    # --- derived instruction counts (filled by from_geometry) ---
    wave_num_m: int = field()
    wave_num_n: int = field()
    wave_size: int = field()
    a_buffer_load_inst_num: int = field()
    b_buffer_load_inst_num: int = field()
    a_lds_write_inst_num: int = field()
    b_lds_write_inst_num: int = field()
    a_lds_read_inst_num: int = field()
    b_lds_read_inst_num: int = field()
    c_mfma_inst_num: int = field()

    @classmethod
    def from_geometry(
        cls,
        *,
        atom: MfmaAtom,
        block_size: int,
        m_per_block: int,
        n_per_block: int,
        k_per_block: int,
        m_repeat: int,
        n_repeat: int,
        a_buffer_load_width: int,
        b_buffer_load_width: int,
        a_lds_write_width: Optional[int] = None,
        b_lds_write_width: Optional[int] = None,
        a_lds_read_width: Optional[int] = None,
        b_lds_read_width: Optional[int] = None,
        a_dtype: Optional[str] = None,
        b_dtype: Optional[str] = None,
        a_packed_size: int = 1,
        b_packed_size: int = 1,
    ) -> "HotLoopInstList":
        """Build the inst list from tile geometry + dtype + ``atom`` timing.

        The LDS read/write widths default to the atom's K-pack
        (``a_lds_read_width=atom.k_per_xdlops`` etc., matching the comp_v4
        ``A_LDS_Read_Width = KPerXDL`` convention and the common AK1==KPerXDL
        case); pass explicit widths to model a different AK1/BK1. The operand
        dtypes default to ``atom.dtype_in``.
        """
        a_dtype = a_dtype or atom.dtype_in
        b_dtype = b_dtype or atom.dtype_in
        k_pack = atom.k_per_xdlops
        a_lds_write_width = k_pack if a_lds_write_width is None else a_lds_write_width
        b_lds_write_width = k_pack if b_lds_write_width is None else b_lds_write_width
        a_lds_read_width = k_pack if a_lds_read_width is None else a_lds_read_width
        b_lds_read_width = k_pack if b_lds_read_width is None else b_lds_read_width

        m_per_xdl = atom.m
        n_per_xdl = atom.n
        k_per_xdl = atom.k_per_xdlops

        wave_num_m = m_per_block // (m_repeat * m_per_xdl)
        wave_num_n = n_per_block // (n_repeat * n_per_xdl)
        wave_size = block_size // wave_num_m // wave_num_n

        a_buffer_load_inst_num = (
            m_per_block * k_per_block // (block_size * a_buffer_load_width)
        )
        b_buffer_load_inst_num = (
            n_per_block * k_per_block // (block_size * b_buffer_load_width)
        )
        a_lds_write_inst_num = (
            m_per_block * k_per_block // (block_size * a_lds_write_width)
        )
        b_lds_write_inst_num = (
            n_per_block * k_per_block // (block_size * b_lds_write_width)
        )
        a_lds_read_inst_num = (
            wave_num_n * m_per_block * k_per_block // (block_size * a_lds_read_width)
        )
        b_lds_read_inst_num = (
            wave_num_m * n_per_block * k_per_block // (block_size * b_lds_read_width)
        )
        c_mfma_inst_num = (
            m_per_block
            * n_per_block
            * k_per_block
            // (block_size // wave_size)
            // (m_per_xdl * n_per_xdl * k_per_xdl)
        )

        return cls(
            block_size=block_size,
            m_per_block=m_per_block,
            n_per_block=n_per_block,
            k_per_block=k_per_block,
            a_buffer_load_width=a_buffer_load_width,
            b_buffer_load_width=b_buffer_load_width,
            a_lds_write_width=a_lds_write_width,
            b_lds_write_width=b_lds_write_width,
            a_lds_read_width=a_lds_read_width,
            b_lds_read_width=b_lds_read_width,
            m_repeat=m_repeat,
            n_repeat=n_repeat,
            m_per_xdl=m_per_xdl,
            n_per_xdl=n_per_xdl,
            k_per_xdl=k_per_xdl,
            a_dtype_bytes=_dtype_bytes(a_dtype),
            b_dtype_bytes=_dtype_bytes(b_dtype),
            a_packed_size=a_packed_size,
            b_packed_size=b_packed_size,
            mfma_cycle=atom.mfma_cycle,
            is_f4f6=atom.is_f4f6,
            wave_num_m=wave_num_m,
            wave_num_n=wave_num_n,
            wave_size=wave_size,
            a_buffer_load_inst_num=a_buffer_load_inst_num,
            b_buffer_load_inst_num=b_buffer_load_inst_num,
            a_lds_write_inst_num=a_lds_write_inst_num,
            b_lds_write_inst_num=b_lds_write_inst_num,
            a_lds_read_inst_num=a_lds_read_inst_num,
            b_lds_read_inst_num=b_lds_read_inst_num,
            c_mfma_inst_num=c_mfma_inst_num,
        )

    # ---- ds_read2 16-byte heuristic + issue/rate derivations ----

    def _a_read16(self) -> bool:
        return self.a_lds_read_width * self.a_dtype_bytes // self.a_packed_size == 16

    def _b_read16(self) -> bool:
        return self.b_lds_read_width * self.b_dtype_bytes // self.b_packed_size == 16

    @property
    def num_ds_read_inst_a(self) -> int:
        """A ds_read count after the ds_read2 halving (CK v3:167-170)."""
        return (
            self.a_lds_read_inst_num
            if self._a_read16()
            else self.a_lds_read_inst_num // 2
        )

    @property
    def num_ds_read_inst_b(self) -> int:
        return (
            self.b_lds_read_inst_num
            if self._b_read16()
            else self.b_lds_read_inst_num // 2
        )

    @property
    def ds_read_a_issue_cycle(self) -> int:
        """8 cycles for a 16-byte ds_read, else 4 (CK v3:185-186)."""
        return 8 if self._a_read16() else 4

    @property
    def ds_read_b_issue_cycle(self) -> int:
        return 8 if self._b_read16() else 4

    @property
    def ds_read_a_mfma_rate(self) -> int:
        """ds_reads that fit in one MFMA's shadow (CK v3:189-190)."""
        c = self.ds_read_a_issue_cycle
        return (self.mfma_cycle - 4 + 2 * c - 1) // (2 * c)

    @property
    def ds_read_b_mfma_rate(self) -> int:
        c = self.ds_read_b_issue_cycle
        return (self.mfma_cycle - 4 + 2 * c - 1) // (2 * c)

    @property
    def num_dsread_a_mfma(self) -> int:
        rate = self.ds_read_a_mfma_rate
        return (self.num_ds_read_inst_a + rate - 1) // rate

    @property
    def num_dsread_b_mfma(self) -> int:
        rate = self.ds_read_b_mfma_rate
        return (self.num_ds_read_inst_b + rate - 1) // rate


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

    name: str = field(default="mem")
    emit_hints: bool = field(default=False)
    setprio_level: Optional[int] = field(default=None)
    mode: str = field(default="default")
    compute_high_prio: int = field(default=1)
    compute_low_prio: int = field(default=0)

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

    def emit_hotloop_v3(
        self,
        b: IRBuilder,
        inst_list: HotLoopInstList,
        *,
        force: bool = False,
    ) -> None:
        """Emit the v3 two-stage ``sched_group_barrier`` HotLoop schedule.

        Exact reproduction of the classic-CK
        ``blockwise_gemm_pipeline_xdlops_v3.hpp:162-267`` HotLoopScheduler
        (identical to ck_tile ``gemm_pipeline_ag_bg_cr_comp_v3.hpp:335-389``):

        * **Stage 1** — for each A buffer-load, emit ``num_dswrite_per_issue_a``
          ``(DS-write, MFMA)`` pairs, then one VMEM read, then
          ``num_mfma_per_issue - num_dswrite_per_issue_a`` MFMAs; repeat for B.
        * **Stage 2** — drain the remaining A then B ds_reads at
          ``ds_read_mfma_rate`` per MFMA, with the final group carrying the
          remainder.

        Issued once per K-tile (matching the C++, which calls it once per hot
        iteration). No-op unless ``emit_hints`` (or ``force``). Uses only the
        existing ``b.sched_group_barrier`` op — no new IR.
        """
        if not (self.emit_hints or force):
            return

        il = inst_list
        num_buffer_load_inst_a = il.a_buffer_load_inst_num
        num_buffer_load_inst_b = il.b_buffer_load_inst_num
        num_ds_write_inst_a = il.a_lds_write_inst_num
        num_ds_write_inst_b = il.b_lds_write_inst_num
        num_mfma_inst = il.c_mfma_inst_num
        num_dsread_a_mfma = il.num_dsread_a_mfma
        num_dsread_b_mfma = il.num_dsread_b_mfma
        ds_read_a_mfma_rate = il.ds_read_a_mfma_rate
        ds_read_b_mfma_rate = il.ds_read_b_mfma_rate
        num_ds_read_inst_a = il.num_ds_read_inst_a
        num_ds_read_inst_b = il.num_ds_read_inst_b

        # stage 1
        num_mfma_stage1 = num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma)
        num_mfma_per_issue = num_mfma_stage1 // (
            num_buffer_load_inst_a + num_buffer_load_inst_b
        )
        num_dswrite_per_issue_a = num_ds_write_inst_a // num_buffer_load_inst_a
        num_dswrite_per_issue_b = num_ds_write_inst_b // num_buffer_load_inst_b

        for _ in range(num_buffer_load_inst_a):
            for _ in range(num_dswrite_per_issue_a):
                b.sched_group_barrier(DS_WRITE, 1, 0)
                b.sched_group_barrier(MFMA, 1, 0)
            b.sched_group_barrier(VMEM_READ, 1, 0)
            b.sched_group_barrier(MFMA, num_mfma_per_issue - num_dswrite_per_issue_a, 0)
        for _ in range(num_buffer_load_inst_b):
            for _ in range(num_dswrite_per_issue_b):
                b.sched_group_barrier(DS_WRITE, 1, 0)
                b.sched_group_barrier(MFMA, 1, 0)
            b.sched_group_barrier(VMEM_READ, 1, 0)
            b.sched_group_barrier(MFMA, num_mfma_per_issue - num_dswrite_per_issue_b, 0)

        # stage 2
        for i in range(num_dsread_a_mfma):
            if (num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >= (
                ds_read_a_mfma_rate
            ):
                b.sched_group_barrier(DS_READ, ds_read_a_mfma_rate, 0)
            else:
                b.sched_group_barrier(
                    DS_READ,
                    num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                    0,
                )
            b.sched_group_barrier(MFMA, 1, 0)

        for i in range(num_dsread_b_mfma):
            if (num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >= (
                ds_read_b_mfma_rate
            ):
                b.sched_group_barrier(DS_READ, ds_read_b_mfma_rate, 0)
            else:
                b.sched_group_barrier(
                    DS_READ,
                    num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                    0,
                )
            b.sched_group_barrier(MFMA, 1, 0)

    # ck_tile spells this the same; keep the comp_v3 name as an alias so call
    # sites can use either vocabulary.
    emit_compv3_hotloop = emit_hotloop_v3

    def emit_compv4_hotloop(
        self,
        b: IRBuilder,
        inst_list: HotLoopInstList,
        *,
        force: bool = False,
    ) -> None:
        """Emit the comp_v4 single-issue HotLoop schedule.

        Port of ck_tile ``gemm_pipeline_ag_bg_cr_comp_v4.hpp:259-277``. Unlike
        v3's two-stage split, v4 issues one combined per-buffer-load group:
        ``MFMA,1 / DSread,(reads/issue) / MFMA,1 / DSwrite,(writes/issue) /
        MFMA,1 / VMEM,1 / MFMA,(C_MFMA/issue - 3)`` then a trailing
        ``sched_barrier(0)`` fence. Counts come from the same
        :class:`HotLoopInstList` (v4 sets the LDS read/write width to KPerXDL,
        which is the ``from_geometry`` default). No-op unless ``emit_hints`` (or
        ``force``).
        """
        if not (self.emit_hints or force):
            return

        il = inst_list
        num_ds_read_inst = il.num_ds_read_inst_a + il.num_ds_read_inst_b
        num_ds_write_inst = il.a_lds_write_inst_num + il.b_lds_write_inst_num
        num_buffer_load_inst = il.a_buffer_load_inst_num + il.b_buffer_load_inst_num
        num_issue = num_buffer_load_inst

        for _ in range(num_buffer_load_inst):
            b.sched_group_barrier(MFMA, 1, 0)
            b.sched_group_barrier(DS_READ, num_ds_read_inst // num_issue, 0)
            b.sched_group_barrier(MFMA, 1, 0)
            b.sched_group_barrier(DS_WRITE, num_ds_write_inst // num_issue, 0)
            b.sched_group_barrier(MFMA, 1, 0)
            b.sched_group_barrier(VMEM_READ, 1, 0)
            b.sched_group_barrier(MFMA, il.c_mfma_inst_num // num_issue - 3, 0)
        b.sched_barrier(0)

    def assert_expected_ir(self, stats: LlvmIrStats) -> None:
        """Lightweight sanity check against lowered LLVM IR stats."""
        if self.emit_hints and stats.sched_group_barriers == 0:
            raise AssertionError(
                f"schedule policy {self.name} expected sched_group_barrier ops"
            )
