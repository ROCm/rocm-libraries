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

import math
from dataclasses import dataclass
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

# WMMA matrix ops reuse the MFMA sched-group class. The AMDGPU
# ``sched_group_barrier`` MFMA mask (0x008) matches both MFMA *and* WMMA
# instructions on the GFX12 programming model (LLVM ``SIInstrInfo::isMFMAorWMMA``
# backs ``SchedGroupMask::MFMA``), so the same 0x008 bit groups the WMMA pipe on
# RDNA3/4 (gfx11/gfx12) and gfx1250-class gfx1250. Aliased for call-site clarity.
WMMA = MFMA


@dataclass(frozen=True)
class HotLoopInstList:
    """Per-iteration instruction counts for the XDLOPS GEMM hot loop.

    Pure-arithmetic port of CK's
    ``BlockwiseGemmXdlops_pipeline_hotloop_inst``
    (``ck/utility/blkgemmpipe_scheduler.hpp:20-107``) plus the ck_tile
    ``HotLoopScheduler`` derivations
    (``gemm_pipeline_ag_bg_cr_comp_v3.hpp:269-318``). Given the block tile
    geometry, the per-buffer vector / LDS widths, the operand dtypes and the
    :class:`~rocke.helpers.atoms.MfmaAtom` timing, it computes every count the
    two-stage scheduler needs: A/B buffer-load, A/B LDS write/read, the C MFMA
    count, and the derived ds_read rates.

    All widths are in **elements** (e.g. ``a_buffer_load_width=8`` = an
    8-element global load, ``a_lds_read_width=8`` = AK1). ``a_repeat`` /
    ``b_repeat`` are MRepeat / NRepeat (how many XDL tiles one wave covers along
    M / N). The MFMA M/N/K and per-shape cycle come from ``atom``.

    Constructed via :meth:`from_geometry` which fills the derived fields.
    """

    # --- raw geometry inputs (mirrors the C++ template params) ---
    block_size: int
    m_per_block: int
    n_per_block: int
    k_per_block: int
    a_buffer_load_width: int
    b_buffer_load_width: int
    a_lds_write_width: int
    b_lds_write_width: int
    a_lds_read_width: int
    b_lds_read_width: int
    m_repeat: int
    n_repeat: int
    m_per_xdl: int
    n_per_xdl: int
    k_per_xdl: int
    a_dtype_bytes: int
    b_dtype_bytes: int
    a_packed_size: int
    b_packed_size: int
    mfma_cycle: int
    is_f4f6: bool

    # --- derived instruction counts (filled by from_geometry) ---
    wave_num_m: int
    wave_num_n: int
    wave_size: int
    a_buffer_load_inst_num: int
    b_buffer_load_inst_num: int
    a_lds_write_inst_num: int
    b_lds_write_inst_num: int
    a_lds_read_inst_num: int
    b_lds_read_inst_num: int
    c_mfma_inst_num: int

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
class WmmaHotLoopInstList:
    """Per-K-tile instruction counts for a WMMA-class GEMM hot loop (wave32).

    The RDNA / GFX12 sibling of :class:`HotLoopInstList`, usable by any WMMA
    matmul (universal GEMM WMMA path, block-scaled GEMM, the fused-MoE WMMA
    mega, WMMA attention). WMMA's per-lane operand fragments are wider than
    MFMA's (A/B carry the full atom-K, e.g. ``<16 x half>`` for the gfx1250
    16x16x32 atom) and fp16/bf16 LDS reads cap at 8 lanes (16 B), so one operand
    fragment lowers to ``ceil(frag_bytes / 16)`` ``ds_read`` instructions. These
    are the concrete per-K-tile tallies the WMMA emitters actually produce, so
    the scheduler groups line up with the real instruction stream rather than a
    wave64 MFMA-cycle model (WMMA has no ``C_MFMA_Inst_Cycle`` table).

    Counts are split by region so a caller can schedule the *compute* region
    (``ds_read`` + ``wmma``, the common single-LDS-buffer / prefetch layout where
    the LDS stores + global loads sit in a different barrier region) or a fused
    single region (all four classes).
    """

    n_wmma: int  # WMMA ops per K-tile (all warp atoms x k_atoms x B-operands)
    n_ds_read: int  # ds_read insts (operand fragment LDS loads)
    n_ds_write: int  # ds_write insts (operand LDS stores; 0 if separate region)
    n_vmem_read: int  # global/buffer loads (0 if separate region)

    @classmethod
    def from_geometry(
        cls,
        *,
        block_size: int,
        m_per_block: int,
        n_per_block: int,
        k_per_block: int,
        m_repeat: int,
        n_repeat: int,
        m_per_wmma: int,
        n_per_wmma: int,
        k_per_wmma: int,
        a_frag_len: int,
        b_frag_len: int,
        num_b_operands: int = 1,
        a_buffer_load_width: int = 0,
        b_buffer_load_width: int = 0,
        a_dtype_bytes: int = 2,
        b_dtype_bytes: int = 2,
        lds_read_bytes: int = 16,
        include_loads: bool = False,
    ) -> "WmmaHotLoopInstList":
        """Build the WMMA inst list from tile geometry + fragment widths.

        ``m_repeat`` / ``n_repeat`` are the per-warp WMMA-atom counts along M / N
        (``mfmas_per_warp_m`` / ``_n``); ``num_b_operands`` is the number of B
        matrices fed from one shared A read (1 for a plain GEMM, 2 for the fused
        gate+up). ``a_frag_len`` / ``b_frag_len`` are the per-lane operand vector
        widths (16 for the gfx1250 16x16x32 fp16/bf16 atom). With
        ``include_loads=True`` the LDS-store + global-load counts are filled (for
        a fused single-region schedule); otherwise they are 0 (compute-only
        region, the default for prefetch / single-buffer loops).
        """
        k_atoms = k_per_block // k_per_wmma
        n_wmma = m_repeat * n_repeat * num_b_operands * k_atoms

        a_reads_per_frag = max(
            1, math.ceil(a_frag_len * a_dtype_bytes / lds_read_bytes)
        )
        b_reads_per_frag = max(
            1, math.ceil(b_frag_len * b_dtype_bytes / lds_read_bytes)
        )
        # A fragment is loaded once per (m-atom, k-atom) and shared across the B
        # operands; each B operand loads its own (n-atom, k-atom) fragments.
        n_ds_read = (
            m_repeat * k_atoms * a_reads_per_frag
            + num_b_operands * n_repeat * k_atoms * b_reads_per_frag
        )

        n_ds_write = 0
        n_vmem_read = 0
        if include_loads and a_buffer_load_width and b_buffer_load_width:
            a_vecs = m_per_block * k_per_block // (block_size * a_buffer_load_width)
            b_vecs = n_per_block * k_per_block // (block_size * b_buffer_load_width)
            n_vmem_read = a_vecs + num_b_operands * b_vecs
            n_ds_write = n_vmem_read

        return cls(
            n_wmma=int(n_wmma),
            n_ds_read=int(n_ds_read),
            n_ds_write=int(n_ds_write),
            n_vmem_read=int(n_vmem_read),
        )


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
        if pipeline in ("wmma_v1", "wmma"):
            # WMMA intrawave schedule: ds_read/wmma interleave in the compute
            # region (no MFMA-cycle model; see :class:`WmmaHotLoopInstList`).
            return cls(name="wmma_v1", emit_hints=True, mode="intrawave")
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

    # ---- WMMA-class schedules (wave32; GFX11/GFX12/gfx1250) ----

    def emit_wmma_compute_schedule(
        self,
        b: IRBuilder,
        inst_list: WmmaHotLoopInstList,
        *,
        force: bool = False,
    ) -> None:
        """Intrawave ``ds_read`` -> ``wmma`` interleave for the compute region.

        The WMMA-class analog of the MFMA intrawave hint, for the common loop
        layout where the operand LDS stores + global loads sit in a *separate*
        barrier region (single-LDS-buffer / software-prefetch k-loops) so the
        compute region between syncs holds only the operand ``ds_read`` fragment
        loads and the ``wmma`` ops. The hints spread the ds_reads evenly across
        the WMMA groups (one read group then its dependent WMMA), so the post-RA
        scheduler keeps the matrix pipe fed instead of stalling on LDS latency,
        then a trailing ``sched_barrier(0)`` fences the cluster so surrounding
        VMEM / ds_write of the next tile cannot float into it. Issued once per
        K-tile. No-op unless ``emit_hints`` (or ``force``); a no-op also when the
        tile has no reads or no WMMAs.

        Uses the WMMA (== MFMA, 0x008) sched-group mask, which the GFX12 LLVM
        backend matches against WMMA via ``isMFMAorWMMA``.
        """
        if not (self.emit_hints or force):
            return
        n_read = int(inst_list.n_ds_read)
        n_wmma = int(inst_list.n_wmma)
        if n_read <= 0 or n_wmma <= 0:
            return
        per = (n_read + n_wmma - 1) // n_wmma  # ceil: reads spread over wmma groups
        remaining = n_read
        for _ in range(n_wmma):
            if remaining > 0:
                r = min(per, remaining)
                b.sched_group_barrier(DS_READ, r, 0)
                remaining -= r
            b.sched_group_barrier(WMMA, 1, 0)
        if remaining > 0:
            b.sched_group_barrier(DS_READ, remaining, 0)
        b.sched_barrier(0)

    def emit_wmma_hotloop(
        self,
        b: IRBuilder,
        inst_list: WmmaHotLoopInstList,
        *,
        force: bool = False,
    ) -> None:
        """v4-style single-region WMMA schedule (loads + stores + compute fused).

        For WMMA loops that keep the global load, LDS store, ds_read and WMMA of
        one K-tile in a *single* barrier region. Issues one combined group per
        buffer-load (``WMMA / DS_READ / WMMA / DS_WRITE / WMMA / VMEM / WMMA``),
        mirroring :meth:`emit_compv4_hotloop`, then a ``sched_barrier(0)`` fence.
        Falls back to :meth:`emit_wmma_compute_schedule` for WMMA-light tiles
        whose ``n_wmma // num_issue`` would make the trailing group negative
        (numerically identical -- both are pure scheduling hints). No-op unless
        ``emit_hints`` (or ``force``).
        """
        if not (self.emit_hints or force):
            return
        il = inst_list
        num_issue = il.n_vmem_read
        if num_issue <= 0 or il.n_wmma // num_issue < 3:
            self.emit_wmma_compute_schedule(b, il, force=force)
            return
        for _ in range(num_issue):
            b.sched_group_barrier(WMMA, 1, 0)
            b.sched_group_barrier(DS_READ, il.n_ds_read // num_issue, 0)
            b.sched_group_barrier(WMMA, 1, 0)
            b.sched_group_barrier(DS_WRITE, il.n_ds_write // num_issue, 0)
            b.sched_group_barrier(WMMA, 1, 0)
            b.sched_group_barrier(VMEM_READ, 1, 0)
            b.sched_group_barrier(WMMA, il.n_wmma // num_issue - 3, 0)
        b.sched_barrier(0)

    def assert_expected_ir(self, stats: LlvmIrStats) -> None:
        """Lightweight sanity check against lowered LLVM IR stats."""
        if self.emit_hints and stats.sched_group_barriers == 0:
            raise AssertionError(
                f"schedule policy {self.name} expected sched_group_barrier ops"
            )
