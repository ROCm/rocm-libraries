# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Flag-free gfx1250 MX GEMM: ``D = alpha * A @ B.T + beta * C`` (RCR).

Derived from the Step 6.14 kernel of :mod:`.mxfp4_tdm_gemm`, with its
feature flags turned into structure. Always on: double-buffered TDM staging split over wave
parity, scales staged by TDM (unless ``scales`` "registers"), padded and segment-interleaved
LDS, hoisted descriptors, the tile k+2 refill inside tile k, the split cluster barrier,
coarse LDS waits, a loop unrolled to one tile per LDS buffer and the vector
epilogue. The tile, wave group, cluster and the scheduling choices below are
tuning parameters; LDS padding, multicast, convert width, WMMA order (unless
``wmma_reuse``), refill position and wait counts are derived from them.

A and B are MXFP4 (E2M1), MXFP6 (E2M3), MXFP8 (E4M3) or MXBF8 (E5M2), with
E8M0 scales per 32 K (``a_dtype`` / ``b_dtype``: ``"fp4"``, ``"fp6"``,
``"fp8"``, ``"bf8"``). Both operands have the same width; FP6 and the 8-bit
types use the 16x16x128 WMMA. FP6 packs four K values in three bytes, so its
rows are 96 B per 128 K.

Tile-local rows: rocKE A row ``m`` and rocKE B row ``n`` (an N index). The
WMMA's 32 M and 16 N positions map to tile rows through :class:`RowMaps`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from math import gcd

from ...core.arch import ArchTarget
from ...core.arch.wmma_scale import scaled_matrix_layout
from ...core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...core.storage import MatrixFragmentLayout
from ...helpers.compile import (
    EXPERT_SCHEDULING_ATTR,
    REGCLASS_PRIORITY_ATTR,
    SOURCE_ORDER_ATTR,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from .mxfp4_tdm_gemm import IntOps, _IrOps, tdm_pad_fields

__all__ = [
    "IntOps",
    "LdsPad",
    "MxGemmLdsLayout",
    "MxGemmSpec",
    "build_mx_gemm",
    "derived_lds_pads",
    "is_valid_spec",
    "mx_gemm_grid",
    "mx_gemm_lds_layout",
    "mx_gemm_signature",
    "validate",
]

_ATOM_K = 128
_WAVE = 32
_WAVES = 4
_SCALE_GROUPS_PER_STEP = _ATOM_K // 32


@dataclass(frozen=True)
class _MxType:
    """An A or B element type: ``bits`` per element, ``layout`` the dtype of
    its scaled-WMMA fragment layout, and the ``mma`` shapes and ``tile_k``
    it supports."""

    bits: int
    layout: str
    mma: tuple[str, ...]
    tile_k: tuple[int, ...]


_MX_TYPES = {
    "fp4": _MxType(4, "fp4e2m1", ("32x16", "16x16"), (128, 256, 512, 1024)),
    "fp6": _MxType(6, "fp6e2m3", ("16x16",), (128, 256, 384, 512)),
    "fp8": _MxType(8, "fp8e4m3", ("16x16",), (128, 256)),
    "bf8": _MxType(8, "bf8e5m2", ("16x16",), (128, 256)),
}
MX_DTYPES = tuple(_MX_TYPES)


@dataclass(frozen=True)
class _MmaAtom:
    """One scaled WMMA of ``m`` x ``n`` x 128. A lane holds ``a_rows`` A rows
    (lane % 16, then + 16) and B row lane % 16, each row as the 16 B K chunks
    of :func:`~rocke.core.arch.wmma_scale.scaled_matrix_layout`, zero-padded
    to ``frag_words`` (16x16x128 takes 16-word operands; FP4 fills 8).
    ``op_id`` names the dtypes as ``{a}`` and ``{b}``. ``c_frag_len`` FP32
    accumulators per lane."""

    m: int
    n: int
    op_id: str
    c_frag_len: int
    a_rows: int
    frag_words: int
    c_dtype: str = "fp32"


_ATOMS = {
    "32x16": _MmaAtom(32, 16, "wmma_scale_f32_32x16x128_f4", 16, 2, 8),
    "16x16": _MmaAtom(
        16, 16, "wmma_gfx1250_f32_16x16x128_{a}_{b}_scale_e8m0_e8m0_k32", 8, 1, 16
    ),
}
MMA_SHAPES = tuple(_ATOMS)


@functools.cache
def _atom(mma: str, a_dtype: str, b_dtype: str) -> _MmaAtom:
    atom = _ATOMS[mma]
    return replace(atom, op_id=atom.op_id.format(a=a_dtype, b=b_dtype))


TILE_MN = (16, 32, 64, 128, 256)
TILE_K = (128, 256, 384, 512, 1024)
K_READS = ("chunked", "contiguous")
WMMA_REUSE = ("none", "a", "b", "ab")
SCALES = ("lds", "registers")
WAVE_GROUPS = ((2, 2), (4, 1), (1, 4))
OUT_DTYPES = ("fp8e4m3", "fp16", "bf16")
_OUT_TYPES = {"fp8e4m3": FP8E4M3, "fp16": F16, "bf16": BF16}
_OUT_BYTES = {"fp8e4m3": 1, "fp16": 2, "bf16": 2}
_DEFAULT_SIGNAL_WMMAS = 6

# TensileLite (calcLdsPad / calcLdsBlockSizePerPad / get_mxs_mt_config) for a
# K-major FP4 operand on gfx1250: pad max(GlobalReadVectorWidth 2,
# LocalReadVectorWidth 32) FP4 elements = 16 B, one pass of the 64 banks.
# The 8-bit types keep it: 16 B is also one lane's FP8 chunk read.
_DATA_PAD_BYTES = 16
_BANK_CYCLE_BYTES = 256
_MXS_BLOCK_BYTES = 256
_MXS_PAD_BYTES = 16
_B_SCALE_REGION_ALIGN = 512
# Longest pad block the TDM pad_interval field encodes; A/B blocks above it
# use TDM iterate mode (one iteration per block).
_TDM_PAD_PERIOD_MAX = 1024
_TDM_PAD_MAX_BYTES = 128 * 4
_LDS_PAD_STEP_BYTES = 8
# s_wait_dscnt immediates are 6 bits on gfx1250. A larger count is clamped
# to it, as rocisa's MaxDscnt: the wait only gets stricter.
_DSCNT_MAX = 63
# Global offsets are i32: A and B in bytes, C and D in elements.
_I32_SPAN = 1 << 31

_GL2_LINE_BYTES = 256
# th:TH_LOAD_NT (bits 0..2) | scope:SCOPE_SE (bits 3..4), as in the winner.
_GL2_PREFETCH_CPOL = 1 | (1 << 3)
_FP8_MAX = 448.0
# sched_barrier mask: VALU (0x2), SALU (0x4) and TRANS (0x400) may cross;
# WMMAs and memory operations may not.
_SCHED_ALU_ONLY = 0x2 | 0x4 | 0x400
# llvm.amdgcn.s.cluster.barrier as inline asm (see mxfp4_tdm_gemm), whole
# and split into its signal and wait halves.
_CLUSTER_BARRIER_ASM = (
    "s_barrier_signal_isfirst -1\n"
    "s_barrier_wait -1\n"
    "s_cbranch_scc0 1\n"
    "s_barrier_signal -3\n"
    "s_barrier_wait -3"
)
_CLUSTER_WAIT_ASM = "s_barrier_wait -3"
# Wave 0 alone signals, with no LDS drain. $0 is the wave index.
_CLUSTER_SIGNAL_WAVE0_ASM = (
    "s_cmp_eq_u32 $0, 0\n" "s_cbranch_scc0 1\n" "s_barrier_signal -3"
)


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LdsPad:
    """``pad_bytes`` of LDS padding after every ``block_bytes`` written
    (TensileLite LdsPad / LdsBlockSizePerPad, in bytes)."""

    pad_bytes: int = 0
    block_bytes: int = 0

    @property
    def on(self) -> bool:
        return self.pad_bytes > 0


@dataclass(frozen=True)
class MxGemmSpec:
    """Problem and tuning parameters. M, N and K are compile-time: the loop's
    tile-pair count comes from ``K``.

    ``*_lds_pad`` override the derived padding of one operand (``None``:
    derived, ``LdsPad()``: none). ``cluster_signal_wmmas`` defaults
    (``None``) to 6, clamped to the tile. Wave 0 alone signals the cluster,
    without draining LDS reads; the cluster wait publishes them.
    ``half_b_order`` places the half-B reads of the next K-step (see
    :class:`TileCompute`).

    ``k_reads`` is how a lane reads its FP6 K values of a row and K-step:
    ``"chunked"`` reads the WMMA's two 24 B K chunks as they are, in 8 B
    reads (the chunks are only 8 B aligned); ``"contiguous"`` reads the lane
    half's 48 B as three 16 B reads. A and B are then regrouped alike (K
    groups 0, 1, 2, 3 of the K-step feed the WMMA's groups 0, 2, 1, 3), which
    leaves the product unchanged once each scale word's bytes 1 and 2 are
    swapped. FP4 and the 8-bit types read their 16 B chunks either way.

    ``wmma_reuse`` (``local_prefetch`` 0, the 16x16 WMMA) orders each
    K-step's WMMAs and flags every WMMA whose next WMMA takes the same
    operand, so the hardware may keep that operand instead of reading it
    again from VGPRs: ``"b"`` N-repeat outer, M-repeat inner, flagging B;
    ``"a"`` M-repeat outer, N-repeat inner, flagging A; ``"ab"`` N-repeat
    outer with the M-repeats in alternating direction, flagging B within an
    N-repeat and A where the direction turns. ``"none"`` issues them M-repeat
    outer, without flags.

    ``scales`` is where the WMMAs' E8M0 scale words come from: ``"lds"``
    from shared memory, where the TDM stages them with the tile;
    ``"registers"`` (``local_prefetch`` 0, the 16x16 WMMA) from global memory
    straight into registers, one tile ahead. There lanes 0-15 hold the
    scales of the first half of the wave's M-repeats (N-repeats) and lanes
    16-31 those of the second half, and each WMMA selects its half
    (:class:`ScaleRegisters`).
    """

    name: str
    M: int
    N: int
    K: int
    out_dtype: str = "fp8e4m3"
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 512
    wave_group: tuple[int, int] = (2, 2)
    cluster_dim: tuple[int, int, int] = (4, 4, 1)
    gl2_prefetch_distance: int = 4
    cluster_signal_wmmas: int | None = None
    a_lds_pad: LdsPad | None = None
    b_lds_pad: LdsPad | None = None
    a_scale_lds_pad: LdsPad | None = None
    b_scale_lds_pad: LdsPad | None = None
    local_prefetch: int = 1
    half_plr: str = "b"
    half_b_order: str = "ahead"
    mma: str = "32x16"
    a_dtype: str = "fp4"
    b_dtype: str = "fp4"
    k_reads: str = "chunked"
    wmma_reuse: str = "none"
    scales: str = "lds"
    block_k: int = field(default=32, init=False)

    @property
    def atom(self) -> _MmaAtom:
        return _atom(self.mma, self.a_dtype, self.b_dtype)

    @property
    def elem_bits(self) -> int:
        """Bits of an A and a B element (``validate`` requires them equal)."""
        return _MX_TYPES[self.a_dtype].bits

    def k_bytes(self, k, ops=IntOps):
        """Bytes of ``k`` K values of an A or B row; ``k`` is an int, or an
        IR value with ``ops``. FP4 packs two values in a byte, FP6 four in
        three bytes; ``k`` is a multiple of 4 (K is one of 32)."""
        unit = gcd(self.elem_bits, 8)
        return ops.mul(ops.div(k, 8 // unit), self.elem_bits // unit)

    @property
    def step_bytes(self) -> int:
        """Bytes of one K-step of an A or B row."""
        return self.k_bytes(_ATOM_K)

    @property
    def fragment_layout(self) -> MatrixFragmentLayout:
        """The K chunks each lane reads of one A or B row and K-step."""
        return scaled_matrix_layout(
            _MX_TYPES[self.a_dtype].layout, self.atom.frag_words
        )

    @property
    def chunk_offsets(self) -> tuple[int, ...]:
        """Byte offsets, in one K-step of a row, of lane group 0's K chunks;
        lane group ``g`` reads ``g * chunk_bytes`` further."""
        layout = self.fragment_layout
        stride = layout.lane_groups * layout.chunk_bytes
        return tuple(j * stride for j in range(layout.chunks_per_lane))

    @property
    def contiguous_k(self) -> bool:
        return self.k_reads == "contiguous"

    @property
    def lane_half_bytes(self) -> int:
        """Bytes from lane half 0's first K byte of a row and K-step to lane
        half 1's."""
        layout = self.fragment_layout
        if self.contiguous_k:
            return layout.chunks_per_lane * layout.chunk_bytes
        return layout.chunk_bytes

    @property
    def lane_reads(self) -> tuple[tuple[int, int], ...]:
        """``(byte offset, words)`` of a lane's LDS reads of one A or B row
        and K-step, from its lane half's first byte, in fragment order: a
        16 B read per K chunk, FP6's 24 B chunks as three 8 B reads, or the
        lane half's contiguous 48 B as three 16 B reads (``k_reads``)."""
        layout = self.fragment_layout
        if self.contiguous_k:
            run = layout.chunks_per_lane * layout.chunk_bytes
            return tuple((offset, 4) for offset in range(0, run, 16))
        piece = 16 if layout.chunk_bytes % 16 == 0 else 8
        return tuple(
            (chunk + offset, piece // 4)
            for chunk in self.chunk_offsets
            for offset in range(0, layout.chunk_bytes, piece)
        )

    @property
    def dtype_tag(self) -> str:
        """``mxfp4``, ``mxfp6``, ``mxfp8``, ``mxbf8``, or ``mx<a><b>`` for
        mixed types."""
        tail = "" if self.b_dtype == self.a_dtype else self.b_dtype
        return f"mx{self.a_dtype}{tail}"

    @property
    def warp_m(self) -> int:
        return self.wave_group[0]

    @property
    def warp_n(self) -> int:
        return self.wave_group[1]

    @property
    def wave_repeat_m(self) -> int:
        return self.tile_m // (self.warp_m * self.atom.m)

    @property
    def wave_repeat_n(self) -> int:
        return self.tile_n // (self.warp_n * self.atom.n)

    @property
    def block_size(self) -> int:
        return _WAVE * _WAVES

    @property
    def k_steps(self) -> int:
        return self.tile_k // _ATOM_K

    @property
    def tail_steps(self) -> int:
        """K-steps of the last, partial tile when ``tile_k`` does not divide K,
        counting a partial last K-step when 128 does not divide K."""
        return -(-(self.K % self.tile_k) // _ATOM_K)

    @property
    def edge_m(self) -> bool:
        """The last M tile is partial: its loads are clipped, its stores masked."""
        return self.M % self.tile_m != 0

    @property
    def edge_n(self) -> bool:
        return self.N % self.tile_n != 0

    @property
    def wide_a(self) -> bool:
        """A's byte offsets, whole tiles counted, pass i32: address it in i64."""
        return _round_up(self.M, self.tile_m) * self.packed_k > _I32_SPAN

    @property
    def wide_b(self) -> bool:
        return _round_up(self.N, self.tile_n) * self.packed_k > _I32_SPAN

    @property
    def wide_d(self) -> bool:
        return (
            _round_up(self.M, self.tile_m) * _round_up(self.N, self.tile_n) > _I32_SPAN
        )

    @property
    def store_width(self) -> int:
        """Elements per D store and C load: at most 16 B and a divisor of N,
        so a vector is wholly inside or outside N and stays aligned."""
        width = min(self.wave_repeat_n, 16 // _OUT_BYTES[self.out_dtype])
        while self.N % width:
            width //= 2
        return width

    @property
    def packed_k(self) -> int:
        """Bytes of an A or B row."""
        return self.k_bytes(self.K)

    @property
    def has_cluster(self) -> bool:
        return tuple(self.cluster_dim[:2]) != (1, 1)

    @property
    def launch_cluster(self) -> tuple[int, int, int] | None:
        return tuple(self.cluster_dim) if self.has_cluster else None

    @property
    def half_b(self) -> bool:
        return self.local_prefetch == 1 and self.half_plr == "b"

    @property
    def signal_limit(self) -> int:
        per_step = self.wave_repeat_m * self.wave_repeat_n
        return per_step // 2 if self.half_b else per_step

    @property
    def signal_wmmas(self) -> int:
        if self.cluster_signal_wmmas is not None:
            return self.cluster_signal_wmmas
        return min(_DEFAULT_SIGNAL_WMMAS, self.signal_limit)

    @property
    def fp8_convert_width(self) -> int:
        return 8 if self.wave_repeat_n % 8 == 0 else 4

    @property
    def scale_regs(self) -> bool:
        return self.scales == "registers"

    @property
    def a_scale_reads(self) -> int:
        return 0 if self.scale_regs else max(1, self.wave_repeat_m // 4)

    @property
    def b_scale_reads(self) -> int:
        return 0 if self.scale_regs else max(1, self.wave_repeat_n // 4)

    @property
    def row_reads(self) -> int:
        """LDS read instructions of one A row or B fragment and K-step: one
        per 16 B read, one per pair of FP6's 8 B reads (LLVM pairs them into
        ``ds_load_2addr_b64``; unpaired, the waits counted from this are only
        stricter)."""
        wide = sum(1 for _, words in self.lane_reads if words == 4)
        return wide + (len(self.lane_reads) - wide + 1) // 2

    @property
    def reads_per_step(self) -> int:
        """LDS reads of one K-step: scales, then ``row_reads`` of each A row
        and B fragment (two for FP4, four for 8-bit types, three for FP6)."""
        return (
            self.a_scale_reads
            + self.b_scale_reads
            + self.row_reads
            * (self.atom.a_rows * self.wave_repeat_m + self.wave_repeat_n)
        )

    @property
    def half_b_late(self) -> bool:
        return self.half_b and self.half_b_order == "late"

    @property
    def b_half_reads(self) -> int:
        """LDS reads of half of one K-step's B fragments."""
        return self.row_reads * (self.wave_repeat_n // 2)

    @property
    def head_reads(self) -> int:
        """LDS reads of one K-step's scales and A rows."""
        return self.reads_per_step - 2 * self.b_half_reads

    @property
    def dscnt_waits(self) -> tuple[int, ...]:
        """The K-loop's ``s_wait_dscnt`` counts before clamping: the reads
        issued after the ones each wait needs (per half with half-B)."""
        if not self.local_prefetch:
            return (0,)
        if self.half_b_late:
            return (self.b_half_reads, self.head_reads)
        return (self.reads_per_step,)

    @property
    def late_pending_reads(self) -> int:
        """Reads of the other buffer that may stay in flight at the cluster
        late wait, which must drain this buffer's."""
        if not self.local_prefetch:
            return 0
        return self.head_reads if self.half_b_late else self.reads_per_step

    @property
    def lds_layout(self) -> MxGemmLdsLayout:
        return mx_gemm_lds_layout(self)

    @property
    def lds_bytes(self) -> int:
        return 2 * self.lds_layout.buffer_bytes

    def kernel_name(self) -> str:
        cx, cy, _ = self.cluster_dim
        parts = [
            f"M{self.M}N{self.N}K{self.K}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"c{cx}x{cy}",
        ]
        if self.out_dtype != "fp8e4m3":
            parts.append(self.out_dtype)
        if self.wave_group != (2, 2):
            parts.append(f"wg{self.warp_m}x{self.warp_n}")
        if self.gl2_prefetch_distance != 4:
            parts.append(f"pf{self.gl2_prefetch_distance}")
        if self.cluster_signal_wmmas is not None:
            parts.append(f"cs{self.cluster_signal_wmmas}")
        for tag, pad in (
            ("pa", self.a_lds_pad),
            ("pb", self.b_lds_pad),
            ("psa", self.a_scale_lds_pad),
            ("psb", self.b_scale_lds_pad),
        ):
            if pad is not None:
                parts.append(f"{tag}{pad.pad_bytes}x{pad.block_bytes}")
        return kernel_name_join(
            self.name,
            self.dtype_tag,
            *parts,
            flags={
                "lp0": self.local_prefetch == 0,
                "hplrnone": self.half_plr == "none",
                "hblate": self.half_b_late,
                "mma16": self.mma == "16x16",
                "kcont": self.contiguous_k,
                "ra": self.wmma_reuse == "a",
                "rb": self.wmma_reuse == "b",
                "rab": self.wmma_reuse == "ab",
                "sreg": self.scale_regs,
            },
        )


# ---------------------------------------------------------------------------
# LDS layout
# ---------------------------------------------------------------------------


def _tensile_data_block(k_bytes: int, vector_width: int) -> int:
    """calcLdsBlockSizePerPad for an UnrollMajorLDS operand: the K row,
    rounded to a bank cycle, times VectorWidth once that is longer."""
    block = _round_up(k_bytes, _BANK_CYCLE_BYTES)
    if k_bytes * vector_width > _BANK_CYCLE_BYTES:
        block = _round_up(k_bytes * vector_width, _BANK_CYCLE_BYTES)
    return block


def _tensile_scale_pad(vector_width: int) -> LdsPad:
    """get_mxs_mt_config(MatrixInstK=128, MXBlock=32, VectorWidth)."""
    per_block = (_ATOM_K // 32) * vector_width
    if vector_width < 4 or per_block % 16 or (per_block // 16) & 1:
        return LdsPad()
    return LdsPad(_MXS_PAD_BYTES, _MXS_BLOCK_BYTES)


def _bank_spread_pad(row_bytes: int) -> LdsPad:
    """16 B after every ``p`` rows of ``row_bytes``, ``p`` the row count after
    which a row's first bank repeats (64 banks of 4 B). Each pad moves the
    next ``p`` rows by one 16 B read, so the 16 rows of a lane half start on
    different banks. FP6 rows (96 B per 128 K) take it: their padded blocks
    are not a power of two, so TDM writes them in iterate mode."""
    banks = _BANK_CYCLE_BYTES // 4
    rows = banks // gcd(row_bytes // 4, banks)
    return LdsPad(_DATA_PAD_BYTES, rows * row_bytes)


def derived_lds_pads(spec: MxGemmSpec) -> dict[str, LdsPad]:
    """TensileLite's padding with its A/B relabeled to rocKE B/A and
    VectorWidth taken as the operand's wave repeat, as in the benchmark
    solution (MIWaveTile [8, 4], VectorWidthA 8, VectorWidthB 4). FP6 data
    takes :func:`_bank_spread_pad` instead."""
    k_bytes = spec.k_bytes(spec.tile_k)
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    if spec.elem_bits == 6:
        a_pad = b_pad = _bank_spread_pad(k_bytes)
    else:
        a_pad = LdsPad(_DATA_PAD_BYTES, _tensile_data_block(k_bytes, rm))
        b_pad = LdsPad(_DATA_PAD_BYTES, _tensile_data_block(k_bytes, rn))
    return {
        "a": a_pad,
        "b": b_pad,
        "a_scale": _tensile_scale_pad(rm),
        "b_scale": _tensile_scale_pad(rn),
    }


def resolved_lds_pads(spec: MxGemmSpec) -> dict[str, LdsPad]:
    derived = derived_lds_pads(spec)
    overrides = {
        "a": spec.a_lds_pad,
        "b": spec.b_lds_pad,
        "a_scale": spec.a_scale_lds_pad,
        "b_scale": spec.b_scale_lds_pad,
    }
    return {k: derived[k] if overrides[k] is None else overrides[k] for k in derived}


@dataclass(frozen=True)
class _OperandLayout:
    rows: int
    base: int
    stride: int
    pad: LdsPad
    iterate_rows: int
    iterate_stride: int


@dataclass(frozen=True)
class MxGemmLdsLayout:
    """One LDS buffer: ``[B0][A0 A1][B1][B scales][A scales]``.

    Each operand is two TDM components (one per wave pair) of ``rows`` rows;
    component ``c`` starts at ``base + c * stride``. Data pads above 1024 B
    are TDM iterate mode: ``iterate_rows`` rows per iteration, iterations
    ``iterate_stride`` apart. The offset methods take ints or IR values
    through ``ops`` so host tests and the builder share the address math.
    """

    row_bytes: int
    k_steps: int
    a: _OperandLayout
    b: _OperandLayout
    a_scale: _OperandLayout
    b_scale: _OperandLayout
    buffer_bytes: int

    def operand(self, name: str) -> _OperandLayout:
        return getattr(self, name)

    @staticmethod
    def component_base(ops, layout: _OperandLayout, component):
        return ops.add(layout.base, ops.mul(component, layout.stride))

    def row_offset(self, ops, name: str, row):
        """LDS byte offset of row ``row`` of operand ``"a"`` / ``"b"``."""
        op = self.operand(name)
        component = ops.div(row, op.rows)
        local = ops.mod(row, op.rows)
        if op.iterate_rows:
            linear = ops.add(
                ops.mul(ops.div(local, op.iterate_rows), op.iterate_stride),
                ops.mul(ops.mod(local, op.iterate_rows), self.row_bytes),
            )
        else:
            linear = ops.mul(local, self.row_bytes)
            if op.pad.on:
                linear = ops.add(
                    linear,
                    ops.mul(ops.div(linear, op.pad.block_bytes), op.pad.pad_bytes),
                )
        return ops.add(self.component_base(ops, op, component), linear)

    def scale_offset(self, ops, name: str, step, row):
        """LDS byte offset of the four E8M0 scales of row ``row`` at K-step
        ``step`` (``name`` is ``"a_scale"`` / ``"b_scale"``)."""
        op = self.operand(name)
        linear = ops.add(ops.mul(step, op.rows * 4), ops.mul(ops.mod(row, op.rows), 4))
        if op.pad.on:
            linear = ops.add(
                linear, ops.mul(ops.div(linear, op.pad.block_bytes), op.pad.pad_bytes)
            )
        return ops.add(self.component_base(ops, op, ops.div(row, op.rows)), linear)


def _pad_period(block_bytes: int) -> bool:
    """The TDM pad_interval field encodes a pad after every ``block_bytes``:
    a power of two of 8..1024 B. Other blocks are TDM iterate mode."""
    return _is_pow2(block_bytes) and 8 <= block_bytes <= _TDM_PAD_PERIOD_MAX


def _data_component(rows: int, row_bytes: int, pad: LdsPad) -> tuple[int, int, int]:
    """(component bytes, iterate rows, iterate stride) of one data component."""
    if pad.on and not _pad_period(pad.block_bytes):
        iterate_rows = pad.block_bytes // row_bytes
        stride = pad.block_bytes + pad.pad_bytes
        return -(-rows // iterate_rows) * stride, iterate_rows, stride
    size = rows * row_bytes
    if pad.on:
        size += size // pad.block_bytes * pad.pad_bytes
    return size, 0, 0


def mx_gemm_lds_layout(spec: MxGemmSpec) -> MxGemmLdsLayout:
    pads = resolved_lds_pads(spec)
    row_bytes = spec.k_bytes(spec.tile_k)
    steps = spec.k_steps
    a_rows, b_rows = spec.tile_m // 2, spec.tile_n // 2
    a_comp, a_it, a_it_stride = _data_component(a_rows, row_bytes, pads["a"])
    b_comp, b_it, b_it_stride = _data_component(b_rows, row_bytes, pads["b"])

    def scale_component(rows: int, pad: LdsPad) -> int:
        size = steps * rows * 4
        return size + (size // pad.block_bytes * pad.pad_bytes if pad.on else 0)

    a_scale_comp = scale_component(a_rows, pads["a_scale"])
    b_scale_comp = scale_component(b_rows, pads["b_scale"])
    if spec.scale_regs:
        a_scale_comp = b_scale_comp = 0
    a_total = 2 * a_comp
    b_base, a_base = 0, b_comp
    b_stride = b_comp + a_total
    data_end = a_total + 2 * b_comp
    b_scale_region = 2 * b_scale_comp
    if pads["b_scale"].on:
        b_scale_region = _round_up(b_scale_region, _B_SCALE_REGION_ALIGN)
    b_scale_base = data_end
    a_scale_base = b_scale_base + b_scale_region
    return MxGemmLdsLayout(
        row_bytes=row_bytes,
        k_steps=steps,
        a=_OperandLayout(a_rows, a_base, a_comp, pads["a"], a_it, a_it_stride),
        b=_OperandLayout(b_rows, b_base, b_stride, pads["b"], b_it, b_it_stride),
        a_scale=_OperandLayout(
            a_rows, a_scale_base, a_scale_comp, pads["a_scale"], 0, 0
        ),
        b_scale=_OperandLayout(
            b_rows, b_scale_base, b_scale_comp, pads["b_scale"], 0, 0
        ),
        buffer_bytes=a_scale_base + 2 * a_scale_comp,
    )


# ---------------------------------------------------------------------------
# Validation, signature, grid
# ---------------------------------------------------------------------------


def _pad_reason(
    name: str, pad: LdsPad, *, row_bytes: int, rows: int, chunk: int
) -> str | None:
    if pad.pad_bytes < 0 or pad.block_bytes < 0:
        return f"{name} LDS pad must be non-negative"
    if not pad.on:
        return None
    if pad.pad_bytes % _LDS_PAD_STEP_BYTES or pad.pad_bytes > _TDM_PAD_MAX_BYTES:
        return f"{name} LDS pad must be a multiple of 8 B up to {_TDM_PAD_MAX_BYTES} B"
    block = pad.block_bytes
    if not _pad_period(block):
        if row_bytes == 0 or block < 8:
            return f"{name} LDS pad block must be a power of two of 8..1024 B"
        if block % row_bytes or rows % (block // row_bytes):
            return (
                f"{name} LDS pad block must be a power of two of 8..1024 B, or "
                "whole rows dividing the component"
            )
    if block % chunk:
        return f"{name} LDS pad block must be a multiple of {chunk} B (one lane's read)"
    return None


def validate(spec: MxGemmSpec, arch: str = "gfx1250") -> str | None:
    """``None`` if ``spec`` builds, else the reason it does not."""
    if arch != "gfx1250":
        return f"mx_gemm is gfx1250-only (got {arch!r})"
    if min(spec.M, spec.N, spec.K) <= 0:
        return "M, N and K must be positive"
    if spec.K % spec.block_k:
        return (
            f"K must be a multiple of {spec.block_k}: each E8M0 scale covers "
            f"{spec.block_k} K values (got K={spec.K})"
        )
    if spec.out_dtype not in OUT_DTYPES:
        return f"out_dtype must be one of {OUT_DTYPES}"
    if spec.tile_m not in TILE_MN or spec.tile_n not in TILE_MN:
        return f"tile_m and tile_n must be in {TILE_MN}"
    if spec.a_dtype not in _MX_TYPES or spec.b_dtype not in _MX_TYPES:
        return f"a_dtype and b_dtype must be in {MX_DTYPES}"
    mx_type = _MX_TYPES[spec.a_dtype]
    if _MX_TYPES[spec.b_dtype].bits != mx_type.bits:
        return "a_dtype and b_dtype must have the same width (no mixed FP4, FP6, 8-bit)"
    if spec.tile_k not in mx_type.tile_k:
        return f"tile_k must be in {mx_type.tile_k} for {spec.a_dtype}"
    if tuple(spec.wave_group) not in WAVE_GROUPS:
        return f"wave_group must be one of {WAVE_GROUPS} (4 waves)"
    if spec.mma not in mx_type.mma:
        return f"mma must be one of {mx_type.mma} for {spec.a_dtype}"
    if spec.k_reads not in K_READS:
        return f"k_reads must be one of {K_READS}"
    if spec.contiguous_k and mx_type.bits != 6:
        return "k_reads 'contiguous' is for FP6 (other types read whole 16 B chunks)"
    if spec.wmma_reuse not in WMMA_REUSE:
        return f"wmma_reuse must be one of {WMMA_REUSE}"
    if spec.wmma_reuse != "none" and spec.local_prefetch != 0:
        return f"wmma_reuse {spec.wmma_reuse!r} needs local_prefetch 0"
    if spec.wmma_reuse != "none" and spec.mma != "16x16":
        return f"wmma_reuse {spec.wmma_reuse!r} is for the 16x16 WMMA"
    if spec.scales not in SCALES:
        return f"scales must be one of {SCALES}"
    if spec.scale_regs and spec.local_prefetch != 0:
        return "scales 'registers' needs local_prefetch 0"
    if spec.scale_regs and spec.mma != "16x16":
        return "scales 'registers' is for the 16x16 WMMA"
    atom = spec.atom
    if spec.tile_m % (spec.warp_m * atom.m) or spec.tile_n % (spec.warp_n * atom.n):
        return f"the tile must hold whole {spec.mma} WMMA atoms per wave"
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    # A, B and D offsets switch to i64 past 2^31 (wide_*); the scales' stay
    # i32. A partial last tile's rows reach the next whole tile.
    mn_span = max(_round_up(spec.M, spec.tile_m), _round_up(spec.N, spec.tile_n))
    if mn_span * -(-spec.K // _ATOM_K) * 4 > _I32_SPAN:
        return "A and B scale bytes must each be at most 2^31 (i32 offsets)"
    # K < 2 * tile_k is supported but is a poor fit for this pipeline:
    # the tile k+2 refills and GL2 prefetches past K re-load the last tile, and
    # with one or two tiles they are most of the loads. Prefer a short-K kernel.
    if spec.K < spec.tile_k:
        return "K must hold at least one full tile"
    if spec.local_prefetch not in (0, 1):
        return "local_prefetch must be 0 or 1"
    if spec.half_plr not in ("b", "none"):
        return "half_plr must be 'b' or 'none'"
    if spec.half_plr == "b" and (spec.local_prefetch != 1 or rn % 2):
        return "half_plr 'b' needs local_prefetch 1 and an even wave_repeat_n"
    if spec.half_b_order not in ("ahead", "late"):
        return "half_b_order must be 'ahead' or 'late'"
    if spec.half_b_order == "late" and not spec.half_b:
        return "half_b_order 'late' needs half_plr 'b'"
    if not 0 <= spec.signal_wmmas <= spec.signal_limit:
        return f"cluster_signal_wmmas must be in 0..{spec.signal_limit}"
    if spec.gl2_prefetch_distance != 0 and not 2 <= spec.gl2_prefetch_distance <= 8:
        return "gl2_prefetch_distance must be 0 (off) or 2..8 tiles"
    cluster = tuple(spec.cluster_dim)
    if len(cluster) != 3 or any(d < 1 for d in cluster):
        return "cluster_dim must be a positive (x, y, z) triple"
    cx, cy, cz = cluster
    if cz != 1 or cx * cy > 16:
        return "cluster_dim must be 2D with at most 16 workgroups"
    gx, gy, _ = mx_gemm_grid(spec)
    if gx % cx or gy % cy:
        return f"grid {gx}x{gy} must be divisible by cluster_dim {cx}x{cy}"
    if (
        spec.has_cluster
        and not ArchTarget.from_gfx("gfx1250").memory.has_cluster_launch
    ):
        return "gfx1250 reports no cluster launch support"
    pads = resolved_lds_pads(spec)
    row_bytes = spec.k_bytes(spec.tile_k)
    for name, rows, chunk, data in (
        ("a", spec.tile_m // 2, 16, True),
        ("b", spec.tile_n // 2, 16, True),
        ("a_scale", spec.tile_m // 2, min(16, 4 * rm), False),
        ("b_scale", spec.tile_n // 2, min(16, 4 * rn), False),
    ):
        pad = pads[name]
        reason = _pad_reason(
            name, pad, row_bytes=row_bytes if data else 0, rows=rows, chunk=chunk
        )
        if reason:
            return reason
        if (
            data
            and pad.on
            and _pad_period(pad.block_bytes)
            and pad.block_bytes % row_bytes
        ):
            return f"{name} LDS pad block must be whole {row_bytes} B rows"
    target = ArchTarget.from_gfx("gfx1250")
    accumulators = rm * rn * atom.c_frag_len
    if accumulators > target.limits.vgprs:
        return (
            f"{accumulators} accumulators per lane exceed {target.limits.vgprs} VGPRs"
        )
    if spec.lds_bytes > target.lds_capacity_bytes:
        return f"LDS {spec.lds_bytes} B exceeds {target.lds_capacity_bytes} B"
    return None


def is_valid_spec(spec: MxGemmSpec, arch: str = "gfx1250") -> tuple[bool, str]:
    reason = validate(spec, arch)
    return (False, reason) if reason else (True, "ok")


def mx_gemm_signature(spec: MxGemmSpec) -> list[dict]:
    return (
        SignatureBuilder()
        .ptr("A", "i8")
        .ptr("B", "i8")
        .ptr("A_scale", "i8")
        .ptr("B_scale", "i8")
        .ptr("C", spec.out_dtype)
        .ptr("D", spec.out_dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("alpha", "f32")
        .scalar("beta", "f32")
        .build()
    )


def mx_gemm_grid(spec: MxGemmSpec) -> tuple[int, int, int]:
    return ceil_div_grid((spec.N, spec.tile_n), (spec.M, spec.tile_m))


# ---------------------------------------------------------------------------
# Kernel roles
# ---------------------------------------------------------------------------


@dataclass
class _Kernel:
    """Values every role reads: the builder, spec, layout and the kernel's
    parameters and per-lane indices."""

    b: IRBuilder
    ops: _IrOps
    spec: MxGemmSpec
    layout: MxGemmLdsLayout
    params: dict
    c0: Value = None
    c1: Value = None
    tile_k_v: Value = None
    packed_k: Value = None
    tid: Value = None
    lane: Value = None
    lane_col: Value = None
    lane_half: Value = None
    wave: Value = None
    wave_m: Value = None
    wave_n: Value = None
    block_m: Value = None
    block_n: Value = None
    cluster_wx: Value = None
    cluster_wy: Value = None
    shared: Value = None
    shared_addr: Value = None

    def buffer_base(self, parity):
        return self.ops.mul(parity, self.layout.buffer_bytes)

    def fence(self) -> None:
        # Keep LLVM from moving LDS reads or WMMAs across the refill; ALU
        # work (descriptor and address math) may cross.
        self.b.inline_asm("", "~{memory}")
        self.b.sched_barrier(_SCHED_ALU_ONLY)

    def cluster_sync(self) -> None:
        if self.spec.has_cluster:
            self.b.inline_asm(_CLUSTER_BARRIER_ASM, "~{scc},~{memory}")
        else:
            self.b.s_barrier_bare()

    def offset64(self, row, stride, col):
        """``row * stride + col`` in i64, for offsets past 2^31."""
        b, ops = self.b, self.ops
        wide = b.mul(b.zext(ops.v(row), I64), b.zext(ops.v(stride), I64))
        return b.add(wide, b.zext(ops.v(col), I64))

    def k_atoms(self):
        """WMMA K-steps, the last one partial when 128 does not divide K;
        the swizzled scales hold that many rows, zero-padded."""
        ops, K = self.ops, self.params["K"]
        if self.spec.K % _ATOM_K:
            return ops.div(ops.add(K, _ATOM_K - 1), _ATOM_K)
        return ops.div(K, _ATOM_K)

    def clamp_k(self, k):
        b = self.b
        return b.select(
            b.cmp_lt(k, self.params["K"]), k, b.sub(self.params["K"], self.tile_k_v)
        )


class RowMaps:
    """WMMA positions to tile rows. A lane's M-repeats are adjacent A rows,
    so its A scales of a K-step are contiguous; its N-repeats are adjacent
    D columns, so one accumulator slot is one vector store."""

    def __init__(self, k: _Kernel):
        self.k = k

    def m_row(self, rm, m_local):
        ops, rm_count = self.k.ops, self.k.spec.wave_repeat_m
        base = ops.mul(self.k.wave_m, rm_count * self.k.spec.atom.m)
        return ops.add(base, ops.add(ops.mul(m_local, rm_count), rm))

    def n_row(self, rn, n_local):
        ops, rn_count = self.k.ops, self.k.spec.wave_repeat_n
        base = ops.mul(self.k.wave_n, rn_count * self.k.spec.atom.n)
        return ops.add(base, ops.add(ops.mul(n_local, rn_count), rn))


class TdmLoader:
    """Tile loads: even waves copy one rocKE B component, odd waves one A
    component (data, then scales unless ``scales`` "registers"); descriptors
    are built once and each tile patches only its LDS address, global
    address and remaining sizes."""

    def __init__(self, k: _Kernel):
        self.k = k
        b, ops, spec = k.b, k.ops, k.spec
        self.a_mask: int | Value = 0
        self.b_mask: int | Value = 0
        if spec.has_cluster:
            cluster_x, cluster_y, _ = spec.cluster_dim
            # Grid x walks N tiles and y walks M tiles: workgroups with the
            # same cluster x share rocKE B, those with the same y share A.
            col_bits = sum(1 << (row * cluster_x) for row in range(cluster_y))
            row_bits = (1 << cluster_x) - 1
            self.b_mask = b.shl(b.const_i32(col_bits), k.cluster_wx)
            self.a_mask = b.shl(b.const_i32(row_bits), ops.mul(k.cluster_wy, cluster_x))

    def lds_ptr(self, offset):
        k = self.k
        return k.b.smem_ptr_add(
            k.shared_addr, k.b.zext(k.ops.v(offset), k.shared_addr.type)
        )

    def _operand(self, operand: str):
        k, p = self.k, self.k.params
        if operand == "a":
            return k.layout.a, k.block_m, p["M"], self.a_mask
        return k.layout.b, k.block_n, p["N"], self.b_mask

    def _rows_left(self, operand: str, row0):
        """Rows from ``row0`` to the end of the operand; in a partial last
        tile a component can start past the end and then gets 0."""
        k = self.k
        b, ops = k.b, k.ops
        _, _, extent, _ = self._operand(operand)
        left = b.sub(extent, ops.v(row0))
        edge = k.spec.edge_m if operand == "a" else k.spec.edge_n
        if not edge:
            return left
        return b.select(b.cmp_lt(ops.v(row0), extent), left, k.c0)

    def data_descriptor(self, operand: str, parity, component, k_byte, remaining_k):
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        layout, block, _, mask = self._operand(operand)
        base_ptr = p["A"] if operand == "a" else p["B"]
        lds = ops.add(
            k.buffer_base(parity), k.layout.component_base(ops, layout, component)
        )
        rows = layout.rows
        row0 = ops.add(block, ops.mul(component, rows))
        if k.spec.wide_a if operand == "a" else k.spec.wide_b:
            element = k.offset64(row0, k.packed_k, k_byte)
        else:
            element = ops.add(ops.mul(row0, k.packed_k), k_byte)
        rows_left = None
        extra: dict = {}
        tile_rows = rows
        if layout.iterate_rows:
            tile_rows = layout.iterate_rows
            extra.update(
                iterate_count=rows // tile_rows,
                lds_increment=layout.iterate_stride,
                global_increment=ops.v(ops.mul(k.packed_k, tile_rows)),
            )
            if k.spec.edge_m if operand == "a" else k.spec.edge_n:
                # Each iteration moves tile_rows rows whatever tensor_dim1 says,
                # so walk ceil(rows left / tile_rows), at least one: with no
                # rows left, tensor_dim1 0 keeps that one off memory.
                rows_left = self._rows_left(operand, row0)
                held = b.smax(b.smin(rows_left, b.const_i32(rows)), k.c1)
                extra["iterations"] = ops.div(ops.add(held, tile_rows - 1), tile_rows)
        elif layout.pad.on:
            extra.update(tdm_pad_fields(layout.pad.block_bytes, layout.pad.pad_bytes))
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=remaining_k,
            tensor_dim1=(
                self._rows_left(operand, row0) if rows_left is None else rows_left
            ),
            row_stride=k.packed_k,
            tile_dim0=k.layout.row_bytes,
            tile_dim1=tile_rows,
            workgroup_mask=mask,
            **extra,
        )

    def scale_descriptor(self, operand: str, parity, component, k_step):
        """Swizzled scales ``[K/128][MN][4]``: one row per K-step, 4 B per MN."""
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        _, block, extent, mask = self._operand(operand)
        layout = k.layout.a_scale if operand == "a" else k.layout.b_scale
        base_ptr = p["A_scale"] if operand == "a" else p["B_scale"]
        lds = ops.add(
            k.buffer_base(parity), k.layout.component_base(ops, layout, component)
        )
        extra = (
            tdm_pad_fields(layout.pad.block_bytes, layout.pad.pad_bytes)
            if layout.pad.on
            else {}
        )
        rows = layout.rows
        row_stride = ops.mul(extent, 4)
        mn0 = ops.add(block, ops.mul(component, rows))
        col0 = ops.mul(mn0, 4)
        element = ops.add(ops.mul(k_step, row_stride), col0)
        global_ptr = b.global_ptr_add(base_ptr, ops.v(element))
        lds_ptr = self.lds_ptr(lds)
        edge = k.spec.edge_m if operand == "a" else k.spec.edge_n
        if edge:
            bytes_left = ops.v(ops.mul(self._rows_left(operand, mn0), 4))
        else:
            bytes_left = b.sub(ops.v(row_stride), ops.v(col0))
        return b.tdm_descriptor_2d(
            global_ptr,
            lds_ptr,
            elem_bytes=1,
            tensor_dim0=bytes_left,
            tensor_dim1=b.sub(k.k_atoms(), ops.v(k_step)),
            row_stride=ops.v(row_stride),
            tile_dim0=rows * 4,
            tile_dim1=k.layout.k_steps,
            workgroup_mask=mask,
            **extra,
        )

    def hoist(self) -> None:
        """Both descriptors for tile 0, buffer 0, selected by wave parity."""
        k = self.k
        b, ops, c0, p = k.b, k.ops, k.c0, k.params
        even0 = b.cmp_eq(ops.mod(k.wave, 2), c0)
        component0 = ops.div(k.wave, 2)

        def selected(b_desc, a_desc):
            return [b.select(even0, x, y) for x, y in zip(b_desc, a_desc)]

        self.data = selected(
            self.data_descriptor("b", c0, component0, c0, b.sub(k.packed_k, c0)),
            self.data_descriptor("a", c0, component0, c0, b.sub(k.packed_k, c0)),
        )
        if k.spec.scale_regs:
            return
        self.scale = selected(
            self.scale_descriptor("b", c0, component0, c0),
            self.scale_descriptor("a", c0, component0, c0),
        )
        self.scale_stride = b.select(
            even0, ops.v(ops.mul(p["N"], 4)), ops.v(ops.mul(p["M"], 4))
        )

    def _patched(self, desc, parity, byte_offset, words):
        """``desc`` with its LDS address moved to buffer ``parity``, its
        global address advanced by ``byte_offset`` and the group-1 words in
        ``words`` (index -> value) replaced."""
        k = self.k
        b, ops = k.b, k.ops
        d0, d1, *rest = desc
        lds = b.add(b.vec_extract(d0, 1), ops.v(k.buffer_base(parity)))
        lo, hi = b.vec_extract(d0, 2), b.vec_extract(d0, 3)
        address = b.lor(
            b.shl(b.zext(b.land(hi, b.const_i32(0x1FFFFFF)), I64), b.const_i64(32)),
            b.zext(lo, I64),
        )
        address = b.add(address, b.zext(ops.v(byte_offset), I64))
        new_hi = b.lor(
            b.land(
                b.trunc(b.lshr(address, b.const_i64(32)), I32), b.const_i32(0x1FFFFFF)
            ),
            b.land(hi, b.const_i32(-0x2000000)),
        )
        d0 = b.vec_pack([b.vec_extract(d0, 0), lds, b.trunc(address, I32), new_hi], I32)
        d1 = b.vec_pack([words.get(i, b.vec_extract(d1, i)) for i in range(8)], I32)
        return (d0, d1, *rest)

    def issue(self, k0, parity) -> None:
        """This wave's data and scale loads of the tile at ``k0`` into buffer ``parity``."""
        k = self.k
        b, ops = k.b, k.ops
        k_byte = k.spec.k_bytes(k0, ops)
        remaining_k = b.sub(k.packed_k, ops.v(k_byte))
        c16 = b.const_i32(16)
        data = self.data
        w2 = b.vec_extract(data[1], 2)
        b.tensor_load_to_lds(
            *self._patched(
                data,
                parity,
                k_byte,
                {
                    1: b.shl(remaining_k, c16),
                    2: b.lor(
                        b.land(w2, b.const_i32(-0x10000)), b.lshr(remaining_k, c16)
                    ),
                },
            )
        )
        if k.spec.scale_regs:
            return
        k_step = ops.v(ops.div(k0, _ATOM_K))
        scale = self.scale
        steps_left = b.sub(ops.v(k.k_atoms()), k_step)
        s2, s3 = b.vec_extract(scale[1], 2), b.vec_extract(scale[1], 3)
        b.tensor_load_to_lds(
            *self._patched(
                scale,
                parity,
                b.mul(k_step, self.scale_stride),
                {
                    2: b.lor(b.land(s2, b.const_i32(0xFFFF)), b.shl(steps_left, c16)),
                    3: b.lor(
                        b.land(s3, b.const_i32(-0x10000)), b.lshr(steps_left, c16)
                    ),
                },
            )
        )


class Gl2Prefetcher:
    """PrefetchGL2: the cluster's A/B and scale lines of a tile ahead, split
    over every thread of the cluster, one 256 B line each."""

    def __init__(self, k: _Kernel):
        self.k = k

    def prefetch(self, k_ahead, ahead_parity: int) -> None:
        """``ahead_parity`` is the parity of the prefetched tile's index: with
        128 B rows one line holds an even tile and the next one, so the data
        lines are only prefetched for even tiles. Rows that neither hold
        whole lines nor divide one (FP6: 96 B per 128 K) take, per row, the
        ``ceil(row / line)`` lines that end at the tile's last byte: the
        tiles of a row then reach every line, a line two tiles share once
        for each."""
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        if not spec.gl2_prefetch_distance:
            return
        row_bytes = k.layout.row_bytes
        cluster_x, cluster_y, _ = spec.cluster_dim
        # Branch-free: any branch in the loop body makes LLVM drain the LDS
        # read pipeline. Past the end, the last tile is prefetched again.
        k_ahead = b.select(
            b.cmp_lt(k_ahead, p["K"]), k_ahead, b.sub(p["K"], k.tile_k_v)
        )
        k_byte = spec.k_bytes(k_ahead, ops)
        whole_lines = row_bytes % _GL2_LINE_BYTES == 0
        straddle = not whole_lines and _GL2_LINE_BYTES % row_bytes != 0
        if straddle:
            data_lines = -(-row_bytes // _GL2_LINE_BYTES)
            last_byte = ops.add(k_byte, row_bytes - 1)
            if spec.tail_steps:
                # The K tail's tile ends at the row's end.
                last_byte = b.smin(ops.v(last_byte), b.sub(k.packed_k, k.c1))
        else:
            data_lines = row_bytes // _GL2_LINE_BYTES
            if not data_lines and ahead_parity == 0:
                data_lines = 1
        if spec.has_cluster:
            cluster_m0 = b.sub(
                ops.v(k.block_m), ops.v(ops.mul(k.cluster_wy, spec.tile_m))
            )
            cluster_n0 = b.sub(
                ops.v(k.block_n), ops.v(ops.mul(k.cluster_wx, spec.tile_n))
            )
        else:
            cluster_m0, cluster_n0 = k.block_m, k.block_n

        def last_inside(value, limit, edge: bool):
            """``value``, or ``limit - 1`` past a partial tile's end."""
            if not edge:
                return value
            return b.select(
                b.cmp_lt(ops.v(value), ops.v(limit)),
                ops.v(value),
                b.sub(ops.v(limit), k.c1),
            )

        def data_line(base_ptr, row0, extent, edge, wide):
            def address(index):
                row = ops.add(row0, ops.div(index, data_lines))
                row = last_inside(row, extent, edge)
                line = ops.mul(ops.mod(index, data_lines), _GL2_LINE_BYTES)
                if not straddle:
                    offset = ops.add(k_byte, line)
                elif isinstance(line, int) and line == 0:
                    offset = last_byte
                else:
                    offset = b.sub(ops.v(last_byte), ops.v(line))
                if wide:
                    return b.global_ptr_add(
                        base_ptr, k.offset64(row, k.packed_k, offset)
                    )
                return b.global_ptr_add(
                    base_ptr, ops.v(ops.add(ops.mul(row, k.packed_k), offset))
                )

            return address

        def scale_line(base_ptr, extent, mn0, mn_count, edge):
            lines_per_row = mn_count * 4 // _GL2_LINE_BYTES
            k_step = ops.div(k_ahead, _ATOM_K)
            # A partial last tile has fewer K-steps than the k_steps rows
            # prefetched for it; a partial M/N tile has lines past the end.
            clamp = edge or bool(spec.tail_steps)
            scale_bytes = ops.mul(k.k_atoms(), ops.mul(extent, 4)) if clamp else None

            def address(index):
                row = ops.add(k_step, ops.div(index, lines_per_row))
                offset = ops.add(
                    ops.mul(mn0, 4),
                    ops.mul(ops.mod(index, lines_per_row), _GL2_LINE_BYTES),
                )
                element = ops.add(ops.mul(row, ops.mul(extent, 4)), offset)
                element = last_inside(element, scale_bytes, clamp)
                return b.global_ptr_add(base_ptr, ops.v(element))

            return address, k.layout.k_steps * lines_per_row

        categories = []
        if data_lines:
            categories += [
                (
                    cluster_y * spec.tile_m * data_lines,
                    data_line(p["A"], cluster_m0, p["M"], spec.edge_m, spec.wide_a),
                ),
                (
                    cluster_x * spec.tile_n * data_lines,
                    data_line(p["B"], cluster_n0, p["N"], spec.edge_n, spec.wide_b),
                ),
            ]
        # Scales are prefetched only for an operand whose cluster scale row
        # is whole lines.
        if (cluster_x * spec.tile_n * 4) % _GL2_LINE_BYTES == 0:
            b_scale_address, b_scale_lines = scale_line(
                p["B_scale"], p["N"], cluster_n0, cluster_x * spec.tile_n, spec.edge_n
            )
            categories.append((b_scale_lines, b_scale_address))
        if (cluster_y * spec.tile_m * 4) % _GL2_LINE_BYTES == 0:
            a_scale_address, a_scale_lines = scale_line(
                p["A_scale"], p["M"], cluster_m0, cluster_y * spec.tile_m, spec.edge_m
            )
            categories.append((a_scale_lines, a_scale_address))
        starts = []
        total = 0
        for count, _ in categories:
            starts.append(total)
            total += count
        pool = cluster_x * cluster_y * spec.block_size
        if spec.has_cluster:
            flat = ops.add(k.cluster_wx, ops.mul(k.cluster_wy, cluster_x))
            thread = ops.add(ops.mul(flat, spec.block_size), k.tid)
        else:
            thread = k.tid

        def prefetch_round(round_start: int, index) -> None:
            live = [
                (start, count, address)
                for start, (count, address) in zip(starts, categories)
                if start < round_start + pool and start + count > round_start
            ]
            # Ranges are visited last to first, so each lane ends up with
            # the first range whose end lies above its index.
            pointer = None
            for start, count, address in reversed(live):
                local = b.sub(ops.v(index), ops.v(start)) if start else index
                if pointer is None:
                    pointer = address(local)
                    continue
                in_range = b.cmp_lt(ops.v(index), ops.v(start + count))
                pointer = b.select(in_range, address(local), pointer)
            b.global_prefetch(pointer, cachepolicy=_GL2_PREFETCH_CPOL)

        for round_start in range(0, total, pool):
            remaining = total - round_start
            if remaining >= pool:
                index = ops.add(thread, round_start)
            else:
                # Every lane covers one of the remaining lines; some lines
                # are prefetched more than once.
                index = ops.add(ops.mod(thread, remaining), round_start)
            prefetch_round(round_start, index)


class ScaleRegisters:
    """``scales`` "registers": a tile's E8M0 scale words from the swizzled
    scales ``[K/128][MN][4]`` straight into registers, one buffer load of
    up to 16 B per lane, operand and K-step (all 32 lanes read 512
    contiguous bytes on the large tiles). With ``r`` repeats of an operand
    (M-repeats for A, N-repeats for B) and ``h = r // 2`` (1 for one
    repeat), the lanes of half ``s`` hold the words of repeats ``s * h`` to
    ``s * h + h - 1`` of their row; repeat ``i`` is word ``i % h`` with scale
    selector ``i // h``. Words past the last K-step read 0; rows past M
    (N) read words of another K-step, which only scale rows of D that are
    not stored."""

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows
        b, ops, p = k.b, k.ops, k.params
        self.rsrc = {
            name: b.buffer_rsrc(
                p[f"{name.upper()}_scale"],
                ops.v(ops.mul(k.k_atoms(), ops.mul(p[extent], 4))),
            )
            for name, extent in (("a", "M"), ("b", "N"))
        }

    @staticmethod
    def half(repeats: int) -> int:
        return repeats // 2 if repeats >= 2 else 1

    @classmethod
    def place(cls, repeats: int, index: int) -> tuple[int, int]:
        """(word, scale selector) of repeat ``index``."""
        h = cls.half(repeats)
        return index % h, index // h

    def _operand(self, name: str, k_step):
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        repeats = spec.wave_repeat_m if name == "a" else spec.wave_repeat_n
        h = self.half(repeats)
        first = ops.mul(k.lane_half, h) if repeats >= 2 else 0
        row = (self.rows.m_row if name == "a" else self.rows.n_row)(first, k.lane_col)
        extent = p["M"] if name == "a" else p["N"]
        offset = ops.mul(ops.add(ops.mul(k_step, extent), row), 4)
        words = []
        for j in range(0, h, 4):
            n = min(4, h - j)
            loaded = b.buffer_load_vN(
                self.rsrc[name], ops.v(ops.add(offset, 4 * j)), k.c0, I32, n
            )
            words += [b.vec_extract(loaded, i) for i in range(n)]
        return words

    def load(self, k0) -> list:
        """Every K-step's A then B words of the tile at ``k0`` (a tile past
        the end reads 0)."""
        ops = self.k.ops
        first = ops.div(k0, _ATOM_K)
        out = []
        for step in range(self.k.spec.k_steps):
            k_step = ops.add(first, step)
            out += self._operand("a", k_step) + self._operand("b", k_step)
        return out


class LdsReader:
    """LDS to registers: the scales of one K-step in 16 B reads, then each A
    row and B fragment as its ``lane_reads`` (one 16 B read per K chunk: two
    for FP4, four for 8-bit; FP6 six 8 B or three 16 B reads)."""

    # v_perm_b32 selector swapping bytes 1 and 2 of a word (``k_reads``
    # "contiguous": the WMMA's K groups 1 and 2 hold the K-step's groups 2, 1).
    _SWAP_GROUPS_1_2 = 0x03010200

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows

    def load_i32(self, offset, n: int = 4):
        return self.k.b.smem_load_vN(
            self.k.shared, self.k.ops.v(offset), dtype=I32, n=n
        )

    def _lane_k(self, step):
        k = self.k
        return k.ops.add(
            step * k.spec.step_bytes, k.ops.mul(k.lane_half, k.spec.lane_half_bytes)
        )

    def _row_fragment(self, buffer, operand: str, row, lane_k):
        k, ops = self.k, self.k.ops
        base = ops.add(ops.add(buffer, k.layout.row_offset(ops, operand, row)), lane_k)
        (first, words), *rest = k.spec.lane_reads
        frag = self.load_i32(ops.add(base, first), words)
        for offset, words in rest:
            frag = k.b.vec_concat(frag, self.load_i32(ops.add(base, offset), words))
        padding = k.spec.atom.frag_words - sum(w for _, w in k.spec.lane_reads)
        if padding:
            frag = k.b.vec_concat(frag, k.b.vector_splat(k.b.const_i32(0), padding))
        return frag

    def a_fragment(self, buffer, rm, step):
        k = self.k
        lane_k = self._lane_k(step)
        if k.spec.atom.a_rows == 1:
            return self._row_fragment(
                buffer, "a", self.rows.m_row(rm, k.lane_col), lane_k
            )
        frags = [
            self._row_fragment(buffer, "a", self.rows.m_row(rm, m_local), lane_k)
            for m_local in (k.lane_col, k.ops.add(k.lane_col, 16))
        ]
        return k.b.vec_concat(frags[0], frags[1])

    def b_fragment(self, buffer, rn, step):
        lane_k = self._lane_k(step)
        row = self.rows.n_row(rn, self.k.lane_col)
        return self._row_fragment(buffer, "b", row, lane_k)

    def scales(self, buffer, step):
        """Per-lane E8M0 words of one K-step: one per M- and per N-repeat."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        a_lane = k.lane if spec.atom.a_rows == 2 else k.lane_col
        a_base = ops.add(
            buffer,
            k.layout.scale_offset(ops, "a_scale", step, self.rows.m_row(0, a_lane)),
        )
        a_scales = []
        if rm_count >= 4:
            for chunk in range(rm_count // 4):
                words = self.load_i32(ops.add(a_base, chunk * 16))
                a_scales += [b.vec_extract(words, i) for i in range(4)]
        else:
            words = self.load_i32(a_base, n=rm_count)
            a_scales = [b.vec_extract(words, i) for i in range(rm_count)]
        b_base = ops.add(
            buffer,
            k.layout.scale_offset(ops, "b_scale", step, self.rows.n_row(0, k.lane_col)),
        )
        b_scales = []
        if rn_count >= 4:
            for chunk in range(rn_count // 4):
                words = self.load_i32(ops.add(b_base, chunk * 16))
                b_scales += [b.vec_extract(words, i) for i in range(4)]
        else:
            words = self.load_i32(b_base, n=rn_count)
            b_scales = [b.vec_extract(words, i) for i in range(rn_count)]
        if spec.contiguous_k:
            swap = b.const_i32(self._SWAP_GROUPS_1_2)
            a_scales = [b.perm_b32(w, w, swap) for w in a_scales]
            b_scales = [b.perm_b32(w, w, swap) for w in b_scales]
        return a_scales, b_scales

    def head(self, buffer, step):
        a_scales, b_scales = self.scales(buffer, step)
        a_frags = [
            self.a_fragment(buffer, rm, step) for rm in range(self.k.spec.wave_repeat_m)
        ]
        return a_frags, a_scales, b_scales

    def b_group(self, buffer, step, first: int, count: int):
        return [self.b_fragment(buffer, rn, step) for rn in range(first, first + count)]


@dataclass
class _Hooks:
    """Refill events inside one tile (see :class:`Pipeline`)."""

    refill: object = None
    signal: object = None
    late_refill: object = None


class TileCompute:
    """One K tile of WMMAs with its LDS reads and refill events.

    ``local_prefetch`` 1: K-step s+1 is read during K-step s; the last K-step
    reads the next tile's K-step 0 from the other buffer (``head``). With
    ``half_plr`` ``"b"`` B is read in halves, two waits per K-step, B-outer
    WMMAs; ``half_b_order`` places K-step s+1's reads:

    - ``"ahead"``: all before the first wait but B's second half, which
      follows the first half's WMMAs. Three B half-sets live; each wait
      leaves one K-step of reads in flight.
    - ``"late"``: wait, scales and A, half 0's WMMAs; wait, B half 0, half
      1's WMMAs; B half 1. Two B half-sets live; the waits leave B's second
      half, then the next scales and A, in flight.
    """

    def __init__(self, k: _Kernel, reader: LdsReader):
        self.k = k
        self.reader = reader

    def _mma_fn(self, out):
        k = self.k
        rn_count = k.spec.wave_repeat_n
        atom = k.spec.atom

        def mma(
            rm,
            rn,
            a_frag,
            b_frag,
            a_scale,
            b_scale,
            *,
            reuse_a=False,
            reuse_b=False,
            sel_a=0,
            sel_b=0,
        ):
            index = rm * rn_count + rn
            attrs = {}
            if reuse_a:
                attrs["reuse_a"] = True
            if reuse_b:
                attrs["reuse_b"] = True
            if sel_a:
                attrs["scale_sel_a"] = sel_a
            if sel_b:
                attrs["scale_sel_b"] = sel_b
            out[index] = k.b.mma(
                atom, a_frag, b_frag, out[index], a_scale, b_scale, attrs=attrs or None
            )

        return mma

    def _wait(self, count: int) -> None:
        self.k.b.inline_asm(f"s_wait_dscnt {min(count, _DSCNT_MAX)}", "~{memory}")

    def head_parts(self) -> list[int]:
        spec = self.k.spec
        rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
        if spec.half_b:
            return [rm, rm, rn, rn // 2, rn // 2]
        return [rm, rm, rn, rn]

    def first_head(self, parity):
        """Tile 0's K-step 0 reads, issued before the first tile."""
        k = self.k
        a_frags, a_scales, b_scales = self.reader.head(k.buffer_base(parity), 0)
        if k.spec.half_b:
            half = k.spec.wave_repeat_n // 2
            groups = [
                self.reader.b_group(k.buffer_base(parity), 0, w * half, half)
                for w in (0, 1)
            ]
        else:
            groups = [
                self.reader.b_group(k.buffer_base(parity), 0, 0, k.spec.wave_repeat_n)
            ]
        return a_frags, a_scales, b_scales, groups

    @staticmethod
    def flatten(head) -> list:
        a_frags, a_scales, b_scales, groups = head
        return [*a_frags, *a_scales, *b_scales, *(v for g in groups for v in g)]

    def unflatten(self, values):
        values = list(values)
        parts = []
        for count in self.head_parts():
            parts.append(values[:count])
            values = values[count:]
        return parts[0], parts[1], parts[2], parts[3:]

    def run(
        self,
        parity,
        accumulators,
        *,
        hooks: _Hooks,
        head=None,
        next_parity=None,
        steps=None,
        scales=None,
    ):
        """Returns ``(accumulators, head)``; ``head`` is the next tile's
        K-step 0 reads (``None`` with ``local_prefetch`` 0). ``steps`` is the
        tile's K-step count (all of them unless it is the K tail); ``scales``
        the tile's scale words with ``scales`` "registers"."""
        steps = self.k.spec.k_steps if steps is None else steps
        if self.k.spec.local_prefetch == 0:
            return self._run_direct(parity, accumulators, hooks, steps, scales)
        if self.k.spec.half_b_late:
            return self._run_half_b_late(
                parity, accumulators, head, next_parity, hooks, steps
            )
        if self.k.spec.half_b:
            return self._run_half_b(
                parity, accumulators, head, next_parity, hooks, steps
            )
        return self._run_full(parity, accumulators, head, next_parity, hooks, steps)

    @staticmethod
    def _signal_before_late(hooks, steps: int) -> None:
        """A one-K-step tile has no second-to-last K-step to signal in: it
        signals just before its cluster wait."""
        if steps == 1 and hooks.signal is not None:
            hooks.signal()

    def _run_half_b(self, parity, accumulators, head, next_parity, hooks, steps):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        half = rn_count // 2
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        *head, b_halves = head
        next_buffer = k.buffer_base(next_parity)
        for step in range(steps):
            a_frags, a_scales, b_scales = head
            last = step + 1 == steps
            if not last:
                head = reader.head(buffer, step + 1)
                next_first = reader.b_group(buffer, step + 1, 0, half)
            else:
                hooks.refill()
                head = reader.head(next_buffer, 0)
                next_first = reader.b_group(next_buffer, 0, 0, half)
            for which in (0, 1):
                if which == 1 and not last:
                    next_second = reader.b_group(buffer, step + 1, half, half)
                elif which == 1:
                    next_second = reader.b_group(next_buffer, 0, half, half)
                if which == 1 and last:
                    self._signal_before_late(hooks, steps)
                    hooks.late_refill()
                signal_here = (
                    hooks.signal is not None and which == 1 and step + 2 == steps
                )
                self._wait(spec.reads_per_step)
                order = [(rm, j) for j in range(half) for rm in range(rm_count)]
                issued = 0
                for rm, j in order:
                    rn = which * half + j
                    if signal_here and issued == spec.signal_wmmas:
                        hooks.signal()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_halves[which][j],
                        a_scales[rm],
                        b_scales[rn],
                    )
                if signal_here and issued == spec.signal_wmmas:
                    hooks.signal()
            if not last:
                b_halves = [next_first, next_second]
        return out, (*head, [next_first, next_second])

    def _run_half_b_late(self, parity, accumulators, head, next_parity, hooks, steps):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, half = spec.wave_repeat_m, spec.wave_repeat_n // 2
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        *head, b_halves = head
        next_buffer = k.buffer_base(next_parity)
        waits = spec.dscnt_waits
        for step in range(steps):
            a_frags, a_scales, b_scales = head
            last = step + 1 == steps
            source = (next_buffer, 0) if last else (buffer, step + 1)
            next_first = None
            for which in (0, 1):
                if which == 1 and last:
                    self._signal_before_late(hooks, steps)
                    hooks.late_refill()
                signal_here = (
                    hooks.signal is not None and which == 1 and step + 2 == steps
                )
                self._wait(waits[which])
                if which == 0:
                    if last:
                        hooks.refill()
                    head = reader.head(*source)
                else:
                    next_first = reader.b_group(*source, 0, half)
                issued = 0
                for j in range(half):
                    for rm in range(rm_count):
                        rn = which * half + j
                        if signal_here and issued == spec.signal_wmmas:
                            hooks.signal()
                        issued += 1
                        mma(
                            rm,
                            rn,
                            a_frags[rm],
                            b_halves[which][j],
                            a_scales[rm],
                            b_scales[rn],
                        )
                if signal_here and issued == spec.signal_wmmas:
                    hooks.signal()
            b_halves = [next_first, reader.b_group(*source, half, half)]
        return out, (*head, b_halves)

    def _run_full(self, parity, accumulators, head, next_parity, hooks, steps):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        a_frags, a_scales, b_scales, (b_frags,) = head
        next_buffer = k.buffer_base(next_parity)
        for step in range(steps):
            last = step + 1 == steps
            if not last:
                ahead = (
                    *reader.head(buffer, step + 1),
                    reader.b_group(buffer, step + 1, 0, rn_count),
                )
            else:
                hooks.refill()
                ahead = (
                    *reader.head(next_buffer, 0),
                    reader.b_group(next_buffer, 0, 0, rn_count),
                )
                self._signal_before_late(hooks, steps)
                hooks.late_refill()
            signal_here = hooks.signal is not None and step + 2 == steps
            self._wait(spec.reads_per_step)
            issued = 0
            for rm in range(rm_count):
                for rn in range(rn_count):
                    if signal_here and issued == spec.signal_wmmas:
                        hooks.signal()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_frags[rn],
                        a_scales[rm],
                        b_scales[rn],
                    )
            if signal_here and issued == spec.signal_wmmas:
                hooks.signal()
            a_frags, a_scales, b_scales, b_frags = ahead
        return out, (a_frags, a_scales, b_scales, [b_frags])

    @staticmethod
    def wmma_order(reuse: str, rm_count: int, rn_count: int):
        """``(rm, rn)`` of each WMMA of a K-step and whether it flags A and B
        for the next one (``wmma_reuse``)."""
        if reuse in ("b", "ab"):
            order = []
            for rn in range(rn_count):
                rms = list(range(rm_count))
                if reuse == "ab" and rn % 2:
                    rms.reverse()
                order += [(rm, rn) for rm in rms]
        else:
            order = [(rm, rn) for rm in range(rm_count) for rn in range(rn_count)]
        nxt = order[1:] + [None]
        keep_a = [
            reuse in ("a", "ab") and after is not None and after[0] == rm
            for (rm, _), after in zip(order, nxt)
        ]
        keep_b = [
            reuse in ("b", "ab") and after is not None and after[1] == rn
            for (_, rn), after in zip(order, nxt)
        ]
        return order, keep_a, keep_b

    def _run_direct(self, parity, accumulators, hooks, steps, scales=None):
        """local_prefetch 0: each K-step reads, drains and computes. The
        tile's last read is the last K-step's: its refill (GL2 prefetch, the
        signal) follows the drain, the cluster wait and TDM come
        ``cluster_signal_wmmas`` WMMAs later (at once without a cluster).
        With ``scales`` "registers" the scale words come in ``scales``."""
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        late_at = spec.signal_wmmas if spec.has_cluster else 0
        order, keep_a, keep_b = self.wmma_order(spec.wmma_reuse, rm_count, rn_count)
        if spec.scale_regs:
            ha = ScaleRegisters.half(rm_count)
            hb = ScaleRegisters.half(rn_count)
            words = list(scales)
            if spec.contiguous_k:
                # Once per tile, a tile after the words were loaded.
                swap = k.b.const_i32(LdsReader._SWAP_GROUPS_1_2)
                words = [k.b.perm_b32(w, w, swap) for w in words]
        for step in range(steps):
            last = step + 1 == steps
            if spec.scale_regs:
                a_frags = [
                    reader.a_fragment(buffer, rm, step) for rm in range(rm_count)
                ]
                at = step * (ha + hb)
                a_scales, b_scales = words[at : at + ha], words[at + ha : at + ha + hb]
            else:
                a_frags, a_scales, b_scales = reader.head(buffer, step)
            b_frags = reader.b_group(buffer, step, 0, rn_count)
            self._wait(0)
            if last:
                hooks.refill()
                if hooks.signal is not None:
                    hooks.signal()
            issued = 0
            for (rm, rn), reuse_a, reuse_b in zip(order, keep_a, keep_b):
                if last and issued == late_at:
                    hooks.late_refill()
                issued += 1
                if spec.scale_regs:
                    (wa, sel_a), (wb, sel_b) = (
                        ScaleRegisters.place(rm_count, rm),
                        ScaleRegisters.place(rn_count, rn),
                    )
                    a_scale, b_scale = a_scales[wa], b_scales[wb]
                else:
                    a_scale, b_scale, sel_a, sel_b = a_scales[rm], b_scales[rn], 0, 0
                mma(
                    rm,
                    rn,
                    a_frags[rm],
                    b_frags[rn],
                    a_scale,
                    b_scale,
                    reuse_a=reuse_a,
                    reuse_b=reuse_b,
                    sel_a=sel_a,
                    sel_b=sel_b,
                )
            if last and issued == late_at:
                hooks.late_refill()
        return out, None


class Pipeline:
    """PrefetchGlobalRead=2: the TDM runs two tiles ahead. Tile k refills its
    own buffer with tile k+2 once every workgroup of the cluster has read
    it; tile k+1 is waited for (and published) at the start of tile k's last
    K-step, with tile k+2's loads still in flight. Past the end the last
    tile is loaded again into a buffer nobody reads (SuppressNoLoadLoop).
    The loop runs tile pairs, one tile per buffer, so buffer offsets fold;
    an odd tile count peels one tile."""

    def __init__(
        self, k: _Kernel, tdm: TdmLoader, gl2: Gl2Prefetcher, tile: TileCompute
    ):
        self.k = k
        self.tdm = tdm
        self.gl2 = gl2
        self.tile = tile
        self.scales = ScaleRegisters(k, tile.reader.rows) if k.spec.scale_regs else None

    def _prefetch_ahead(self, k0, parity: int) -> None:
        distance = self.k.spec.gl2_prefetch_distance
        if distance:
            self.gl2.prefetch(
                self.k.ops.v(self.k.ops.add(k0, distance * self.k.spec.tile_k)),
                (parity + distance) % 2,
            )

    def hooks(self, k0, parity, static_parity: int) -> _Hooks:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        two_tiles = 2 * spec.tile_k

        def signal() -> None:
            k.fence()
            b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            k.fence()

        def refill() -> None:
            self._prefetch_ahead(k0, static_parity)
            k.fence()
            if spec.local_prefetch:
                # Tile k+1 is read next, from this point: it is the only TDM
                # load in flight, and the barrier below publishes it.
                b.s_wait_tensorcnt(0)
                b.s_barrier_bare()
                k.fence()

        def late() -> None:
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            # No wave drained its reads of this buffer before the signal;
            # the workgroup barrier publishes them here.
            pending = min(spec.late_pending_reads, _DSCNT_MAX)
            b.inline_asm(f"s_wait_dscnt {pending}", "~{memory}")
            b.s_barrier_bare()
            self.tdm.issue(k.clamp_k(ops.v(ops.add(k0, two_tiles))), parity)
            k.fence()

        return _Hooks(
            refill=refill, signal=signal if spec.has_cluster else None, late_refill=late
        )

    def wait_next_tile(self) -> None:
        b = self.k.b
        b.s_wait_tensorcnt(2)
        b.s_barrier_bare()
        b.inline_asm("", "~{memory}")

    def run(self):
        k = self.k
        b, ops, spec, c0, c1 = k.b, k.ops, k.spec, k.c0, k.c1
        n_acc = spec.wave_repeat_m * spec.wave_repeat_n
        read_ahead = spec.local_prefetch == 1
        tile = self.tile
        k.cluster_sync()
        self.tdm.issue(c0, c0)
        if spec.K == spec.tile_k:
            # One tile: buffer 1 gets tile 0 again and is never read. The
            # refill and GL2 prefetches load it again too; see validate().
            self.tdm.issue(k.clamp_k(k.tile_k_v), c1)
        else:
            self.tdm.issue(k.tile_k_v, c1)
        for ahead in range(2, spec.gl2_prefetch_distance):
            self.gl2.prefetch(ops.v(ahead * spec.tile_k), ahead % 2)
        if read_ahead:
            self.wait_next_tile()
        accumulators = [b.zero_vec_f32(spec.atom.c_frag_len) for _ in range(n_acc)]
        head_values = tile.flatten(tile.first_head(c0)) if read_ahead else []
        regs = self.scales
        # Register scales: tile 0's here, each tile loads the next one's.
        scale_values = regs.load(c0) if regs else []
        start, start_parity = c0, 0

        def loop_tile(k_tile, parity: int, acc, head_vals, scale_vals, steps=None):
            if not read_ahead:
                self.wait_next_tile()
            next_scales = regs.load(ops.add(k_tile, spec.tile_k)) if regs else []
            updated, head = tile.run(
                parity,
                acc,
                hooks=self.hooks(k_tile, parity, parity),
                head=tile.unflatten(head_vals) if read_ahead else None,
                next_parity=1 - parity,
                steps=steps,
                scales=scale_vals,
            )
            return updated, tile.flatten(head) if read_ahead else [], next_scales

        def carried(acc, head_vals, scale_vals):
            return (
                [(f"acc{i}", v) for i, v in enumerate(acc)]
                + [(f"head{i}", v) for i, v in enumerate(head_vals)]
                + [(f"scale{i}", v) for i, v in enumerate(scale_vals)]
            )

        full_tiles = spec.K // spec.tile_k
        if (full_tiles - start_parity) % 2:
            accumulators, head_values, scale_values = loop_tile(
                start, start_parity, accumulators, head_values, scale_values
            )
            start, start_parity = ops.add(start, spec.tile_k), 1 - start_parity
        n_head = len(head_values)
        loop = b.scf_for_iter(
            ops.v(start),
            b.const_i32(full_tiles * spec.tile_k) if spec.tail_steps else k.params["K"],
            b.const_i32(2 * spec.tile_k),
            carried(accumulators, head_values, scale_values),
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc = list(values[:n_acc])
            head_vals = list(values[n_acc : n_acc + n_head])
            scale_vals = list(values[n_acc + n_head :])
            for pair_index, parity in enumerate((start_parity, 1 - start_parity)):
                acc, head_vals, scale_vals = loop_tile(
                    ops.add(k0, pair_index * spec.tile_k),
                    parity,
                    acc,
                    head_vals,
                    scale_vals,
                )
            b.scf_yield(*acc, *head_vals, *scale_vals)
        results = loop.results[:n_acc]
        if spec.tail_steps:
            # The K tail is one more tile: its TDM loads are clipped at K and
            # only its first tail_steps K-steps are read.
            results, _, _ = loop_tile(
                b.const_i32(full_tiles * spec.tile_k),
                full_tiles % 2,
                list(results),
                list(loop.results[n_acc : n_acc + n_head]),
                list(loop.results[n_acc + n_head :]),
                steps=spec.tail_steps,
            )
        # No workgroup may exit while a TDM load into its LDS is in flight:
        # with multicast, peers' loads write here too.
        b.s_wait_tensorcnt(0)
        if spec.has_cluster:
            k.cluster_sync()
        return results


class Epilogue:
    """``alpha * acc + beta * C`` stored as one vector per accumulator slot;
    the ``beta == 0`` path never reads C. FP8 saturates as hipBLASLt (+-inf
    to NaN, NaN kept, finite values clamped to +-448); FP16 and BF16 round
    to nearest even without clamping, as its v_cvt_pk_f16_f32 and
    v_cvt_pk_bf16_f32."""

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows

    def _saturate(self, value):
        b = self.k.b
        value = b.fma(value, b.const_f32(0.0), value)
        return b.fminimum(
            b.fmaximum(value, b.const_f32(-_FP8_MAX)), b.const_f32(_FP8_MAX)
        )

    def _pack_fp8(self, values):
        b = self.k.b
        if self.k.spec.fp8_convert_width == 8:
            width, convert = 8, b.cvt_pk8_fp8_f32x8
        else:
            width, convert = 4, b.cvt_pk_fp8_f32x4
        if len(values) < width:
            # Fewer values than one convert: pad with zeros, keep the first.
            zeros = [b.const_f32(0.0)] * (width - len(values))
            part = convert(b.vec_pack(list(values) + zeros, F32))
            return b.vec_pack(
                [b.vec_extract(part, i) for i in range(len(values))], FP8E4M3
            )
        packed = None
        for chunk in range(len(values) // width):
            part = convert(b.vec_pack(values[chunk * width : (chunk + 1) * width], F32))
            packed = part if packed is None else b.vec_concat(packed, part)
        return packed

    def _store(self, final, read_c: bool) -> None:
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        out_type = _OUT_TYPES[spec.out_dtype]
        out_bytes = _OUT_BYTES[spec.out_dtype]
        width = spec.store_width
        chunks = range(0, rn_count, width)
        fp8 = spec.out_dtype == "fp8e4m3"
        col = ops.add(k.block_n, self.rows.n_row(0, k.lane_col))
        for rm in range(rm_count):
            for slot in range(spec.atom.c_frag_len):
                m_local = ops.add((slot // 8) * 16 + slot % 8, ops.mul(k.lane_half, 8))
                row = ops.add(k.block_m, self.rows.m_row(rm, m_local))
                c_ptr, d_ptr = p["C"], p["D"]
                if spec.wide_d:
                    # Row base in i64, the column offset in i32.
                    row_bytes = k.offset64(row, ops.mul(p["N"], out_bytes), 0)
                    d_ptr = b.global_ptr_add(p["D"], row_bytes)
                    if read_c:
                        c_ptr = b.global_ptr_add(p["C"], row_bytes)
                    index = ops.v(col)
                else:
                    index = ops.v(ops.add(ops.mul(row, p["N"]), col))
                if spec.edge_m or spec.edge_n:
                    self._store_masked(
                        final, read_c, rm, slot, row, col, (c_ptr, d_ptr, index)
                    )
                    continue
                values = [
                    b.fmul(p["alpha"], b.vec_extract(final[rm * rn_count + rn], slot))
                    for rn in range(rn_count)
                ]
                if read_c:
                    c_words = [
                        (
                            b.global_load(
                                c_ptr,
                                ops.v(ops.add(index, begin)),
                                out_type,
                                align=out_bytes,
                            )
                            if width == 1
                            else b.global_load_vN(
                                c_ptr,
                                ops.v(ops.add(index, begin)),
                                out_type,
                                width,
                                align=width * out_bytes,
                            )
                        )
                        for begin in chunks
                    ]

                    def c_value(i, words=c_words):
                        if width == 1:
                            c = words[i]
                        else:
                            c = b.vec_extract(words[i // width], i % width)
                        return b.cvt_fp8_to_f32(c) if fp8 else b.cast_to_f32(c)

                    values = [
                        b.fadd(value, b.fmul(p["beta"], c_value(i)))
                        for i, value in enumerate(values)
                    ]
                if fp8:
                    values = [self._saturate(v) for v in values]
                for begin in chunks:
                    part = values[begin : begin + width]
                    if fp8:
                        packed = self._pack_fp8(part)
                    elif spec.out_dtype == "fp16":
                        packed = b.vec_pack([b.trunc_f32_to_f16(v) for v in part], F16)
                    else:
                        packed = b.vec_pack(
                            [b.trunc_f32_to_bf16(v) for v in part], BF16
                        )
                    if width == 1:
                        b.global_store(
                            d_ptr,
                            ops.v(ops.add(index, begin)),
                            b.vec_extract(packed, 0),
                            align=out_bytes,
                        )
                        continue
                    b.global_store_vN(
                        d_ptr,
                        ops.v(ops.add(index, begin)),
                        packed,
                        width,
                        align=width * out_bytes,
                    )

    def _store_masked(self, final, read_c: bool, rm: int, slot: int, row, col, at_row):
        """One accumulator slot of a partial tile: each vector of ``width``
        columns is read and written only if its row and columns are in D.
        ``at_row`` is the C and D pointers and the slot's index from them."""
        c_ptr, d_ptr, index = at_row
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        rn_count, width = spec.wave_repeat_n, spec.store_width
        out_type = _OUT_TYPES[spec.out_dtype]
        out_bytes = _OUT_BYTES[spec.out_dtype]
        fp8 = spec.out_dtype == "fp8e4m3"
        row_inside = b.cmp_lt(ops.v(row), p["M"]) if spec.edge_m else None
        for begin in range(0, rn_count, width):
            inside = row_inside
            if spec.edge_n:
                col_inside = b.cmp_lt(ops.v(ops.add(col, begin)), p["N"])
                inside = b.land(row_inside, col_inside) if spec.edge_m else col_inside
            with b.scf_if(inside):
                at = ops.v(ops.add(index, begin))
                values = [
                    b.fmul(p["alpha"], b.vec_extract(final[rm * rn_count + rn], slot))
                    for rn in range(begin, begin + width)
                ]
                if read_c:
                    if width == 1:
                        c_values = [b.global_load(c_ptr, at, out_type, align=out_bytes)]
                    else:
                        words = b.global_load_vN(
                            c_ptr, at, out_type, width, align=width * out_bytes
                        )
                        c_values = [b.vec_extract(words, i) for i in range(width)]
                    c_values = [
                        b.cvt_fp8_to_f32(c) if fp8 else b.cast_to_f32(c)
                        for c in c_values
                    ]
                    values = [
                        b.fadd(v, b.fmul(p["beta"], c))
                        for v, c in zip(values, c_values)
                    ]
                if fp8:
                    packed = self._pack_fp8([self._saturate(v) for v in values])
                elif spec.out_dtype == "fp16":
                    packed = b.vec_pack([b.trunc_f32_to_f16(v) for v in values], F16)
                else:
                    packed = b.vec_pack([b.trunc_f32_to_bf16(v) for v in values], BF16)
                if width == 1:
                    b.global_store(d_ptr, at, b.vec_extract(packed, 0), align=out_bytes)
                else:
                    b.global_store_vN(d_ptr, at, packed, width, align=width * out_bytes)

    def emit(self, final) -> None:
        b, beta = self.k.b, self.k.params["beta"]
        # GW_B0 / GW_B1.
        with b.scf_if_else(b.fcmp("oeq", beta, b.const_f32(0.0))) as (then_, else_):
            with then_:
                self._store(final, read_c=False)
            with else_:
                self._store(final, read_c=True)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_mx_gemm(spec: MxGemmSpec, *, arch: str = "gfx1250") -> KernelDef:
    reason = validate(spec, arch)
    if reason:
        raise ValueError(f"invalid mx_gemm spec: {reason}")
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    b.kernel.attrs[EXPERT_SCHEDULING_ATTR] = True
    b.kernel.attrs[SOURCE_ORDER_ATTR] = True
    # Without it the 256x256x512 tile spills the read-ahead scales (8 VGPRs at
    # the Step 6.14 problem).
    b.kernel.attrs[REGCLASS_PRIORITY_ATTR] = True
    if spec.has_cluster:
        b.set_cluster_dims(*spec.cluster_dim)
    ops = _IrOps(b)
    out_type = _OUT_TYPES[spec.out_dtype]
    params = {
        "A": b.param("A", PtrType(I8, "global"), noalias=True, readonly=True, align=16),
        "B": b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16),
        "A_scale": b.param(
            "A_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4
        ),
        "B_scale": b.param(
            "B_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4
        ),
        "C": b.param(
            "C",
            PtrType(out_type, "global"),
            noalias=True,
            readonly=True,
            align=_OUT_BYTES[spec.out_dtype],
        ),
        "D": b.param(
            "D",
            PtrType(out_type, "global"),
            noalias=True,
            writeonly=True,
            align=_OUT_BYTES[spec.out_dtype],
        ),
        "M": b.param("M", I32),
        "N": b.param("N", I32),
        "K": b.param("K", I32),
        "alpha": b.param("alpha", F32),
        "beta": b.param("beta", F32),
    }
    k = _Kernel(b=b, ops=ops, spec=spec, layout=spec.lds_layout, params=params)
    k.c0 = b.const_i32(0)
    k.c1 = b.const_i32(1)
    k.tile_k_v = b.const_i32(spec.tile_k)
    k.packed_k = spec.k_bytes(params["K"], ops)
    # Global-scale group count; unused with TDM scales, kept so the IR
    # matches the Step 6 kernel's.
    ops.div(params["K"], 32)

    k.tid = b.thread_id_x()
    k.lane = ops.mod(k.tid, _WAVE)
    k.lane_col = ops.mod(k.lane, 16)
    k.lane_half = ops.div(k.lane, 16)
    # Wave-uniform, so the TDM descriptors built from it stay scalar.
    k.wave = b.readfirstlane(ops.div(k.tid, _WAVE))
    k.wave_n = ops.mod(k.wave, spec.warp_n)
    k.wave_m = ops.div(k.wave, spec.warp_n)

    block_x, block_y = b.block_id_x(), b.block_id_y()
    k.block_m = ops.mul(block_y, spec.tile_m)
    k.block_n = ops.mul(block_x, spec.tile_n)
    if spec.has_cluster:
        k.cluster_wx = b.cluster_workgroup_id("x")
        k.cluster_wy = b.cluster_workgroup_id("y")

    rows = RowMaps(k)
    tdm = TdmLoader(k)
    k.shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    k.shared_addr = b.smem_addr_of(k.shared)
    tdm.hoist()
    reader = LdsReader(k, rows)
    pipeline = Pipeline(k, tdm, Gl2Prefetcher(k), TileCompute(k, reader))
    final = pipeline.run()
    Epilogue(k, rows).emit(final)
    b.ret()
    return b.kernel
