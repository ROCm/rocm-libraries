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
    "tiled_transfer",
    "validate",
    "wave_coords",
]

_ATOM_K = 128
_WAVE = 32
_WAVES = 4
_SCALE_GROUPS_PER_STEP = _ATOM_K // 32
# K values in one FP6 pack (12 bytes): FP6 K is a multiple of this.
_FP6_PACK_K = 16


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
    "fp6": _MxType(6, "fp6e2m3", ("16x16",), (128, 256, 384, 512, 1024)),
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
SCALE_SWAPS = ("read", "wait")
READ_ISSUES = ("block", "interleave")
WALKS = ("tile", "panels", "halves", "ring")
READ_BASES = ("loop", "fixed")
WAVE_MAPS = ("rows", "pairs")
TILED_OPERANDS = ("none", "a", "b", "ab")
# walk "panels": a workgroup's 256x256 output in four 128x128 passes over
# 128-row panels of A and B, one panel per LDS slot, each panel split over
# the four waves (one component of 32 rows each). walk "halves": two
# 128x256 halves, an A panel slot each and B panels in slots of two.
_PANEL_ROWS = 128
_PANEL_SLOTS = 4
LDS_BUFFERS = (2, 3)
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
# sched_barrier mask: everything but VALU may cross (SALU 0x4, WMMA 0x8,
# VMEM 0x10..0x40, DS 0x80..0x200, TRANS 0x400).
_SCHED_NO_VALU = 0x7FC
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
# The next 32 instruction cache lines (128 B each, 4 KiB) from this PC, as
# mxfp4_tdm_gemm's epilogue prefetch. Address translations drain first so the
# prefetch is not caught in an XNACK replay.
_INST_PREFETCH_ASM = "s_wait_xcnt 0\ns_prefetch_inst_pc_rel 0, null, 31"


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

    ``scale_swap`` is where those swaps go: ``"read"`` right after the
    scale reads, so the compiler waits for the scale reads (and every LDS
    read before them) before it issues the K-step's data reads; ``"wait"``
    (``k_reads`` "contiguous", ``scales`` "lds") right after the first wait
    of the K-step whose WMMAs take the words, behind a scheduling barrier
    VALU may not cross, so the scale reads need no wait of their own. Words
    carried to the next K-step or tile stay in K order until then.

    ``read_issue`` is where the next K-step's LDS reads go with read-ahead
    and no half-B prefetch: ``"block"`` all before the K-step's WMMAs;
    ``"interleave"`` in groups, one after each chunk of the K-step's WMMAs
    (:meth:`TileCompute._run_interleave`), so the reads issue while the
    WMMAs run.

    ``walk`` is how a workgroup covers its output tile: ``"tile"`` loads
    the whole tile's A and B rows per K tile into one LDS buffer;
    ``"panels"`` (FP6, a 256x256x512 tile, read-ahead) covers it with four
    128x128 passes over 128-row panels of A and B, one panel per LDS slot,
    so the LDS needs four panels where a 256x256x512 tile needs two buffers
    of four (:class:`PanelWalk`). Each pass is the 128x128 tile's compute
    (``pass_spec``); each wave keeps the accumulators of all four passes.
    ``"halves"`` (as ``"panels"``) covers it with two 128x256 halves, A
    panel 0 then A panel 1, each with both B panels: a wave computes 64x128
    of a half, four A rows and eight B fragments a K-step, where a pass
    reads four of each for half the WMMAs. A panels have a slot each; B
    panels share a slot of two, one per K tile parity, so the next tile's B
    loads while this tile's is read (:class:`HalfWalk`). ``"ring"`` (FP6, the
    256x256x256 tile, read-ahead with B in halves, ``read_bases`` "fixed",
    ``k_reads`` "contiguous") keeps two 256 B line slots per row of A and B, so the TDM loads whole
    lines of the row-major operands, one line slot of 128 rows at a time,
    each as soon as the cluster has read the line before it in that slot
    (:class:`RingWalk`); its K loop runs a period of 16 K-steps.

    ``read_bases`` is where the LDS read addresses come from: ``"loop"``
    computes each fragment's address next to its reads; ``"fixed"`` (the
    16x16 WMMA, scales without LDS pad) computes one address per buffer and
    A row, B fragment and scale group before the K loop and keeps it in a
    register, so each read is that register plus an immediate. In expert
    scheduling an LDS read that takes a VALU result waits for every WMMA in
    flight; with ``"fixed"`` the loop computes no read addresses.

    ``wave_map`` is how the 2x2 waves share the tile: ``"rows"`` gives wave
    ``w`` A half ``w // 2`` and B half ``w % 2``. ``"pairs"`` (the 16x16
    WMMA, an even ``wave_repeat_n``, scales "lds", walk "tile") gives each
    SIMD pair, waves ``p`` and ``p + 2``, A half ``p``. A wave's N-repeats
    are two groups of adjacent D columns, group ``g`` in B half
    ``(g + p) % 2``; within a B half the pair's waves take one quarter of the
    tile's columns each (``w // 2``). Where the WMMAs run group by group
    (``half_plr`` "b"), the two pairs read different A and B halves at the
    same time. D vectors are then one group wide.

    ``inst_prefetch`` (0: none) issues an instruction prefetch of the next
    4 KiB of code before every ``inst_prefetch``-th WMMA of the kernel's code,
    counted in code order over the whole kernel, so a K loop body longer than
    the instruction cache finds its next code already fetched on every pass.

    ``wmma_reuse`` (the 16x16 WMMA; ``local_prefetch`` 0, or ``"b"`` with B
    in halves) orders each K-step's WMMAs and flags every WMMA whose next
    WMMA takes the same operand, so the hardware may keep that operand
    instead of reading it again from VGPRs: ``"b"`` N-repeat outer, M-repeat
    inner, flagging B; ``"a"`` M-repeat outer, N-repeat inner, flagging A;
    ``"ab"`` N-repeat outer with the M-repeats in alternating direction,
    flagging B within an N-repeat and A where the direction turns.
    ``"none"`` issues them M-repeat outer, without flags. With read-ahead and
    B in halves (``half_plr`` "b"), whose WMMAs already run N-repeat outer
    within a half, ``"b"`` flags every WMMA but the last of each B fragment.

    ``scales`` is where the WMMAs' E8M0 scale words come from: ``"lds"``
    from shared memory, where the TDM stages them with the tile;
    ``"registers"`` (``local_prefetch`` 0, the 16x16 WMMA) from global memory
    straight into registers, one tile ahead. There lanes 0-15 hold the
    scales of the first half of the wave's M-repeats (N-repeats) and lanes
    16-31 those of the second half, and each WMMA selects its half
    (:class:`ScaleRegisters`).

    ``lds_buffers`` is how many tiles the shared memory holds, each in its
    own buffer: the TDM loads tile ``k + lds_buffers`` into tile ``k``'s
    buffer once the cluster has read it, so with 3 a tile load has two tiles
    of compute to land instead of one (:class:`Pipeline`).

    ``tiled_operands`` names the operands (``"a"``, ``"b"``, ``"ab"``; or
    ``"none"``) that the caller stores tile by tile instead of row-major
    (``walk`` "tile", whole M and N tiles, K a multiple of ``tile_k``): the
    operand's rows in groups of one TDM component (``rows``, half the
    tile's rows of the operand), and group ``g``'s rows of K tile ``t``
    one after another from byte ``(g * rows) * packed_k + t * rows *
    row_bytes``. Each TDM load then copies whole 256 B lines
    (:func:`tiled_transfer`) into the LDS image of the row-major operand.
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
    lds_buffers: int = 2
    scale_swap: str = "read"
    read_issue: str = "block"
    walk: str = "tile"
    read_bases: str = "loop"
    wave_map: str = "rows"
    inst_prefetch: int = 0
    tiled_operands: str = "none"
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
        three bytes; ``k`` is a multiple of 4 (FP6 K is one of 16, the
        others' one of 32)."""
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
    def pairs(self) -> bool:
        return self.wave_map == "pairs"

    @property
    def n_group(self) -> int:
        """N-repeats a lane holds as adjacent D columns: all of them, or with
        ``wave_map`` "pairs" each of its two groups."""
        return self.wave_repeat_n // 2 if self.pairs else self.wave_repeat_n

    @property
    def store_width(self) -> int:
        """Elements per D store and C load: at most 16 B and a divisor of N,
        so a vector is wholly inside or outside N and stays aligned."""
        width = min(self.n_group, 16 // _OUT_BYTES[self.out_dtype])
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
        return 8 if self.n_group % 8 == 0 else 4

    @property
    def scale_regs(self) -> bool:
        return self.scales == "registers"

    @property
    def swap_after_wait(self) -> bool:
        return self.contiguous_k and self.scale_swap == "wait"

    @property
    def fixed_bases(self) -> bool:
        return self.read_bases == "fixed"

    @property
    def panels(self) -> bool:
        """``walk`` "panels" or "halves": A and B in 128-row panel slots."""
        return self.walk in ("panels", "halves")

    @property
    def halves(self) -> bool:
        return self.walk == "halves"

    @property
    def ring(self) -> bool:
        return self.walk == "ring"

    def tiled(self, operand: str) -> bool:
        """Whether ``operand`` (``"a"`` / ``"b"``) is stored tile by tile."""
        return self.tiled_operands in (operand, "ab")

    @property
    def pass_spec(self) -> "MxGemmSpec":
        """With ``walk`` "panels", one 128x128 pass as a tile of its own;
        with "halves", one 128x256 half, its B padded as A (both are read
        from panel slots)."""
        if self.halves:
            half = replace(self, tile_m=self.tile_m // 2, walk="tile")
            pads = resolved_lds_pads(half)
            return replace(half, b_lds_pad=pads["a"], b_scale_lds_pad=pads["a_scale"])
        return replace(
            self, tile_m=self.tile_m // 2, tile_n=self.tile_n // 2, walk="tile"
        )

    @property
    def read_groups(self) -> int:
        """Groups of a K-step's reads with ``read_issue`` "interleave": group
        ``i`` holds A row ``i`` and B fragment ``i`` (group 0 the scales
        first)."""
        return max(self.wave_repeat_m, self.wave_repeat_n)

    def group_reads(self, group: int) -> int:
        """LDS read instructions of a read group, counted as ``reads_per_step``
        counts them."""
        rows = (group < self.wave_repeat_m) + (group < self.wave_repeat_n)
        scales = self.a_scale_reads + self.b_scale_reads if group == 0 else 0
        return scales + rows * self.row_reads

    @property
    def tdm_loads(self) -> int:
        """TDM loads a wave issues per tile: its data component, then its
        scale component unless the scales go to registers."""
        return 1 if self.scale_regs else 2

    @property
    def a_scale_reads(self) -> int:
        return 0 if self.scale_regs else max(1, self.wave_repeat_m // 4)

    @property
    def b_scale_reads(self) -> int:
        """One read per 16 B, or per group of N-repeats if shorter."""
        if self.scale_regs:
            return 0
        groups = self.wave_repeat_n // self.n_group
        return groups * max(1, self.n_group // 4)

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
        """One LDS buffer, or with ``walk`` "panels" one panel slot."""
        if self.panels:
            return mx_gemm_panel_layout(self)
        return mx_gemm_lds_layout(self)

    @property
    def lds_bytes(self) -> int:
        if self.ring:
            return RingWalk.lds_bytes()
        if self.halves:
            return HalfWalk.lds_bytes(self.lds_layout)
        buffers = _PANEL_SLOTS if self.panels else self.lds_buffers
        return buffers * self.lds_layout.buffer_bytes

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
                "lb3": self.lds_buffers == 3,
                "swait": self.scale_swap == "wait",
                "ril": self.read_issue == "interleave",
                "pw": self.walk == "panels",
                "pwh": self.halves,
                "rg": self.ring,
                "rbf": self.fixed_bases,
                "wmp": self.pairs,
                f"ipf{self.inst_prefetch}": self.inst_prefetch > 0,
                f"tl{self.tiled_operands}": self.tiled_operands != "none",
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


def _bank_spread_pad(row_bytes: int, lane_rows: int) -> LdsPad:
    """The 16 lanes of a lane half read rows ``lane_rows`` apart (RowMaps):
    16 B after every ``p`` such lane steps, ``p`` the steps after which a
    lane's first bank repeats (64 banks of 4 B). Each pad moves the next
    ``p`` lanes by one 16 B read, so the 16 lanes start on different banks.
    FP6 rows (96 B per 128 K) take it: their padded blocks are not a power of
    two, so TDM writes them in iterate mode."""
    step = lane_rows * row_bytes
    banks = _BANK_CYCLE_BYTES // 4
    steps = banks // gcd(step // 4, banks)
    return LdsPad(_DATA_PAD_BYTES, steps * step)


def derived_lds_pads(spec: MxGemmSpec) -> dict[str, LdsPad]:
    """TensileLite's padding with its A/B relabeled to rocKE B/A and
    VectorWidth taken as the operand's wave repeat, as in the benchmark
    solution (MIWaveTile [8, 4], VectorWidthA 8, VectorWidthB 4). FP6 data
    takes :func:`_bank_spread_pad` instead, its lanes ``wave_repeat_m`` A rows
    and ``n_group`` B rows apart."""
    k_bytes = spec.k_bytes(spec.tile_k)
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    if spec.elem_bits == 6:
        a_pad = _bank_spread_pad(k_bytes, rm)
        b_pad = _bank_spread_pad(k_bytes, spec.n_group)
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


def mx_gemm_panel_layout(spec: MxGemmSpec) -> MxGemmLdsLayout:
    """``walk`` "panels": one LDS slot, ``[data components][scale
    components]`` of one 128-row panel of A or of B, a component of 32 rows
    per wave, padded as the 128x128 pass tile's operands. A and B read the
    same offsets, each from its own slot. With "halves" a B slot holds both
    B panels, ``[8 data components][8 scale components]``: rows 0..255 of
    B, read as the 128x256 half tile's B."""
    pass_spec = spec.pass_spec
    pads = resolved_lds_pads(pass_spec)
    row_bytes = spec.k_bytes(spec.tile_k)
    rows = _PANEL_ROWS // _WAVES
    comp, iterate_rows, iterate_stride = _data_component(rows, row_bytes, pads["a"])
    scale_size = spec.k_steps * rows * 4
    scale_pad = pads["a_scale"]
    if scale_pad.on:
        scale_size += scale_size // scale_pad.block_bytes * scale_pad.pad_bytes
    data = _OperandLayout(rows, 0, comp, pads["a"], iterate_rows, iterate_stride)
    scale_base = _WAVES * comp
    scales = _OperandLayout(rows, scale_base, scale_size, scale_pad, 0, 0)
    b_scales = scales
    if spec.halves:
        b_scales = replace(scales, base=2 * _WAVES * comp)
    return MxGemmLdsLayout(
        row_bytes=row_bytes,
        k_steps=spec.k_steps,
        a=data,
        b=data,
        a_scale=scales,
        b_scale=b_scales,
        buffer_bytes=_round_up(scale_base + _WAVES * scale_size, _BANK_CYCLE_BYTES),
    )


def tiled_transfer(layout: MxGemmLdsLayout, name: str) -> tuple[int, int] | None:
    """``(tile_dim0, tile_dim1)`` of one TDM transfer of operand ``name``
    stored tile by tile (``tiled_operands``), so that every transfer is
    whole 256 B lines and lands as the row-major operand's: an iterate-mode
    block as lines; a padded component as its rows, as the row-major
    operand's transfer, when the rows are whole lines; an unpadded
    component as lines. ``None`` if the transfers cannot be whole lines."""
    op = layout.operand(name)
    row = layout.row_bytes
    if op.iterate_rows:
        block = op.iterate_rows * row
    elif op.pad.on:
        return (row, op.rows) if row % _GL2_LINE_BYTES == 0 else None
    else:
        block = op.rows * row
    if block % _GL2_LINE_BYTES:
        return None
    return _GL2_LINE_BYTES, block // _GL2_LINE_BYTES


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
    if spec.a_dtype == "fp6":
        # CK Tile's FP6 contract: rows of whole 16-value packs; the last
        # E8M0 scale covers the K % 32 values left, the TDM zero-fills past K.
        if spec.K % _FP6_PACK_K:
            return (
                f"FP6 K must be a multiple of {_FP6_PACK_K}: A and B rows are "
                f"whole {_FP6_PACK_K}-value packs (got K={spec.K})"
            )
    elif spec.K % spec.block_k:
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
        if spec.wmma_reuse != "b":
            return f"wmma_reuse {spec.wmma_reuse!r} needs local_prefetch 0"
        if not spec.half_b:
            return "wmma_reuse 'b' with local_prefetch 1 needs half_plr 'b'"
    if spec.wmma_reuse != "none" and spec.mma != "16x16":
        return f"wmma_reuse {spec.wmma_reuse!r} is for the 16x16 WMMA"
    if spec.scales not in SCALES:
        return f"scales must be one of {SCALES}"
    if spec.scale_regs and spec.local_prefetch != 0:
        return "scales 'registers' needs local_prefetch 0"
    if spec.scale_regs and spec.mma != "16x16":
        return "scales 'registers' is for the 16x16 WMMA"
    if spec.lds_buffers not in LDS_BUFFERS:
        return f"lds_buffers must be one of {LDS_BUFFERS}"
    if spec.scale_swap not in SCALE_SWAPS:
        return f"scale_swap must be one of {SCALE_SWAPS}"
    if spec.scale_swap == "wait" and (not spec.contiguous_k or spec.scale_regs):
        return "scale_swap 'wait' needs k_reads 'contiguous' and scales 'lds'"
    if spec.read_issue not in READ_ISSUES:
        return f"read_issue must be one of {READ_ISSUES}"
    if spec.read_issue == "interleave":
        if spec.local_prefetch != 1 or spec.half_plr != "none":
            return "read_issue 'interleave' needs local_prefetch 1 and half_plr 'none'"
        if spec.atom.a_rows != 1:
            return "read_issue 'interleave' is for the 16x16 WMMA"
        if spec.contiguous_k and not spec.swap_after_wait:
            # The groups hold reads only: the swaps go after the wait.
            return "read_issue 'interleave' with 'contiguous' reads needs scale_swap 'wait'"
    if spec.walk not in WALKS:
        return f"walk must be one of {WALKS}"
    if spec.ring:
        # The rest as the 256x256x256 tile (RingWalk.reason).
        return RingWalk.reason(spec, arch)
    if spec.panels:
        reason = _panel_reason(spec, arch)
        if reason:
            return reason
    if spec.read_bases not in READ_BASES:
        return f"read_bases must be one of {READ_BASES}"
    if spec.fixed_bases:
        if spec.atom.a_rows != 1:
            return "read_bases 'fixed' is for the 16x16 WMMA"
        layout = spec.lds_layout
        if layout.a_scale.pad.on or layout.b_scale.pad.on:
            # A K-step's scales are then not one stride from the last.
            return "read_bases 'fixed' needs the scales without LDS pad"
    row = spec.k_bytes(spec.tile_k)
    if spec.lds_buffers != 2 and row < _GL2_LINE_BYTES and _GL2_LINE_BYTES % row == 0:
        # The GL2 prefetch of such rows follows the tile index's parity,
        # which only two buffers keep static.
        return (
            f"lds_buffers {spec.lds_buffers} cannot take {row} B rows "
            f"(a fraction of a {_GL2_LINE_BYTES} B line)"
        )
    atom = spec.atom
    if spec.tile_m % (spec.warp_m * atom.m) or spec.tile_n % (spec.warp_n * atom.n):
        return f"the tile must hold whole {spec.mma} WMMA atoms per wave"
    if spec.inst_prefetch < 0:
        return "inst_prefetch must be 0 (none) or a positive WMMA count"
    if spec.wave_map not in WAVE_MAPS:
        return f"wave_map must be one of {WAVE_MAPS}"
    if spec.pairs:
        if tuple(spec.wave_group) != (2, 2) or atom.a_rows != 1:
            return "wave_map 'pairs' needs wave_group 2x2 and the 16x16 WMMA"
        if spec.wave_repeat_n % 2 or spec.scale_regs or spec.walk != "tile":
            return (
                "wave_map 'pairs' needs an even wave_repeat_n, scales 'lds' and "
                "walk 'tile'"
            )
    if spec.tiled_operands not in TILED_OPERANDS:
        return f"tiled_operands must be one of {TILED_OPERANDS}"
    if spec.tiled_operands != "none":
        if spec.walk != "tile":
            return "tiled_operands needs walk 'tile'"
        if spec.K % spec.tile_k:
            return "tiled_operands needs K to be a multiple of tile_k"
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    # A, B and D offsets switch to i64 past 2^31 (wide_*); the scales' stay
    # i32. A partial last tile's rows reach the next whole tile.
    mn_span = max(_round_up(spec.M, spec.tile_m), _round_up(spec.N, spec.tile_n))
    if mn_span * -(-spec.K // _ATOM_K) * 4 > _I32_SPAN:
        return "A and B scale bytes must each be at most 2^31 (i32 offsets)"
    # K < 2 * tile_k is supported but is a poor fit for this pipeline:
    # the tile k+2 refills and GL2 prefetches past K re-load the last tile, and
    # with one or two tiles they are most of the loads. Prefer a short-K kernel.
    # FP6 also takes K < tile_k: one partial tile, and the loads past K
    # re-load it (clamp_k).
    if spec.K < spec.tile_k and spec.a_dtype != "fp6":
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
    for name, edge, wide in (
        ("a", spec.edge_m, spec.wide_a),
        ("b", spec.edge_n, spec.wide_b),
    ):
        if not spec.tiled(name):
            continue
        if edge or wide:
            return f"tiled operand {name} needs whole tiles and at most 2 GiB"
        if tiled_transfer(spec.lds_layout, name) is None:
            return (
                f"tiled operand {name}: its TDM transfers are not whole "
                f"{_GL2_LINE_BYTES} B lines"
            )
    return None


def _panel_reason(spec: MxGemmSpec, arch: str) -> str | None:
    """Why ``walk`` "panels" or "halves" cannot take ``spec``, else ``None``."""
    walk = f"walk {spec.walk!r}"
    if spec.a_dtype != "fp6" or spec.b_dtype != "fp6":
        return f"{walk} is for FP6"
    side = 2 * _PANEL_ROWS
    if (spec.tile_m, spec.tile_n, spec.tile_k) != (side, side, 512):
        return f"{walk} needs the 256x256x512 tile"
    if tuple(spec.wave_group) != (2, 2) or spec.mma != "16x16":
        return f"{walk} needs wave_group 2x2 and the 16x16 WMMA"
    if spec.local_prefetch != 1:
        return f"{walk} needs local_prefetch 1"
    if spec.lds_buffers != 2 or spec.scale_regs:
        # The slots replace the buffers; scales are staged in the slots.
        return f"{walk} takes no lds_buffers and scales 'lds'"
    if spec.edge_m or spec.edge_n:
        return f"{walk} needs M and N to be multiples of 256"
    if spec.K % (2 * spec.tile_k):
        # The loop runs a K tile of each pass order (each B slot with
        # "halves").
        return f"{walk} needs K to be a multiple of {2 * spec.tile_k}"
    if spec.wide_a or spec.wide_b:
        return f"{walk} needs A and B under 2 GiB each"
    reason = validate(spec.pass_spec, arch)
    return f"{walk} pass tile: {reason}" if reason else None


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
        """LDS offset of buffer ``parity``; with ``walk`` "panels" a pass's
        (A slot offset, B slot offset) as is."""
        if isinstance(parity, tuple):
            return parity
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
        """``k``, or past K the last whole tile's start (tile 0 when K is
        shorter than a tile)."""
        b = self.b
        inside = b.cmp_lt(k, self.params["K"])
        return b.select(inside, k, self.last_tile_k())

    def last_tile_k(self):
        if self.spec.K < self.spec.tile_k:
            return self.c0
        return self.b.sub(self.params["K"], self.tile_k_v)


def wave_coords(spec: MxGemmSpec, wave, ops=IntOps):
    """``(wave_m, wave_n)`` of wave ``wave`` in the wave group (``wave_map``)."""
    if spec.pairs:
        # Waves p and p + 2 (SIMD pair p) take A half p.
        return ops.mod(wave, 2), ops.div(wave, 2)
    wave_n = ops.mod(wave, spec.warp_n)
    return ops.div(wave, spec.warp_n), wave_n


class RowMaps:
    """WMMA positions to tile rows. A lane's M-repeats are adjacent A rows,
    so its A scales of a K-step are contiguous; its N-repeats are adjacent
    D columns (each group of ``n_group`` with ``wave_map`` "pairs"), so one
    accumulator slot is one vector store per group."""

    def __init__(self, k: _Kernel):
        self.k = k

    def m_row(self, rm, m_local):
        ops, rm_count = self.k.ops, self.k.spec.wave_repeat_m
        base = ops.mul(self.k.wave_m, rm_count * self.k.spec.atom.m)
        return ops.add(base, ops.add(ops.mul(m_local, rm_count), rm))

    def n_row(self, rn, n_local):
        k = self.k
        ops, spec = k.ops, k.spec
        if spec.pairs:
            # Group g of SIMD pair p in B half (g + p) % 2, the quarter of the
            # tile's columns that wave_n picks in it.
            group = spec.n_group
            half, j = divmod(rn, group)
            side = k.wave_m if half == 0 else ops.mod(ops.add(k.wave_m, 1), 2)
            base = ops.add(
                ops.mul(side, spec.tile_n // 2),
                ops.mul(k.wave_n, group * spec.atom.n),
            )
            return ops.add(base, ops.add(ops.mul(n_local, group), j))
        rn_count = spec.wave_repeat_n
        base = ops.mul(k.wave_n, rn_count * spec.atom.n)
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
        if k.spec.tiled(operand):
            return self._tiled_descriptor(operand, lds, row0, k_byte)
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

    def _tiled_descriptor(self, operand: str, lds, row0, k_byte):
        """``tiled_operands``: the component's rows of the K tile at
        ``k_byte``, stored one after another from ``row0 * packed_k + k_byte
        * rows``, in ``tiled_transfer`` units; the extents run from there to
        the operand's end, so a tile needs no new ones."""
        k = self.k
        b, ops = k.b, k.ops
        layout, _, _, mask = self._operand(operand)
        base_ptr = k.params["A"] if operand == "a" else k.params["B"]
        unit, units = tiled_transfer(k.layout, operand)
        rows = layout.rows
        element = ops.add(ops.mul(row0, k.packed_k), ops.mul(k_byte, rows))
        extra: dict = {}
        if layout.iterate_rows:
            extra.update(
                iterate_count=rows // layout.iterate_rows,
                lds_increment=layout.iterate_stride,
                global_increment=ops.v(unit * units),
            )
        elif layout.pad.on:
            extra.update(tdm_pad_fields(layout.pad.block_bytes, layout.pad.pad_bytes))
        left = ops.mul(self._rows_left(operand, row0), k.packed_k)
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=ops.v(unit),
            tensor_dim1=ops.v(ops.div(left, unit)),
            row_stride=ops.v(unit),
            tile_dim0=unit,
            tile_dim1=units,
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
        self.even = even0
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

    def _lds_offset(self, parity):
        return self.k.buffer_base(parity)

    def _patched(self, desc, parity, byte_offset, words):
        """``desc`` with its LDS address moved to buffer ``parity``, its
        global address advanced by ``byte_offset`` and the group-1 words in
        ``words`` (index -> value) replaced."""
        k = self.k
        b, ops = k.b, k.ops
        d0, d1, *rest = desc
        lds = b.add(b.vec_extract(d0, 1), ops.v(self._lds_offset(parity)))
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

    def _tiled_patch(self, k_byte, words):
        """``(global byte offset, group-1 words)`` of this wave's data load of
        the K tile at ``k_byte`` with ``tiled_operands``: a tiled operand's
        tile is ``k_byte * rows`` into its row group and keeps the extents it
        was built with; even waves load B, odd waves A."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        rows = {op: k.layout.operand(op).rows for op in ("a", "b")}

        def offset(operand):
            return ops.mul(k_byte, rows[operand]) if spec.tiled(operand) else k_byte

        if spec.tiled_operands == "ab" and rows["a"] == rows["b"]:
            byte_offset = offset("a")
        else:
            byte_offset = b.select(self.even, ops.v(offset("b")), ops.v(offset("a")))
        if words:
            # One operand is row-major: its waves take the new words.
            kept = {i: b.vec_extract(self.data[1], i) for i in words}
            row_b = not spec.tiled("b")
            words = {
                i: b.select(self.even, *((w, kept[i]) if row_b else (kept[i], w)))
                for i, w in words.items()
            }
        return byte_offset, words

    def issue(self, k0, parity) -> None:
        """This wave's data and scale loads of the tile at ``k0`` into buffer ``parity``."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        k_byte = spec.k_bytes(k0, ops)
        row_major = not (spec.tiled("a") and spec.tiled("b"))
        remaining_k = b.sub(k.packed_k, ops.v(k_byte)) if row_major else None
        c16 = b.const_i32(16)
        data = self.data
        words = {}
        if row_major:
            w2 = b.vec_extract(data[1], 2)
            words = {
                1: b.shl(remaining_k, c16),
                2: b.lor(b.land(w2, b.const_i32(-0x10000)), b.lshr(remaining_k, c16)),
            }
        byte_offset = k_byte
        if spec.tiled_operands != "none":
            byte_offset, words = self._tiled_patch(k_byte, words)
        b.tensor_load_to_lds(*self._patched(data, parity, byte_offset, words))
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


class PanelTdmLoader(TdmLoader):
    """``walk`` "panels" and "halves": panel loads. Wave ``w`` copies
    component ``w`` (32 rows) of a 128-row A or B panel, its data then its
    scales, into a panel slot; the descriptors of panel 0 at LDS offset 0
    are built once per operand and each load patches the LDS offsets, the K
    tile and the panel's rows."""

    def _lds_offset(self, at):
        # Panel loads take LDS byte offsets, not buffer parities.
        return at

    def hoist(self) -> None:
        k = self.k
        b, ops, c0, p = k.b, k.ops, k.c0, k.params
        self.data = {
            op: self.data_descriptor(op, c0, k.wave, c0, b.sub(k.packed_k, c0))
            for op in ("a", "b")
        }
        self.scale = {
            op: self.scale_descriptor(op, c0, k.wave, c0) for op in ("a", "b")
        }
        self.scale_stride = {
            "a": ops.v(ops.mul(p["M"], 4)),
            "b": ops.v(ops.mul(p["N"], 4)),
        }

    def issue_panel(self, operand: str, panel: int, at, k0) -> None:
        """This wave's loads of ``operand``'s panel ``panel`` (rows from
        ``panel * 128``) of the K tile at ``k0``; ``at`` is the LDS offset
        of its data and of its scales (``PanelWalk.at``, ``HalfWalk.at``)."""
        k = self.k
        b, ops = k.b, k.ops
        k_byte = k.spec.k_bytes(k0, ops)
        remaining_k = b.sub(k.packed_k, ops.v(k_byte))
        c16 = b.const_i32(16)
        data = self.data[operand]
        w2 = b.vec_extract(data[1], 2)
        rows = ops.mul(k.packed_k, panel * _PANEL_ROWS)
        b.tensor_load_to_lds(
            *self._patched(
                data,
                at[0],
                ops.add(k_byte, rows),
                {
                    1: b.shl(remaining_k, c16),
                    2: b.lor(
                        b.land(w2, b.const_i32(-0x10000)), b.lshr(remaining_k, c16)
                    ),
                },
            )
        )
        k_step = ops.v(ops.div(k0, _ATOM_K))
        scale = self.scale[operand]
        steps_left = b.sub(ops.v(k.k_atoms()), k_step)
        s2, s3 = b.vec_extract(scale[1], 2), b.vec_extract(scale[1], 3)
        b.tensor_load_to_lds(
            *self._patched(
                scale,
                at[1],
                ops.add(
                    b.mul(k_step, self.scale_stride[operand]), panel * _PANEL_ROWS * 4
                ),
                {
                    2: b.lor(b.land(s2, b.const_i32(0xFFFF)), b.shl(steps_left, c16)),
                    3: b.lor(
                        b.land(s3, b.const_i32(-0x10000)), b.lshr(steps_left, c16)
                    ),
                },
            )
        )


class RingTdmLoader(TdmLoader):
    """``walk`` "ring": a wave's loads of its part (even waves a B part, odd
    waves an A part, component ``wave // 2``; :class:`RingWalk`). A line
    load copies line ``L`` of the part's 128 rows (256 B at byte ``256 L``
    of each row) into a line slot: 8 rows per iteration, the 256 B pad after
    every 256 B making the 512 B pitch, 16 iterations of a group and its
    16 B pad. A scale load copies a scale tile, the part's 512 B of each of
    8 K-steps. The descriptors are built once for line 0 and scale tile 0
    in their slot 0; each load patches the LDS offset, the global address
    and the extent left."""

    def _lds_offset(self, at):
        # Ring loads take LDS byte offsets, not buffer parities.
        return at

    def _line_descriptor(self, operand: str, component):
        k = self.k
        b, ops = k.b, k.ops
        _, block, _, mask = self._operand(operand)
        base_ptr = k.params["A"] if operand == "a" else k.params["B"]
        rows = RingWalk.PART_ROWS
        row0 = ops.add(block, ops.mul(component, rows))
        lds = ops.add(
            RingWalk.data_base(operand, 0), ops.mul(component, RingWalk.part_bytes())
        )
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(ops.mul(row0, k.packed_k))),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=k.packed_k,
            tensor_dim1=self._rows_left(operand, row0),
            row_stride=k.packed_k,
            tile_dim0=RingWalk.LINE,
            tile_dim1=RingWalk.GROUP_ROWS,
            workgroup_mask=mask,
            iterate_count=rows // RingWalk.GROUP_ROWS,
            lds_increment=RingWalk.group_bytes(),
            global_increment=ops.v(ops.mul(k.packed_k, RingWalk.GROUP_ROWS)),
            **tdm_pad_fields(RingWalk.LINE, RingWalk.PITCH - RingWalk.LINE),
        )

    def _scale_descriptor(self, operand: str, component):
        """Swizzled scales ``[K/128][MN][4]``: the part's 512 B of each of a
        scale tile's K-steps."""
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        _, block, extent, mask = self._operand(operand)
        base_ptr = p["A_scale"] if operand == "a" else p["B_scale"]
        rows = RingWalk.PART_ROWS
        lds = ops.add(
            RingWalk.scale_base(operand, 0),
            ops.mul(component, RingWalk.scale_ring_bytes()),
        )
        row_stride = ops.mul(extent, 4)
        col0 = ops.mul(ops.add(block, ops.mul(component, rows)), 4)
        pad = RingWalk.SCALE_PAD
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(col0)),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=b.sub(ops.v(row_stride), ops.v(col0)),
            tensor_dim1=ops.v(k.k_atoms()),
            row_stride=ops.v(row_stride),
            tile_dim0=rows * 4,
            tile_dim1=RingWalk.SCALE_STEPS,
            workgroup_mask=mask,
            **tdm_pad_fields(pad.block_bytes, pad.pad_bytes),
        )

    def hoist(self) -> None:
        """Both descriptors of this wave's part, selected by wave parity."""
        k = self.k
        b, ops, c0, p = k.b, k.ops, k.c0, k.params
        even0 = b.cmp_eq(ops.mod(k.wave, 2), c0)
        self.even = even0
        component0 = ops.div(k.wave, 2)

        def selected(b_desc, a_desc):
            return [b.select(even0, x, y) for x, y in zip(b_desc, a_desc)]

        self.data = selected(
            self._line_descriptor("b", component0),
            self._line_descriptor("a", component0),
        )
        self.scale = selected(
            self._scale_descriptor("b", component0),
            self._scale_descriptor("a", component0),
        )
        self.scale_stride = b.select(
            even0, ops.v(ops.mul(p["N"], 4)), ops.v(ops.mul(p["M"], 4))
        )

    def issue_line(self, k0, line: int) -> None:
        """Line ``line`` of the period at ``k0`` (``RingWalk.refills``
        counts lines from the period's first) into its slot. Past the row's
        end it loads one of the row's last two lines again, into a slot no
        K-step reads any more."""
        k = self.k
        b, ops = k.b, k.ops
        byte = ops.v(ops.add(k.spec.k_bytes(k0, ops), line * RingWalk.LINE))
        again = b.sub(k.packed_k, ops.v(RingWalk.PITCH - line % 2 * RingWalk.LINE))
        byte = b.select(b.cmp_lt(byte, k.packed_k), byte, again)
        left = b.sub(k.packed_k, byte)
        c16 = b.const_i32(16)
        w2 = b.vec_extract(self.data[1], 2)
        words = {
            1: b.shl(left, c16),
            2: b.lor(b.land(w2, b.const_i32(-0x10000)), b.lshr(left, c16)),
        }
        at = line % RingWalk.SLOTS * RingWalk.LINE
        b.tensor_load_to_lds(*self._patched(self.data, at, byte, words))

    def issue_scales(self, k0, tile: int) -> None:
        """Scale tile ``tile`` of the period at ``k0`` into its slot; past
        the end one of the last two tiles again."""
        k = self.k
        b, ops = k.b, k.ops
        steps = RingWalk.SCALE_STEPS
        k_step = ops.v(ops.add(ops.div(k0, _ATOM_K), tile * steps))
        atoms = ops.v(k.k_atoms())
        again = b.sub(atoms, ops.v(steps * (RingWalk.SCALE_SLOTS - tile % 2)))
        k_step = b.select(b.cmp_lt(k_step, atoms), k_step, again)
        steps_left = b.sub(atoms, k_step)
        c16 = b.const_i32(16)
        s2, s3 = b.vec_extract(self.scale[1], 2), b.vec_extract(self.scale[1], 3)
        at = tile % RingWalk.SCALE_SLOTS * RingWalk.scale_tile_bytes()
        b.tensor_load_to_lds(
            *self._patched(
                self.scale,
                at,
                b.mul(k_step, self.scale_stride),
                {
                    2: b.lor(b.land(s2, b.const_i32(0xFFFF)), b.shl(steps_left, c16)),
                    3: b.lor(
                        b.land(s3, b.const_i32(-0x10000)), b.lshr(steps_left, c16)
                    ),
                },
            )
        )

    def issue_load(self, k0, load) -> None:
        kind, index = load
        if kind == "line":
            self.issue_line(k0, index)
        else:
            self.issue_scales(k0, index)


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
        k_ahead = k.clamp_k(k_ahead)
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
        cluster_m0, cluster_n0 = self._cluster_origin()

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

        def tiled_line(base_ptr, row0, rows):
            """``tiled_operands``: each row group's tile is whole lines."""
            group_lines = rows * row_bytes // _GL2_LINE_BYTES

            def address(index):
                group_row = ops.add(row0, ops.mul(ops.div(index, group_lines), rows))
                offset = ops.add(
                    ops.mul(group_row, k.packed_k),
                    ops.add(
                        ops.mul(k_byte, rows),
                        ops.mul(ops.mod(index, group_lines), _GL2_LINE_BYTES),
                    ),
                )
                return b.global_ptr_add(base_ptr, ops.v(offset))

            return address

        categories = []
        operands = (
            ("a", cluster_y * spec.tile_m, p["A"], cluster_m0, p["M"], spec.edge_m),
            ("b", cluster_x * spec.tile_n, p["B"], cluster_n0, p["N"], spec.edge_n),
        )
        for name, cluster_rows, base_ptr, row0, extent, edge in operands:
            wide = spec.wide_a if name == "a" else spec.wide_b
            if spec.tiled(name):
                rows = k.layout.operand(name).rows
                categories.append(
                    (
                        cluster_rows * row_bytes // _GL2_LINE_BYTES,
                        tiled_line(base_ptr, row0, rows),
                    )
                )
            elif data_lines:
                categories.append(
                    (
                        cluster_rows * data_lines,
                        data_line(base_ptr, row0, extent, edge, wide),
                    )
                )
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
        self._distribute(categories)

    def _cluster_origin(self):
        """The first A and B rows of the workgroup's cluster."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        if not spec.has_cluster:
            return k.block_m, k.block_n
        cluster_m0 = b.sub(ops.v(k.block_m), ops.v(ops.mul(k.cluster_wy, spec.tile_m)))
        cluster_n0 = b.sub(ops.v(k.block_n), ops.v(ops.mul(k.cluster_wx, spec.tile_n)))
        return cluster_m0, cluster_n0

    def _distribute(self, categories) -> None:
        """One line prefetch per thread of the cluster for each of the
        ``(count, address)`` categories' lines, in rounds of the cluster's
        threads."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        cluster_x, cluster_y, _ = spec.cluster_dim
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

    def prefetch_ring(self, k0, loads) -> None:
        """``walk`` "ring": the cluster's lines of the ``loads`` of the
        period at ``k0`` (``("line", L)``: line ``L`` of every A and B row,
        ``("scales", t)``: scale tile ``t`` of A and B). Past the end the last
        line or scale tile is prefetched again."""
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        cluster_x, cluster_y, _ = spec.cluster_dim
        cluster_m0, cluster_n0 = self._cluster_origin()
        operands = (
            (cluster_y * spec.tile_m, p["A"], p["A_scale"], p["M"], cluster_m0),
            (cluster_x * spec.tile_n, p["B"], p["B_scale"], p["N"], cluster_n0),
        )
        steps = RingWalk.SCALE_STEPS
        categories = []
        for kind, index in loads:
            if kind == "line":
                byte = ops.v(ops.add(spec.k_bytes(k0, ops), index * RingWalk.LINE))
                last = b.sub(k.packed_k, ops.v(RingWalk.LINE))
                byte = b.smin(byte, last)
                for rows, base_ptr, _, _, row0 in operands:

                    def line(i, base_ptr=base_ptr, row0=row0, byte=byte):
                        row = ops.add(row0, i)
                        offset = ops.add(ops.mul(row, k.packed_k), byte)
                        return b.global_ptr_add(base_ptr, ops.v(offset))

                    categories.append((rows, line))
                continue
            k_step = ops.v(ops.add(ops.div(k0, _ATOM_K), index * steps))
            k_step = b.smin(k_step, b.sub(ops.v(k.k_atoms()), ops.v(steps)))
            for rows, _, base_ptr, extent, mn0 in operands:
                per_step = rows * 4 // _GL2_LINE_BYTES

                def scale_line(
                    i,
                    base_ptr=base_ptr,
                    extent=extent,
                    mn0=mn0,
                    per_step=per_step,
                    k_step=k_step,
                ):
                    row = ops.add(k_step, ops.div(i, per_step))
                    offset = ops.add(
                        ops.add(ops.mul(row, ops.mul(extent, 4)), ops.mul(mn0, 4)),
                        ops.mul(ops.mod(i, per_step), _GL2_LINE_BYTES),
                    )
                    return b.global_ptr_add(base_ptr, ops.v(offset))

                categories.append((steps * per_step, scale_line))
        self._distribute(categories)


class ScaleRegisters:
    """``scales`` "registers": a tile's E8M0 scale words from the swizzled
    scales ``[K/128][MN][4]`` straight into registers, one global load of
    up to 16 B per lane, operand and K-step (all 32 lanes read 512
    contiguous bytes on the large tiles). With ``r`` repeats of an operand
    (M-repeats for A, N-repeats for B) and ``h = r // 2`` (1 for one
    repeat), the lanes of half ``s`` hold the words of repeats ``s * h`` to
    ``s * h + h - 1`` of their row of the workgroup's tile; repeat ``i`` is word ``i % h`` with scale
    selector ``i // h``. Every address stays inside the tensor: K-steps past
    the last one read the last one, and rows past M (N) read a row of the
    tensor; those words only scale K-steps that are not computed and rows
    of D that are not stored. When M (N) is not a multiple of ``h`` each
    word is its own load."""

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows
        # Per operand: (byte offset, word groups) of each of a K-step's
        # loads, made at the first load (before the K loop).
        self._offsets = {}
        self._identity = None

    @staticmethod
    def half(repeats: int) -> int:
        return repeats // 2 if repeats >= 2 else 1

    @classmethod
    def place(cls, repeats: int, index: int) -> tuple[int, int]:
        """(word, scale selector) of repeat ``index``."""
        h = cls.half(repeats)
        return index % h, index // h

    def _row_offsets(self, name: str) -> list:
        """The lane's byte offsets into a K-step's row of scales and the
        words each load takes: one load per group of up to four of the
        lane's words, or per word when M (N) is not a multiple of ``h``.
        Made at the first load, before the K loop."""
        if name in self._offsets:
            return self._offsets[name]
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        a = name == "a"
        repeats = spec.wave_repeat_m if a else spec.wave_repeat_n
        h = self.half(repeats)
        first = ops.mul(k.lane_half, h) if repeats >= 2 else 0
        row = ops.v(
            ops.add(
                k.block_m if a else k.block_n,
                (self.rows.m_row if a else self.rows.n_row)(first, k.lane_col),
            )
        )
        extent = p["M"] if a else p["N"]
        if self._identity is None:
            self._identity = b.inline_asm(
                f"s_mov_b32 $0, {LdsReader._IDENTITY:#x}", "=s", result_type=I32
            )
        if (spec.M if a else spec.N) % h:
            last = b.sub(extent, k.c1)
            offsets = [
                (ops.v(ops.mul(b.smin(ops.v(ops.add(row, j)), last), 4)), [(0, 1)])
                for j in range(h)
            ]
        else:
            # Groups of h rows are whole: one past the end reads the last
            # group.
            row = b.smin(row, b.sub(extent, b.const_i32(h)))
            groups = [(j, min(4, h - j)) for j in range(0, h, 4)]
            offsets = [(ops.v(ops.mul(row, 4)), groups)]
        self._offsets[name] = offsets
        return offsets

    def _words(self, base, offsets) -> list:
        """The lane's words of one operand and K-step: each load takes
        ``base`` (SGPR) and a lane offset (VGPR) of ``offsets``."""
        k = self.k
        b, ops = k.b, k.ops
        words = []
        for offset, groups in offsets:
            at = b.global_ptr_add(base, offset)
            for j, n in groups:
                if n == 1:
                    words.append(b.global_load(at, ops.v(j), I32, align=4))
                    continue
                loaded = b.global_load_vN(at, ops.v(j), I32, n, align=4 * n)
                words += [b.vec_extract(loaded, i) for i in range(n)]
        return words

    def load(self, k0) -> list:
        """Every K-step's A then B words of the tile at ``k0``."""
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        first = ops.div(k0, _ATOM_K)
        last = b.sub(ops.v(k.k_atoms()), k.c1)
        # The tile's copy of each lane offset: a v_perm_b32 that returns it
        # through an identity selector LLVM cannot see into and also takes
        # the tile's first K-step, so it is made next to the tile's loads,
        # which take it as a 32-bit offset, rather than once before the K
        # loop as a 64-bit one; all the tile's loads read that register.
        offsets = {}
        for name in ("a", "b"):
            offsets[name] = [
                (b.perm_b32(ops.v(first), offset, self._identity), groups)
                for offset, groups in self._row_offsets(name)
            ]
        bases = []
        for step in range(k.spec.k_steps):
            k_step = b.smin(ops.v(ops.add(first, step)), last)
            for ptr, extent in ((p["A_scale"], p["M"]), (p["B_scale"], p["N"])):
                step_bytes = ops.mul(ops.mul(k_step, extent), 4)
                bases.append(b.global_ptr_add(ptr, ops.v(step_bytes)))
        # Each K-step's rows of scales as SGPR bases from one asm LLVM cannot
        # see into: they stay out of the lane offsets, and live together, so
        # no load computes its address into registers an earlier load may
        # still be reading.
        count = len(bases)
        bases = b.inline_asm_multi(
            "",
            ",".join(["=s"] * count + [str(i) for i in range(count)]),
            bases,
            result_types=[base.type for base in bases],
            sideeffect=False,
        )
        out = []
        for step in range(k.spec.k_steps):
            out += self._words(bases[2 * step], offsets["a"])
            out += self._words(bases[2 * step + 1], offsets["b"])
        return out


class LdsReader:
    """LDS to registers: the scales of one K-step in 16 B reads, then each A
    row and B fragment as its ``lane_reads`` (one 16 B read per K chunk: two
    for FP4, four for 8-bit; FP6 six 8 B or three 16 B reads)."""

    # v_perm_b32 selector swapping bytes 1 and 2 of a word (``k_reads``
    # "contiguous": the WMMA's K groups 1 and 2 hold the K-step's groups 2, 1).
    _SWAP_GROUPS_1_2 = 0x03010200
    # v_perm_b32 selector returning its second source unchanged.
    _IDENTITY = 0x03020100

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows
        # id(fragment) -> (fragment, its reads), for fragment_pair's.
        self._parts = {}
        # The reads of the last scales() call.
        self.scale_reads = []
        # read_bases "fixed": (operand, buffer offset, index) -> address.
        self.fixed = None

    def parts(self, frag):
        """The reads a :meth:`fragment_pair` fragment was made of (the
        registers its WMMAs read; not the zero padding), else ``None``."""
        entry = self._parts.get(id(frag))
        return list(entry[1]) if entry and entry[0] is frag else None

    @staticmethod
    def _operand_buffer(buffer, operand: str):
        """``buffer``, or of a pass's (A slot, B slot) the operand's slot."""
        if isinstance(buffer, tuple):
            return buffer[operand.startswith("b")]
        return buffer

    def load_i32(self, offset, n: int = 4):
        return self.k.b.smem_load_vN(
            self.k.shared, self.k.ops.v(offset), dtype=I32, n=n
        )

    def _lane_k(self, step):
        k = self.k
        return k.ops.add(
            step * k.spec.step_bytes, k.ops.mul(k.lane_half, k.spec.lane_half_bytes)
        )

    def fix_bases(self, buffers) -> None:
        """``read_bases`` "fixed", before the K loop: for each buffer (or
        pass's slots) one register per A row, B fragment and scale group,
        holding the buffer offset plus the lane's row and lane-half offsets.
        A read adds its K-step and chunk, which it takes as an immediate."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        half = ops.mul(k.lane_half, spec.lane_half_bytes)
        self.fixed = {}
        identity = b.inline_asm(
            f"s_mov_b32 $0, {self._IDENTITY:#x}", "=s", result_type=I32
        )

        def keep(value):
            # Opaque to LLVM (the selector is), so it stays in a register
            # instead of being computed again next to the reads. An asm VGPR
            # operand would hold it in v0-v255 for the whole loop.
            value = ops.v(value)
            return b.perm_b32(value, value, identity)

        for buffer in buffers:
            for operand, count, row_of in (
                ("a", spec.wave_repeat_m, self.rows.m_row),
                ("b", spec.wave_repeat_n, self.rows.n_row),
            ):
                offset = self._operand_buffer(buffer, operand)
                for index in range(count):
                    if (operand, offset, index) not in self.fixed:
                        row = k.layout.row_offset(
                            ops, operand, row_of(index, k.lane_col)
                        )
                        self.fixed[operand, offset, index] = keep(
                            ops.add(ops.add(offset, row), half)
                        )
                name = f"{operand}_scale"
                # wave_map "pairs": one B scale base per group of N-repeats.
                firsts = [0]
                if operand == "b" and spec.pairs:
                    firsts.append(spec.n_group)
                for first in firsts:
                    key = (name, offset) if first == 0 else (name, offset, first)
                    if not spec.scale_regs and key not in self.fixed:
                        row = row_of(first, k.lane_col)
                        self.fixed[key] = keep(
                            ops.add(offset, k.layout.scale_offset(ops, name, 0, row))
                        )

    def _fixed_base(self, buffer, operand: str, index, step):
        offset = self._operand_buffer(buffer, operand)
        return self.k.ops.add(
            self.fixed[operand, offset, index], step * self.k.spec.step_bytes
        )

    def _fixed_scale_base(self, buffer, name: str, step, first: int = 0):
        layout = self.k.layout.operand(name)
        offset = self._operand_buffer(buffer, name)
        key = (name, offset) if first == 0 else (name, offset, first)
        return self.k.ops.add(self.fixed[key], step * layout.rows * 4)

    def _row_fragment(self, buffer, operand: str, row, lane_k):
        k, ops = self.k, self.k.ops
        buffer = self._operand_buffer(buffer, operand)
        base = ops.add(ops.add(buffer, k.layout.row_offset(ops, operand, row)), lane_k)
        return self._read_fragment(base)

    def _read_fragment(self, base):
        k, ops = self.k, self.k.ops
        (first, words), *rest = k.spec.lane_reads
        frag = self.load_i32(ops.add(base, first), words)
        for offset, words in rest:
            frag = k.b.vec_concat(frag, self.load_i32(ops.add(base, offset), words))
        padding = k.spec.atom.frag_words - sum(w for _, w in k.spec.lane_reads)
        if padding:
            frag = k.b.vec_concat(frag, k.b.vector_splat(k.b.const_i32(0), padding))
        return frag

    def _assemble(self, parts):
        """A fragment from its reads (in ``lane_reads`` order) and padding."""
        b = self.k.b
        frag = parts[0]
        for part in parts[1:]:
            frag = b.vec_concat(frag, part)
        padding = self.k.spec.atom.frag_words - sum(
            w for _, w in self.k.spec.lane_reads
        )
        if padding:
            frag = b.vec_concat(frag, b.vector_splat(b.const_i32(0), padding))
        return frag

    def fragment_pair(self, buffer, step, rm, rn):
        """A row ``rm`` and B fragment ``rn`` of a K-step (16x16 WMMA; either
        may be ``None``), their reads alternating: A's first, B's first,
        A's second, ..."""
        k, ops = self.k, self.k.ops
        lane_k = None if self.fixed is not None else self._lane_k(step)
        bases = []
        for index, operand, row_of in (
            (rm, "a", self.rows.m_row),
            (rn, "b", self.rows.n_row),
        ):
            if index is None:
                continue
            if self.fixed is not None:
                bases.append(self._fixed_base(buffer, operand, index, step))
                continue
            buffer_op = self._operand_buffer(buffer, operand)
            row = row_of(index, k.lane_col)
            bases.append(
                ops.add(
                    ops.add(buffer_op, k.layout.row_offset(ops, operand, row)), lane_k
                )
            )
        parts = [[] for _ in bases]
        for offset, words in k.spec.lane_reads:
            for base, part in zip(bases, parts):
                part.append(self.load_i32(ops.add(base, offset), words))
        frags = []
        for part in parts:
            frags.append(self._assemble(part))
            self._parts[id(frags[-1])] = (frags[-1], part)
        frags = iter(frags)
        return (
            next(frags) if rm is not None else None,
            next(frags) if rn is not None else None,
        )

    def a_fragment(self, buffer, rm, step):
        k = self.k
        if self.fixed is not None:
            return self._read_fragment(self._fixed_base(buffer, "a", rm, step))
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
        if self.fixed is not None:
            return self._read_fragment(self._fixed_base(buffer, "b", rn, step))
        lane_k = self._lane_k(step)
        row = self.rows.n_row(rn, self.k.lane_col)
        return self._row_fragment(buffer, "b", row, lane_k)

    def scales(self, buffer, step):
        """Per-lane E8M0 words of one K-step: one per M- and per N-repeat."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        self.scale_reads = []
        if self.fixed is not None:
            a_base = self._fixed_scale_base(buffer, "a_scale", step)
        else:
            a_lane = k.lane if spec.atom.a_rows == 2 else k.lane_col
            a_base = ops.add(
                self._operand_buffer(buffer, "a"),
                k.layout.scale_offset(ops, "a_scale", step, self.rows.m_row(0, a_lane)),
            )
        a_scales = []
        if rm_count >= 4:
            for chunk in range(rm_count // 4):
                words = self.load_i32(ops.add(a_base, chunk * 16))
                self.scale_reads.append(words)
                a_scales += [b.vec_extract(words, i) for i in range(4)]
        else:
            words = self.load_i32(a_base, n=rm_count)
            if rm_count > 1:
                # One word is the scale itself (an asm operand cannot be a
                # one-element vector).
                self.scale_reads.append(words)
            a_scales = [b.vec_extract(words, i) for i in range(rm_count)]
        b_scales = []
        # One run of words per group of adjacent N-repeats (two with
        # wave_map "pairs").
        group = spec.n_group
        for first in range(0, rn_count, group):
            if self.fixed is not None:
                b_base = self._fixed_scale_base(buffer, "b_scale", step, first)
            else:
                b_base = ops.add(
                    self._operand_buffer(buffer, "b"),
                    k.layout.scale_offset(
                        ops, "b_scale", step, self.rows.n_row(first, k.lane_col)
                    ),
                )
            if group >= 4:
                for chunk in range(group // 4):
                    words = self.load_i32(ops.add(b_base, chunk * 16))
                    self.scale_reads.append(words)
                    b_scales += [b.vec_extract(words, i) for i in range(4)]
            else:
                words = self.load_i32(b_base, n=group)
                if group > 1:
                    # One word is the scale itself (an asm operand cannot be
                    # a one-element vector).
                    self.scale_reads.append(words)
                b_scales += [b.vec_extract(words, i) for i in range(group)]
        if spec.contiguous_k and not spec.swap_after_wait:
            return self.swap_groups(a_scales, b_scales)
        return a_scales, b_scales

    def swap_groups(self, a_scales, b_scales):
        """The words with bytes 1 and 2 swapped (``k_reads`` "contiguous")."""
        b = self.k.b
        swap = b.const_i32(self._SWAP_GROUPS_1_2)
        return (
            [b.perm_b32(w, w, swap) for w in a_scales],
            [b.perm_b32(w, w, swap) for w in b_scales],
        )

    def head(self, buffer, step):
        a_scales, b_scales = self.scales(buffer, step)
        a_frags = [
            self.a_fragment(buffer, rm, step) for rm in range(self.k.spec.wave_repeat_m)
        ]
        return a_frags, a_scales, b_scales

    def b_group(self, buffer, step, first: int, count: int):
        return [self.b_fragment(buffer, rn, step) for rn in range(first, first + count)]


class RingReader(LdsReader):
    """``walk`` "ring": reads from the line slots. ``step`` is the K-step
    of a period (``buffer`` is unused). Per operand a lane keeps a base, its
    row group in the part it reads plus its lane half's offset, and a split
    base, lane half 1's 512 B lower; each read takes one of them, ``512 *
    repeat`` and its immediate (:meth:`RingWalk.read`). A lane's scales of a
    K-step are its base in the scale ring plus the K-step's offset."""

    def fix_bases(self, buffers=None) -> None:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        identity = b.inline_asm(
            f"s_mov_b32 $0, {self._IDENTITY:#x}", "=s", result_type=I32
        )

        def keep(value):
            # As LdsReader.fix_bases: opaque to LLVM, kept in a register.
            value = ops.v(value)
            return b.perm_b32(value, value, identity)

        # The lane's row group: rows 8 * lane_col .. 8 * lane_col + 7.
        group = ops.mul(k.lane_col, RingWalk.group_bytes())
        half = ops.mul(k.lane_half, spec.lane_half_bytes)
        self.fixed = {}
        for operand, component in (("a", k.wave_m), ("b", k.wave_n)):
            part = ops.add(
                RingWalk.data_base(operand, 0),
                ops.mul(component, RingWalk.part_bytes()),
            )
            base = ops.v(ops.add(ops.add(part, group), half))
            split = b.sub(base, ops.v(ops.mul(k.lane_half, RingWalk.PITCH)))
            self.fixed[operand] = keep(base)
            self.fixed[operand, "split"] = keep(split)
            scales = ops.add(
                RingWalk.scale_base(operand, 0),
                ops.mul(component, RingWalk.scale_ring_bytes()),
            )
            row = ops.mul(k.lane_col, RingWalk.GROUP_ROWS)
            self.fixed[f"{operand}_scale"] = keep(
                ops.add(scales, RingWalk.scale_offset(0, row, ops))
            )

    def _fixed_scale_base(self, buffer, name: str, step, first: int = 0):
        offset = RingWalk.scale_offset(step % RingWalk.PERIOD, 0)
        return self.k.ops.add(self.fixed[name], offset)

    def _ring_fragment(self, operand: str, repeat: int, step: int):
        k = self.k
        spec = k.spec
        parts = []
        for offset, words in spec.lane_reads:
            split, imm = RingWalk.read(
                step % RingWalk.PERIOD, offset, spec.lane_half_bytes
            )
            base = self.fixed[operand, "split"] if split else self.fixed[operand]
            at = k.ops.add(base, RingWalk.PITCH * repeat + imm)
            parts.append(self.load_i32(at, words))
        return self._assemble(parts)

    def a_fragment(self, buffer, rm, step):
        return self._ring_fragment("a", rm, step)

    def b_fragment(self, buffer, rn, step):
        return self._ring_fragment("b", rn, step)


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
        # WMMAs emitted so far, for ``inst_prefetch``.
        self.wmmas = 0

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
            every = k.spec.inst_prefetch
            if every and self.wmmas % every == 0:
                k.b.inline_asm(_INST_PREFETCH_ASM, "")
            self.wmmas += 1
            out[index] = k.b.mma(
                atom, a_frag, b_frag, out[index], a_scale, b_scale, attrs=attrs or None
            )

        return mma

    def _wait(self, count: int) -> None:
        self.k.b.inline_asm(f"s_wait_dscnt {min(count, _DSCNT_MAX)}", "~{memory}")

    def _wmma_scales(self, a_scales, b_scales):
        """A K-step's scale words as its WMMAs take them, right after the
        K-step's first wait: with ``scale_swap`` "wait" the swaps come here,
        and stay below the wait (VALU may not cross the barrier)."""
        if not self.k.spec.swap_after_wait:
            return a_scales, b_scales
        self.k.b.sched_barrier(_SCHED_NO_VALU)
        return self.reader.swap_groups(a_scales, b_scales)

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
        if self.k.spec.read_issue == "interleave":
            return self._run_interleave(
                parity, accumulators, head, next_parity, hooks, steps
            )
        return self._run_full(parity, accumulators, head, next_parity, hooks, steps)

    @staticmethod
    def _signal_before_late(hooks, steps: int) -> None:
        """A one-K-step tile has no second-to-last K-step to signal in: it
        signals just before its cluster wait."""
        if steps == 1 and hooks.signal is not None:
            hooks.signal()

    @staticmethod
    def _hooks_at(hooks, step: int, steps: int):
        """``(refill, late_refill, signal before the late refill, signal
        after the half's first WMMAs)`` of K-step ``step``, each ``None`` if
        the K-step has none. A :class:`_Hooks` refills in the tile's last
        K-step and signals in the one before (just before the late refill
        with one K-step); a callable gives each K-step its own
        (``walk`` "ring"), its signal placed as a second-to-last K-step's."""
        if callable(hooks):
            each = hooks(step)
            return each.refill, each.late_refill, None, each.signal
        last = step + 1 == steps
        return (
            hooks.refill if last else None,
            hooks.late_refill if last else None,
            hooks.signal if last and steps == 1 else None,
            hooks.signal if step + 2 == steps else None,
        )

    def _run_half_b(self, parity, accumulators, head, next_parity, hooks, steps):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        half = rn_count // 2
        reuse_b = spec.wmma_reuse == "b"
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        *head, b_halves = head
        next_buffer = k.buffer_base(next_parity)
        for step in range(steps):
            a_frags, a_scales, b_scales = head
            last = step + 1 == steps
            refill, late, signal_early, signal = self._hooks_at(hooks, step, steps)
            if refill is not None:
                refill()
            if not last:
                head = reader.head(buffer, step + 1)
                next_first = reader.b_group(buffer, step + 1, 0, half)
            else:
                head = reader.head(next_buffer, 0)
                next_first = reader.b_group(next_buffer, 0, 0, half)
            for which in (0, 1):
                if which == 1 and not last:
                    next_second = reader.b_group(buffer, step + 1, half, half)
                elif which == 1:
                    next_second = reader.b_group(next_buffer, 0, half, half)
                if which == 1 and late is not None:
                    if signal_early is not None:
                        signal_early()
                    late()
                signal_here = signal is not None and which == 1
                self._wait(spec.reads_per_step)
                if which == 0:
                    a_scales, b_scales = self._wmma_scales(a_scales, b_scales)
                order = [(rm, j) for j in range(half) for rm in range(rm_count)]
                issued = 0
                for rm, j in order:
                    rn = which * half + j
                    if signal_here and issued == spec.signal_wmmas:
                        signal()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_halves[which][j],
                        a_scales[rm],
                        b_scales[rn],
                        reuse_b=reuse_b and rm + 1 < rm_count,
                    )
                if signal_here and issued == spec.signal_wmmas:
                    signal()
            if not last:
                b_halves = [next_first, next_second]
        return out, (*head, [next_first, next_second])

    def _run_half_b_late(self, parity, accumulators, head, next_parity, hooks, steps):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, half = spec.wave_repeat_m, spec.wave_repeat_n // 2
        reuse_b = spec.wmma_reuse == "b"
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
            refill, late, signal_early, signal = self._hooks_at(hooks, step, steps)
            next_first = None
            for which in (0, 1):
                if which == 1 and late is not None:
                    if signal_early is not None:
                        signal_early()
                    late()
                signal_here = signal is not None and which == 1
                self._wait(waits[which])
                if which == 0:
                    a_scales, b_scales = self._wmma_scales(a_scales, b_scales)
                    if refill is not None:
                        refill()
                    head = reader.head(*source)
                else:
                    next_first = reader.b_group(*source, 0, half)
                issued = 0
                for j in range(half):
                    for rm in range(rm_count):
                        rn = which * half + j
                        if signal_here and issued == spec.signal_wmmas:
                            signal()
                        issued += 1
                        mma(
                            rm,
                            rn,
                            a_frags[rm],
                            b_halves[which][j],
                            a_scales[rm],
                            b_scales[rn],
                            reuse_b=reuse_b and rm + 1 < rm_count,
                        )
                if signal_here and issued == spec.signal_wmmas:
                    signal()
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
            a_scales, b_scales = self._wmma_scales(a_scales, b_scales)
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
    def l_order(rm_count: int, rn_count: int) -> list[tuple[int, int]]:
        """A K-step's WMMAs ``(rm, rn)`` in L order: part ``i`` adds A row
        ``i`` with B fragments ``0..i``, then B fragment ``i`` with A rows
        ``0..i-1``, so part ``i`` needs read group ``i`` and the groups
        before it only."""
        order = []
        for i in range(max(rm_count, rn_count)):
            if i < rm_count:
                order += [(i, j) for j in range(min(i + 1, rn_count))]
            if i < rn_count:
                order += [(j, i) for j in range(min(i, rm_count))]
        return order

    def _hold(self, a_frags, b_frags, scales, scale_reads) -> None:
        """A use of a K-step's operands after its last WMMA: none of their
        registers frees up, so no read of the next K-step's groups can
        overwrite one a WMMA of this K-step is still reading. With it the
        scale reads the swaps took (``scale_swap`` "wait"): in expert
        scheduling a read overwriting a register a VALU took waits for every
        WMMA issued since. Fragments carried in through the loop (reads not
        known) are left out."""
        values = list(scales) + list(scale_reads)
        for frag in [*a_frags, *b_frags]:
            values += self.reader.parts(frag) or []
        self.k.b.inline_asm("", ",".join("v" * len(values)), values)

    def _run_interleave(self, parity, accumulators, head, next_parity, hooks, steps):
        """``read_issue`` "interleave": the WMMAs of K-step ``s`` run in L
        order (:meth:`l_order`) in ``read_groups`` chunks; read group ``g``
        of K-step ``s+1`` follows chunk ``min(g, groups - 2)``, so no group
        follows the K-step's last chunk. K-step ``s`` ends with a use of its
        operands (:meth:`_hold`): the groups land in registers K-step
        ``s-1`` left, whose last WMMAs are at least a chunk old, and issue
        while the WMMAs run; each group arrives ahead of the K-step ``s+1``
        WMMAs that need it. The last K-step reads the next tile's K-step 0
        from the other buffer; the cluster late wait (and TDM refill)
        follows its first group. The signal follows the second-to-last
        K-step's last group, the buffer's last read."""
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        buffer = k.buffer_base(parity)
        next_buffer = k.buffer_base(next_parity)
        out = list(accumulators)
        mma = self._mma_fn(out)
        a_frags, a_scales, b_scales, (b_frags,) = head
        # Carried in with the head: the scale reads are not known.
        scale_reads = []
        order = self.l_order(rm_count, rn_count)
        groups = spec.read_groups
        size, extra = divmod(len(order), groups)
        chunks, at = [], 0
        for g in range(groups):
            n = size + (1 if g < extra else 0)
            chunks.append(order[at : at + n])
            at += n
        for step in range(steps):
            last = step + 1 == steps
            source = (next_buffer, 0) if last else (buffer, step + 1)
            if last:
                hooks.refill()
            signal_here = hooks.signal is not None and step + 2 == steps
            self._wait(spec.reads_per_step)
            a_scales, b_scales = self._wmma_scales(a_scales, b_scales)
            next_a, next_b = [None] * rm_count, [None] * rn_count
            for c, chunk in enumerate(chunks):
                for rm, rn in chunk:
                    mma(rm, rn, a_frags[rm], b_frags[rn], a_scales[rm], b_scales[rn])
                for g in range(groups):
                    if min(g, max(groups - 2, 0)) != c:
                        continue
                    if g == 0:
                        next_scales = reader.scales(*source)
                        next_scale_reads = reader.scale_reads
                    a_frag, b_frag = reader.fragment_pair(
                        *source,
                        g if g < rm_count else None,
                        g if g < rn_count else None,
                    )
                    if a_frag is not None:
                        next_a[g] = a_frag
                    if b_frag is not None:
                        next_b[g] = b_frag
                    if last and g == 0:
                        self._signal_before_late(hooks, steps)
                        hooks.late_refill(spec.group_reads(0))
                    if signal_here and g == groups - 1:
                        hooks.signal()
            self._hold(a_frags, b_frags, [*a_scales, *b_scales], scale_reads)
            a_frags, b_frags = next_a, next_b
            a_scales, b_scales = next_scales
            scale_reads = next_scale_reads
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
            a_scales, b_scales = self._wmma_scales(a_scales, b_scales)
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
    """PrefetchGlobalRead=``lds_buffers``: the TDM runs that many tiles
    ahead. Tile k refills its own buffer with tile k+B (B buffers) once
    every workgroup of the cluster has read it; a tile is waited for (and
    published) at its start, or with read-ahead at the start of the tile
    before's last K-step, with the B-1 tiles after it still in flight. Past
    the end the last tile is loaded again into a buffer nobody reads
    (SuppressNoLoadLoop). The loop runs B tiles, one per buffer, so buffer
    offsets fold; a tile count that B does not divide peels the rest."""

    def __init__(
        self, k: _Kernel, tdm: TdmLoader, gl2: Gl2Prefetcher, tile: TileCompute
    ):
        self.k = k
        self.tdm = tdm
        self.gl2 = gl2
        self.tile = tile
        self.scales = ScaleRegisters(k, tile.reader.rows) if k.spec.scale_regs else None

    def _prefetch_ahead(self, k0, buffer: int) -> None:
        # With two buffers the buffer is the tile index's parity; with more,
        # validate() leaves only rows whose prefetch does not use it.
        distance = self.k.spec.gl2_prefetch_distance
        if distance:
            self.gl2.prefetch(
                self.k.ops.v(self.k.ops.add(k0, distance * self.k.spec.tile_k)),
                (buffer + distance) % 2,
            )

    def hooks(self, k0, parity, static_parity: int) -> _Hooks:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        ahead = spec.lds_buffers * spec.tile_k

        def signal() -> None:
            k.fence()
            b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            k.fence()

        def refill() -> None:
            self._prefetch_ahead(k0, static_parity)
            k.fence()
            if spec.local_prefetch:
                # Tile k+1 is read next, from this point: only the tiles
                # after it may still be loading, and the barrier below
                # publishes it.
                b.s_wait_tensorcnt((spec.lds_buffers - 2) * spec.tdm_loads)
                b.s_barrier_bare()
                k.fence()

        def late(pending: int | None = None) -> None:
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            # No wave drained its reads of this buffer before the signal;
            # the workgroup barrier publishes them here. ``pending``: the
            # other buffer's reads issued after this buffer's last one.
            if pending is None:
                pending = spec.late_pending_reads
            pending = min(pending, _DSCNT_MAX)
            b.inline_asm(f"s_wait_dscnt {pending}", "~{memory}")
            b.s_barrier_bare()
            self.tdm.issue(k.clamp_k(ops.v(ops.add(k0, ahead))), parity)
            k.fence()

        return _Hooks(
            refill=refill, signal=signal if spec.has_cluster else None, late_refill=late
        )

    def wait_next_tile(self) -> None:
        # The later tiles' loads may stay in flight; this tile's must be done.
        b, spec = self.k.b, self.k.spec
        b.s_wait_tensorcnt((spec.lds_buffers - 1) * spec.tdm_loads)
        b.s_barrier_bare()
        b.inline_asm("", "~{memory}")

    def run(self):
        k = self.k
        b, ops, spec, c0, c1 = k.b, k.ops, k.spec, k.c0, k.c1
        n_acc = spec.wave_repeat_m * spec.wave_repeat_n
        read_ahead = spec.local_prefetch == 1
        tile = self.tile
        buffers = spec.lds_buffers
        k.cluster_sync()
        self.tdm.issue(c0, c0)
        for index in range(1, buffers):
            k_index = k.tile_k_v if index == 1 else ops.v(index * spec.tile_k)
            if spec.K <= index * spec.tile_k:
                # Fewer tiles than buffers: this buffer gets the last tile
                # again and is never read. The refill and GL2 prefetches
                # load it again too; see validate().
                k_index = k.clamp_k(k_index)
            self.tdm.issue(k_index, c1 if index == 1 else index)
        for ahead in range(buffers, spec.gl2_prefetch_distance):
            self.gl2.prefetch(ops.v(ahead * spec.tile_k), ahead % 2)
        if read_ahead:
            self.wait_next_tile()
        first = c0
        if spec.fixed_bases:
            tile.reader.fix_bases([k.buffer_base(p) for p in range(buffers)])
            first = 0
        accumulators = [b.zero_vec_f32(spec.atom.c_frag_len) for _ in range(n_acc)]
        head_values = tile.flatten(tile.first_head(first)) if read_ahead else []
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
                next_parity=(parity + 1) % buffers,
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
        for _ in range(full_tiles % buffers):
            accumulators, head_values, scale_values = loop_tile(
                start, start_parity, accumulators, head_values, scale_values
            )
            start, start_parity = ops.add(start, spec.tile_k), start_parity + 1
        n_head = len(head_values)
        loop = b.scf_for_iter(
            ops.v(start),
            b.const_i32(full_tiles * spec.tile_k) if spec.tail_steps else k.params["K"],
            b.const_i32(buffers * spec.tile_k),
            carried(accumulators, head_values, scale_values),
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc = list(values[:n_acc])
            head_vals = list(values[n_acc : n_acc + n_head])
            scale_vals = list(values[n_acc + n_head :])
            for index in range(buffers):
                acc, head_vals, scale_vals = loop_tile(
                    ops.add(k0, index * spec.tile_k),
                    (start_parity + index) % buffers,
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
                full_tiles % buffers,
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


class PanelWalk:
    """``walk`` "panels": the order of a 256x256 tile's four 128x128 passes
    and the panel loads around them. A pass ``(a, b)`` multiplies A panel
    ``a`` by B panel ``b``; each panel has its own slot. Consecutive passes
    share a panel, and the order alternates with the K tile's parity so
    that the panels a tile frees first are the ones the next tile's first
    pass needs. After its last use in a tile a panel's slot is refilled
    with the next tile's panel, in the order that tile first uses them; the
    wait before a pass covers the panels it uses first. Every wave issues
    two TDM loads per panel (its component's data and scales)."""

    ORDERS = (
        ((0, 0), (0, 1), (1, 1), (1, 0)),
        ((0, 1), (0, 0), (1, 0), (1, 1)),
    )
    LOADS_PER_PANEL = 2

    @staticmethod
    def slot(operand: str, panel: int) -> int:
        return panel if operand == "a" else 2 + panel

    @classmethod
    def at(cls, layout: MxGemmLdsLayout, operand: str, panel: int) -> tuple[int, int]:
        """LDS offsets of a panel's data and scales: its slot's."""
        offset = cls.slot(operand, panel) * layout.buffer_bytes
        return offset, offset

    @classmethod
    def first_use(cls, parity: int) -> list[tuple[str, int]]:
        """A tile's panels in the order its passes first use them."""
        seen = []
        for a, b in cls.ORDERS[parity]:
            for panel in (("a", a), ("b", b)):
                if panel not in seen:
                    seen.append(panel)
        return seen

    @classmethod
    def first_pass(cls, parity: int, panel) -> int:
        operand, index = panel
        return next(
            j
            for j, pair in enumerate(cls.ORDERS[parity])
            if pair["ab".index(operand)] == index
        )

    @classmethod
    def refills(cls, parity: int) -> list[list[tuple[str, int]]]:
        """Per pass, the panels whose last use in the tile it is, in the
        order the next tile first uses them."""
        last = {}
        for j, (a, b) in enumerate(cls.ORDERS[parity]):
            last["a", a] = j
            last["b", b] = j
        upcoming = cls.first_use(1 - parity)
        return [
            sorted(
                (panel for panel, j in last.items() if j == pass_index),
                key=upcoming.index,
            )
            for pass_index in range(len(cls.ORDERS[parity]))
        ]

    @classmethod
    def waits(cls) -> dict:
        """``s_wait_tensorcnt`` counts: ``"prologue"`` before tile 0's first
        pass, and ``(parity, j)`` before pass ``j`` of a tile of that parity
        (``None``: no panel to wait for). Read-ahead places the wait before
        pass ``j`` ahead of the refills after pass ``j - 1``. The counts
        repeat from the second K tile on; tile 0 differs only before its
        first pass."""
        issued, pending, counts = [], [], {}
        issued += [(0, *panel) for panel in cls.first_use(0)]
        for tile in range(5):
            parity = tile % 2
            for j, (a, b) in enumerate(cls.ORDERS[parity]):
                new = [
                    (tile, *panel)
                    for panel in (("a", a), ("b", b))
                    if cls.first_pass(parity, panel) == j
                ]
                count = None
                if new:
                    last = max(issued.index(panel) for panel in new)
                    count = (len(issued) - 1 - last) * cls.LOADS_PER_PANEL
                key = "prologue" if tile == 0 and j == 0 else (parity, j)
                if key in counts:
                    assert counts[key] == count, (key, counts[key], count)
                counts[key] = count
                issued += pending
                pending = [(tile + 1, *panel) for panel in cls.refills(parity)[j]]
        return counts


class PanelPipeline:
    """``walk`` "panels" (read-ahead): the K loop over pairs of tiles (one
    of each pass order, :class:`PanelWalk`), each tile four passes of the
    128x128 compute. A pass's last K-step reads the next pass's K-step 0
    after the wait for the panels that pass uses first; when it frees
    panels, the cluster was signalled in its second-to-last K-step and the
    late refill loads the next tile's panels into the freed slots."""

    def __init__(
        self,
        k: _Kernel,
        kp: _Kernel,
        tdm: PanelTdmLoader,
        gl2: Gl2Prefetcher,
        tile: TileCompute,
    ):
        self.k = k
        self.kp = kp
        self.tdm = tdm
        self.gl2 = gl2
        self.tile = tile
        self.counts = PanelWalk.waits()

    def bases(self, pair):
        a, b = pair
        kp = self.kp
        return (
            kp.buffer_base(PanelWalk.slot("a", a)),
            kp.buffer_base(PanelWalk.slot("b", b)),
        )

    def _wait(self, count) -> None:
        if count is None:
            return
        b = self.k.b
        b.s_wait_tensorcnt(count)
        b.s_barrier_bare()

    def hooks(self, k_tile, parity: int, j: int) -> _Hooks:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        refills = PanelWalk.refills(parity)[j]
        upcoming = (parity, j + 1) if j + 1 < 4 else (1 - parity, 0)

        def signal() -> None:
            k.fence()
            b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            k.fence()

        def refill() -> None:
            if j == 0 and spec.gl2_prefetch_distance:
                distance = spec.gl2_prefetch_distance
                self.gl2.prefetch(
                    ops.v(ops.add(k_tile, distance * spec.tile_k)),
                    (parity + distance) % 2,
                )
            k.fence()
            self._wait(self.counts[upcoming])
            k.fence()

        def late(pending: int | None = None) -> None:
            if not refills:
                return
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            if pending is None:
                pending = self.kp.spec.late_pending_reads
            b.inline_asm(f"s_wait_dscnt {min(pending, _DSCNT_MAX)}", "~{memory}")
            b.s_barrier_bare()
            k_next = k.clamp_k(ops.v(ops.add(k_tile, spec.tile_k)))
            for operand, panel in refills:
                self.tdm.issue_panel(
                    operand, panel, PanelWalk.at(self.kp.layout, operand, panel), k_next
                )
            k.fence()

        return _Hooks(
            refill=refill,
            signal=signal if spec.has_cluster and refills else None,
            late_refill=late,
        )

    def run(self) -> list[list]:
        k, tile = self.k, self.tile
        b, ops, spec, c0 = k.b, k.ops, k.spec, k.c0
        n_acc = self.kp.spec.wave_repeat_m * self.kp.spec.wave_repeat_n
        k.cluster_sync()
        for operand, panel in PanelWalk.first_use(0):
            self.tdm.issue_panel(
                operand, panel, PanelWalk.at(self.kp.layout, operand, panel), c0
            )
        for ahead in range(1, spec.gl2_prefetch_distance):
            self.gl2.prefetch(ops.v(ahead * spec.tile_k), ahead % 2)
        self._wait(self.counts["prologue"])
        b.inline_asm("", "~{memory}")
        if spec.fixed_bases:
            tile.reader.fix_bases([self.bases(pair) for pair in PanelWalk.ORDERS[0]])
        acc = [
            [b.zero_vec_f32(spec.atom.c_frag_len) for _ in range(n_acc)]
            for _ in range(4)
        ]
        head_values = tile.flatten(tile.first_head(self.bases(PanelWalk.ORDERS[0][0])))
        n_head = len(head_values)
        carried = [
            (f"acc{q}_{i}", v)
            for q, values in enumerate(acc)
            for i, v in enumerate(values)
        ] + [(f"head{i}", v) for i, v in enumerate(head_values)]
        loop = b.scf_for_iter(
            c0,
            k.params["K"],
            b.const_i32(2 * spec.tile_k),
            carried,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc = [list(values[q * n_acc : (q + 1) * n_acc]) for q in range(4)]
            head_vals = list(values[4 * n_acc :])
            for parity in (0, 1):
                k_tile = ops.add(k0, parity * spec.tile_k)
                order = PanelWalk.ORDERS[parity]
                for j, (a, b_panel) in enumerate(order):
                    upcoming = (
                        order[j + 1] if j + 1 < 4 else PanelWalk.ORDERS[1 - parity][0]
                    )
                    q = 2 * a + b_panel
                    acc[q], head = tile.run(
                        self.bases((a, b_panel)),
                        acc[q],
                        hooks=self.hooks(k_tile, parity, j),
                        head=tile.unflatten(head_vals),
                        next_parity=self.bases(upcoming),
                    )
                    head_vals = tile.flatten(head)
            b.scf_yield(*(v for values in acc for v in values), *head_vals)
        results = list(loop.results[: 4 * n_acc])
        assert len(loop.results) == 4 * n_acc + n_head
        # No workgroup may exit while a TDM load into its LDS is in flight:
        # with multicast, peers' loads write here too.
        b.s_wait_tensorcnt(0)
        if spec.has_cluster:
            k.cluster_sync()
        return [results[q * n_acc : (q + 1) * n_acc] for q in range(4)]


class HalfWalk:
    """``walk`` "halves": the order of a 256x256 tile's two 128x256 halves
    and the panel loads around them. Half ``h`` multiplies A panel ``h`` by
    both B panels. The LDS holds ``[A0][A1][B slot 0][B slot 1]``: a slot
    per A panel and a slot of both B panels per K tile parity. After half 0
    the A0 slot is refilled with the next tile's A0; after half 1 the A1
    slot with the next tile's A1, then the tile's B slot with B two tiles
    ahead, so the next tile's B loads while this tile's is read. Every wave
    issues two TDM loads per panel (its component's data and scales)."""

    LOADS_PER_PANEL = 2

    @staticmethod
    def pair_bytes(layout: MxGemmLdsLayout) -> int:
        """Bytes of a B slot (both panels and their scales)."""
        scales = layout.b_scale
        return _round_up(scales.base + 2 * _WAVES * scales.stride, _BANK_CYCLE_BYTES)

    @classmethod
    def lds_bytes(cls, layout: MxGemmLdsLayout) -> int:
        return 2 * layout.buffer_bytes + 2 * cls.pair_bytes(layout)

    @classmethod
    def bases(cls, layout: MxGemmLdsLayout, half: int, parity: int) -> tuple:
        """(A slot, B slot) offsets of half ``half`` of a tile of parity
        ``parity``."""
        a_slot = half * layout.buffer_bytes
        return a_slot, 2 * layout.buffer_bytes + parity * cls.pair_bytes(layout)

    @classmethod
    def at(
        cls, layout: MxGemmLdsLayout, operand: str, panel: int, parity: int
    ) -> tuple[int, int]:
        """LDS offsets of a panel's data and scales: A panel ``panel`` in
        its slot, B panel ``panel`` in the B slot of ``parity`` (components
        ``4 * panel`` on)."""
        a_slot, b_slot = cls.bases(layout, panel, parity)
        if operand == "a":
            return a_slot, a_slot
        first = panel * _WAVES
        return (
            b_slot + first * layout.b.stride,
            b_slot + first * layout.b_scale.stride,
        )

    @staticmethod
    def prologue() -> list[tuple[str, int, int]]:
        """Loads before the loop as ``(operand, panel, K tile)``: tile 0's
        half-0 panels, then what a tile's end loads (A1 of tile 0, B of
        tile 1)."""
        return [("a", 0, 0), ("b", 0, 0), ("b", 1, 0)] + [
            (operand, panel, ahead - 1) for operand, panel, ahead in HalfWalk.refills(1)
        ]

    @staticmethod
    def refills(half: int) -> list[tuple[str, int, int]]:
        """Loads after half ``half``'s last read, ``(operand, panel, K
        tiles ahead)``, in issue order."""
        if half == 0:
            return [("a", 0, 1)]
        return [("a", 1, 1), ("b", 0, 2), ("b", 1, 2)]

    @staticmethod
    def needs(half: int) -> list[tuple[str, int]]:
        return [("a", 0), ("b", 0), ("b", 1)] if half == 0 else [("a", 1)]

    @classmethod
    def waits(cls) -> dict:
        """``s_wait_tensorcnt`` counts: ``"prologue"`` before tile 0's half
        0, and ``h`` before half ``h`` of a tile. Read-ahead places the wait
        before a half ahead of the refills after the half before it. The
        counts repeat from tile 0's half 1 on."""
        issued = [(tile, operand, panel) for operand, panel, tile in cls.prologue()]
        pending, counts = [], {}
        for tile in range(5):
            for half in (0, 1):
                last = max(issued.index((tile, *panel)) for panel in cls.needs(half))
                count = (len(issued) - 1 - last) * cls.LOADS_PER_PANEL
                key = "prologue" if tile == 0 and half == 0 else half
                if key in counts:
                    assert counts[key] == count, (key, counts[key], count)
                counts[key] = count
                issued += pending
                pending = [
                    (tile + ahead, operand, panel)
                    for operand, panel, ahead in cls.refills(half)
                ]
        return counts


class HalfPipeline(PanelPipeline):
    """``walk`` "halves" (read-ahead): the K loop over pairs of tiles (one
    per B slot, :class:`HalfWalk`), each tile two halves of the 128x256
    compute. A half's last K-step reads the next half's K-step 0 after the
    wait for the panels it uses first; the cluster was signalled in its
    second-to-last K-step and the late refill loads the panels it frees."""

    def __init__(
        self,
        k: _Kernel,
        kp: _Kernel,
        tdm: PanelTdmLoader,
        gl2: Gl2Prefetcher,
        tile: TileCompute,
    ):
        super().__init__(k, kp, tdm, gl2, tile)
        self.counts = HalfWalk.waits()

    def bases(self, pair):
        half, parity = pair
        return HalfWalk.bases(self.kp.layout, half, parity)

    def hooks(self, k_tile, parity: int, half: int) -> _Hooks:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec

        def signal() -> None:
            k.fence()
            b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            k.fence()

        def refill() -> None:
            if half == 0 and spec.gl2_prefetch_distance:
                distance = spec.gl2_prefetch_distance
                self.gl2.prefetch(
                    ops.v(ops.add(k_tile, distance * spec.tile_k)),
                    (parity + distance) % 2,
                )
            k.fence()
            self._wait(self.counts[1 - half])
            k.fence()

        def late(pending: int | None = None) -> None:
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            if pending is None:
                pending = self.kp.spec.late_pending_reads
            b.inline_asm(f"s_wait_dscnt {min(pending, _DSCNT_MAX)}", "~{memory}")
            b.s_barrier_bare()
            k_next = {}
            for operand, panel, ahead in HalfWalk.refills(half):
                if ahead not in k_next:
                    k_next[ahead] = k.clamp_k(
                        ops.v(ops.add(k_tile, ahead * spec.tile_k))
                    )
                at = HalfWalk.at(self.kp.layout, operand, panel, parity)
                self.tdm.issue_panel(operand, panel, at, k_next[ahead])
            k.fence()

        return _Hooks(
            refill=refill,
            signal=signal if spec.has_cluster else None,
            late_refill=late,
        )

    def run(self) -> list[list]:
        k, tile = self.k, self.tile
        b, ops, spec, c0 = k.b, k.ops, k.spec, k.c0
        layout = self.kp.layout
        n_acc = self.kp.spec.wave_repeat_m * self.kp.spec.wave_repeat_n
        k.cluster_sync()
        for operand, panel, k_tile in HalfWalk.prologue():
            at = HalfWalk.at(layout, operand, panel, k_tile % 2)
            k0 = ops.v(k_tile * spec.tile_k) if k_tile else c0
            self.tdm.issue_panel(operand, panel, at, k0)
        for ahead in range(1, spec.gl2_prefetch_distance):
            self.gl2.prefetch(ops.v(ahead * spec.tile_k), ahead % 2)
        self._wait(self.counts["prologue"])
        b.inline_asm("", "~{memory}")
        if spec.fixed_bases:
            tile.reader.fix_bases(
                [self.bases((half, parity)) for parity in (0, 1) for half in (0, 1)]
            )
        acc = [
            [b.zero_vec_f32(spec.atom.c_frag_len) for _ in range(n_acc)]
            for _ in range(2)
        ]
        head_values = tile.flatten(tile.first_head(self.bases((0, 0))))
        n_head = len(head_values)
        carried = [
            (f"acc{h}_{i}", v)
            for h, values in enumerate(acc)
            for i, v in enumerate(values)
        ] + [(f"head{i}", v) for i, v in enumerate(head_values)]
        loop = b.scf_for_iter(
            c0,
            k.params["K"],
            b.const_i32(2 * spec.tile_k),
            carried,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc = [list(values[h * n_acc : (h + 1) * n_acc]) for h in range(2)]
            head_vals = list(values[2 * n_acc :])
            for parity in (0, 1):
                k_tile = ops.add(k0, parity * spec.tile_k)
                for half in (0, 1):
                    upcoming = (1, parity) if half == 0 else (0, 1 - parity)
                    acc[half], head = tile.run(
                        self.bases((half, parity)),
                        acc[half],
                        hooks=self.hooks(k_tile, parity, half),
                        head=tile.unflatten(head_vals),
                        next_parity=self.bases(upcoming),
                    )
                    head_vals = tile.flatten(head)
            b.scf_yield(*(v for values in acc for v in values), *head_vals)
        results = list(loop.results[: 2 * n_acc])
        assert len(loop.results) == 2 * n_acc + n_head
        # No workgroup may exit while a TDM load into its LDS is in flight:
        # with multicast, peers' loads write here too.
        b.s_wait_tensorcnt(0)
        if spec.has_cluster:
            k.cluster_sync()
        return [results[h * n_acc : (h + 1) * n_acc] for h in range(2)]


class RingWalk:
    """``walk`` "ring" (FP6, the 256x256x256 tile, read-ahead with B in
    halves): A and B stay row-major and every TDM load is whole 256 B lines.

    The tile's four 128-row parts (A rows 0..127 and 128..255, B likewise;
    a TDM component each, loaded by one wave as :class:`TdmLoader`'s) keep
    two 256 B line slots per row. Line ``L`` of a row's K bytes (bytes
    ``256 L`` to ``256 L + 255``) lands in slot ``L % 2``, so byte ``x`` of
    a row sits at ``x % 512`` of the row's 512 B. 16 B follow every 8 rows:
    a lane half's 16 lanes read rows 8 apart (:class:`RowMaps`), and each pad
    moves the next 8 rows onto the next 16 B bank group. K-step ``q`` reads
    bytes ``96 q`` to ``96 q + 95``; the offsets repeat every ``PERIOD``
    K-steps (six lines). One TDM load fills one line slot of a part: 128
    rows of 256 B, a 256 B pad after every 256 B, iterate mode over the 16
    groups of 8 rows. Line ``L + 2`` goes into slot ``L % 2`` in the K-step
    that last reads line ``L``, once the cluster has read it.

    A part's scales are a ring of ``SCALE_SLOTS`` tiles of ``SCALE_STEPS``
    K-steps (512 B a K-step, a 16 B pad after every 256 B); tile ``t + 2``
    goes into tile ``t``'s slot in the K-step that last reads tile ``t``. A
    period holds a whole number of scale tiles, so each K-step of a period
    reads the same slots in every period. The LDS holds the four scale
    rings, then the four data parts.

    A lane reads its 8 rows of a part (its M- or N-repeats) from one 8-row
    group, 512 B apart: from its base (the group plus its lane half's
    offset) plus ``512 * repeat`` and the read's immediate, or from the
    split base, lane half 1's base less 512 B, where only lane half 1's
    bytes wrap past the row's 512 B (:meth:`read`)."""

    LINE = _GL2_LINE_BYTES
    SLOTS = 2
    PITCH = SLOTS * LINE
    GROUP_ROWS = 8
    ROW_PAD = _DATA_PAD_BYTES
    PART_ROWS = _PANEL_ROWS
    STEP_BYTES = _ATOM_K * 6 // 8
    PERIOD = PITCH // gcd(STEP_BYTES, PITCH)
    PERIOD_LINES = PERIOD * STEP_BYTES // LINE
    SCALE_STEPS = 8
    SCALE_SLOTS = 2
    SCALE_PAD = LdsPad(_MXS_PAD_BYTES, _MXS_BLOCK_BYTES)
    PARTS = (("a", 0), ("a", 1), ("b", 0), ("b", 1))

    # -- LDS layout ----------------------------------------------------------

    @classmethod
    def group_bytes(cls) -> int:
        return cls.GROUP_ROWS * cls.PITCH + cls.ROW_PAD

    @classmethod
    def part_bytes(cls) -> int:
        return cls.PART_ROWS // cls.GROUP_ROWS * cls.group_bytes()

    @classmethod
    def row_offset(cls, row, ops=IntOps):
        """Offset of row ``row`` of a part from the part's base."""
        return ops.add(
            ops.mul(ops.div(row, cls.GROUP_ROWS), cls.group_bytes()),
            ops.mul(ops.mod(row, cls.GROUP_ROWS), cls.PITCH),
        )

    @classmethod
    def scale_step_bytes(cls) -> int:
        """Bytes of one K-step of a part's scales, with the pad."""
        size = cls.PART_ROWS * 4
        return size + size // cls.SCALE_PAD.block_bytes * cls.SCALE_PAD.pad_bytes

    @classmethod
    def scale_tile_bytes(cls) -> int:
        return cls.SCALE_STEPS * cls.scale_step_bytes()

    @classmethod
    def scale_ring_bytes(cls) -> int:
        return cls.SCALE_SLOTS * cls.scale_tile_bytes()

    @classmethod
    def part(cls, operand: str, component: int) -> int:
        return cls.PARTS.index((operand, component))

    @classmethod
    def scale_base(cls, operand: str, component: int) -> int:
        return cls.part(operand, component) * cls.scale_ring_bytes()

    @classmethod
    def data_base(cls, operand: str, component: int) -> int:
        return (
            len(cls.PARTS) * cls.scale_ring_bytes()
            + cls.part(operand, component) * cls.part_bytes()
        )

    @classmethod
    def lds_bytes(cls) -> int:
        return len(cls.PARTS) * (cls.scale_ring_bytes() + cls.part_bytes())

    @classmethod
    def scale_offset(cls, step: int, row, ops=IntOps):
        """Offset, from the part's scale base, of the four E8M0 scales of
        row ``row`` at K-step ``step``."""
        tile, local = divmod(step, cls.SCALE_STEPS)
        linear = ops.mul(row, 4)
        pad = ops.mul(
            ops.div(linear, cls.SCALE_PAD.block_bytes), cls.SCALE_PAD.pad_bytes
        )
        start = (tile % cls.SCALE_SLOTS) * cls.scale_tile_bytes()
        return ops.add(start + local * cls.scale_step_bytes(), ops.add(linear, pad))

    @classmethod
    def at(cls, load, operand: str, component: int) -> int:
        """LDS offset where a part's load ``load`` lands: ``("line", L)``
        at its slot of row 0, ``("scales", t)`` at its scale tile."""
        kind, index = load
        if kind == "line":
            return cls.data_base(operand, component) + (index % cls.SLOTS) * cls.LINE
        start = (index % cls.SCALE_SLOTS) * cls.scale_tile_bytes()
        return cls.scale_base(operand, component) + start

    # -- The K stream --------------------------------------------------------

    @classmethod
    def offset(cls, step: int) -> int:
        """Where K-step ``step``'s first byte of a row sits in the row's
        512 B."""
        return step * cls.STEP_BYTES % cls.PITCH

    @classmethod
    def lines(cls, step: int) -> range:
        """The lines K-step ``step`` reads."""
        start = step * cls.STEP_BYTES
        return range(start // cls.LINE, (start + cls.STEP_BYTES - 1) // cls.LINE + 1)

    @classmethod
    def first(cls, line: int) -> int:
        """The first K-step that reads line ``line``."""
        return max(0, (cls.LINE * line - cls.STEP_BYTES) // cls.STEP_BYTES + 1)

    @classmethod
    def last(cls, line: int) -> int:
        """The last K-step that reads line ``line``."""
        return (cls.LINE * (line + 1) - 1) // cls.STEP_BYTES

    @classmethod
    def read(cls, step: int, offset: int, half_bytes: int) -> tuple[bool, int]:
        """``(split, immediate)`` of a lane's read at ``offset`` from its
        lane half's first byte of K-step ``step``, lane half 1 being
        ``half_bytes`` further: from the lane's base, or with ``split`` from
        the split base, which only lane half 1's bytes need. The reader adds
        ``512 * repeat``. A read never crosses the 512 B of a row: its offset
        and ``half_bytes`` are multiples of its size."""
        position = cls.offset(step) + offset
        if position >= cls.PITCH:
            return False, position - cls.PITCH
        return position + half_bytes >= cls.PITCH, position

    # -- Loads and waits -----------------------------------------------------

    @classmethod
    def first_use(cls, load) -> int:
        kind, index = load
        return cls.first(index) if kind == "line" else index * cls.SCALE_STEPS

    @classmethod
    def prologue(cls) -> list[tuple[str, int]]:
        """Loads before the loop: the first two lines and scale tiles, in the
        order the K-steps first read them (a line before scales)."""
        loads = [("line", line) for line in range(cls.SLOTS)]
        loads += [("scales", tile) for tile in range(cls.SCALE_SLOTS)]
        return sorted(loads, key=lambda load: (cls.first_use(load), load[0]))

    @classmethod
    def needs(cls, step: int) -> list[tuple[str, int]]:
        """Loads that K-step ``step`` reads first."""
        loads = [("line", line) for line in cls.lines(step) if cls.first(line) == step]
        if step % cls.SCALE_STEPS == 0:
            loads.append(("scales", step // cls.SCALE_STEPS))
        return loads

    @classmethod
    def refills(cls, step: int) -> list[tuple[str, int]]:
        """Loads of K-step ``step``'s late refill, in issue order: line
        ``L + 2`` for the line ``L`` it last reads, then scale tile ``t + 2``
        if it is the last K-step of tile ``t``. A wave issues them for its
        part; ``step`` counts from the first K tile, so a period ``p`` later
        the lines are ``6 p`` and the scale tiles ``2 p`` further."""
        loads = [
            ("line", line + cls.SLOTS)
            for line in cls.lines(step)
            if cls.last(line) == step
        ]
        if (step + 1) % cls.SCALE_STEPS == 0:
            loads.append(("scales", step // cls.SCALE_STEPS + cls.SCALE_SLOTS))
        return loads

    @classmethod
    def signals(cls, step: int) -> bool:
        """Whether K-step ``step`` signals the cluster, once it has issued
        the next K-step's reads: the next K-step's late refill reloads a
        slot those reads were the last of."""
        return bool(cls.refills(step + 1))

    @classmethod
    def waits(cls) -> dict:
        """``s_wait_tensorcnt`` counts: ``"prologue"`` before K-step 0's
        reads, and ``q`` (``0 <= q < PERIOD``) in K-step ``q`` of a period,
        before the next K-step's reads and ahead of the K-step's late refill
        (``None``: nothing new to wait for). The counts are the same in every
        period."""
        issued = list(cls.prologue())

        def count(needed):
            if not needed:
                return None
            return len(issued) - 1 - max(issued.index(load) for load in needed)

        counts = {"prologue": count(cls.needs(0))}
        for step in range(3 * cls.PERIOD):
            value = count(cls.needs(step + 1))
            key = step % cls.PERIOD
            if key in counts:
                assert counts[key] == value, (key, counts[key], value)
            counts[key] = value
            issued += cls.refills(step)
        return counts

    # -- Validation ----------------------------------------------------------

    @classmethod
    def reason(cls, spec: MxGemmSpec, arch: str = "gfx1250") -> str | None:
        """Why ``walk`` "ring" cannot take ``spec``, else ``None``."""
        walk = "walk 'ring'"
        if spec.a_dtype != "fp6" or spec.b_dtype != "fp6":
            return f"{walk} is for FP6"
        if (spec.tile_m, spec.tile_n, spec.tile_k) != (256, 256, 256):
            return f"{walk} needs the 256x256x256 tile"
        if tuple(spec.wave_group) != (2, 2) or spec.mma != "16x16" or spec.pairs:
            return f"{walk} needs wave_group 2x2, the 16x16 WMMA and wave_map 'rows'"
        if not spec.half_b:
            return f"{walk} needs local_prefetch 1 and half_plr 'b'"
        if spec.lds_buffers != 2 or spec.scale_regs:
            # The ring replaces the buffers; scales are staged in it.
            return f"{walk} takes no lds_buffers and scales 'lds'"
        if not spec.fixed_bases:
            return f"{walk} needs read_bases 'fixed'"
        if not spec.contiguous_k:
            # Chunked reads are 8 B pairs whose two offsets reach 2040 B,
            # under the ring's immediates.
            return f"{walk} needs k_reads 'contiguous'"
        if spec.tiled_operands != "none":
            return f"{walk} reads row-major A and B (tiled_operands 'none')"
        pads = (
            spec.a_lds_pad,
            spec.b_lds_pad,
            spec.a_scale_lds_pad,
            spec.b_scale_lds_pad,
        )
        if any(pad is not None for pad in pads):
            return f"{walk} pads its LDS itself (no *_lds_pad)"
        if spec.edge_m or spec.edge_n:
            return f"{walk} needs M and N to be multiples of 256"
        period_k = cls.PERIOD * _ATOM_K
        if spec.K % period_k:
            # The loop runs whole periods of the ring.
            return f"{walk} needs K to be a multiple of {period_k}"
        if spec.wide_a or spec.wide_b:
            return f"{walk} needs A and B under 2 GiB each"
        # The rest as the 256x256x256 tile with loop read bases: the ring's
        # scales keep one stride per K-step whatever their pad.
        reason = validate(replace(spec, walk="tile", read_bases="loop"), arch)
        if reason:
            return f"{walk} tile: {reason}"
        capacity = ArchTarget.from_gfx(arch).lds_capacity_bytes
        if cls.lds_bytes() > capacity:
            return f"{walk}: LDS {cls.lds_bytes()} B exceeds {capacity} B"
        return None


class RingPipeline:
    """``walk`` "ring" (read-ahead, B in halves): the K loop over periods of
    16 K-steps, each the 256x256 tile's compute with :class:`RingWalk`'s
    events in its K-steps: the TDM wait before the next K-step's reads, the
    cluster signal once a slot's last reads are issued, and the late refill
    of the slot in the next K-step. ``gl2_prefetch_distance`` lines ahead of
    each line refill (and one scale tile ahead of a scale refill) go to GL2
    with it."""

    # The reader's buffer: the ring has one, the K-step picks the offsets.
    RING = ()

    def __init__(
        self, k: _Kernel, tdm: RingTdmLoader, gl2: Gl2Prefetcher, tile: TileCompute
    ):
        self.k = k
        self.tdm = tdm
        self.gl2 = gl2
        self.tile = tile
        self.counts = RingWalk.waits()

    def _prefetched(self, loads) -> list:
        """The loads ``gl2_prefetch_distance`` lines (one scale tile) after
        ``loads``."""
        distance = self.k.spec.gl2_prefetch_distance
        if not distance:
            return []
        return [
            (kind, index + (distance if kind == "line" else 1)) for kind, index in loads
        ]

    def hooks(self, k0, step: int) -> _Hooks:
        k = self.k
        b, spec = k.b, k.spec
        count = self.counts[step]
        loads = RingWalk.refills(step)

        def refill() -> None:
            k.fence()
            b.s_wait_tensorcnt(count)
            b.s_barrier_bare()
            k.fence()

        def signal() -> None:
            k.fence()
            b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            k.fence()

        def late(pending: int | None = None) -> None:
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            # Every wave's reads of the slots before the barrier; the next
            # K-step's reads (other slots) may stay in flight.
            if pending is None:
                pending = spec.late_pending_reads
            b.inline_asm(f"s_wait_dscnt {min(pending, _DSCNT_MAX)}", "~{memory}")
            b.s_barrier_bare()
            for load in loads:
                self.tdm.issue_load(k0, load)
            ahead = self._prefetched(loads)
            if ahead:
                self.gl2.prefetch_ring(k0, ahead)
            k.fence()

        has_signal = spec.has_cluster and RingWalk.signals(step)
        return _Hooks(
            refill=refill if count is not None else None,
            signal=signal if has_signal else None,
            late_refill=late if loads else None,
        )

    def run(self) -> list:
        k, tile = self.k, self.tile
        b, spec, c0 = k.b, k.spec, k.c0
        n_acc = spec.wave_repeat_m * spec.wave_repeat_n
        k.cluster_sync()
        prologue = RingWalk.prologue()
        for load in prologue:
            self.tdm.issue_load(c0, load)
        distance = spec.gl2_prefetch_distance
        if distance:
            lines = [("line", RingWalk.SLOTS + i) for i in range(distance)]
            self.gl2.prefetch_ring(c0, lines + [("scales", RingWalk.SCALE_SLOTS)])
        b.s_wait_tensorcnt(self.counts["prologue"])
        b.s_barrier_bare()
        b.inline_asm("", "~{memory}")
        tile.reader.fix_bases()
        acc = [b.zero_vec_f32(spec.atom.c_frag_len) for _ in range(n_acc)]
        head_values = tile.flatten(tile.first_head(self.RING))
        n_head = len(head_values)
        carried = [(f"acc{i}", v) for i, v in enumerate(acc)] + [
            (f"head{i}", v) for i, v in enumerate(head_values)
        ]
        loop = b.scf_for_iter(
            c0,
            k.params["K"],
            b.const_i32(RingWalk.PERIOD * _ATOM_K),
            carried,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc = list(values[:n_acc])
            head = tile.unflatten(list(values[n_acc:]))
            acc, head = tile.run(
                self.RING,
                acc,
                hooks=lambda step: self.hooks(k0, step),
                head=head,
                next_parity=self.RING,
                steps=RingWalk.PERIOD,
            )
            b.scf_yield(*acc, *tile.flatten(head))
        assert len(loop.results) == n_acc + n_head
        results = list(loop.results[:n_acc])
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

    def _pack_16(self, values):
        """``values`` (FP32) as an FP16 or BF16 vector. A lone value goes
        through an optimization barrier first: without it the conversion of
        ``alpha * acc`` may be selected as one multiply-convert (v_fma_mixlo),
        which rounds the exact product once, so a product whose FP32
        rounding is a tie between two FP16/BF16 values can end on the odd
        one. Pairs convert with v_cvt_pk_*_f32, after the FP32 rounding."""
        b = self.k.b
        if len(values) == 1:
            values = [b.optimization_barrier(values[0])]
        if self.k.spec.out_dtype == "fp16":
            return b.vec_pack([b.trunc_f32_to_f16(v) for v in values], F16)
        return b.vec_pack([b.trunc_f32_to_bf16(v) for v in values], BF16)

    def _store(self, final, read_c: bool) -> None:
        k = self.k
        ops, spec = k.ops, k.spec
        group = spec.n_group
        # (first N-repeat, D column) of each group of adjacent columns.
        groups = [
            (first, ops.add(k.block_n, self.rows.n_row(first, k.lane_col)))
            for first in range(0, spec.wave_repeat_n, group)
        ]
        for rm in range(spec.wave_repeat_m):
            for slot in range(spec.atom.c_frag_len):
                for first, col in groups:
                    self._store_group(final, read_c, rm, slot, first, col)

    def _store_group(self, final, read_c: bool, rm: int, slot: int, first: int, col):
        """One accumulator slot's group of N-repeats from ``first``, at D
        column ``col``."""
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        rn_count = spec.wave_repeat_n
        out_type = _OUT_TYPES[spec.out_dtype]
        out_bytes = _OUT_BYTES[spec.out_dtype]
        width = spec.store_width
        chunks = range(0, spec.n_group, width)
        fp8 = spec.out_dtype == "fp8e4m3"
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
                final, read_c, rm, slot, row, (first, col), (c_ptr, d_ptr, index)
            )
            return
        values = [
            b.fmul(p["alpha"], b.vec_extract(final[rm * rn_count + rn], slot))
            for rn in range(first, first + spec.n_group)
        ]
        if read_c:
            c_words = [
                (
                    b.global_load(
                        c_ptr, ops.v(ops.add(index, begin)), out_type, align=out_bytes
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
            packed = self._pack_fp8(part) if fp8 else self._pack_16(part)
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

    def _store_masked(
        self, final, read_c: bool, rm: int, slot: int, row, group, at_row
    ):
        """One accumulator slot's group of a partial tile: each vector of
        ``width`` columns is read and written only if its row and columns are
        in D. ``group`` is the group's first N-repeat and D column, ``at_row``
        the C and D pointers and the group's index from them."""
        c_ptr, d_ptr, index = at_row
        first, col = group
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        rn_count, width = spec.wave_repeat_n, spec.store_width
        out_type = _OUT_TYPES[spec.out_dtype]
        out_bytes = _OUT_BYTES[spec.out_dtype]
        fp8 = spec.out_dtype == "fp8e4m3"
        row_inside = b.cmp_lt(ops.v(row), p["M"]) if spec.edge_m else None
        for begin in range(0, spec.n_group, width):
            inside = row_inside
            if spec.edge_n:
                col_inside = b.cmp_lt(ops.v(ops.add(col, begin)), p["N"])
                inside = b.land(row_inside, col_inside) if spec.edge_m else col_inside
            with b.scf_if(inside):
                at = ops.v(ops.add(index, begin))
                values = [
                    b.fmul(p["alpha"], b.vec_extract(final[rm * rn_count + rn], slot))
                    for rn in range(first + begin, first + begin + width)
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
                else:
                    packed = self._pack_16(values)
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


def _build_panels(k: _Kernel) -> None:
    """``walk`` "panels": the passes run on ``kp``, the kernel seen as one
    128x128 pass tile over panel slots; each quadrant's accumulators are
    stored as that tile's, from its corner of the workgroup's output. With
    "halves" ``kp`` is one 128x256 half and each half is stored as such."""
    b, ops, spec = k.b, k.ops, k.spec
    k.shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    k.shared_addr = b.smem_addr_of(k.shared)
    kp = replace(k, spec=spec.pass_spec, layout=spec.lds_layout)
    tdm = PanelTdmLoader(kp)
    tdm.hoist()
    tile = TileCompute(kp, LdsReader(kp, RowMaps(kp)))
    if spec.halves:
        final = HalfPipeline(k, kp, tdm, Gl2Prefetcher(k), tile).run()
        for half in (0, 1):
            kq = replace(kp, block_m=ops.add(k.block_m, half * _PANEL_ROWS))
            Epilogue(kq, RowMaps(kq)).emit(final[half])
        return
    final = PanelPipeline(k, kp, tdm, Gl2Prefetcher(k), tile).run()
    for a, b_panel in ((0, 0), (0, 1), (1, 0), (1, 1)):
        kq = replace(
            kp,
            block_m=ops.add(k.block_m, a * _PANEL_ROWS),
            block_n=ops.add(k.block_n, b_panel * _PANEL_ROWS),
        )
        Epilogue(kq, RowMaps(kq)).emit(final[2 * a + b_panel])


def _build_ring(k: _Kernel) -> None:
    """``walk`` "ring": the 256x256x256 tile's compute over the line slots
    (:class:`RingPipeline`), stored as the tile's."""
    b = k.b
    k.shared = b.smem_alloc(I8, [k.spec.lds_bytes], name_hint="ab_tdm")
    k.shared_addr = b.smem_addr_of(k.shared)
    rows = RowMaps(k)
    tdm = RingTdmLoader(k)
    tdm.hoist()
    tile = TileCompute(k, RingReader(k, rows))
    final = RingPipeline(k, tdm, Gl2Prefetcher(k), tile).run()
    Epilogue(k, rows).emit(final)


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
    k.wave_m, k.wave_n = wave_coords(spec, k.wave, ops)

    block_x, block_y = b.block_id_x(), b.block_id_y()
    k.block_m = ops.mul(block_y, spec.tile_m)
    k.block_n = ops.mul(block_x, spec.tile_n)
    if spec.has_cluster:
        k.cluster_wx = b.cluster_workgroup_id("x")
        k.cluster_wy = b.cluster_workgroup_id("y")

    if spec.panels:
        _build_panels(k)
        b.ret()
        return b.kernel
    if spec.ring:
        _build_ring(k)
        b.ret()
        return b.kernel
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
