# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Flattened-pixel implicit-GEMM forward convolution on wave64 MFMA (gfx950).

A dense NHWC forward convolution with any filter, stride, dilation and
padding, as a GEMM of output channels (MFMA A = weights) by flattened output
pixels ``X = N * Ho * Wo`` (MFMA B = input), reducing over taps x 64-channel
chunks:

* **Tile.** A block computes X256 x K``bk`` with 8 waves in two ping-pong
  groups (group g owns pixel half g); a wave owns 64 pixels x ``16 * kt``
  channels (``kt`` = 8 for ``bk`` 256, 4 for ``bk`` 128). One *step* is one
  tap and one 64-channel chunk: per 32-channel half the wave reads its 4 input
  and ``kt`` weight fragments from LDS and runs ``4 * kt`` ``mfma_16x16x32``.
* **LDS-DMA for both operands.** Two stages of (input X256 x 64 channels,
  weights ``bk`` x 64 channels), 16-byte ``buffer_load ... lds`` with the
  granule swizzle applied on the global side. im2col reads a per-wave LDS
  offset table built at each tile change (per tap and pixel: the input
  fibre's byte offset, or an out-of-range offset for padding, which zero
  fills); the chunk is the scalar offset.
* **Ping-pong DMA schedule.** Barrier instances alternate between the
  groups' memory and MFMA phases; the DMA of step ``j + 1`` is issued after
  instance ``2j`` and waited right before ``2j + 2`` (group 0 issues at its
  memory-phase start and waits at its MFMA-phase end; group 1, one barrier
  behind, forms its offsets in its memory phase, spreads its DMAs over its
  MFMA phase and waits at its memory-phase end), so every DMA has about a
  step to land. The kernel must not spill: scratch stores share the DMAs'
  vmcnt, so a spill puts a full DMA drain into the step.
* **Tap skipping.** A tile's output rows (and columns, when it lies inside
  one output row) bound the rows / columns each tap reads; the taps whose
  window is entirely padding form a prefix / suffix of the filter rows and
  columns and are skipped (atrous layers skip up to a third of the steps).
* **Persistent grid** of up to ``PERSIST`` blocks; block b takes item
  ``i * G + (b + i * ROT) % G`` in round i (the rotation spreads the tile
  classes of tap skipping over the blocks); the step stream, and so the DMA
  look-ahead, runs across items.
* **Split-C.** With ``split_c`` > 1 an item is (tile, piece of the chunks)
  and the block writes an fp32 slab of the packed ``[split_c, N, Ho, Wo, K]``
  workspace that ``direct_split_reduce`` folds into the output.
* **Register epilogue.** The output-channel order inside each 16-channel MFMA
  tile is interleaved by the weight DMA so that one ``permlane32_swap`` per
  dword pair gives every lane 8 consecutive channels: 16-byte stores.

AOT: batch, extents, C and K are kernargs; the filter, padding, stride,
dilation, tile and split are baked. Limits: C % 64 == 0, K % 8 == 0, at most
32 taps, packed tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, I8, I32, I64, IRBuilder, KernelDef, Value
from rocke.core.ir import VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    pad_w_reason,
    problem_dims_reason,
)

__all__ = ["FwdFlexMfmaSpec", "build_fwd_flex_mfma", "fwd_flex_mfma_reason"]

_WAVE = 64
_CK = 64  # channels per chunk
_BX = 256  # pixels per block tile
_XT = 4  # 16-pixel MFMA tiles per wave
PERSIST = 256  # blocks of the persistent grid (one per gfx950 CU)
ROT = 97  # per-round rotation of the item walk


class _Div:
    """Division by a runtime divisor 1 <= d < 2^24 of numerators in
    [0, 2^31): q = (mulhi(n, m) + n) >> l with l = bit length of d - 1 and
    m = floor(2^32 (2^l - d) / d) + 1, formed once per kernel (the generic
    signed division expands to ~30 instructions per use). The magic numbers
    come from :func:`_divs`."""

    def __init__(self, b: IRBuilder, d: Value, ml):
        self.b, self.d = b, d
        self.m, self.l = ml

    def div(self, n: Value) -> Value:
        b = self.b
        return b.lshr(b.add(b.umul_hi_i32(n, self.m), n), self.l)

    def mod(self, n: Value) -> Value:
        b = self.b
        return b.sub(n, b.mul(self.div(n), self.d))


def _divs(b: IRBuilder, lane: Value, ds: List[Value]) -> List[_Div]:
    """:class:`_Div` of each divisor in ``ds`` with the magic numbers formed
    together, lane i for divisor i (one VALU sequence instead of one per
    divisor), then read back by v_readlane. The magic's quotient comes in two
    16-bit halves, each an f32 estimate (|error| < 1 / 64) rounded to an
    integer and corrected once by its remainder, instead of a 64-bit division
    (~120 instructions)."""
    c0, c1 = b.const_i32(0), b.const_i32(1)
    d = c1
    for i in reversed(range(len(ds))):
        d = b.select(b.cmp_eq(lane, b.const_i32(i)), ds[i], d)
    # bit length of d - 1 from the exponent of its (exact) float
    e = b.lshr(b.bitcast(b.sitofp_f32(b.sub(d, c1)), I32), b.const_i32(23))
    l = b.smax(b.sub(e, b.const_i32(126)), c0)
    a = b.sub(b.shl(c1, l), d)
    rd = b.fmul(b.rcp_fast(b.sitofp_f32(d)), b.const_f32(65536.0))

    def half(n: Value):
        """(floor(n 2^16 / d), remainder) for 0 <= n < d."""
        est = b.fmul(b.sitofp_f32(n), rd)
        # round to nearest: the float of 2^23 + est has an ulp of 1
        q = b.sub(
            b.bitcast(b.fadd(est, b.const_f32(8388608.0)), I32), b.const_i32(0x4B000000)
        )
        # the remainder is small (|r| < d): its low 32 bits, from i64
        # products (an i32 product of these would overflow)
        r = b.trunc(
            b.sub(
                b.shl(b.zext(n, I64), b.const_i64(16)),
                b.mul(b.zext(q, I64), b.zext(d, I64)),
            ),
            I32,
        )
        # the estimate is never low: only a round-up to correct
        neg = b.cmp_lt(r, c0)
        return b.sub(q, b.select(neg, c1, c0)), b.add(r, b.select(neg, d, c0))

    qh, r = half(a)
    ql, _ = half(r)
    m = b.add(b.shl(qh, b.const_i32(16)), b.add(ql, c1))
    return [
        _Div(b, ds[i], (b.readlane(m, b.const_i32(i)), b.readlane(l, b.const_i32(i))))
        for i in range(len(ds))
    ]


def _dma_lds(b: IRBuilder, rsrc: Value, dst: Value, voff: Value, soff: Value) -> None:
    """16-byte ``buffer_load ... lds`` of every lane into LDS at ``dst``. As
    inline asm the waitcnt pass does not track it: the kernels' explicit
    waits are the only ones (the pass's own waits before fragment reads of
    the other stage cost a DMA round trip per step)."""
    b.inline_asm(
        "s_mov_b32 m0, $3\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds",
        "v,s,s,s,~{m0},~{memory}",
        [voff, rsrc, soff, b.readfirstlane(b.trunc(dst, I32))],
    )


def _dma_lds_at(
    b: IRBuilder, rsrc: Value, base: Value, imm: int, voff: Value, soff: Value
) -> None:
    """:func:`_dma_lds` into LDS at the wave-uniform ``base`` (an SGPR) plus
    the constant ``imm``: one SGPR per stage instead of one per instruction."""
    b.inline_asm(
        f"s_add_u32 m0, $3, {imm}\n\tbuffer_load_dwordx4 $0, $1, $2 offen lds",
        "v,s,s,s,~{m0},~{scc},~{memory}",
        [voff, rsrc, soff, base],
    )


@dataclass(frozen=True)
class FwdFlexMfmaSpec:
    """One configuration of :func:`build_fwd_flex_mfma`."""

    problem: DirectConvProblem
    bk: int = 256  # output channels per block tile (256, 192 (bx 256), 128 or 64)
    split_c: int = 1  # pieces of the channel chunks (fp32 slabs when > 1)
    # renumber the blocks so each of the 8 XCDs (blocks dealt round-robin)
    # walks a contiguous run of the tile order:
    # neighbouring X tiles (shared input rows) and the K tiles of one X tile
    # then share an L2.
    # xcd 2 (sw): each XCD walks one contiguous run of the tiles (an even
    # share of the last round); xcd 3 (sw, probe): xcd without the rotation
    xcd: bool = False
    # with xcd, the tile list is K-major (tile = tk * nxt + tx), so the
    # contiguous run one XCD walks covers a K-tile share and all X tiles:
    # each XCD reads 1/8 of the weights (weight-stationary L2).
    kmaj: bool = False
    # a channel tail (C % 64 != 0, C % 8 == 0): ceil(C / 64) chunks, the
    # last chunk's granules past C load zeros (input and weights).
    c_tail: bool = False
    # pixels per block tile, 256 or 128 (128: a 1 x 4 wave grid per group,
    # for small outputs)
    bx: int = 256
    # 2 (X128 x K256, bx = 128, bk = 256: the groups take the two
    # 32-channel halves of every chunk from three shared stages) or 4-way
    # in-block split of the reduction (X128 x K128 tile, bx = bk =
    # 128): the two wave groups take alternate channel chunks into their own
    # LDS stages, the two wave pairs of a group the two 32-channel halves of
    # a chunk; every wave owns 64 pixels x 128 channels, and the four partial
    # sums meet in LDS at the tile's end (small layers: twice the MFMAs per
    # step and no fp32 workspace)
    cs: int = 1
    # B is a dgrad's own KRSC filter, i.e. W[C][R][S][K] in this pass's
    # terms (dgrad via forward without the weight transpose pre-pass): a tap
    # reads the flipped filter position, the weights stage as K halves of
    # [64 reduction rows][128 outputs] (256-byte rows of 8 groups of 16
    # outputs, group g of row r at slot g ^ (r & 3) ^ 4 ((r >> 3) & 1)) and
    # the A fragments come from transposing LDS reads, conflict-free, two per
    # fragment, their lane part formed once
    wt: bool = False
    # a strided dgrad in one launch (needs wt; the problem is the dgrad's
    # own, H/W = dX): output phase (a, b) of the stride (dX rows a mod s,
    # columns b mod s) is a stride-1 forward over dY with the filter taps of
    # that phase; the pixel space is phase-major (each phase padded to whole
    # tiles), so a tile runs only its phase's taps and stores with stride s
    ds: bool = False
    # ds on a one-row dX (H = 1, a 1 x S filter, no row padding): the row
    # stride is moot, so the phases run along the columns only (a 1-D
    # dgrad with a column stride, e.g. 1 x 16 stride 8)
    h1: bool = False
    # ds where one output phase has every tap and the others none (a
    # 1 x 1 stride-2 dgrad): the items cover that phase only and the
    # epilogue also stores the zeros of the tapless phases' pixels next to
    # each stored one (one launch, no memset of dX)
    zf: bool = False
    # zf with light zero items (0: off): the compute items store no
    # zeros; the tapless phases' pixels are stored by a store-only loop after
    # a block's items, ZL_PX phase pixels per unit, dealt by cost to the
    # blocks with one item fewer (a compute item ~ zl units; the grid fills
    # every CU when the items alone do not)
    zl: int = 0
    # an output channel count that is not a multiple of 8 (e.g. 27, 36):
    # the epilogue stores 16-bit elements, each masked by K (the 16-byte
    # stores would be misaligned and cross into the next pixel); weight rows
    # past K read zeros as in any partial K tile
    k_tail: bool = False
    # flattened reduction for C % 64 != 0 (C % 8 == 0): a step is 64
    # consecutive elements of the (tap, channel) reduction, so a C48 3x3
    # layer runs ceil(9 * 48 / 64) = 7 steps instead of 9 zero-padded ones.
    # Each lane's 8-channel granule finds its own tap (granule index / (C / 8));
    # the weights are the KRSC rows read linearly. No tap skipping.
    flat: bool = False
    # a wave owns 128 pixels x bk / 4 channels (8 x-tiles, the group's
    # four waves side by side along K) instead of 64 x bk / 2: the weight
    # fragments are the smaller operand, half as many A reads per step (the
    # reference's dgrad wave shape; with wt every A fragment costs two
    # transposing reads). bx 256, bk 128 / 256, cs 1.
    xt8: bool = False
    # a grouped conv (groups > 1): the group count is the kernarg
    # p_groups, the per-group C / K are p_total_c / p_total_k over it; the K
    # tiles of every group follow each other (item tile K = group * K tiles
    # per group + tile), the input / output pixel strides stay the totals and
    # the channel bases move by group * C / K. cs 1, split_c 1 only.
    grp: bool = False
    # positions-inner sweep: the filter taps of a chunk unrolled (each
    # step's tap, offset-table row and weight tap are constants; no per-step
    # cursor), the chunks of an item a loop of plain chunks (no channel-tail
    # selects) and the last chunk (channel tail and the item crossing); the
    # stage parity is a scalar. Every tap runs (padding taps load zeros).
    sw: bool = False
    name: str = "fwd_flex_mfma"
    wave_size: int = 64

    @property
    def kt(self) -> int:
        if self.xt8:
            return self.bk // 64
        return self.bk // (16 * (4 // (self.bx // 128)))

    @property
    def threads_per_block(self) -> int:
        return 8 * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}x{p.PAD_W}",
            f"s{p.stride}",
            f"d{p.dil_h}x{p.dil_w}",
            f"k{self.bk}",
            f"sc{self.split_c}",
            f"x{self.bx}" if self.bx != 256 else "",
            f"zl{self.zl}" if self.zl else "",
            flags={
                "bf16": p.dtype == "bf16",
                "xcd": self.xcd,
                "xc2": self.xcd == 2,
                "xr0": self.xcd == 3,
                "km": self.kmaj,
                "ct": self.c_tail,
                "cs4": self.cs == 4,
                "cs2": self.cs == 2,
                "wt": self.wt,
                "ds": self.ds,
                "h1": self.h1,
                "zf": self.zf,
                "kx": self.k_tail,
                "fr": self.flat,
                "x8": self.xt8,
                "gr": self.grp,
                "sw": self.sw,
            },
        )

    def items(self) -> int:
        p = self.problem
        if self.ds:
            sh, sw = ds_strides(self)
            nxp = p.N * ceil_div(p.H, sh) * ceil_div(p.W, sw)
            return (
                len(ds_order(self))
                * ceil_div(nxp, self.bx)
                * ceil_div(p.total_c, self.bk)
                * self.split_c
            )
        tiles = (
            ceil_div(p.N * p.Ho * p.Wo, self.bx) * p.groups * ceil_div(p.kpg, self.bk)
        )
        return tiles * self.split_c

    def grid(self) -> Tuple[int, int, int]:
        if self.zl:
            p = self.problem
            sh, sw = ds_strides(self)
            nxp = p.N * ceil_div(p.H, sh) * ceil_div(p.W, sw)
            return (min(self.items() + ceil_div(nxp, ZL_PX), PERSIST), 1, 1)
        return (min(self.items(), PERSIST), 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    @property
    def workspace_bytes(self) -> int:
        p = self.problem
        if self.split_c == 1:
            return 0
        if self.ds:
            return 4 * self.split_c * p.N * p.H * p.W * p.total_c
        return 4 * self.split_c * p.N * p.Ho * p.Wo * p.total_k


def ds_phases(spec: FwdFlexMfmaSpec):
    """Per axis, per output phase a of a ``ds`` dgrad: (taps T, input offset
    pa, filter position of tap 0): dX[s i + a] = sum_t dY[i - pa + t] *
    W[f0 - s t], t < T."""
    p = spec.problem
    out = []
    for n, pad, st in zip((p.KH, p.KW), (p.PAD, p.PAD_W), ds_strides(spec)):
        ax = []
        for a in range(st):
            r0 = (a + pad) % st
            t = (n - r0 + st - 1) // st
            q = (a + pad - r0) // st
            ax.append((t, t - 1 - q, r0 + st * (t - 1)))
        out.append(ax)
    return out


def ds_order(spec: FwdFlexMfmaSpec):
    """The output phases (a, b) a ``ds`` dgrad runs items for, in falling tap
    count (``zf``: the phases with taps only)."""
    sh, sw = ds_strides(spec)
    ph, pw = ds_phases(spec)
    out = sorted(
        ((a, c) for a in range(sh) for c in range(sw)),
        key=lambda ac: -ph[ac[0]][0] * pw[ac[1]][0],
    )
    if spec.zf:
        out = [(a, c) for a, c in out if ph[a][0] * pw[c][0] > 0]
    return out


def ds_zero_deltas(spec: FwdFlexMfmaSpec):
    """``zf``: (row, column) offsets from the pixels of the one phase with
    taps to those of each tapless phase."""
    sh, sw = ds_strides(spec)
    ((a0, b0),) = ds_order(spec)
    return [
        (a - a0, c - b0) for a in range(sh) for c in range(sw) if (a, c) != (a0, b0)
    ]


# zl: phase pixels of a light zero unit (8 per wave)
ZL_PX = 64

# zf: the dX pixel index of the epilogue's LDS table keeps the low ZF_BITS
# bits; bit ZF_BITS + d flags that tapless neighbour d lies inside dX
ZF_BITS = 26


def ds_strides(spec: FwdFlexMfmaSpec) -> Tuple[int, int]:
    """Row and column phases of a ``ds`` dgrad (h1: one row phase)."""
    p = spec.problem
    return (1 if spec.h1 else p.stride), p.stride


def _tap_grid(spec: FwdFlexMfmaSpec) -> Tuple[int, int]:
    """Rows and columns of the taps a step walks (a ``ds`` phase's at most)."""
    p = spec.problem
    if spec.ds:
        ph, pw = ds_phases(spec)
        return max(t for t, _, _ in ph), max(t for t, _, _ in pw)
    return p.KH, p.KW


def _tab_taps(spec: FwdFlexMfmaSpec) -> int:
    """Taps of the DMA offset table, rounded up to the 8 lanes building it."""
    tr, ts = _tap_grid(spec)
    return ceil_div(tr * ts, 8) * 8


def _tab_bytes(spec: FwdFlexMfmaSpec) -> int:
    """The offset table: [tap][wave][pixel of 8][instruction] i32."""
    return _tab_taps(spec) * 8 * 8 * _nxi(spec) * 4


def _tab_db(spec: FwdFlexMfmaSpec, arch: str = "gfx950") -> bool:
    """Two offset-table buffers by item parity (main builder, not ds, when the
    second buffer fits in LDS): group 1 builds the rows of all eight waves, at
    its cursor's crossing into item k only when not built ahead, else after
    its epilogue of item k - 2 (its waves idle there); group 0 builds its
    rows of item 1 in its prologue."""
    if spec.cs != 1 or spec.ds or (spec.wt and spec.c_tail):
        # (wt with c_tail: VGPR spills on bk 256)
        return False
    stage = (spec.bx * _CK + _w_elems(spec)) * 2
    return ArchTarget.from_gfx(arch).fits_lds(2 * stage + 2 * _tab_bytes(spec))


def _wt_half(spec: FwdFlexMfmaSpec) -> int:
    """Outputs of a ``wt`` weight half (bk 192: two halves of 96, the last
    two groups of each 256-byte row unused)."""
    return 96 if spec.bk == 192 else 128


def _w_elems(spec: FwdFlexMfmaSpec) -> int:
    """Weight elements of one stage (``wt``: whole 128-output rows)."""
    if spec.wt:
        return spec.bk // _wt_half(spec) * _CK * 128
    return spec.bk * _CK


def _nxi(spec: FwdFlexMfmaSpec) -> int:
    """Input DMA instructions per wave per step (8 pixel rows each)."""
    return 4 if spec.cs == 4 else spec.bx // 64


def fwd_flex_mfma_reason(spec: FwdFlexMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    if spec.ds:
        if spec.cs == 4:
            return "ds: no cs 4 (its kernel has no dgrad phases)"
        if p.stride < 2 or p.is_dilated or p.groups != 1 or p.pad_w < -1:
            return "ds: a dense strided dgrad without dilation"
        if spec.h1 and (p.H != 1 or p.KH != 1 or p.PAD != 0):
            return "h1: one dX row, a 1 x S filter and no row padding"
        if spec.zf:
            if spec.split_c != 1 or spec.h1 or not spec.wt:
                return "zf: split_c 1 and wt only, no h1"
            if len(ds_order(spec)) != 1 or len(ds_zero_deltas(spec)) + ZF_BITS > 31:
                return "zf: one output phase with taps, the others none"
            if spec.zl and (spec.zl < 0 or spec.cs != 1 or spec.xcd or spec.k_tail):
                return "zl: zf with cs 1, no xcd / k_tail"
            if p.N * p.H * p.W >= 1 << ZF_BITS:
                return f"zf: dX pixels exceed 2^{ZF_BITS}"
            if any(pa < -64 for ax in ds_phases(spec) for t, pa, _ in ax if t > 0):
                return "ds: an input offset out of range"
        elif any(t < 1 or pa < -64 for ax in ds_phases(spec) for t, pa, _ in ax):
            return "ds: an output phase without taps"
        if p.Ho < 1 or p.Wo < 1:
            return "empty dY"
        if 2 * p.N * p.H * p.W * p.total_c > I32_MAX:
            return f"dX exceeds {I32_MAX} B, the i32 buffer range"
        if 2 * p.total_k * p.KH * p.KW * p.total_c > I32_MAX:
            return f"weights exceed {I32_MAX} B, the i32 buffer range"
        if spec.workspace_bytes > I32_MAX:
            return f"workspace exceeds {I32_MAX} B, the i32 byte-count kernarg"
        tr, ts = _tap_grid(spec)
        if tr * ts > 32:
            return "ds: more than 32 taps in a phase"
        lds = (
            2 * (spec.bx * _CK + _w_elems(spec)) * 2
            + _tab_bytes(spec)
            + 3 * spec.bx * 4
        )
        if not ArchTarget.from_gfx(arch).fits_lds(lds):
            return f"LDS {lds} B too large"
        # the forward over dY the kernel runs (taps checked above)
        from dataclasses import replace as _rp

        p = DirectConvProblem(
            N=p.N,
            H=p.Ho,
            W=p.Wo,
            groups=1,
            cpg=p.kpg,
            kpg=p.cpg,
            KH=1,
            KW=1,
            PAD=0,
            stride=1,
            dtype=p.dtype,
        )
        spec = _rp(spec, problem=p, ds=False)
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "fwd_flex_mfma is a wave64 MFMA kernel"
    if target.async_lds_max_dwords < 4:
        return f"fwd_flex_mfma needs the 16-byte LDS DMA (absent on {arch})"
    if spec.bx not in (128, 256) or (spec.bx == 128 and spec.bk == 64):
        return "bx must be 256, or 128 with bk 128 / 256"
    if spec.cs not in (1, 2, 4):
        return "cs must be 1, 2 or 4"
    if spec.cs == 2:
        if spec.bx != 128 or spec.bk != 256 or spec.c_tail or spec.ds or spec.flat:
            return "cs 2: bx 128 and bk 256 only, no c_tail, ds or flat"
        if (p.KH * p.KW) % 3:
            return (
                "cs 2: the tap count must be a multiple of 3 (a tap's stage is fixed)"
            )
        if p.total_c % _CK:
            return f"cs 2: C must be a multiple of {_CK}"
    if spec.cs == 4:
        if spec.bx != 128 or spec.bk != 128 or spec.c_tail:
            return "cs 4: bx 128 and bk 128 only, no c_tail"
        if p.total_c % _CK or p.total_c < 2 * _CK * spec.split_c:
            return f"cs 4: C must be a multiple of {_CK} and >= {2 * _CK * spec.split_c} (two chunks per piece)"
    if spec.wt and (spec.bk not in (128, 192, 256) or (spec.c_tail and spec.cs == 4)):
        return "wt: bk 128 / 192 / 256 only, no c_tail with cs 4"
    stage = (spec.bx * _CK + _w_elems(spec)) * 2 * (2 if spec.cs == 4 else 1)
    lds = (3 if spec.cs == 2 else 2) * stage + _tab_bytes(spec)
    if not target.fits_lds(lds):
        return f"LDS {lds} B > {target.lds_capacity_bytes}"
    if p.dtype not in ("fp16", "bf16"):
        return "fp16/bf16 only"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing mfma_f32_16x16x32_{ab} on {arch}"
    why = problem_dims_reason(p)
    if why is not None:
        return why
    if spec.xcd not in (False, True, 2, 3):
        return "xcd: off, on, 2 (contiguous XCD runs) or 3 (no rotation)"
    if spec.xcd in (2, 3) and (not spec.sw or spec.ds or spec.zl or spec.cs != 1):
        return "xcd 2 / 3: sw, cs 1, no ds / zl"
    if spec.kmaj and (not spec.xcd or spec.ds or spec.grp):
        return "kmaj: needs xcd, no ds, no grp"
    if spec.sw and (spec.cs != 1 or spec.ds or spec.flat or p.KH * p.KW < 3):
        return "sw: cs 1 only, no ds or flat, at least 3 taps"
    if spec.grp:
        if p.groups < 2:
            return "grp: grouped problems only"
        if (
            spec.ds
            or spec.flat
            or spec.k_tail
            or spec.xt8
            or spec.cs != 1
            or spec.split_c != 1
        ):
            return "grp: cs 1 and split_c 1, no ds, flat, k_tail or xt8"
    elif p.groups != 1:
        return "groups must be 1"
    # the reduction / output channels of one group (all of them when dense)
    CC, KK = (p.cpg, p.kpg) if spec.grp else (p.total_c, p.total_k)
    if p.stride < 1 or p.dil_h < 1 or p.dil_w < 1:
        return (
            f"stride and dilation must be >= 1 "
            f"(got s{p.stride} d{p.dil_h}x{p.dil_w})"
        )
    why = pad_w_reason(p)
    if why is not None:
        return why
    if spec.bk not in (64, 128, 192, 256):
        return "bk must be 64, 128, 192 or 256"
    if spec.bk == 192 and spec.bx != 256:
        return "bk 192 needs bx 256"
    if spec.xt8 and (spec.bx != 256 or spec.bk not in (128, 256) or spec.cs != 1):
        return "xt8: bx 256, bk 128 / 256, cs 1 only"
    if spec.flat:
        if spec.c_tail or spec.wt or spec.ds or spec.cs != 1:
            return "flat: no c_tail, wt, ds or cs"
        if p.total_c % 8 or p.total_c % _CK == 0:
            return f"flat: C {p.total_c} must be a multiple of 8 and not of {_CK}"
        if spec.split_c > ceil_div(p.KH * p.KW * p.total_c, _CK):
            return "flat: split_c exceeds the reduction chunks"
    elif spec.c_tail:
        if CC % 8 or CC % _CK == 0:
            return f"c_tail: C {CC} must be a multiple of 8 and not of {_CK}"
    elif CC % _CK:
        return f"C {CC} must be a multiple of {_CK}"
    if spec.k_tail:
        if KK % 8 == 0:
            return f"k_tail: K {KK} must not be a multiple of 8"
        if spec.cs != 1 or spec.wt or spec.ds:
            return "k_tail: no cs, wt or ds"
    elif KK % 8:
        return f"K {KK} must be a multiple of 8 (16-byte output writes)"
    if spec.split_c < 1 or (not spec.flat and spec.split_c > ceil_div(CC, _CK)):
        return f"split_c {spec.split_c} must be in [1, C / {_CK}]"
    if p.KH * p.KW > 32:
        return "more than 32 taps (the DMA offset table's LDS)"
    if p.Ho < 1 or p.Wo < 1:
        return "empty output"
    # every buffer offset is an i32 over the whole tensor (or workspace slab)
    nx = p.N * p.Ho * p.Wo
    for what, nbytes in (
        ("input", 2 * p.N * p.H * p.W * p.total_c),
        ("output", 2 * nx * p.total_k),
        ("weights", 2 * p.total_k * p.KH * p.KW * p.total_c),
        ("workspace slab", 4 * nx * p.total_k if spec.split_c > 1 else 0),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    # the whole workspace's size is an i32 kernarg of the conv and the fold
    if spec.workspace_bytes > I32_MAX:
        return f"workspace exceeds {I32_MAX} B, the i32 byte-count kernarg"
    return None


def build_fwd_flex_mfma(spec: FwdFlexMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_flex_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdFlexMfmaSpec for {arch}: {why}")
    if spec.cs == 4:
        return _build_cs4(spec, arch)
    if spec.cs == 2:
        return _build_cs2(spec, arch)
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    BK, KT, SP = spec.bk, spec.kt, spec.split_c
    BX = spec.bx
    XT = 8 if spec.xt8 else _XT  # 16-pixel MFMA tiles per wave
    WKN = BK // (KT * 16)  # waves of a group along K
    WXN = 4 // WKN  # ... and along X
    assert WKN * WXN == 4 and WXN * XT * 16 * 2 == BX
    NXI, NWI = BX // 64, BK // 64  # DMA instructions per wave per step
    R, S = p.KH, p.KW
    SH, PH, PW, DH, DW = p.stride, p.PAD, p.PAD_W, p.dil_h, p.dil_w
    IN_E, W_E = BX * _CK, _w_elems(spec)  # elements per stage
    WT = spec.wt
    SW = spec.sw  # positions-inner sweep (see FwdFlexMfmaSpec.sw)
    NT = R * S  # sw: the taps of a chunk
    if WT:
        # wt: NH halves of HK outputs, each [64 reduction rows][128]; a wave
        # DMAs pieces q = wave * NWI + i of 4 rows (1 KiB), half q // 16
        HK = _wt_half(spec)
        NH = BK // HK
        NWI = 2 * NH

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    DS = spec.ds
    params = emit_direct_params(b, direction="dgrad" if DS else "fwd", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    def dv(a: Value, n: int) -> Value:
        return b.div(a, ci(n))

    def md(a: Value, n: int) -> Value:
        return b.mod(a, ci(n))

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_N, p_Hi, p_Wi = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]
    TR, TS = _tap_grid(spec)
    if DS:
        # the forward over dY: input dY [N, Ho, Wo, K], output dX [N, H, W, C]
        p_Hi, p_Wi, p_Ho, p_Wo = p_Ho, p_Wo, p_Hi, p_Wi
        C, K = K, C
        SDH, SDW = ds_strides(spec)
        PHS_H, PHS_W = ds_phases(spec)
        PORD = ds_order(spec)
    # input / output pixel strides (the totals); grp: C, K become one group's
    XS, YS = C, K
    GRP = spec.grp
    if GRP:
        assert not DS and SP == 1 and spec.cs == 1 and not spec.flat and not spec.k_tail
        NG = params["p_groups"]
        C, K = b.div(C, NG), b.div(K, NG)

    tid = b.thread_id_x()
    lane = md(tid, _WAVE)
    wave = b.readfirstlane(dv(tid, _WAVE))
    g = dv(wave, 4)
    wq = md(wave, 4)
    wk = md(wq, WKN)
    wx = dv(wq, WKN)

    def lane_bases():
        """Lane-derived LDS bases, formed once (the fragment reads add
        constants): input granule, input / weight fragment rows, table row."""
        h_r16, h_q = md(lane, 16), dv(lane, 16)
        h_l8 = dv(lane, 8)
        h_gran = b.mul(b.xor(md(lane, 8), h_l8), ci(16))
        h_xin, h_kw = [], []
        for h in range(2):
            sl = b.mul(b.xor(b.add(ci(h * 4), h_q), md(h_r16, 8)), ci(8))
            xrow = b.add(b.add(b.mul(g, ci(BX // 2)), b.mul(wx, ci(XT * 16))), h_r16)
            krow = b.add(b.mul(wk, ci(KT * 16)), h_r16)
            h_xin.append(b.add(b.mul(xrow, ci(_CK)), sl))
            h_kw.append(b.add(b.mul(krow, ci(_CK)), sl))
        h_tab = b.mul(b.add(b.mul(wave, ci(8)), h_l8), ci(NXI))
        return h_gran, h_xin, h_kw, h_tab

    def opaque_lane() -> Value:
        # lane-derived addressing is recomputed where it is used instead of
        # being hoisted into long-lived VGPRs
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    if DS:
        # phase-major pixels: phase (a, b) holds dX[n, SDH i + a, SDW j + b]
        # for i < Hq, j < Wq, padded to TPP whole tiles
        Hq = b.div(b.add(p_Ho, ci(SDH - 1)), ci(SDH))
        Wq = b.div(b.add(p_Wo, ci(SDW - 1)), ci(SDW))
        hwq = b.mul(Hq, Wq)
        NXP = b.mul(p_N, hwq)
        TPP = b.div(b.add(NXP, ci(BX - 1)), ci(BX))
        nxt = b.mul(TPP, ci(len(PORD)))
        NX = b.mul(p_N, b.mul(p_Ho, p_Wo))  # dX pixels (split-C slab rows)
    else:
        hw = b.mul(p_Ho, p_Wo)
        NX = b.mul(p_N, hw)
        nxt = b.div(b.add(NX, ci(BX - 1)), ci(BX))

    def phase_sel(idx: Value, vals) -> Value:
        out = ci(vals[0])
        for i, v in enumerate(vals[1:], 1):
            out = b.select(b.cmp_eq(idx, ci(i)), ci(v), out)
        return out

    def ds_pix(tx: Value, xi: Value):
        """(valid, n, i, j, phase row a, phase column b) of the tile's pixel xi
        (offset from the start of tile tx)."""
        ph = b.div(tx, TPP)
        xl = b.add(b.mul(b.sub(tx, b.mul(ph, TPP)), ci(BX)), xi)
        n = b.div(xl, hwq)
        rem = b.sub(xl, b.mul(n, hwq))
        i = b.div(rem, Wq)
        j = b.sub(rem, b.mul(i, Wq))
        return b.cmp_lt(xl, NXP), n, i, j, *phase_ab(ph)

    def phase_ab(ph: Value):
        """Phase row a and column b of the ph-th phase in PORD."""
        return phase_sel(ph, [a for a, _ in PORD]), phase_sel(ph, [c for _, c in PORD])

    nkt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    if GRP:
        # grp: K tile = group * (K tiles per group) + tile
        nktg = nkt
        nkt = b.mul(nktg, NG)
    TI = b.mul(b.mul(nxt, nkt), ci(SP))
    G = b.smin(TI, ci(PERSIST))
    # ds forms its pixel indices in ds_pix (no hw)
    dl = ([] if DS else [hw]) + [p_Wo, nkt, G]
    if spec.flat:
        dl.append(b.div(C, ci(8)))
    if GRP:
        dl.append(nktg)
    fd = _divs(b, lane, dl)
    f_hw = None if DS else fd.pop(0)
    f_wo, f_nkt = fd.pop(0), fd.pop(0)
    f_g = fd.pop(0)
    f_cg0 = fd.pop(0) if spec.flat else None
    f_nktg = fd.pop(0) if GRP else None

    def grp_tk(tk: Value):
        """grp: (group, its K tile) of the item's K tile."""
        gi = f_nktg.div(tk)
        return gi, b.sub(tk, b.mul(gi, nktg))

    FLAT = spec.flat
    if FLAT:
        CG = b.div(C, ci(8))  # granules per tap
        RSG = b.mul(ci(R * S), CG)  # granules of a reduction row
        f_cg = f_cg0
        nch = b.div(b.add(RSG, ci(7)), ci(8))
    elif spec.c_tail:
        nch = b.div(b.add(C, ci(_CK - 1)), ci(_CK))
    else:
        nch = b.div(C, ci(_CK))
    RSC = b.mul(ci(R * S), C)
    RSK = b.mul(ci(R * S), K)  # wt: a reduction row of the KRSC filter
    blk = b.block_id_x()
    XR = spec.xcd == 2
    # xcd 3: the xcd walk without the per-round rotation
    rot = 0 if spec.xcd == 3 else ROT
    if XR:
        # xcd 2: XCD x (block % 8) owns the contiguous run of tiles
        # [x q + min(x, r), + q + (x < r)) of TI = 8 q + r and its block w
        # (block / 8) takes run[w + i G / 8] in round i, so a round's tiles
        # of one XCD neighbour its previous round's and every XCD gets the
        # same share of the last round (G % 8 != 0: one round, dealt order)
        is8 = b.cmp_eq(b.mod(G, ci(8)), c0)
        x8 = b.select(is8, b.mod(blk, ci(8)), c0)
        w8 = b.select(is8, b.div(blk, ci(8)), blk)
        wpx = b.select(is8, b.div(G, ci(8)), G)
        q8 = b.select(is8, b.div(TI, ci(8)), TI)
        r8 = b.sub(TI, b.mul(q8, b.select(is8, ci(8), c1)))
        run0 = b.add(b.add(b.mul(x8, q8), b.smin(x8, r8)), w8)
        rlen = b.add(q8, b.select(b.cmp_lt(x8, r8), c1, c0))
        rounds = b.div(b.add(b.sub(rlen, w8), b.sub(wpx, c1)), wpx)
    elif spec.xcd:
        # 8 XCDs; only when G is a multiple of 8 (else keep the dealt order)
        g8 = b.div(G, ci(8))
        xr = b.add(b.mul(b.mod(blk, ci(8)), g8), b.div(blk, ci(8)))
        blk = b.select(b.cmp_eq(b.mod(G, ci(8)), c0), xr, blk)
    # the rounds from the grid size's magic (no signed division)
    full = f_g.div(TI)
    part = b.sub(TI, b.mul(full, G))
    if DS:
        # the items run in falling tap count (phase order PORD); a block takes
        # slot b in even rounds and G - 1 - b in odd ones, so the long tiles of
        # one round pair with the short ones of the next
        def slot(i: Value) -> Value:
            return b.select(b.cmp_eq(b.mod(i, c2), c1), b.sub(b.sub(G, c1), blk), blk)

        last_ok = b.cmp_lt(slot(full), part)
    elif not XR:
        last_ok = b.cmp_lt(f_g.mod(b.add(blk, b.mul(full, ci(rot)))), part)
    if not XR:
        rounds = b.add(full, b.select(last_ok, c1, c0))

    def tile_of(i: Value) -> Value:
        if DS:
            return b.add(b.mul(i, G), slot(i))
        if XR:
            return b.add(run0, b.mul(i, wpx))
        return b.add(b.mul(i, G), f_g.mod(b.add(blk, b.mul(i, ci(rot)))))

    # ---- item -> (tile, piece, valid tap ranges) -----------------------------
    def item_info(t: Value) -> Dict[str, Value]:
        tl = b.div(t, ci(SP))
        pc = b.mod(t, ci(SP))
        if spec.kmaj:
            tk = b.div(tl, nxt)
            tx = b.sub(tl, b.mul(tk, nxt))
        else:
            tx = f_nkt.div(tl)
            tk = b.sub(tl, b.mul(tx, nkt))
        if DS:
            ph = b.div(tx, TPP)
            pa, pb = phase_ab(ph)
            fb = b.add(
                b.mul(phase_sel(pa, [f for _, _, f in PHS_H]), ci(S)),
                phase_sel(pb, [f for _, _, f in PHS_W]),
            )
            ch0 = b.div(b.mul(pc, nch), ci(SP))
            ch1 = b.div(b.mul(b.add(pc, c1), nch), ci(SP))
            return dict(
                pc=pc,
                tx=tx,
                tk=tk,
                r0=c0,
                s0=c0,
                ch0=ch0,
                ch1=ch1,
                fb=fb,
                r1=phase_sel(pa, [t for t, _, _ in PHS_H]),
                s1=phase_sel(pb, [t for t, _, _ in PHS_W]),
            )
        if SW:
            # every tap runs (padding taps load zeros)
            out = dict(
                pc=pc,
                tx=tx,
                tk=tk,
                r0=c0,
                r1=ci(R),
                s0=c0,
                s1=ci(S),
                ch0=b.div(b.mul(pc, nch), ci(SP)),
                ch1=b.div(b.mul(b.add(pc, c1), nch), ci(SP)),
            )
            if GRP:
                out["gi"], out["tk"] = grp_tk(tk)
            return out
        x0 = b.mul(tx, ci(BX))
        x1 = b.sub(b.smin(b.add(x0, ci(BX)), NX), c1)
        n0, n1 = f_hw.div(x0), f_hw.div(x1)
        same = b.cmp_eq(n0, n1)
        q0 = b.sub(x0, b.mul(n0, hw))
        q1 = b.sub(x1, b.mul(n0, hw))
        ho0 = b.select(same, f_wo.div(q0), c0)
        ho1 = b.select(same, f_wo.div(q1), b.sub(p_Ho, c1))
        row = b.land(same, b.cmp_eq(ho0, ho1))
        wo0 = b.select(row, b.sub(q0, b.mul(ho0, p_Wo)), c0)
        wo1 = b.select(row, b.sub(q1, b.mul(ho1, p_Wo)), b.sub(p_Wo, c1))
        # valid taps are an interval: a prefix of the rows reads only the top
        # padding (lo < 0 at the tile's last row) and a suffix only the bottom
        lo_h = b.sub(b.mul(ho1, ci(SH)), ci(PH))
        hi_h = b.sub(b.mul(ho0, ci(SH)), ci(PH))
        lo_w = b.sub(b.mul(wo1, ci(SH)), ci(PW))
        hi_w = b.sub(b.mul(wo0, ci(SH)), ci(PW))

        # the counts in closed form: #{r < n: lo + r d < 0} =
        # min(ceil(max(-lo, 0) / d), n), #{r < n: hi + r d < E} =
        # min(ceil(max(E - hi, 0) / d), n)
        def cnt(v: Value, d: int, n: int) -> Value:
            v = b.smax(v, c0)
            if d > 1:
                v = b.div(b.add(v, ci(d - 1)), ci(d))
            return b.smin(v, ci(n))

        r0, r1 = cnt(b.sub(c0, lo_h), DH, R), cnt(b.sub(p_Hi, hi_h), DH, R)
        s0, s1 = cnt(b.sub(c0, lo_w), DW, S), cnt(b.sub(p_Wi, hi_w), DW, S)
        if FLAT:  # one cursor tap; the chunks walk the whole reduction
            r0, r1, s0, s1 = c0, c1, c0, c1
        # a tile reading only padding still runs one (zero) step
        empty = b.lor(b.cmp_ge(r0, r1), b.cmp_ge(s0, s1))
        r0, r1 = b.select(empty, c0, r0), b.select(empty, c1, r1)
        s0, s1 = b.select(empty, c0, s0), b.select(empty, c1, s1)
        ch0 = b.div(b.mul(pc, nch), ci(SP))
        ch1 = b.div(b.mul(b.add(pc, c1), nch), ci(SP))
        out = dict(pc=pc, tx=tx, tk=tk, r0=r0, r1=r1, s0=s0, s1=s1, ch0=ch0, ch1=ch1)
        if GRP:
            out["gi"], out["tk"] = grp_tk(tk)
        return out

    # ---- DMA cursor ------------------------------------------------------------
    SC = (
        (
            ["i", "t", "ch", "ch1"]
            if SW
            else ["i", "t", "ch", "ch1", "r", "s", "r0", "r1", "s0", "s1"]
        )
        + (["fb"] if DS else [])
        + (["wk0"] if spec.wt else [])
    )
    LN = ["wo", "wo1"] if WT and NWI >= 4 else ["wo"]
    # Input DMA offsets: a per-wave LDS table [tap][wave][pixel of 8][instr]
    # built at each tile change (byte offset of the tap's input fibre, or the
    # zero-fill offset): the cursor keeps no
    # per-pixel registers and a DMA takes one table read plus the lane's
    # granule. A wave reads only its own rows, so no barrier guards it.
    RSP = _tab_taps(spec)
    # wt: the weight stages first, at LDS offsets below 64 KiB, so their bases
    # fold into the transposing reads' offsets
    if WT and SW:
        # sw: both stages in one allocation (the stage is a scalar offset)
        w_s_wt = [b.smem_alloc(I8, [1, 2 * W_E * 2], name_hint="flex_w")] * 2
    else:
        w_s_wt = (
            [b.smem_alloc(I8, [1, W_E * 2], name_hint=f"flex_w{i}") for i in range(2)]
            if WT
            else None
        )
    TB2 = _tab_db(spec, arch)
    TABN = RSP * 64 * NXI  # elements of one table buffer
    tab_s = b.smem_alloc(I32, [1, (2 if TB2 else 1) * TABN], name_hint="flex_tab")
    # TB2: the group of the code being built and its round (set by run)
    tctx = {"g1": False, "rnd": None}
    if DS:
        # ds: the dX pixel (or -1) of each tile pixel, formed with the offset
        # table and read by the epilogue; 3 buffers by item (a buffer is
        # rewritten 3 items later, past every wave's epilogue of its item)
        opix_s = b.smem_alloc(I32, [1, 3 * BX], name_hint="flex_opix")

    def tab_index(tap: Value, l8: Value, wv: Value, buf) -> Value:
        e = b.mul(b.add(b.mul(b.add(b.mul(tap, ci(8)), wv), ci(8)), l8), ci(NXI))
        return e if buf is None else b.add(e, b.mul(buf, ci(TABN)))

    def build_tab(info, ln: Value, l8: Value, wv=None, buf=None) -> None:
        """Wave wv's (default: this wave's) table rows for the tile, into
        buffer buf (TB2): lane (l8, c) writes taps c, c + 8, ... of its NXI
        pixels (byte offset of the tap's input fibre, or the zero-fill
        offset)."""
        wv = wave if wv is None else wv
        c = md(ln, 8)
        pix = []
        for i in range(NXI):
            x = b.add(
                b.mul(info["tx"], ci(BX)),
                b.add(b.mul(b.add(b.mul(wv, ci(NXI)), ci(i)), ci(8)), l8),
            )
            if DS:
                xi = b.add(b.mul(b.add(b.mul(wave, ci(NXI)), ci(i)), ci(8)), l8)
                xok, n, qi, qj, pa, pb = ds_pix(info["tx"], xi)
                hb = b.sub(qi, phase_sel(pa, [v for _, v, _ in PHS_H]))
                wb = b.sub(qj, phase_sel(pb, [v for _, v, _ in PHS_W]))
                base = b.add(b.mul(b.add(b.mul(n, p_Hi), hb), p_Wi), wb)
                pix.append((xok, hb, wb, base))
                oh = b.add(b.mul(qi, ci(SDH)), pa)
                ow = b.add(b.mul(qj, ci(SDW)), pb)
                ook = b.land(xok, b.land(b.cmp_lt(oh, p_Ho), b.cmp_lt(ow, p_Wo)))
                op = b.add(b.mul(b.add(b.mul(n, p_Ho), oh), p_Wo), ow)
                if spec.zf:
                    for d, (da, db) in enumerate(ds_zero_deltas(spec)):
                        zok = b.land(
                            b.cmp_lt(b.add(oh, ci(da)), p_Ho),
                            b.cmp_lt(b.add(ow, ci(db)), p_Wo),
                        )
                        op = b.add(op, b.select(zok, ci(1 << (ZF_BITS + d)), c0))
                b.smem_store_vN(
                    opix_s,
                    [c0, b.add(b.mul(md(info["rnd"], 3), ci(BX)), xi)],
                    b.select(ook, op, ci(-1)),
                    1,
                )
                continue
            n = f_hw.div(x)
            rem = b.sub(x, b.mul(n, hw))
            ho = f_wo.div(rem)
            wo = b.sub(rem, b.mul(ho, p_Wo))
            hb = b.sub(b.mul(ho, ci(SH)), ci(PH))
            wb = b.sub(b.mul(wo, ci(SH)), ci(PW))
            base = b.add(b.mul(b.add(b.mul(n, p_Hi), hb), p_Wi), wb)
            pix.append((b.cmp_lt(x, NX), hb, wb, base))
        gc = b.mul(info["gi"], C) if GRP else None
        for kk in range(RSP // 8):
            t = b.add(c, ci(8 * kk))
            r = b.div(t, ci(TS))
            s_ = b.sub(t, b.mul(r, ci(TS)))
            dh, dw = b.mul(r, ci(DH)), b.mul(s_, ci(DW))
            tok = b.cmp_lt(t, ci(TR * TS))
            toff = b.add(b.mul(dh, p_Wi), dw)
            ents = []
            for xok, hb, wb, base in pix:
                ok = b.land(
                    b.land(tok, xok),
                    b.land(
                        b.cmp_ult(b.add(hb, dh), p_Hi), b.cmp_ult(b.add(wb, dw), p_Wi)
                    ),
                )
                fo = b.mul(b.add(base, toff), XS)
                if GRP:
                    fo = b.add(fo, gc)
                ents.append(b.select(ok, b.mul(fo, c2), oob))
            val = ents[0] if NXI == 1 else b.vec_pack(ents, I32)
            b.smem_store_vN(tab_s, [c0, tab_index(t, l8, wv, buf)], val, NXI)

    def build_tab_all(info, k: Value) -> None:
        """TB2, group 1: the rows of this wave and of its group-0 partner
        (wave - 4) for item k, into buffer k & 1."""
        ln = opaque_lane()
        l8 = dv(ln, 8)
        buf = b.land(k, c1)
        build_tab(info, ln, l8, wave, buf)
        ln = opaque_lane()
        build_tab(info, ln, dv(ln, 8), b.sub(wave, ci(4)), buf)

    def lanes_of(info, first: bool = False) -> Dict[str, Value]:
        """The cursor's lane parts (and the item's offset-table rows)."""
        ln = opaque_lane()
        l8 = dv(ln, 8)
        c8 = b.xor(md(ln, 8), l8)
        out = {}
        if first or DS:
            build_tab(info, ln, l8, None, c0 if TB2 else None)
        elif TB2:
            if tctx["g1"]:
                # built here unless built after the epilogue of item k - 2
                k, rnd_ = info["rnd"], tctx["rnd"]
                # item 1: group 1's own rows (group 0 built its rows in its
                # prologue)
                with b.scf_if(b.land(b.cmp_eq(k, c1), b.cmp_lt(k, rounds))):
                    build_tab(info, ln, l8, None, c1)
                need = b.land(b.cmp_lt(k, rounds), b.cmp_ge(k, b.add(rnd_, c2)))
                with b.scf_if(need):
                    build_tab_all(info, k)
        else:
            # no table for the item past the block's last (the cursor enters
            # it when the last item's steps run out)
            with b.scf_if(b.cmp_lt(info["rnd"], rounds)):
                build_tab(info, ln, l8)
        # weight rows: wave w DMAs LDS rows (w * NWI + i) * 8 + l / 8. Row
        # 4g + j of a 16-row tile holds channel (g & 1) * 8 + (g >> 1) * 4 + j,
        # so the permlane32_swap pairs of the epilogue (lane rows 0/2 and 1/3)
        # hold 8 consecutive channels. Row + 8 is channel + 4 and row + 16
        # channel + 16, so the lane part is one offset and instruction i adds
        # a scalar.
        # Tail rows (k >= K) fall past B_bytes: zero fill, never stored.
        m = l8
        gq = dv(m, 4)
        chan = b.add(b.mul(gq, ci(8)), md(m, 4))
        if NWI == 1:
            # bk 64: one weight DMA per wave; row group w holds channels
            # (w // 2) * 16 + (w % 2) * 4 + chan (the interleave the wider
            # tiles reach through their per-instruction channel offsets)
            wbase = b.add(b.mul(dv(wave, 2), ci(16)), b.mul(md(wave, 2), ci(4)))
            k = b.add(b.add(b.mul(info["tk"], ci(BK)), wbase), chan)
        elif spec.wt:
            k = None
        elif NWI == 3:
            # bk 192: a wave's 3 DMAs straddle the 16-row tiles, so the
            # instruction's channel offset carries the wave too
            k = b.add(b.mul(info["tk"], ci(BK)), chan)
        else:
            k = b.add(b.add(b.mul(info["tk"], ci(BK)), b.mul(wave, ci(NWI * 8))), chan)
        if GRP and k is not None:
            # the group's filter rows (tail rows read the next group's: never stored)
            k = b.add(k, b.mul(info["gi"], K))
        if WT:
            # LDS slot l % 16 of row 4 q + l // 16 (piece q) takes the global
            # granule of its group: slot group (l % 16) // 2 ^ swz(row), swz =
            # (row & 3) | 4 ((row >> 3) & 1) with row & 3 = l // 16 and
            # (row >> 3) & 1 = (q >> 1) & 1 (qb). The lane part (row in the
            # piece, column) is formed per tile, the rest is the soffset; the
            # columns past K read other rows' weights, of outputs never stored.
            w16, s16 = dv(ln, 16), md(ln, 16)
            cb = b.add(b.mul(b.xor(dv(s16, 2), w16), ci(16)), b.mul(md(s16, 2), ci(8)))

            def wlane(qb: Value) -> Value:
                col = b.xor(cb, b.mul(qb, ci(64)))
                v = b.mul(b.add(b.mul(w16, RSK), col), c2)
                return v if HK == 128 else b.select(b.cmp_lt(col, ci(HK)), v, oob)

            if NWI >= 4:
                out["wo"], out["wo1"] = wlane(c0), wlane(c1)
            else:  # q = 2 wave + i: qb = wave & 1
                out["wo"] = wlane(md(wave, 2))
            # the scalar part of the tile: its first output, the wave's half
            # and rows (formed in VALU from the item's divisions: made scalar)
            q0 = b.mul(wave, ci(NWI))
            wk = b.add(
                b.add(b.mul(info["tk"], ci(BK)), b.mul(dv(q0, 16), ci(HK))),
                b.mul(b.mul(md(q0, 16), ci(4)), RSK),
            )
            if GRP:
                # the group's reduction rows of the KRSC filter
                wk = b.add(wk, b.mul(b.mul(info["gi"], C), RSK))
            out["wk0"] = b.readfirstlane(b.mul(wk, c2))
        else:
            out["wo"] = b.mul(b.add(b.mul(k, RSC), b.mul(c8, ci(8))), c2)
        return out

    def cursor_at(i: Value, first: bool = False) -> Dict[str, Value]:
        t = tile_of(i)
        info = item_info(t)
        info["rnd"] = i
        cur = dict(
            i=i, t=t, ch=info["ch0"], ch1=info["ch1"], r=info["r0"], s=info["s0"]
        )
        cur.update({k: info[k] for k in ("r0", "r1", "s0", "s1")})
        if DS:
            cur["fb"] = info["fb"]
        cur.update(lanes_of(info, first))
        return cur

    def cur_adv(cur) -> Dict[str, Value]:
        s = b.add(cur["s"], c1)
        ws = b.cmp_ge(s, cur["s1"])
        s = b.select(ws, cur["s0"], s)
        r = b.add(cur["r"], b.select(ws, c1, c0))
        wr = b.cmp_ge(r, cur["r1"])
        r = b.select(wr, cur["r0"], r)
        ch = b.add(cur["ch"], b.select(wr, c1, c0))
        wc = b.cmp_ge(ch, cur["ch1"])
        keep = dict(cur, s=s, r=r, ch=ch)
        names = SC + LN
        branch = b.scf_if_else(wc, [I32] * len(names))
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                nxt = cursor_at(b.add(cur["i"], c1))
                b.scf_yield(*[nxt[k] for k in names])
            with else_ctx:
                b.scf_yield(*[keep[k] for k in names])
        return branch.results

    # ---- LDS --------------------------------------------------------------------
    # one allocation per stage and operand; the DMAs are inline asm
    # (_dma_lds), so the explicit vmcnt waits of the schedule order them
    # against the fragment reads
    if SW:
        # one allocation per operand, stage st at st * (stage size)
        in_s = [b.smem_alloc(io_type, [1, 2 * IN_E], name_hint="flex_in")] * 2
        w_s = w_s_wt or [b.smem_alloc(io_type, [1, 2 * W_E], name_hint="flex_w")] * 2
    else:
        in_s = [
            b.smem_alloc(io_type, [1, IN_E], name_hint=f"flex_in{i}") for i in range(2)
        ]
        w_s = w_s_wt or [
            b.smem_alloc(io_type, [1, W_E], name_hint=f"flex_w{i}") for i in range(2)
        ]
    in_a = [b.smem_addr_of(v) for v in in_s]
    w_a = [b.smem_addr_of(v) for v in w_s]

    def sv(st) -> Value:
        """sw: the stage as a value (a static stage or a scalar parity)."""
        return ci(st) if isinstance(st, int) else st

    def s_other(st):
        return 1 - st if isinstance(st, int) else b.xor(st, c1)

    # the lane-derived LDS bases, formed per group in run (from an opaque
    # lane, not formed above the group branch and held across it)
    h_gran, h_xin, h_kw, h_tab = (None,) * 4

    def dma(cur, st: int):
        """This wave's NXI input + NWI weight DMAs of the cursor's step into stage st."""
        dma_issue(dma_prep(cur), st)

    def dma_issue(ops, st):
        for kind, i, rsrc, voff, soff in ops:
            base, n = (in_a, NXI) if kind == "x" else (w_a, NWI)
            wv = b.add(wave, ci(4)) if kind == "w4" else wave
            if SW:
                # the wave's first slot of the stage as one SGPR, the
                # instruction's slot the m0 add's immediate
                sb = IN_E * 2 if kind == "x" else W_E * 2
                off = b.add(b.mul(wv, ci(n * 1024)), b.mul(sv(st), ci(sb)))
                sbase = b.readfirstlane(
                    b.trunc(b.smem_ptr_add(base[0], b.zext(off, I64)), I32)
                )
                _dma_lds_at(b, rsrc, sbase, i * 1024, voff, soff)
                continue
            off = b.mul(b.add(b.mul(wv, ci(n)), ci(i)), ci(1024))
            dst = b.smem_ptr_add(base[st], b.zext(off, I64))
            _dma_lds(b, rsrc, dst, voff, soff)

    def dma_prep(cur, parts: str = "xw", stap=None, ct: bool = True):
        """(operand, instruction, rsrc, voffset, soffset) of the cursor's DMAs
        (``parts``: "x" input, "w" weights; sw: ``stap`` the static tap,
        ``ct`` whether the chunk may be the channel tail's)."""
        ops = []
        CT = spec.c_tail and ct
        live = b.cmp_lt(cur["t"], TI)
        xr = b.buffer_rsrc(params["A"], b.select(live, params["A_bytes"], c0))
        if stap is not None:
            tap = ci(stap)
        else:
            tap = b.add(b.mul(cur["r"], ci(TS)), cur["s"])
        if CT:
            # this lane's granule lies past C in the last chunk: zeros
            tl_ln = opaque_lane()
            tl_c8 = b.xor(md(tl_ln, 8), dv(tl_ln, 8))
            g_ok = b.cmp_lt(
                b.add(b.mul(cur["ch"], ci(_CK)), b.mul(md(tl_c8, 8), ci(8))), C
            )
        if FLAT:
            # this lane's granule of the step: tap and channel granule
            fl_ln = opaque_lane()
            fl_g = b.add(b.mul(cur["ch"], ci(8)), b.xor(md(fl_ln, 8), dv(fl_ln, 8)))
            g_ok = b.cmp_lt(fl_g, RSG)
            tap = b.select(g_ok, f_cg.div(fl_g), c0)
            fl_gran = b.mul(b.sub(fl_g, b.mul(tap, CG)), ci(16))
        if "x" in parts:
            if stap is not None:
                ta = b.add(h_tab, ci(stap * 64 * NXI))
            else:
                ta = b.add(b.mul(tap, ci(64 * NXI)), h_tab)
            if TB2:
                ta = b.add(ta, b.mul(b.land(cur["i"], c1), ci(TABN)))
            ent = b.smem_load_vN(tab_s, c0, ta, dtype=I32, n=NXI)
            soff = c0 if FLAT else b.mul(cur["ch"], ci(_CK * 2))
            for i in range(NXI):
                e = ent if NXI == 1 else b.vec_extract(ent, i)
                voff = b.add(e, fl_gran if FLAT else h_gran)
                if CT or FLAT:
                    voff = b.select(g_ok, voff, oob)
                ops.append(("x", i, xr, voff, soff))
        if "w" not in parts:
            return ops
        wr = b.buffer_rsrc(params["B"], b.select(live, params["B_bytes"], c0))
        if WT:
            # piece q = wave * NWI + i: rows 4 (q % 16) + l // 16 of the
            # chunk, outputs k0 + HK (q // 16) + column of the flipped tap
            if DS:
                tapf = b.sub(
                    cur["fb"],
                    b.add(b.mul(cur["r"], ci(SDH * S)), b.mul(cur["s"], ci(SDW))),
                )
            elif stap is not None:
                tapf = ci(R * S - 1 - stap)
            else:
                tapf = b.sub(ci(R * S - 1), tap)
            row0 = b.add(
                b.mul(cur["ch"], ci(_CK)), b.mul(md(b.mul(wave, ci(NWI)), 16), ci(4))
            )
            sbase = b.add(
                b.mul(b.add(b.mul(b.mul(cur["ch"], ci(_CK)), RSK), b.mul(tapf, K)), c2),
                cur["wk0"],
            )
            if CT:
                # rows past C (the last chunk) read zeros
                ct_l16 = dv(opaque_lane(), 16)
            for i in range(NWI):
                voff = cur["wo1"] if NWI >= 4 and (i >> 1) & 1 else cur["wo"]
                if CT:
                    rok = b.cmp_lt(b.add(b.add(row0, ci(4 * i)), ct_l16), C)
                    voff = b.select(rok, voff, oob)
                soff = b.add(sbase, b.mul(RSK, ci(8 * i))) if i else sbase
                ops.append(("w", i, wr, voff, soff))
            return ops
        if DS:
            # the transposed filter (C/K swapped, flipped): position R*S-1-f
            tap = b.sub(
                ci(R * S - 1),
                b.sub(
                    cur["fb"],
                    b.add(b.mul(cur["r"], ci(SDH * S)), b.mul(cur["s"], ci(SDW))),
                ),
            )
        if FLAT:  # the KRSC row read linearly
            woff = b.mul(cur["ch"], ci(_CK * 2))
        else:
            woff = b.mul(b.add(b.mul(tap, C), b.mul(cur["ch"], ci(_CK))), c2)
        for i in range(NWI):
            wsh = ci(((i & 1) * 4 + (i >> 1) * 16) * 2)
            if NWI == 3:
                ii = b.add(b.mul(wave, ci(3)), ci(i))
                wsh = b.mul(
                    b.add(b.mul(dv(ii, 2), ci(16)), b.mul(md(ii, 2), ci(4))), c2
                )
            ops.append(
                (
                    "w",
                    i,
                    wr,
                    b.select(g_ok, cur["wo"], oob) if CT or FLAT else cur["wo"],
                    b.add(woff, b.mul(RSC, wsh)),
                )
            )
        return ops

    def frags(st, hook=None):
        """The step's A / B fragments from stage st; ``hook(i)`` is called
        before the i-th LDS read (when given). sw: st is a scalar parity, the
        stage's element offset is added to the lane bases."""
        a, bb = [None, None], [None, None]
        nrd = [0]
        if SW:
            xo = b.mul(sv(st), ci(IN_E))
            wo_ = b.mul(sv(st), ci(W_E * 2 if WT else W_E))
            xin = [b.add(v, xo) for v in h_xin]
            kw = None if WT else [b.add(v, wo_) for v in h_kw]
            st = 0
        else:
            xin, kw, wo_ = h_xin, h_kw, None

        def tick():
            if hook is not None:
                hook(nrd[0])
            nrd[0] += 1

        for h in range(2):
            bl = []
            for xt in range(XT):
                tick()
                bl.append(
                    b.smem_load_vN(
                        in_s[st],
                        c0,
                        b.add(xin[h], ci(xt * 16 * _CK)),
                        dtype=io_type,
                        n=8,
                    )
                )
            bb[h] = bl
            if WT:
                if h == 0:
                    # KT 8: the XORs stay in the step (the 8 addresses kept
                    # in registers spill)
                    tl = wt_l
                    if KT >= 8:
                        tl = b.inline_asm("", "=v,0", [wt_l], I32, sideeffect=True)
                    if wo_ is not None:
                        # (the stage offset, 2^15, is above the XORed bits)
                        tl = b.add(tl, wo_)
                a[h] = [wt_frag(st, h, kt, tl) for kt in range(KT)]
                continue
            al = []
            for kt in range(KT):
                tick()
                al.append(
                    b.smem_load_vN(
                        w_s[st], c0, b.add(kw[h], ci(kt * 16 * _CK)), dtype=io_type, n=8
                    )
                )
            a[h] = al
        return a, bb

    if WT:
        # lane (tb, ti, tj) of a transposing read takes row 8 tb + ti (+ 4) of
        # a 32-row half and outputs P(tj) * 4.. of a 16-output group, P(tj) =
        # 2 (tj & 1) + (tj >> 1) (the epilogue's interleave: MFMA row 4 g + j
        # holds output (g & 1) * 8 + (g >> 1) * 4 + j); group kt of the wave's
        # first one g0 sits at slot (g0 + kt) ^ swz = (g0 ^ swz) ^ kt, so a
        # fragment's byte address (the stage is a byte array at an offset below
        # 64 KiB, so its base folds into the reads' offsets) is one XOR of a
        # lane constant
        kt0 = b.mul(wk, ci(KT))
        GPH = HK // 16
        wt_wv = b.readfirstlane(b.mul(md(kt0, GPH), ci(32)))
        wt_hv = b.readfirstlane(b.mul(dv(kt0, GPH), ci(_CK * 256)))

        def wt_lane(ln: Value) -> Value:
            def bits(sh: int, mask: int, to: int) -> Value:
                v = b.land(b.lshr(ln, ci(sh)) if sh else ln, ci(mask))
                return b.shl(v, ci(to)) if to else v

            # row 8 tb + ti: tb = l >> 4 at 2048, ti = (l >> 2) & 3 at 256;
            # swz = ti | 4 (tb & 1) at 32; P(tj) at 8
            lp = b.lor(
                b.lor(
                    b.lor(bits(4, 3, 11), bits(2, 3, 8)),
                    b.lor(bits(2, 3, 5), bits(4, 1, 7)),
                ),
                b.lor(bits(0, 1, 4), bits(1, 1, 3)),
            )
            return b.add(b.xor(lp, wt_wv), wt_hv)

        wt_l = wt_lane(lane)

    def wt_frag(st: int, h: int, kt: int, wl: Value) -> Value:
        """A fragment (16 outputs x 32 reduction rows): two transposing reads."""
        base = b.xor(wl, ci(kt * 32)) if kt else wl
        lo = b.ds_read_tr16_b64(
            w_s[st], c0, b.add(base, ci(h * 32 * 256)), dtype=io_type
        )
        hi = b.ds_read_tr16_b64(
            w_s[st], c0, b.add(base, ci((h * 32 + 4) * 256)), dtype=io_type
        )
        return b.vec_concat(lo, hi)

    def mfmas(accs, a, bb, inter=(), start: int = 0):
        """The step's MFMAs; ``inter`` callables are spread evenly between them
        from MFMA ``start`` on (pinned by scheduling barriers)."""
        accs = list(accs)
        n = 2 * XT * KT
        every = (n - start) // len(inter) if inter else 0
        j = 0
        for h in range(2):
            for xt in range(XT):
                for kt in range(KT):
                    q = j - start
                    if every and q >= 0 and q % every == 0 and q // every < len(inter):
                        b.sched_barrier(0)
                        inter[q // every]()
                        b.sched_barrier(0)
                    i = xt * KT + kt
                    # None: an item's first step (C = 0, no zeroed registers)
                    c_in = zero if accs[i] is None else accs[i]
                    accs[i] = _mfma(b, dtype, "16x16x32", a[h][kt], bb[h][xt], c_in)
                    j += 1
        return accs

    def bar(prio: int):
        b.s_setprio(prio)
        b.s_barrier_bare()
        b.sched_barrier(0)

    # ---- epilogue -----------------------------------------------------------
    i32x2 = VectorType(I32, 2)
    half8 = VectorType(BF16 if dtype == "bf16" else F16, 8)

    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])

    def epilogue(accs, t: Value, rnd: Value):
        tl = b.div(t, ci(SP))
        if spec.kmaj:
            tk = b.div(tl, nxt)
            tx = b.sub(tl, b.mul(tk, nxt))
        else:
            tx = f_nkt.div(tl)
            tk = b.sub(tl, b.mul(tx, nkt))
        if GRP:
            gi, tk = grp_tk(tk)
            gk = b.mul(gi, K)  # the group's first output channel
        ln = opaque_lane()
        md(ln, 16), dv(ln, 16)  # (unused, kept: part of the kernel's IR)
        # lane 4 p + j stores pixel p's j-th 16 bytes of the store's 64
        # (bf16: channels 8 j.., held by lane 16 j + p; fp32 slab:
        # channels 4 j.., held by lane 16 q + p, q = 2 (j & 1) + j / 2):
        # one ds_bpermute per dword pair, and a store's lanes then cover
        # 16 64-byte rows, 4 lanes per row (16 lanes per row touched 4x
        # as many cache lines)
        l16, q = dv(ln, 4), md(ln, 4)
        sq = q if SP == 1 else b.add(b.mul(md(q, 2), c2), dv(q, 2))
        src = b.mul(b.add(b.mul(sq, ci(16)), l16), ci(4))

        def permute(words):
            out = []
            for u, v in (words[:2], words[2:]):
                pr = b.ds_bpermute_b64(src, b.vec_bitcast(b.vec_pack([u, v], I32), I64))
                pr = b.vec_bitcast(pr, i32x2)
                out += [b.vec_extract(pr, j) for j in range(2)]
            return out

        xbase = b.add(
            b.add(b.mul(tx, ci(BX)), b.mul(g, ci(BX // 2))),
            b.add(b.mul(wx, ci(XT * 16)), l16),
        )
        kbase = b.add(b.mul(tk, ci(BK)), b.mul(wk, ci(KT * 16)))

        def opix(x: Value):
            """(valid, output pixel) of tile pixel x: x itself, or for ds the
            dX pixel of its phase."""
            if not DS:
                return b.cmp_lt(x, NX), x
            e = b.smem_load_vN(
                opix_s,
                c0,
                b.add(b.mul(md(rnd, 3), ci(BX)), b.sub(x, b.mul(tx, ci(BX)))),
                dtype=I32,
                n=1,
            )
            e = b.vec_extract(e, 0) if isinstance(e.type, VectorType) else e
            if spec.zf:
                zfl[0] = e
                return b.cmp_ge(e, c0), b.land(e, ci((1 << ZF_BITS) - 1))
            return b.cmp_ge(e, c0), e

        zfl = [None]

        if SP == 1:
            kq = b.add(b.mul(dv(q, 2), ci(16)), b.mul(md(q, 2), ci(8)))  # = 8 q
            packed = {}
            for xt in range(XT):
                for cc in range(KT // 2):
                    da = b.vec_bitcast(
                        _trunc_f32(b, dtype, accs[xt * KT + 2 * cc]), i32x2
                    )
                    db = b.vec_bitcast(
                        _trunc_f32(b, dtype, accs[xt * KT + 2 * cc + 1]), i32x2
                    )
                    lo0, hi0 = b.permlane32_swap(
                        b.vec_extract(da, 0), b.vec_extract(db, 0)
                    )
                    lo1, hi1 = b.permlane32_swap(
                        b.vec_extract(da, 1), b.vec_extract(db, 1)
                    )
                    packed[xt, cc] = [lo0, lo1, hi0, hi1]
            # every accumulator narrowed first: the permutes then take
            # registers from the freed half (interleaved, K256 spills)
            b.sched_barrier(0)
            klim = b.sub(K, kbase)  # channels left from the wave's first
            for xt in range(XT):
                xok, xp = opix(b.add(xbase, ci(xt * 16)))
                if not spec.k_tail:
                    # the store offsets as one select per 16-pixel tile, the
                    # channel step as the store's immediate offset (K range:
                    # a compare of the lane's channel with a scalar limit)
                    po = b.mul(xp, YS)
                    if GRP:
                        po = b.add(po, gk)
                    b0 = b.select(xok, b.mul(b.add(po, b.add(kbase, kq)), c2), oob)
                zb = []
                if spec.zf and not spec.zl:
                    # the tapless neighbours' pixels: zeros
                    for d, (da, db) in enumerate(ds_zero_deltas(spec)):
                        zok = b.land(
                            xok, b.cmp_ne(b.land(zfl[0], ci(1 << (ZF_BITS + d))), c0)
                        )
                        dp = b.add(b.mul(ci(da), p_Wo), ci(db))
                        zb.append(
                            b.select(zok, b.add(b0, b.mul(b.mul(dp, K), c2)), oob)
                        )
                for cc in range(KT // 2):
                    words = permute(packed[xt, cc])
                    val = b.vec_bitcast(b.vec_pack(words, I32), half8)
                    k = b.add(b.add(kbase, ci(cc * 32)), kq)
                    if zb:
                        kok = b.cmp_lt(kq, b.sub(klim, ci(cc * 32)))
                        zv = b.vec_bitcast(b.vec_pack([c0] * 4, I32), half8)
                        for z in zb:
                            off = b.add(b.select(kok, z, oob), ci(cc * 64))
                            _buf_store_vN(b, dtype, d_rsrc, off, c0, zv, 4)
                    if spec.k_tail:
                        # 16-bit stores, each masked by K
                        st1 = (
                            b.buffer_store_bf16
                            if dtype == "bf16"
                            else b.buffer_store_f16
                        )
                        e0 = b.mul(b.add(b.mul(xp, K), k), c2)
                        for j in range(8):
                            ok = b.land(xok, b.cmp_lt(b.add(k, ci(j)), K))
                            off = b.select(ok, b.add(e0, ci(2 * j)), oob)
                            st1(d_rsrc, off, c0, b.vec_extract(val, j))
                        continue
                    kok = b.cmp_lt(kq, b.sub(klim, ci(cc * 32)))
                    off = b.add(b.select(kok, b0, oob), ci(cc * 64))
                    _buf_store_vN(b, dtype, d_rsrc, off, c0, val, 4)
        else:
            pc = b.mod(t, ci(SP))
            slab = b.mul(b.mul(NX, K), ci(4))
            base = b.global_ptr_offset(
                params["D"], b.mul(b.zext(pc, I64), b.zext(slab, I64))
            )
            s_rsrc = b.buffer_rsrc(base, slab)
            kq = b.mul(q, ci(4))
            for xt in range(XT):
                xok, xp = opix(b.add(xbase, ci(xt * 16)))
                for kt in range(KT):
                    k = b.add(b.add(kbase, ci(kt * 16)), kq)
                    ok = b.land(xok, b.cmp_lt(k, K))
                    off = b.select(ok, b.mul(b.add(b.mul(xp, K), k), ci(4)), oob)
                    v = accs[xt * KT + kt]
                    w = b.vec_bitcast(v, VectorType(I32, 4))
                    v = b.vec_bitcast(
                        b.vec_pack(
                            permute([b.vec_extract(w, j) for j in range(4)]), I32
                        ),
                        v.type,
                    )
                    if spec.k_tail:
                        # fp32 element stores, each masked by K
                        e0 = b.mul(b.add(b.mul(xp, K), k), ci(4))
                        for j in range(4):
                            okj = b.land(xok, b.cmp_lt(b.add(k, ci(j)), K))
                            b.buffer_store_f32(
                                s_rsrc,
                                b.select(okj, b.add(e0, ci(4 * j)), oob),
                                c0,
                                b.vec_extract(v, j),
                            )
                    else:
                        b.buffer_store_vN_f32(s_rsrc, off, c0, v, 4)
                    b.sched_barrier(0)

    # ---- the two groups ------------------------------------------------------
    zero = b.zero_vec_f32(4)
    names = SC + LN

    nacc = XT * KT
    vec4 = zero.type

    # group 1's DMAs out of its MFMA phase (where they share the memory pipe
    # with group 0's and stall its MFMAs). S6, bk 256 plain: group 0 issues
    # all eight waves' weight rows (wave + 4's are one scalar offset away),
    # group 1 only its input rows (nothing in its MFMA phase); S5, the other
    # kernels except wt: group 1's input rows (read by its own waves only) of
    # step + 1 into the other stage at the start of its memory phase (two
    # phases of latency), its weight rows over the second half of its MFMA
    # phase
    S6 = BK == 256 and BX == 256 and not WT and not DS and not FLAT
    S5 = not S6 and not WT

    def adv_sw(cur, tau: int, last: bool):
        """sw: the cursor's advance in the step of tap tau: it moves to the
        next chunk in the step of tap NT - 2 (into the next item in the last
        chunk), else only its static tap moves."""
        if tau != NT - 2:
            return cur
        if not last:
            return dict(cur, ch=b.add(cur["ch"], c1))
        nxt = cursor_at(b.add(cur["i"], c1))
        return {k: nxt[k] for k in names}

    def step(accs, cur, st, g1: bool, sw=None):
        """One step reading stage st (static; sw: a scalar parity); returns
        (accs, cursor). sw = (tap, last chunk): the step's static tap."""

        def prep(c, parts: str, d: int):
            # the DMAs of step + d (the cursor's position)
            if sw is None:
                return dma_prep(c, parts)
            tau, last = sw
            return dma_prep(c, parts, stap=(tau + d) % NT, ct=last or tau + d >= NT)

        def adv(c):
            if sw is None:
                return dict(zip(names, cur_adv(c)))
            return adv_sw(c, *sw)

        ost = s_other(st)
        if not g1:
            # the DMAs of step j + 1 between the fragment reads (one every
            # 3 reads, S6: 2), the cursor advance after them
            ops = prep(cur, "xw", 1)
            if S6:
                # S6: group 1's weight rows too (wave + 4: channel
                # + 128, one scalar offset)
                ops += [
                    ("w4", i, r_, v_, b.add(s_, b.mul(RSC, ci(256))))
                    for (k_, i, r_, v_, s_) in ops
                    if k_ == "w"
                ]
            # the first DMA before the advance (whose table build needs the
            # registers the fragments take later)
            dma_issue(ops[:1], ost)
            cur = adv(cur)

            nxt = [1]
            every = 2 if S6 else 3

            def hook(i: int):
                if i % every == 0 and nxt[0] < len(ops):
                    dma_issue([ops[nxt[0]]], ost)
                    nxt[0] += 1

            a, bb = frags(st, hook)
            for op in ops[nxt[0] :]:
                dma_issue([op], ost)
            b.s_waitcnt(lgkmcnt=0)
            b.sched_barrier(0)
            bar(1)
            accs = mfmas(accs, a, bb)
            b.sched_barrier(0)
            b.s_waitcnt(vmcnt=0)
            bar(0)
        else:
            # Group 1 advances its cursor at the start of its memory phase,
            # before the fragment reads (only the accumulators are live across
            # a tile change: no spills, whose stores would share the DMAs'
            # vmcnt), forms its DMA offsets there and spreads its DMAs (step +
            # 2 into this stage, read by both groups) over its MFMA phase;
            # group 0 issues all of its own at the barrier this phase starts on.
            # The barrier that ends group 1's MFMA phase opens its next
            # memory phase instead: an item's epilogue runs before it,
            # beside group 0's epilogue, not one phase after it.
            bar(0)
            if S6:
                # only this group's input rows (step + 1, the
                # other stage) at the start of the memory phase; group 0
                # issues all weight rows; no DMAs in the MFMA phase
                dma_issue(prep(cur, "x", 1), ost)
                cur = adv(cur)
                b.sched_barrier(0)
                a, bb = frags(st)
                b.s_waitcnt(lgkmcnt=0)
                b.sched_barrier(0)
                bar(1)
                # the inputs land before the barrier that opens the next
                # memory phase: waited in the MFMA phase's last quarter (a
                # wait after the MFMAs makes the allocator spill)
                n_ = 2 * XT * KT
                accs = mfmas(
                    accs,
                    a,
                    bb,
                    inter=[lambda: b.s_waitcnt(vmcnt=0)],
                    start=n_ - n_ // 4,
                )
                b.sched_barrier(0)
                return accs, cur
            if S5:
                # the input DMAs of step + 1 (the cursor before its advance)
                # into the other stage at the start of the memory phase: their
                # rows (this group's pixel half) are read by this group only,
                # last in its previous memory phase, next after the barrier
                # that opens its next one (waited before it); the weight DMAs
                # of step + 2 stay in the MFMA phase (second half)
                dma_issue(prep(cur, "x", 1), ost)
                cur = adv(cur)
                b.sched_barrier(0)
                a, bb = frags(st)
                wops = prep(cur, "w", 2)
                b.s_waitcnt(vmcnt=NXI, lgkmcnt=0)
                b.sched_barrier(0)
                bar(1)
                # this phase's inputs land before the barrier that opens the
                # next memory phase: waited before the weights are issued (a
                # wait after the MFMAs makes the allocator spill)
                inter = [lambda: b.s_waitcnt(vmcnt=0)] + [
                    (lambda op=op: dma_issue([op], st)) for op in wops
                ]
                accs = mfmas(accs, a, bb, inter=inter, start=(2 * XT * KT) // 2)
                b.sched_barrier(0)
                return accs, cur
            cur = adv(cur)
            b.sched_barrier(0)
            a, bb = frags(st)
            ops = prep(cur, "xw", 2)
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sched_barrier(0)
            bar(1)
            inter = [(lambda op=op: dma_issue([op], st)) for op in ops]
            accs = mfmas(accs, a, bb, inter=inter)
            b.sched_barrier(0)
        return accs, cur

    # the round loop carries the item's step count (the next item's formed
    # before this item's epilogue, its scalar work filling the epilogue's
    # VALU / LDS waits); wt and c_tail bk 256 kernels spill VGPRs with it
    STEPS = not WT and not spec.c_tail and not SW

    # zero-first: an item's first step peeled, its MFMAs on C = 0 (no
    # accumulator zeroing at the item's start); bk <= 128 only (the wider
    # tiles spill)
    ZF = BK <= 128 and not SW

    def chunk_sw(accs, cur, p0, g1: bool, last: bool):
        """sw: the NT steps of one chunk, stage parity p0 at its first."""
        for tau in range(NT):
            sp = p0 if tau % 2 == 0 else b.xor(p0, c1)
            # an opaque copy per step: the stage's lane bases are formed in
            # the step, not shared by every other step (held: VGPR spills)
            sp = b.inline_asm("", "=s,0", [sp], I32, sideeffect=True)
            accs, cur = step(accs, cur, sp, g1, (tau, last))
        return accs, cur, (b.xor(p0, c1) if NT % 2 else p0)

    def item_sw(cur, st, rnd, g1: bool):
        """sw: an item's plain chunks (a loop), then its last chunk."""
        if SP == 1:
            nci = nch
        else:
            pc = b.mod(tile_of(rnd), ci(SP))
            nci = b.sub(
                b.div(b.mul(b.add(pc, c1), nch), ci(SP)), b.div(b.mul(pc, nch), ci(SP))
            )
        accs = [zero] * nacc
        loop = b.scf_for_iter(
            c0,
            b.sub(nci, c1),
            c1,
            [(f"sa{i}", v) for i, v in enumerate(accs)]
            + [(f"sc_{k}", cur[k]) for k in names]
            + [("ssp", st)],
            iv_name="chunk",
            elide_trailing_barrier=False,
        )
        with loop as (_, lc):
            la = list(lc[:nacc])
            lcur = dict(zip(names, lc[nacc : nacc + len(names)]))
            la, lcur, lp = chunk_sw(la, lcur, lc[-1], g1, False)
            b.scf_yield(*la, *[lcur[k] for k in names], lp)
        res = list(loop.results)
        accs, cur = res[:nacc], dict(zip(names, res[nacc : nacc + len(names)]))
        return chunk_sw(accs, cur, res[-1], g1, True)

    def item_steps(accs, cur, steps: Value, p: int, g1: bool, tag: str):
        """All steps of an item that starts on stage p: pairs, then an odd tail
        (ZF: the first step peeled, its MFMAs on C = 0)."""
        if ZF:
            accs, cur = step([None] * nacc, cur, p, g1)
            steps, p = b.sub(steps, c1), 1 - p
        loop = b.scf_for_iter(
            c0,
            b.div(steps, c2),
            c1,
            [(f"{tag}a{i}", v) for i, v in enumerate(accs)]
            + [(f"{tag}c_{k}", cur[k]) for k in names],
            iv_name=f"{tag}pair",
            elide_trailing_barrier=False,
        )
        with loop as (_, lc):
            la = list(lc[:nacc])
            lcur = dict(zip(names, lc[nacc:]))
            la, lcur = step(la, lcur, p, g1)
            la, lcur = step(la, lcur, 1 - p, g1)
            b.scf_yield(*la, *[lcur[k] for k in names])
        res = list(loop.results)
        accs, cur = res[:nacc], dict(zip(names, res[nacc:]))
        tail = b.scf_if_else(
            b.cmp_eq(b.mod(steps, c2), c1), [vec4] * nacc + [I32] * (len(names) + 1)
        )
        with tail as (then_ctx, else_ctx):
            with then_ctx:
                ta, tcur = step(accs, cur, p, g1)
                b.scf_yield(*ta, *[tcur[k] for k in names], ci(1 - p))
            with else_ctx:
                b.scf_yield(*accs, *[cur[k] for k in names], ci(p))
        res = list(tail.results)
        return res[:nacc], dict(zip(names, res[nacc:-1])), res[-1]

    def run(g1: bool):
        nonlocal h_gran, h_xin, h_kw, h_tab, lane
        tctx["g1"], tctx["rnd"] = g1, c0
        # per group, from an opaque lane (not formed above the group branch
        # and held across it)
        lane_ = lane
        lane = opaque_lane()
        h_gran, h_xin, h_kw, h_tab = lane_bases()
        lane = lane_
        cur = cursor_at(c0, True)
        if SW:
            dma_issue(dma_prep(cur, stap=0), 0)
        else:
            dma(cur, 0)
        if TB2 and not g1:
            # group 0's rows of item 1 while it waits for group 1 at the
            # first barrier (group 1 builds its own at its crossing)
            with b.scf_if(b.cmp_lt(c1, rounds)):
                ln1 = opaque_lane()
                build_tab(item_info(tile_of(c1)), ln1, dv(ln1, 8), None, c1)
        if not SW:  # (sw: the cursor's static tap moves to 1)
            cur = dict(zip(names, cur_adv(cur)))
        st1 = dict(stap=1) if SW else {}
        if g1 and not S6:
            # group 1's DMAs of step 1 before the first barrier: the wait
            # leaves them in flight (they are waited at the end of its first
            # memory phase)
            if S5:
                # (step 1's inputs come at the first memory phase)
                dma_issue(dma_prep(cur, "w", **st1), 1)
            else:
                dma_issue(dma_prep(cur, **st1), 1)
            # (ds, TB2: LDS writes other waves read after the barrier)

            n1 = NWI if S5 else NXI + NWI
            if DS or TB2:
                b.s_waitcnt(vmcnt=n1, lgkmcnt=0)
            else:
                b.s_waitcnt(vmcnt=n1)
            b.s_barrier_bare()
        else:
            b.s_waitcnt(vmcnt=0, lgkmcnt=0) if DS or TB2 else b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
        b.sched_barrier(0)

        def steps_of(i: Value) -> Value:
            info = item_info(tile_of(i))
            return b.mul(
                b.mul(b.sub(info["r1"], info["r0"]), b.sub(info["s1"], info["s0"])),
                b.sub(info["ch1"], info["ch0"]),
            )

        outer = b.scf_for_iter(
            c0,
            rounds,
            c1,
            [(f"o_{k}", cur[k]) for k in names]
            + [("o_st", c0)]
            + ([("o_steps", steps_of(c0))] if STEPS else []),
            iv_name="rnd",
            elide_trailing_barrier=False,
        )
        with outer as (rnd, oc):
            cur = dict(zip(names, oc[: len(names)]))
            st = oc[len(names)]
            t = tile_of(rnd)
            # STEPS: the item's step count comes with the round
            if SW:
                steps = None
            elif STEPS:
                steps = oc[len(names) + 1]
            else:
                steps = steps_of(rnd)
            accs = [zero] * nacc
            tctx["rnd"] = rnd
            if SW:
                accs, cur, st = item_sw(cur, st, rnd, g1)
            else:
                par = b.scf_if_else(
                    b.cmp_eq(st, c0), [vec4] * nacc + [I32] * (len(names) + 1)
                )
                with par as (then_ctx, else_ctx):
                    for p, ctx in ((0, then_ctx), (1, else_ctx)):
                        with ctx:
                            pa, pcur, pst = item_steps(accs, cur, steps, p, g1, f"p{p}")
                            b.scf_yield(*pa, *[pcur[k] for k in names], pst)
                res = list(par.results)
                accs, cur, st = res[:nacc], dict(zip(names, res[nacc:-1])), res[-1]
            # the next item's step count before the epilogue (its scalar work
            # fills the epilogue's VALU / LDS waits)
            nxt_steps = [steps_of(b.add(rnd, c1))] if STEPS else []
            epilogue(accs, t, rnd)
            b.sched_barrier(0)
            if TB2 and g1:
                # item rnd + 2's rows while this group idles, unless the
                # cursor already crossed into it (it built them then)
                k2 = b.add(rnd, c2)
                with b.scf_if(
                    b.land(b.cmp_eq(cur["i"], b.add(rnd, c1)), b.cmp_lt(k2, rounds))
                ):
                    build_tab_all(item_info(tile_of(k2)), k2)
                    b.s_waitcnt(lgkmcnt=0)
                b.sched_barrier(0)
            b.scf_yield(*[cur[k] for k in names], st, *nxt_steps)
        b.s_setprio(0)
        b.s_barrier_bare()  # balance the barrier count

    def groups():
        with b.scf_if_else(b.cmp_eq(g, c0)) as (then_ctx, else_ctx):
            with then_ctx:
                run(False)
            with else_ctx:
                run(True)

    if not spec.zl:
        groups()
        return b.kernel

    # ---- zl: light zero units after the items ---------------------------------
    # blocks past G (the items alone fill fewer CUs) run no item
    with b.scf_if(b.cmp_lt(blk, G)):
        groups()
    ZU = b.div(b.add(NXP, ci(ZL_PX - 1)), ci(ZL_PX))
    big = b.cmp_ge(TI, ci(PERSIST))
    GL = b.smin(b.add(TI, ZU), ci(PERSIST))
    xtra = b.sub(GL, G)  # 0 when big
    # light blocks (one item fewer): ds deals slot G - 1 - b in odd rounds,
    # so the heavy ones are b >= G - part (full odd) or b < part (even)
    uneven = b.select(big, b.cmp_gt(part, c0), b.cmp_gt(xtra, c0))
    L = b.select(big, b.select(uneven, b.sub(G, part), G), b.select(uneven, xtra, G))
    odd = b.cmp_eq(b.mod(full, c2), c1)
    o_big = b.select(odd, blk, b.mod(b.add(blk, L), G))
    o_sm = b.select(b.cmp_ge(blk, G), b.sub(blk, G), b.add(blk, xtra))
    o = b.select(big, o_big, o_sm)
    NB = b.select(big, G, GL)
    E = b.select(uneven, ci(spec.zl), c0)
    few = b.cmp_le(ZU, b.mul(L, E))  # the light blocks' slack takes them all
    RZ = b.sub(ZU, b.mul(L, E))

    def first_unit(o_: Value) -> Value:
        ol = b.smin(o_, L)
        a = b.div(b.mul(ol, ZU), L)
        bb = b.add(b.mul(ol, E), b.div(b.mul(o_, RZ), NB))
        return b.select(few, a, bb)

    u0, u1 = first_unit(o), first_unit(b.add(o, c1))
    ((a0, b0),) = PORD
    ZD = ds_zero_deltas(spec)
    ln = opaque_lane()
    pp, jg = dv(ln, 8), md(ln, 8)
    nkc = b.div(b.add(K, ci(63)), ci(64))
    zv = b.vec_bitcast(b.vec_pack([c0] * 4, I32), half8)
    with b.scf_for(u0, u1, c1, iv_name="zu") as u:
        xl = b.add(b.mul(u, ci(ZL_PX)), b.add(b.mul(wave, ci(8)), pp))
        ok = b.cmp_lt(xl, NXP)
        n = b.div(xl, hwq)
        rem = b.sub(xl, b.mul(n, hwq))
        qi = b.div(rem, Wq)
        qj = b.sub(rem, b.mul(qi, Wq))
        oh = b.add(b.mul(qi, ci(SDH)), ci(a0))
        ow = b.add(b.mul(qj, ci(SDW)), ci(b0))
        op = b.add(b.mul(b.add(b.mul(n, p_Ho), oh), p_Wo), ow)
        zb = []
        for da, db in ZD:
            zok = b.land(
                ok,
                b.land(
                    b.cmp_lt(b.add(oh, ci(da)), p_Ho), b.cmp_lt(b.add(ow, ci(db)), p_Wo)
                ),
            )
            zp = b.add(op, b.add(b.mul(ci(da), p_Wo), ci(db)))
            zb.append((zok, b.add(b.mul(b.mul(zp, K), c2), b.mul(jg, ci(16)))))
        with b.scf_for(c0, nkc, c1, iv_name="zk") as kc:
            kok = b.cmp_lt(b.add(b.mul(kc, ci(64)), b.mul(jg, ci(8))), K)
            ko = b.mul(kc, ci(128))
            for zok, zo in zb:
                off = b.select(b.land(zok, kok), b.add(zo, ko), oob)
                _buf_store_vN(b, dtype, d_rsrc, off, c0, zv, 4)
    return b.kernel


def _build_cs4(spec: FwdFlexMfmaSpec, arch: str) -> KernelDef:
    """The ``cs == 4`` kernel: X128 x K128 tiles, the reduction split four
    ways inside the block (see ``FwdFlexMfmaSpec.cs``).

    Group g (waves 4g..4g+3) runs chunks ch0 + g, ch0 + g + 2, ... of the
    item's piece through its own two LDS stages (one step = one tap and one
    chunk), one barrier phase behind / ahead of the other group as in the
    main kernel; wave (xi, sl) of a group reads pixels 64 xi.. of the stage
    and the 32-channel half sl and runs 4 x 8 MFMAs per step. A tile is
    drained at its end: the four (group, half) partial sums of each pixel
    half meet in the stage memory (fp32, two rounds of 32 pixels) and one
    pass sums, converts and stores them (or writes the fp32 slab of a
    split-C piece)."""
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    BX, BK, SP, KT = 128, 128, spec.split_c, 8
    NXI = NWI = 4
    R, S = p.KH, p.KW
    SH, PH, PW, DH, DW = p.stride, p.PAD, p.PAD_W, p.dil_h, p.dil_w
    IN_E = BX * _CK  # input elements of a group's stage
    GE = (BX + BK) * _CK  # elements of a group's stage
    PART = 32 * BK * 2  # one partial region (32 pixels x 128 fp32) in elements

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    def dv(a: Value, n: int) -> Value:
        return b.div(a, ci(n))

    def md(a: Value, n: int) -> Value:
        return b.mod(a, ci(n))

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_N, p_Hi, p_Wi = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]

    tid = b.thread_id_x()
    lane = md(tid, _WAVE)
    wave = b.readfirstlane(dv(tid, _WAVE))
    wq = md(wave, 4)
    xi = md(wq, 2)
    sl = dv(wq, 2)

    def opaque_lane() -> Value:
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    hw = b.mul(p_Ho, p_Wo)
    NX = b.mul(p_N, hw)
    nxt = b.div(b.add(NX, ci(BX - 1)), ci(BX))
    nkt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    TI = b.mul(b.mul(nxt, nkt), ci(SP))
    G = b.smin(TI, ci(PERSIST))
    nch = b.div(C, ci(_CK))
    RSC = b.mul(ci(R * S), C)
    blk = b.block_id_x()
    if spec.xcd:
        # any G: the first 8 * (G // 8) blocks are renumbered XCD-major, the
        # last G % 8 keep their number (a bijection of [0, G))
        g8 = b.div(G, ci(8))
        xr = b.add(b.mul(b.mod(blk, ci(8)), g8), b.div(blk, ci(8)))
        blk = b.select(b.cmp_lt(blk, b.mul(g8, ci(8))), xr, blk)
    full, part = b.div(TI, G), b.mod(TI, G)
    last_ok = b.cmp_lt(b.mod(b.add(blk, b.mul(full, ci(ROT))), G), part)
    rounds = b.add(full, b.select(last_ok, c1, c0))

    def tile_of(i: Value) -> Value:
        return b.add(b.mul(i, G), b.mod(b.add(blk, b.mul(i, ci(ROT))), G))

    def item_info(t: Value) -> Dict[str, Value]:
        tl = b.div(t, ci(SP))
        pc = b.mod(t, ci(SP))
        if spec.kmaj:
            tk = b.div(tl, nxt)
            tx = b.mod(tl, nxt)
        else:
            tx = b.div(tl, nkt)
            tk = b.mod(tl, nkt)
        x0 = b.mul(tx, ci(BX))
        x1 = b.sub(b.smin(b.add(x0, ci(BX)), NX), c1)
        n0, n1 = b.div(x0, hw), b.div(x1, hw)
        same = b.cmp_eq(n0, n1)
        q0 = b.sub(x0, b.mul(n0, hw))
        q1 = b.sub(x1, b.mul(n0, hw))
        ho0 = b.select(same, b.div(q0, p_Wo), c0)
        ho1 = b.select(same, b.div(q1, p_Wo), b.sub(p_Ho, c1))
        row = b.land(same, b.cmp_eq(ho0, ho1))
        wo0 = b.select(row, b.sub(q0, b.mul(ho0, p_Wo)), c0)
        wo1 = b.select(row, b.sub(q1, b.mul(ho1, p_Wo)), b.sub(p_Wo, c1))
        lo_h = b.sub(b.mul(ho1, ci(SH)), ci(PH))
        hi_h = b.sub(b.mul(ho0, ci(SH)), ci(PH))
        lo_w = b.sub(b.mul(wo1, ci(SH)), ci(PW))
        hi_w = b.sub(b.mul(wo0, ci(SH)), ci(PW))

        # the counts in closed form: #{r < n: lo + r d < 0} =
        # min(ceil(max(-lo, 0) / d), n), #{r < n: hi + r d < E} =
        # min(ceil(max(E - hi, 0) / d), n)
        def cnt(v: Value, d: int, n: int) -> Value:
            v = b.smax(v, c0)
            if d > 1:
                v = b.div(b.add(v, ci(d - 1)), ci(d))
            return b.smin(v, ci(n))

        r0, r1 = cnt(b.sub(c0, lo_h), DH, R), cnt(b.sub(p_Hi, hi_h), DH, R)
        s0, s1 = cnt(b.sub(c0, lo_w), DW, S), cnt(b.sub(p_Wi, hi_w), DW, S)
        empty = b.lor(b.cmp_ge(r0, r1), b.cmp_ge(s0, s1))
        r0, r1 = b.select(empty, c0, r0), b.select(empty, c1, r1)
        s0, s1 = b.select(empty, c0, s0), b.select(empty, c1, s1)
        ch0 = b.div(b.mul(pc, nch), ci(SP))
        ch1 = b.div(b.mul(b.add(pc, c1), nch), ci(SP))
        return dict(pc=pc, tx=tx, tk=tk, r0=r0, r1=r1, s0=s0, s1=s1, ch0=ch0, ch1=ch1)

    # DMA offset table, per wave: [tap][wave][pixel of 8][instr]
    RSP = _tab_taps(spec)
    tab_s = b.smem_alloc(I32, [1, RSP * 64 * NXI], name_hint="flex_tab")

    def build_tab(tx: Value) -> None:
        ln = opaque_lane()
        l8 = dv(ln, 8)
        c = md(ln, 8)
        pix = []
        for i in range(NXI):
            x = b.add(
                b.mul(tx, ci(BX)),
                b.add(b.mul(b.add(b.mul(wq, ci(NXI)), ci(i)), ci(8)), l8),
            )
            n = b.div(x, hw)
            rem = b.sub(x, b.mul(n, hw))
            ho = b.div(rem, p_Wo)
            wo = b.sub(rem, b.mul(ho, p_Wo))
            hb = b.sub(b.mul(ho, ci(SH)), ci(PH))
            wb = b.sub(b.mul(wo, ci(SH)), ci(PW))
            base = b.add(b.mul(b.add(b.mul(n, p_Hi), hb), p_Wi), wb)
            pix.append((b.cmp_lt(x, NX), hb, wb, base))
        for kk in range(RSP // 8):
            t = b.add(c, ci(8 * kk))
            r = b.div(t, ci(S))
            s_ = b.sub(t, b.mul(r, ci(S)))
            dh, dw = b.mul(r, ci(DH)), b.mul(s_, ci(DW))
            tok = b.cmp_lt(t, ci(R * S))
            toff = b.add(b.mul(dh, p_Wi), dw)
            ents = []
            for xok, hb, wb, base in pix:
                ok = b.land(
                    b.land(tok, xok),
                    b.land(
                        b.cmp_ult(b.add(hb, dh), p_Hi), b.cmp_ult(b.add(wb, dw), p_Wi)
                    ),
                )
                ents.append(b.select(ok, b.mul(b.mul(b.add(base, toff), C), c2), oob))
            idx = b.mul(b.add(b.mul(b.add(b.mul(t, ci(8)), wave), ci(8)), l8), ci(NXI))
            b.smem_store_vN(tab_s, [c0, idx], b.vec_pack(ents, I32), NXI)

    # one allocation per stage (both groups' halves): the DMA into one stage
    # and the fragment reads of the other stay apart for the waitcnt pass
    st_s = [
        b.smem_alloc(io_type, [1, 2 * GE], name_hint=f"flexcs{i}") for i in range(2)
    ]
    st_a = [b.smem_addr_of(v) for v in st_s]

    h_r16, h_q = md(lane, 16), dv(lane, 16)
    h_l8 = dv(lane, 8)
    h_gran = b.mul(b.xor(md(lane, 8), h_l8), ci(16))
    h_sl = b.mul(b.xor(b.add(b.mul(sl, ci(4)), h_q), md(h_r16, 8)), ci(8))
    # (the per-wave table row base; run_pf2 forms its own in the group's
    # branch, this one is unused but kept: it is part of the kernel's IR)
    b.mul(b.add(b.mul(wave, ci(8)), h_l8), ci(NXI))

    RS = R * S

    C2 = b.mul(C, c2)
    RSK = b.mul(ci(RS), K)  # wt: a reduction row of the KRSC filter

    def run_pf2(g, gb, wlane, frags, bar, reduce_store, zero, vec4, nacc, head):
        """Group g's items, the filter taps of a chunk unrolled (compile-time
        tap offsets, no scalar cursor; every tap runs, padding taps load
        zeros), DMA look-ahead of two steps: a group's DMAs of step j + 2 go
        into the stage step j just read and are spread over step j's MFMA
        phase, their offsets formed before the phase barrier. Lookahead slot
        m of a chunk pair (step m from the pair's start, m < 2 RS + 2) is
        chunk cA + 2 (m // RS), tap m % RS; it is live while that chunk is
        one of the group's (index < NQ) and < ch1, else its DMAs are dead
        (zero fill, kept so every step waits the same vmcnt). ``head`` runs
        at each item's start."""
        NDMA = NXI + NWI
        # formed in the group's branch (held across it, it went to scratch)
        g_tab = b.mul(b.add(b.mul(wave, ci(8)), dv(opaque_lane(), 8)), ci(NXI))

        def ent_of(tap: int) -> Value:
            return b.smem_load_vN(
                tab_s, c0, b.add(g_tab, ci(tap * 64 * NXI)), dtype=I32, n=NXI
            )

        def rsrcs(ok: Value):
            return (
                b.buffer_rsrc(params["A"], b.select(ok, params["A_bytes"], c0)),
                b.buffer_rsrc(params["B"], b.select(ok, params["B_bytes"], c0)),
            )

        live_rs = (
            b.buffer_rsrc(params["A"], params["A_bytes"]),
            b.buffer_rsrc(params["B"], params["B_bytes"]),
        )

        def opaque_s(v: Value) -> Value:
            # a per-step copy: products of it are formed where used, not
            # hoisted and held across the unrolled sweep
            return b.inline_asm("", "=s,0", [v], I32, sideeffect=True)

        def dma_ops(rs, ent, chb: Value, tap: int, wo, st: int, parts: str = "xw"):
            """The DMAs as (rsrc, (LDS SGPR base, immediate), voffset,
            soffset)."""
            xr, wr = rs
            ops = []
            for i in range(NXI if "x" in parts else 0):
                dst = (sbase[st], i * 1024)
                ops.append((xr, dst, b.add(b.vec_extract(ent, i), h_gran), chb))
            if "w" not in parts:
                return ops
            if spec.wt:
                # piece q = wq * 4 + i: reduction rows 4 q + lane // 16 of the
                # flipped tap (lane parts in wo, the K tile's first output k0b)
                wv0, wv1, k0b = wo
                rsk, kk = opaque_s(RSK), opaque_s(K)
                sb = b.add(
                    b.add(
                        b.mul(b.mul(dv(chb, 2), rsk), c2),
                        b.mul(b.mul(ci(RS - 1 - tap), kk), c2),
                    ),
                    b.add(k0b, b.mul(b.mul(wq, ci(4 * NWI * 2)), RSK)),
                )
                for i in range(NWI):
                    dst = (sbase[st], IN_E * 2 + i * 1024)
                    soff = b.add(sb, b.mul(rsk, ci(8 * i))) if i else sb
                    ops.append((wr, dst, wv1 if (i >> 1) & 1 else wv0, soff))
                return ops
            woff = b.add(b.mul(ci(tap), opaque_s(C2)), chb)
            rsc = opaque_s(RSC)
            for i in range(NWI):
                wsh = ((i & 1) * 4 + (i >> 1) * 16) * 2
                dst = (sbase[st], IN_E * 2 + i * 1024)
                ops.append(
                    (wr, dst, wo, b.add(woff, b.mul(rsc, ci(wsh))) if wsh else woff)
                )
            return ops

        # this wave's first DMA slot of each stage as one SGPR (input and
        # weight slots are i KiB past it, the weights' IN_E elements further;
        # 16 held addresses cost SGPR spills and VGPR scratch)
        assert NXI == NWI
        sbase = [
            b.readfirstlane(
                b.trunc(
                    b.smem_ptr_add(
                        st_a[st],
                        b.zext(b.add(ci(gb * 2), b.mul(wq, ci(NXI * 1024))), I64),
                    ),
                    I32,
                )
            )
            for st in range(2)
        ]

        def issue(op):
            rs, dst, voff, soff = op
            _dma_lds_at(b, rs, dst[0], dst[1], voff, soff)

        if spec.wt:
            # transposing A reads of the [64 reduction rows][128 outputs] stage
            # (the main kernel's wt_frag, one 32-row half sl): one lane part,
            # group kt one XOR
            def t_lane():
                ln0 = opaque_lane()
                # the lane fields are disjoint bit ranges (BK 128): row
                # 8 tb + ti at 1024 / 128, swz = ti | 4 (tb & 1) at 16,
                # P(tj) at 4 -- shifts and masks of the (opaque, so
                # signedness-unknown) lane instead of divisions
                assert BK == 128

                def bits(sh: int, mask: int, to: int) -> Value:
                    v = b.land(b.lshr(ln0, ci(sh)) if sh else ln0, ci(mask))
                    return b.shl(v, ci(to)) if to else v

                # in bytes (each field one bit higher): the fragment's
                # XOR stays in the byte domain and the exact halving to
                # the element index folds into the read's address
                lp = b.lor(
                    b.lor(
                        b.lor(bits(4, 3, 11), bits(2, 3, 8)),
                        b.lor(bits(2, 3, 5), bits(4, 1, 7)),
                    ),
                    b.lor(bits(0, 1, 4), bits(1, 1, 3)),
                )
                return b.lor(
                    lp, b.shl(b.add(ci(gb + IN_E), b.mul(sl, ci(32 * BK))), c1)
                )

            t_l0 = t_lane()
            frags_in = frags

            def frags(st: int):
                _, bb = frags_in(st, a_frags=False)
                a = []
                # the kernel's lane part (one VGPR), laundered per step
                t_l = b.inline_asm("", "=v,0", [t_l0], I32, sideeffect=True)
                for kt in range(KT):
                    base = b.lshr(b.xor(t_l, ci(kt * 32)) if kt else t_l, c1)
                    lo = b.ds_read_tr16_b64(st_s[st], c0, base, dtype=io_type)
                    hi = b.ds_read_tr16_b64(
                        st_s[st], c0, b.add(base, ci(4 * BK)), dtype=io_type
                    )
                    a.append(b.vec_concat(lo, hi))
                return a, bb

        def step2(accs, st: int, tgt):
            """Step on stage st; tgt = (rsrcs, chunk bytes, tap, wo) of step j + 2
            (DMAs into stage st during the MFMA phase)."""
            rs, chb, tap, wo = tgt
            ent = ent_of(tap)
            a, bb = frags(st)
            b.s_waitcnt(lgkmcnt=0)
            # the DMA offsets formed before the phase barrier
            ops = dma_ops(rs, ent, chb, tap, wo, st)
            b.sched_barrier(0)
            bar(1)
            every = (_XT * KT) // len(ops)
            accs = list(accs)
            j = 0
            for xt in range(_XT):
                for kt in range(KT):
                    if j % every == 0 and j // every < len(ops):
                        b.sched_barrier(0)
                        issue(ops[j // every])
                        b.sched_barrier(0)
                    i = xt * KT + kt
                    accs[i] = _mfma(b, dtype, "16x16x32", a[kt], bb[xt], accs[i])
                    j += 1
            b.sched_barrier(0)
            b.s_waitcnt(vmcnt=NDMA)
            bar(0)
            return accs

        outer = b.scf_for_iter(
            c0, rounds, c1, [("o_d", c0)], iv_name="rnd", elide_trailing_barrier=False
        )
        with outer as (rnd, oc):
            t = tile_of(rnd)
            info = item_info(t)
            head()
            if spec.wt:
                # slot l % 16 of row 4 q + l // 16 takes group (l % 16) // 2 ^
                # (l // 16) ^ 4 ((q >> 1) & 1) (q = 4 wq + i: bit 1 of i); the
                # columns past K read other rows' weights, of outputs never
                # stored
                wl = opaque_lane()
                w16, s16 = dv(wl, 16), md(wl, 16)
                cb = b.add(
                    b.mul(b.xor(dv(s16, 2), w16), ci(16)), b.mul(md(s16, 2), ci(8))
                )
                wrow = b.mul(w16, RSK)
                wo = (
                    b.mul(b.add(wrow, cb), c2),
                    b.mul(b.add(wrow, b.xor(cb, ci(64))), c2),
                    b.mul(info["tk"], ci(BK * 2)),
                )
            else:
                wo = wlane(info["tk"])
            ch0, ch1 = info["ch0"], info["ch1"]
            NQ = dv(b.add(b.sub(ch1, ch0), c1), 2)  # chunks per group
            chA = b.add(ch0, ci(g))

            def slot(base_q: Value, cbase: Value, m: int):
                q = m // RS
                ch = b.add(cbase, ci(2 * q))
                ok = b.land(b.cmp_lt(b.add(base_q, ci(q)), NQ), b.cmp_lt(ch, ch1))
                return ok, b.mul(ch, ci(_CK * 2)), m % RS

            # the previous item's reduce ended on a barrier: the stages are
            # free. The weight DMAs of steps 0 and 1 need no table: they go
            # first and land while this wave builds its table rows.
            pro = [slot(c0, chA, m) for m in range(2)]
            for m, (ok, chb, tap) in enumerate(pro):
                for op in dma_ops(rsrcs(ok), None, chb, tap, wo, m % 2, parts="w"):
                    issue(op)
            build_tab(info["tx"])
            for m, (ok, chb, tap) in enumerate(pro):
                ent = ent_of(tap)
                b.s_waitcnt(lgkmcnt=0)
                for op in dma_ops(rsrcs(ok), ent, chb, tap, wo, m % 2, parts="x"):
                    issue(op)
            b.s_waitcnt(vmcnt=NXI)
            b.s_barrier_bare()
            if g == 1:
                b.s_barrier_bare()  # group 1 runs one barrier behind
            b.sched_barrier(0)
            accs = [zero] * nacc
            NP = dv(NQ, 2)
            loop = b.scf_for_iter(
                c0,
                NP,
                c1,
                [(f"a{i}", v) for i, v in enumerate(accs)],
                iv_name="cpair",
                elide_trailing_barrier=False,
            )
            with loop as (qp, lc):
                la = list(lc[:nacc])
                q0 = b.mul(qp, c2)
                cA = b.add(chA, b.mul(qp, ci(4)))
                slots = {}
                for m in range(2, 2 * RS + 2):
                    q = m // RS
                    if q not in slots:
                        ok, chb, _ = slot(q0, cA, m)
                        # the pair's first chunk is always live
                        slots[q] = (rsrcs(ok) if q else live_rs, chb)
                for k in range(2 * RS):
                    m = k + 2
                    rs, chb = slots[m // RS]
                    la = step2(la, k % 2, (rs, chb, m % RS, wo))
                b.scf_yield(*la)
            accs = list(loop.results)
            tail = b.scf_if_else(b.cmp_eq(md(NQ, 2), c1), [vec4] * nacc)
            with tail as (then_ctx, else_ctx):
                with then_ctx:
                    ta = accs
                    qT = b.mul(NP, c2)
                    cT = b.add(chA, b.mul(NP, ci(4)))
                    slots = {}
                    for m in range(2, RS + 2):
                        q = m // RS
                        if q not in slots:
                            ok, chb, _ = slot(qT, cT, m)
                            slots[q] = (rsrcs(ok), chb)
                    for k in range(RS):
                        m = k + 2
                        rs, chb = slots[m // RS]
                        ta = step2(ta, k % 2, (rs, chb, m % RS, wo))
                    b.scf_yield(*ta)
                with else_ctx:
                    b.scf_yield(*accs)
            accs = list(tail.results)
            # the last steps' (dead) DMAs land before any wave reuses the stages
            b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            b.s_setprio(0)
            if g == 0:
                b.s_barrier_bare()  # wait out group 1's last MFMA phase
            reduce_store(accs, info)
            b.sched_barrier(0)

            b.scf_yield(c0)

    def run(g: int):
        gb = g * GE  # this group's half of a stage, elements
        # the fragment-read bases (re-formed at each item's head by
        # item_head, from an opaque lane: no kernel-long copy to spill)
        h_xin = b.add(
            b.add(ci(gb), b.mul(b.add(b.mul(xi, ci(64)), h_r16), ci(_CK))), h_sl
        )
        h_kw = b.add(ci(gb + IN_E), b.add(b.mul(h_r16, ci(_CK)), h_sl))
        hh = {"xin": h_xin, "kw": h_kw}

        def item_head() -> None:
            ln2 = opaque_lane()
            r2, q2 = md(ln2, 16), dv(ln2, 16)
            sl2 = b.mul(b.xor(b.add(b.mul(sl, ci(4)), q2), md(r2, 8)), ci(8))
            hh["xin"] = b.add(
                b.add(ci(gb), b.mul(b.add(b.mul(xi, ci(64)), r2), ci(_CK))), sl2
            )
            hh["kw"] = b.add(ci(gb + IN_E), b.add(b.mul(r2, ci(_CK)), sl2))

        def wlane(tk: Value) -> Value:
            ln = opaque_lane()
            l8 = dv(ln, 8)
            c8 = b.xor(md(ln, 8), l8)
            chan = b.add(b.mul(dv(l8, 4), ci(8)), md(l8, 4))
            k = b.add(b.add(b.mul(tk, ci(BK)), b.mul(wq, ci(NWI * 8))), chan)
            return b.mul(b.add(b.mul(k, RSC), b.mul(c8, ci(8))), c2)

        def frags(st: int, a_frags: bool = True):
            a = (
                [
                    b.smem_load_vN(
                        st_s[st],
                        c0,
                        b.add(hh["kw"], ci(kt * 16 * _CK)),
                        dtype=io_type,
                        n=8,
                    )
                    for kt in range(KT)
                ]
                if a_frags
                else []
            )
            bb = [
                b.smem_load_vN(
                    st_s[st],
                    c0,
                    b.add(hh["xin"], ci(xt * 16 * _CK)),
                    dtype=io_type,
                    n=8,
                )
                for xt in range(_XT)
            ]
            return a, bb

        def bar(prio: int):
            b.s_setprio(prio)
            b.s_barrier_bare()
            b.sched_barrier(0)

        nacc = _XT * KT
        zero = b.zero_vec_f32(4)
        vec4 = zero.type

        i32x2 = VectorType(I32, 2)
        i32x4 = VectorType(I32, 4)
        f32x4 = vec4
        half8 = VectorType(BF16 if dtype == "bf16" else F16, 8)
        d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])

        def reduce_store(accs, info):
            ln = opaque_lane()
            q, x16 = dv(ln, 16), md(ln, 16)
            ubase = b.add(b.mul(md(q, 2), c2), dv(q, 2))
            # this wave's partial region: stage g, slot (xi, sl)
            rbase = b.mul(b.add(b.mul(xi, c2), sl), ci(PART))
            # reader: thread -> (pixel half, two (row, 8-channel) items)
            rxi = dv(tid, 256)
            t2 = md(tid, 256)
            tx, tk = info["tx"], info["tk"]
            if SP > 1:
                slab = b.mul(b.mul(NX, K), ci(4))
                sbase = b.global_ptr_offset(
                    params["D"], b.mul(b.zext(info["pc"], I64), b.zext(slab, I64))
                )
                s_rsrc = b.buffer_rsrc(sbase, slab)
            for rr in range(2):
                for xl in range(2):
                    xt = 2 * rr + xl
                    rowv = b.add(ci(xl * 16), x16)
                    for kt in range(KT):
                        u = b.add(ci(kt * 4), ubase)
                        el = b.add(
                            rbase,
                            b.add(
                                b.mul(rowv, ci(BK * 2)), b.mul(b.xor(u, rowv), ci(8))
                            ),
                        )
                        val = b.vec_bitcast(accs[xt * KT + kt], half8)
                        b.smem_store_vN(st_s[g], [c0, el], val, 8)
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                for j in range(2):
                    it = b.add(t2, ci(256 * j))
                    row = dv(it, 16)
                    k8 = md(it, 16)
                    sums = []
                    for hh in range(2):
                        u = b.add(b.mul(k8, c2), ci(hh))
                        off = b.add(b.mul(row, ci(BK * 2)), b.mul(b.xor(u, row), ci(8)))
                        acc = None
                        for gg in range(2):
                            for ss in range(2):
                                el = b.add(
                                    b.mul(b.add(b.mul(rxi, c2), ci(ss)), ci(PART)), off
                                )
                                v = b.vec_bitcast(
                                    b.smem_load_vN(
                                        st_s[gg], c0, el, dtype=io_type, n=8
                                    ),
                                    f32x4,
                                )
                                acc = v if acc is None else b.fadd(acc, v)
                        sums.append(acc)
                    x = b.add(
                        b.mul(tx, ci(BX)),
                        b.add(b.mul(rxi, ci(64)), b.add(ci(32 * rr), row)),
                    )
                    k = b.add(b.mul(tk, ci(BK)), b.mul(k8, ci(8)))
                    ok = b.land(b.cmp_lt(x, NX), b.cmp_lt(k, K))
                    if SP == 1:
                        lo = b.vec_bitcast(_trunc_f32(b, dtype, sums[0]), i32x2)
                        hi = b.vec_bitcast(_trunc_f32(b, dtype, sums[1]), i32x2)
                        val = b.vec_bitcast(
                            b.vec_pack(
                                [
                                    b.vec_extract(lo, 0),
                                    b.vec_extract(lo, 1),
                                    b.vec_extract(hi, 0),
                                    b.vec_extract(hi, 1),
                                ],
                                I32,
                            ),
                            half8,
                        )
                        off = b.select(ok, b.mul(b.add(b.mul(x, K), k), c2), oob)
                        _buf_store_vN(b, dtype, d_rsrc, off, c0, val, 4)
                    else:
                        off = b.select(ok, b.mul(b.add(b.mul(x, K), k), ci(4)), oob)
                        b.buffer_store_vN_f32(s_rsrc, off, c0, sums[0], 4)
                        off2 = b.select(ok, b.add(off, ci(16)), oob)
                        b.buffer_store_vN_f32(s_rsrc, off2, c0, sums[1], 4)
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

        run_pf2(g, gb, wlane, frags, bar, reduce_store, zero, vec4, nacc, item_head)

    with b.scf_if_else(b.cmp_eq(dv(wave, 4), c0)) as (then_ctx, else_ctx):
        with then_ctx:
            run(0)
        with else_ctx:
            run(1)
    return b.kernel


def _build_cs2(spec: FwdFlexMfmaSpec, arch: str) -> KernelDef:
    """The ``cs == 2`` kernel: X128 x K256 tiles, the reduction split two
    ways inside the block.

    Both wave groups read every step (one tap and one 64-channel chunk) from
    one of three shared stages; group g takes the chunk's 32-channel half g,
    and wave (xi, ki) of a group 64 pixels x 128 channels of the tile: 4 x 8
    MFMAs per step from 12 fragment reads. Each group stages half of a step
    (wave w: input rows 16 w.., weight rows 32 w..), group g the half of step
    j + 1 + g while it reads step j, issued between its fragment reads (the
    stage of step j + 1 + g was last read at step j - 2 + g, and group 1 runs
    one barrier phase behind group 0); a step's copies land by the end of its
    MFMA phase. Every filter tap runs (padding taps load zeros); the taps of
    a chunk are unrolled, so a tap count that is a multiple of 3 keeps the
    stage of each tap fixed. A tile is drained at its end: the two groups'
    partial sums of each pixel half meet in the stage memory (fp32, two
    rounds of 32 pixels per wave) and one pass sums, converts and stores
    them (or writes the fp32 slab of a split-C piece)."""
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    BX, BK, SP, KT = 128, 256, spec.split_c, 8
    NXI, NWI = 2, 4  # DMA instructions per wave per step (its group's half)
    R, S = p.KH, p.KW
    RS = R * S
    SH, PH, PW, DH, DW = p.stride, p.PAD, p.PAD_W, p.dil_h, p.dil_w
    IN_E = BX * _CK  # input elements of a stage
    GE = (BX + BK) * _CK  # elements of a stage
    PART = 32 * 128 * 2  # one partial region (32 pixels x 128 fp32) in elements

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    def dv(a: Value, n: int) -> Value:
        return b.div(a, ci(n))

    def md(a: Value, n: int) -> Value:
        return b.mod(a, ci(n))

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_N, p_Hi, p_Wi = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]

    tid = b.thread_id_x()
    lane = md(tid, _WAVE)
    wave = b.readfirstlane(dv(tid, _WAVE))
    wq = md(wave, 4)
    xi = md(wq, 2)
    ki = dv(wq, 2)

    def opaque_lane() -> Value:
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    def opaque_s(v: Value) -> Value:
        return b.inline_asm("", "=s,0", [v], I32, sideeffect=True)

    hw = b.mul(p_Ho, p_Wo)
    NX = b.mul(p_N, hw)
    nxt = b.div(b.add(NX, ci(BX - 1)), ci(BX))
    nkt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    TI = b.mul(b.mul(nxt, nkt), ci(SP))
    G = b.smin(TI, ci(PERSIST))
    nch = b.div(C, ci(_CK))
    RSC = b.mul(ci(RS), C)
    C2 = b.mul(C, c2)
    blk = b.block_id_x()
    if spec.xcd:
        g8 = b.div(G, ci(8))
        xr = b.add(b.mul(b.mod(blk, ci(8)), g8), b.div(blk, ci(8)))
        blk = b.select(b.cmp_lt(blk, b.mul(g8, ci(8))), xr, blk)
    full, part = b.div(TI, G), b.mod(TI, G)
    last_ok = b.cmp_lt(b.mod(b.add(blk, b.mul(full, ci(ROT))), G), part)
    rounds = b.add(full, b.select(last_ok, c1, c0))

    def tile_of(i: Value) -> Value:
        return b.add(b.mul(i, G), b.mod(b.add(blk, b.mul(i, ci(ROT))), G))

    def item_info(t: Value) -> Dict[str, Value]:
        tl = b.div(t, ci(SP))
        pc = b.mod(t, ci(SP))
        ch0 = b.div(b.mul(pc, nch), ci(SP))
        ch1 = b.div(b.mul(b.add(pc, c1), nch), ci(SP))
        if spec.kmaj:
            return dict(pc=pc, tx=b.mod(tl, nxt), tk=b.div(tl, nxt), ch0=ch0, ch1=ch1)
        return dict(pc=pc, tx=b.div(tl, nkt), tk=b.mod(tl, nkt), ch0=ch0, ch1=ch1)

    # DMA offset table, per wave: [tap][wave][pixel of 8][instr]
    RSP = _tab_taps(spec)
    tab_s = b.smem_alloc(I32, [1, RSP * 64 * NXI], name_hint="flex_tab")

    def build_tab(tx: Value) -> None:
        ln = opaque_lane()
        l8 = dv(ln, 8)
        c = md(ln, 8)
        pix = []
        for i in range(NXI):
            x = b.add(
                b.mul(tx, ci(BX)),
                b.add(b.mul(b.add(b.mul(wave, ci(NXI)), ci(i)), ci(8)), l8),
            )
            n = b.div(x, hw)
            rem = b.sub(x, b.mul(n, hw))
            ho = b.div(rem, p_Wo)
            wo = b.sub(rem, b.mul(ho, p_Wo))
            hb = b.sub(b.mul(ho, ci(SH)), ci(PH))
            wb = b.sub(b.mul(wo, ci(SH)), ci(PW))
            base = b.add(b.mul(b.add(b.mul(n, p_Hi), hb), p_Wi), wb)
            pix.append((b.cmp_lt(x, NX), hb, wb, base))
        for kk in range(RSP // 8):
            t = b.add(c, ci(8 * kk))
            r = b.div(t, ci(S))
            s_ = b.sub(t, b.mul(r, ci(S)))
            dh, dw = b.mul(r, ci(DH)), b.mul(s_, ci(DW))
            tok = b.cmp_lt(t, ci(RS))
            toff = b.add(b.mul(dh, p_Wi), dw)
            ents = []
            for xok, hb, wb, base in pix:
                ok = b.land(
                    b.land(tok, xok),
                    b.land(
                        b.cmp_ult(b.add(hb, dh), p_Hi), b.cmp_ult(b.add(wb, dw), p_Wi)
                    ),
                )
                ents.append(b.select(ok, b.mul(b.mul(b.add(base, toff), C), c2), oob))
            idx = b.mul(b.add(b.mul(b.add(b.mul(t, ci(8)), wave), ci(8)), l8), ci(NXI))
            b.smem_store_vN(tab_s, [c0, idx], b.vec_pack(ents, I32), NXI)

    # one allocation per stage: the DMA into one stage and the fragment reads
    # of another stay apart for the waitcnt pass
    st_s = [b.smem_alloc(io_type, [1, GE], name_hint=f"flexc2s{i}") for i in range(3)]
    st_a = [b.smem_addr_of(v) for v in st_s]
    # this wave's first input and weight DMA slots of each stage
    sb_in, sb_w = [
        [
            b.readfirstlane(
                b.trunc(
                    b.smem_ptr_add(
                        st_a[st], b.zext(b.add(ci(o), b.mul(wave, ci(n))), I64)
                    ),
                    I32,
                )
            )
            for st in range(3)
        ]
        for o, n in ((0, NXI * 1024), (IN_E * 2, NWI * 1024))
    ]

    r16 = md(lane, 16)
    h_gran = b.mul(b.xor(md(lane, 8), dv(lane, 8)), ci(16))
    # the group's 32-channel half of the chunk (granules 4 g.. of a row)
    h_sl = b.mul(
        b.xor(b.add(b.mul(dv(wave, 4), ci(4)), dv(lane, 16)), md(r16, 8)), ci(8)
    )
    h_xin = b.add(b.mul(b.add(b.mul(xi, ci(64)), r16), ci(_CK)), h_sl)
    h_kw = b.add(ci(IN_E), b.add(b.mul(b.add(b.mul(ki, ci(128)), r16), ci(_CK)), h_sl))
    h_tab = b.mul(b.add(b.mul(wave, ci(8)), dv(lane, 8)), ci(NXI))

    RSK = b.mul(ci(RS), K)  # wt: a reduction row of the KRSC filter
    GQ = dv(wave, 4)  # the group: its 32-row half of a step's 64

    def wlane_wt(tk: Value):
        """wt: the weight DMA lane parts (piece q = 4 wave + i, half q // 16
        = the group, rows 4 (q % 16) + lane // 16; slot l % 16 of a row takes
        group (l % 16) // 2 ^ (l // 16) ^ 4 ((q >> 1) & 1), i.e. bit 1 of i)
        and the item's first output (bytes, with the wave's half)."""
        wl = opaque_lane()
        w16 = b.land(b.lshr(wl, ci(4)), ci(3))
        s16 = b.land(wl, ci(15))
        cb = b.add(
            b.mul(b.xor(b.lshr(s16, c1), w16), ci(16)), b.mul(b.land(s16, c1), ci(8))
        )
        wrow = b.mul(w16, RSK)
        return (
            b.mul(b.add(wrow, cb), c2),
            b.mul(b.add(wrow, b.xor(cb, ci(64))), c2),
            b.mul(b.add(b.mul(tk, ci(BK)), b.mul(GQ, ci(128))), c2),
        )

    def t_lane() -> Value:
        """wt: a transposing read's element offset in the stage (lane part
        from shifts and masks, formed per step): weight half ki (8192), the
        group's 32-row half g (4096), row 8 tb + ti of the half (128 each),
        group slot swz = ti | 4 (tb & 1) (16), output quad P(tj) (4)."""
        ln0 = opaque_lane()

        def bits(sh: int, mask: int, to: int) -> Value:
            v = b.land(b.lshr(ln0, ci(sh)) if sh else ln0, ci(mask))
            return b.shl(v, ci(to)) if to else v

        # in bytes (every field one bit higher): a fragment's XOR stays in
        # the byte domain, and the element index (byte offset / 2, exact)
        # folds back into the read's address
        lp = b.lor(
            b.lor(
                b.lor(bits(4, 3, 11), bits(2, 3, 8)),
                b.lor(bits(2, 3, 5), bits(4, 1, 7)),
            ),
            b.lor(bits(0, 1, 4), bits(1, 1, 3)),
        )
        return b.lor(
            lp,
            b.shl(b.add(ci(IN_E), b.add(b.mul(ki, ci(8192)), b.mul(GQ, ci(4096)))), c1),
        )

    # wt: the transposing reads' lane part formed once per kernel (one VGPR)
    t_l0 = t_lane() if spec.wt else None

    def wlane(tk: Value) -> Value:
        if spec.wt:
            return wlane_wt(tk)
        ln = opaque_lane()
        l8 = dv(ln, 8)
        c8 = b.xor(md(ln, 8), l8)
        chan = b.add(b.mul(dv(l8, 4), ci(8)), md(l8, 4))
        k = b.add(b.add(b.mul(tk, ci(BK)), b.mul(wave, ci(NWI * 8))), chan)
        return b.mul(b.add(b.mul(k, RSC), b.mul(c8, ci(8))), c2)

    def ent_of(tap: int) -> Value:
        return b.smem_load_vN(
            tab_s, c0, b.add(h_tab, ci(tap * 64 * NXI)), dtype=I32, n=NXI
        )

    xr = b.buffer_rsrc(params["A"], params["A_bytes"])
    wr = b.buffer_rsrc(params["B"], params["B_bytes"])

    def dma_ops(ent: Value, chb: Value, tap: int, wo: Value, st: int):
        """The wave's 6 DMAs of (tap, chunk at chb bytes) into stage st."""
        ops = [
            (xr, sb_in[st], i * 1024, b.add(b.vec_extract(ent, i), h_gran), chb)
            for i in range(NXI)
        ]
        if spec.wt:
            # rows chb / 2 + 16 wq + 4 i + lane // 16 of the flipped tap
            wv0, wv1, k0b = wo
            rsk, kk = opaque_s(RSK), opaque_s(K)
            sb = b.add(
                b.add(
                    b.mul(b.mul(dv(chb, 2), rsk), c2),
                    b.mul(b.mul(ci(RS - 1 - tap), kk), c2),
                ),
                b.add(k0b, b.mul(b.mul(wq, ci(32)), RSK)),
            )
            for i in range(NWI):
                soff = b.add(sb, b.mul(rsk, ci(8 * i))) if i else sb
                ops.append((wr, sb_w[st], i * 1024, wv1 if (i >> 1) & 1 else wv0, soff))
            return ops
        woff = b.add(b.mul(ci(tap), opaque_s(C2)), chb)
        rsc = opaque_s(RSC)
        for i in range(NWI):
            wsh = ((i & 1) * 4 + (i >> 1) * 16) * 2
            ops.append(
                (
                    wr,
                    sb_w[st],
                    i * 1024,
                    wo,
                    b.add(woff, b.mul(rsc, ci(wsh))) if wsh else woff,
                )
            )
        return ops

    def frags(st: int):
        a = [
            b.smem_load_vN(
                st_s[st], c0, b.add(h_kw, ci(kt * 16 * _CK)), dtype=io_type, n=8
            )
            for kt in range(KT)
        ]
        bb = [
            b.smem_load_vN(
                st_s[st], c0, b.add(h_xin, ci(xt * 16 * _CK)), dtype=io_type, n=8
            )
            for xt in range(_XT)
        ]
        return a, bb

    def bar(prio: int):
        b.s_setprio(prio)
        b.s_barrier_bare()
        b.sched_barrier(0)

    nacc = _XT * KT
    zero = b.zero_vec_f32(4)
    vec4 = zero.type

    def step(accs, tap: int, ops, ent_next):
        """One step on stage tap % 3: the fragment reads with this wave's
        copies (ops, or none) between them, the table entries of the next
        step's copies (tap ent_next, or None), then the MFMA phase."""
        st = tap % 3
        e2 = ent_of(ent_next) if ent_next is not None else None
        a, bb = [None] * KT, [None] * _XT
        reads = [("a", kt) for kt in range(KT)] + [("b", xt) for xt in range(_XT)]
        per = ceil_div(len(reads), max(1, len(ops)))
        tl = None
        if spec.wt:
            # the kernel's lane part, laundered per step (else the compiler
            # keeps the 8 XORed addresses live across the sweep)
            tl = b.inline_asm("", "=v,0", [t_l0], I32, sideeffect=True)
        for j, (kind, i) in enumerate(reads):
            if j % per == 0 and j // per < len(ops):
                _dma_lds_at(b, *ops[j // per])
            if kind == "a" and spec.wt:
                # two transposing reads (rows 8 tb + ti and + 4), group i one XOR
                base = b.lshr(b.xor(tl, ci(i * 32)) if i else tl, c1)
                lo = b.ds_read_tr16_b64(st_s[st], c0, base, dtype=io_type)
                hi = b.ds_read_tr16_b64(
                    st_s[st], c0, b.add(base, ci(4 * 128)), dtype=io_type
                )
                a[i] = b.vec_concat(lo, hi)
            elif kind == "a":
                a[i] = b.smem_load_vN(
                    st_s[st], c0, b.add(h_kw, ci(i * 16 * _CK)), dtype=io_type, n=8
                )
            else:
                bb[i] = b.smem_load_vN(
                    st_s[st], c0, b.add(h_xin, ci(i * 16 * _CK)), dtype=io_type, n=8
                )
        b.s_waitcnt(lgkmcnt=0)
        b.sched_barrier(0)
        bar(1)
        accs = list(accs)
        for xt in range(_XT):
            for kt in range(KT):
                i = xt * KT + kt
                accs[i] = _mfma(b, dtype, "16x16x32", a[kt], bb[xt], accs[i])
        b.sched_barrier(0)
        # this step's copies land before any wave's next memory phase
        b.s_waitcnt(vmcnt=0)
        bar(0)
        return accs, e2

    i32x2 = VectorType(I32, 2)
    f32x4 = vec4
    half8 = VectorType(BF16 if dtype == "bf16" else F16, 8)
    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])
    WREG = 16 * 128 * 2  # one wave's partial region (16 pixels x 128 fp32), elements

    def reduce_store(accs, info, g: int):
        """Four rounds (16-pixel tile xt): every wave writes its partial sums
        of xt to region wq of its group's stage, then each (pixel, 8
        channels) item sums the two groups' regions and is stored."""
        ln = opaque_lane()
        q, x16 = dv(ln, 16), md(ln, 16)
        ubase = b.add(b.mul(md(q, 2), c2), dv(q, 2))
        wbase = b.mul(wq, ci(WREG))
        t0 = b.add(b.mul(wave, ci(_WAVE)), opaque_lane())
        tx, tk = info["tx"], info["tk"]
        if SP > 1:
            slab = b.mul(b.mul(NX, K), ci(4))
            sbase = b.global_ptr_offset(
                params["D"], b.mul(b.zext(info["pc"], I64), b.zext(slab, I64))
            )
            s_rsrc = b.buffer_rsrc(sbase, slab)
        for xt in range(_XT):
            for kt in range(KT):
                u = b.add(ci(kt * 4), ubase)
                el = b.add(
                    wbase, b.add(b.mul(x16, ci(256)), b.mul(b.xor(u, x16), ci(8)))
                )
                b.smem_store_vN(
                    st_s[g], [c0, el], b.vec_bitcast(accs[xt * KT + kt], half8), 8
                )
            b.s_waitcnt(lgkmcnt=0)
            b.s_barrier_bare()
            for j in range(2):
                it = b.add(t0, ci(512 * j))
                wp = dv(it, 256)  # the region's wave position (xi, ki)
                row = md(dv(it, 16), 16)
                k8 = md(it, 16)
                sums = []
                for hh in range(2):
                    u = b.add(b.mul(k8, c2), ci(hh))
                    off = b.add(
                        b.mul(wp, ci(WREG)),
                        b.add(b.mul(row, ci(256)), b.mul(b.xor(u, row), ci(8))),
                    )
                    v0 = b.vec_bitcast(
                        b.smem_load_vN(st_s[0], c0, off, dtype=io_type, n=8), f32x4
                    )
                    v1 = b.vec_bitcast(
                        b.smem_load_vN(st_s[1], c0, off, dtype=io_type, n=8), f32x4
                    )
                    sums.append(b.fadd(v0, v1))
                x = b.add(
                    b.mul(tx, ci(BX)),
                    b.add(b.mul(md(wp, 2), ci(64)), b.add(ci(16 * xt), row)),
                )
                k = b.add(
                    b.mul(tk, ci(BK)),
                    b.add(b.mul(dv(wp, 2), ci(128)), b.mul(k8, ci(8))),
                )
                ok = b.land(b.cmp_lt(x, NX), b.cmp_lt(k, K))
                if SP == 1:
                    lo = b.vec_bitcast(_trunc_f32(b, dtype, sums[0]), i32x2)
                    hi = b.vec_bitcast(_trunc_f32(b, dtype, sums[1]), i32x2)
                    val = b.vec_bitcast(
                        b.vec_pack(
                            [
                                b.vec_extract(lo, 0),
                                b.vec_extract(lo, 1),
                                b.vec_extract(hi, 0),
                                b.vec_extract(hi, 1),
                            ],
                            I32,
                        ),
                        half8,
                    )
                    off = b.select(ok, b.mul(b.add(b.mul(x, K), k), c2), oob)
                    _buf_store_vN(b, dtype, d_rsrc, off, c0, val, 4)
                else:
                    off = b.select(ok, b.mul(b.add(b.mul(x, K), k), ci(4)), oob)
                    b.buffer_store_vN_f32(s_rsrc, off, c0, sums[0], 4)
                    off2 = b.select(ok, b.add(off, ci(16)), oob)
                    b.buffer_store_vN_f32(s_rsrc, off2, c0, sums[1], 4)
            b.s_waitcnt(lgkmcnt=0)
            b.s_barrier_bare()

    def run(g: int):
        """Group g's items: group g stages its half of step j + 1 + g during
        step j (its own entries read a step ahead)."""
        outer = b.scf_for_iter(
            c0, rounds, c1, [("o_d", c0)], iv_name="rnd", elide_trailing_barrier=False
        )
        with outer as (rnd, oc):
            t = tile_of(rnd)
            info = item_info(t)
            wo = wlane(info["tk"])
            ch0, ch1 = info["ch0"], info["ch1"]
            # the previous item's reduce ended on a barrier: the stages are
            # free; a wave's table rows are its own
            build_tab(info["tx"])
            cb0 = b.mul(ch0, ci(_CK * 2))
            pro = [0, 1] if g == 1 else [0]  # group 1 also stages its half of step 1
            for m in pro:
                e = ent_of(m)
                b.s_waitcnt(lgkmcnt=0)
                for op in dma_ops(e, cb0, m, wo, m):
                    _dma_lds_at(b, *op)
            ent = ent_of(1 + g)  # the first step's copies (tap 1 + g of each chunk)
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.s_barrier_bare()
            if g == 1:
                b.s_barrier_bare()  # group 1 runs one barrier behind
            b.sched_barrier(0)
            accs = [zero] * nacc
            loop = b.scf_for_iter(
                c0,
                b.sub(b.sub(ch1, ch0), c1),
                c1,
                [(f"a{i}", v) for i, v in enumerate(accs)],
                iv_name="chunk",
                elide_trailing_barrier=False,
            )
            with loop as (cc, lc):
                la = list(lc[:nacc])
                chb = b.mul(b.add(ch0, cc), ci(_CK * 2))
                chn = b.add(chb, ci(_CK * 2))
                e = ent
                for tp in range(RS):
                    tt = tp + 1 + g
                    ops = dma_ops(e, chb if tt < RS else chn, tt % RS, wo, tt % 3)
                    la, e = step(
                        la, tp, ops, (tp + 2 + g) % RS if tp + 1 < RS else None
                    )
                b.scf_yield(*la)
            la = list(loop.results)
            # the last chunk: no copies past the item's last step
            chb = b.mul(b.sub(ch1, c1), ci(_CK * 2))
            e = ent
            for tp in range(RS):
                tt = tp + 1 + g
                ops = dma_ops(e, chb, tt, wo, tt % 3) if tt < RS else []
                la, e = step(la, tp, ops, tp + 2 + g if tp + 2 + g < RS else None)
            b.s_setprio(0)
            if g == 0:
                b.s_barrier_bare()  # wait out group 1's last MFMA phase
            reduce_store(la, info, g)
            b.sched_barrier(0)
            b.scf_yield(c0)

    with b.scf_if_else(b.cmp_eq(dv(wave, 4), c0)) as (then_ctx, else_ctx):
        with then_ctx:
            run(0)
        with else_ctx:
            run(1)
    return b.kernel
