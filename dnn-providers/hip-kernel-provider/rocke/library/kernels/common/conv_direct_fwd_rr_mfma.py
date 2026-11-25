# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Rows-in-registers direct forward 3x3 convolution on wave64 MFMA (gfx950).

A dense NHWC 3x3 stride-1 "same" convolution with large channel counts:

* **Rows in registers.** A block computes a 16 x 32 pixel x 128 output channel
  tile with 8 waves (2 row halves x 2 column halves x 2 channel halves); a wave
  owns 8 rows x 16 pixels x 64 channels (32 fp32x4 accumulators). One *step*
  is one filter column and one 32-channel half of a 64-channel chunk: the wave
  reads the 10 input rows its 8 output rows need (8 + 2 halo) from LDS once
  and reuses each of them for the 3 filter rows -- 96 ``mfma_f32_16x16x32``
  from 10 LDS fragments.
* **Weights straight to VGPRs.** A helper kernel (:func:`build_fwd_rr_wfmt`)
  re-lays the KRSC filter into MFMA A-operand tiles (1 KiB per 16 x 32 tile,
  lane-major), so a step loads its 12 tiles with one 16-byte buffer load per
  lane each; the 4 waves of a channel half read the same tiles (L1 hits).
  LDS holds only the double-buffered input halo tile (18 x 34 pixels x 64
  channels, granule-swizzled, filled by ``buffer_load ... lds``).
* **Ping-pong.** The waves of channel half 1 run one barrier behind those of
  half 0; every step is ``weights + DMA + LDS reads; waitcnt; setprio 1;
  barrier; MFMAs; setprio 0; barrier``, so one half's memory phase overlaps
  the other half's MFMAs. The stagger is kept across tiles.
* **Persistent grid.** ``PERSIST`` blocks walk the tiles; the last chunk of a
  tile DMAs the next tile's first chunk and stores the tile (below), and the
  next tile's first weights load while those stores are in flight.
* **Register epilogue.** The output-channel order inside each 16-channel MFMA
  tile is interleaved by the weight format so that one ``permlane32_swap``
  per dword pair gives every lane 8 consecutive channels of one pixel, and a
  row-shift DPP exchange between pixel halves makes each 16-byte store cover
  8 whole pixels of the wave's 64 channels (full 128-byte lines). No LDS, no
  barrier: a channel half writes its tile while the other half computes.
  The tile's last step runs its MFMAs row by row and issues row ``p - 1``'s
  conversion and stores (non-temporal, except K96) between row ``p``'s MFMAs, so the
  epilogue overlaps the matrix pipe instead of lengthening the next memory
  phase; the stores stay in flight past the chunk's final wait.

The DMA addressing is scalar: a wave always DMAs the same 8-pixel quarter of
a halo row (the per-lane part is one VGPR per tile, the row goes in
``soffset``); the 2 right-hand halo columns are 256-byte dword DMAs. Rows
outside the image use a zero-size buffer descriptor (zero fill).

AOT: batch, extents, C and K are kernargs; the filter, padding and dtype are
baked. Limits: C % 128 == 0, K % 128 == 0. Partial row / pixel tiles at the
bottom and right edges read zeros past the image and mask their stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.core.ir import VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _buf_load_half,
    _buf_load_vN,
    _buf_store_vN,
    _io_type,
    _mfma,
    emit_direct_params,
    problem_dims_reason,
)

__all__ = [
    "FwdRrMfmaSpec",
    "FwdRrWfmtSpec",
    "build_fwd_rr_mfma",
    "build_fwd_rr_wfmt",
    "fwd_rr_format_weights_reference",
    "fwd_rr_mfma_reason",
    "fwd_rr_weight_bytes",
]

_WAVE = 64
_CK = 64  # channels per chunk
_TP, _TQ, _TK = 16, 32, 128  # block tile: rows x pixels x output channels
_WP = 8  # output rows per wave
_NIN = _WP + 2  # input row fragments per step
_TR, _TC = _TP + 2, _TQ + 2  # halo tile (3x3, pad 1)
_PX_B = _CK * 2  # bytes per staged pixel (64 channels)
_BUF_B = 80 * 1024  # bytes per input buffer (2 buffers = the 160 KiB)
_PAD_B = _TR * _TC * _PX_B  # dummy side-DMA slots start here
_DMA_MAIN, _DMA_SIDE = 9, 3  # DMAs per wave and chunk
_DMA_N = _DMA_MAIN + _DMA_SIDE
_DMA_PER_STEP = 3  # steps 0..3 carry the DMA; 4..5 let it land
_STEPS = 6  # 3 filter columns x 2 channel halves
_TILE_B = 1024  # one 16 x 32 A-operand tile
_ST_NT = 2  # store aux: nt (K128 / K64 epilogue; not used for the K96 half-line stores)
KW16S = (4, 3, 2)  # 16-channel MFMA tiles per wave: block K 128 / 96 / 64


def _step_b(kw16: int, nwk: int = 2) -> int:
    """Formatted weight bytes per step: [wave_k nwk][r 3][k16 kw16] tiles."""
    return nwk * 3 * kw16 * _TILE_B


def _stores(kw16: int) -> int:
    """Epilogue stores per wave (per output row: 2, or 1 for a K32 wave)."""
    return _WP * (2 if kw16 > 2 else 1)


PERSIST = 256  # blocks of the persistent grid (one per gfx950 CU)
_XCD_ARCHES = {"gfx950": 8}  # XCDs (blocks dealt round-robin over them)
# each step forms its DMAs' row offsets, row checks and LDS addresses
# from a few per-tile scalars (opaque copies) instead of holding one of each
# per DMA across the chunk loop (those spilled to VGPR lanes: v_readlane /
# v_writelane in the memory phase). Not for unfold (few spills) nor odd_c (its
# two-tile body then spills VGPRs to scratch).
RR_SLIM = 1
# a step's 12 weight tiles from 3 scalar offsets, the rest in the
# instruction's immediate offset (12 s_add per step -> 3)
RR_WIMM = 1

assert _PAD_B + 6 * 256 <= _BUF_B


def fwd_rr_weight_bytes(problem: DirectConvProblem) -> int:
    """Bytes of the formatted weights (same element count as the filter)."""
    return 2 * problem.total_k * 3 * problem.KW * ceil_div(problem.cpg, _CK) * _CK


def _chan(m: int) -> int:
    """Output channel (inside a 16-channel tile) computed by MFMA row ``m``.

    Rows ``4g .. 4g + 3`` hold channels ``(g & 1) * 8 + (g >> 1) * 4 + j``, so
    lane rows 0 and 2 (and 1 and 3) together hold 8 consecutive channels: the
    pairing ``permlane32_swap`` exchanges.
    """
    g, j = divmod(m, 4)
    return (g & 1) * 8 + (g >> 1) * 4 + j


@dataclass(frozen=True)
class FwdRrMfmaSpec:
    """One configuration of :func:`build_fwd_rr_mfma`."""

    problem: DirectConvProblem
    name: str = "fwd_rr_mfma"
    wave_size: int = 64
    kw16: int = 4  # 16-channel MFMA tiles per wave (block K = 32 * kw16)
    odd_c: bool = False  # C / 64 odd (>= 3): tiles alternate the LDS buffer parity
    # one step per filter column covering both 32-channel halves (3 steps
    # per chunk instead of 6): twice the MFMAs per ping-pong phase, for the
    # narrow K64 / K96 blocks whose 6-step phases are shorter than the other
    # group's memory phase.
    c64: bool = False
    # the tile's 32 columns are 32 images of a width-1 layer (a 1xS
    # layer folded to Sx1 over [N*H, W, 1, C]: rows run along the original W)
    unfold: bool = False
    # 4 waves along K (2 x 4 wave grid, 8-row tile, block K = 64 * kw16):
    # half the input re-reads of the K128 block per output channel
    k4w: bool = False
    # C not a multiple of 64 (a multiple of 8): the last chunk's DMA lanes
    # past C read zeros (the weight format zero-pads C to 64)
    c_tail: bool = False
    # C <= 64 (one 64-channel chunk per tile): each tile is a single
    # chunk that DMAs the next tile's chunk, so tiles alternate the LDS buffer
    # (the odd_c two-tile loop)
    one_c: bool = False
    # grouped (runtime group count): a K block lies in one group; its
    # chunks read that group's cpg input channels
    grp: bool = False
    # XCD-local tile walk (gfx950: blocks are dealt round-robin over 8
    # XCDs, each with its own L2). Slot ``blk + round * PERSIST`` of XCD
    # ``blk % 8`` maps to that XCD's contiguous run of the tile order, so the
    # K blocks of one pixel tile and the neighbouring pixel tiles (shared
    # halo rows / columns) are read through one L2, and an XCD's next round
    # continues right after its current one.
    xcd: bool = False
    # chunked columns (k4w): the tile's 32 columns are 4 slots of 8
    # columns, each slot the next 8-column chunk of the [N, ceil(Wo / 8)]
    # chunk sequence (any image); a slot stages its 8 + 2 halo columns, so a
    # width that is no multiple of 32 wastes < 8 columns per image row.
    chunk: bool = False

    @property
    def nwk(self) -> int:  # waves along K
        return 4 if self.k4w else 2

    @property
    def tp(self) -> int:  # output rows per tile
        return 8 if self.k4w else _TP

    @property
    def threads_per_block(self) -> int:
        return 8 * self.wave_size

    @property
    def tk(self) -> int:
        return 16 * self.nwk * self.kw16

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}" if p.PAD_W == p.PAD else f"p{p.PAD}x{p.PAD_W}",
            "" if self.kw16 == 4 else f"k{self.tk}",
            flags={
                "oddc": self.odd_c,
                "c64": self.c64,
                "unf": self.unfold,
                "k4w": self.k4w,
                "ctail": self.c_tail,
                "onec": self.one_c,
                "grp": self.grp,
                "xcd": self.xcd,
                "chk": self.chunk,
                "bf16": p.dtype == "bf16",
            },
        )

    def tiles(self) -> int:
        p = self.problem
        cols = ceil_div(p.N, _TQ) if self.unfold else p.N * ceil_div(p.Wo, _TQ)
        if self.chunk:
            cols = ceil_div(p.N * ceil_div(p.Wo, 8), 4)
        return cols * ceil_div(p.Ho, self.tp) * (p.total_k // self.tk)

    def grid(self) -> Tuple[int, int, int]:
        return (min(self.tiles(), PERSIST), 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    def workspace_bytes(self) -> int:
        return fwd_rr_weight_bytes(self.problem)


def fwd_rr_mfma_reason(spec: FwdRrMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "fwd_rr_mfma is a wave64 MFMA kernel"
    if target.async_lds_max_dwords < 4:
        return f"fwd_rr_mfma needs the 16-byte LDS DMA (absent on {arch})"
    if not target.fits_lds(2 * _BUF_B):
        return f"LDS {2 * _BUF_B} B > {target.lds_capacity_bytes}"
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
    if p.groups != 1 and not spec.grp:
        return "groups must be 1 (grp takes grouped rows)"
    if spec.grp and (p.groups < 2 or p.kpg % spec.tk or spec.unfold):
        return "grp: groups >= 2, a whole number of K blocks per group, no unfold"
    if p.KH != 3 or p.KW not in (3, 1) or (p.stride, p.dil_h, p.dil_w) != (1, 1, 1):
        return "3x3 or 3x1, stride 1, dilation 1 only"
    if p.pad_w < -1 or not (0 <= p.PAD <= 2 and 0 <= p.PAD_W <= p.KW - 1):
        return (
            f"padding must be 0..2 (rows) and 0..KW-1 (columns) (got {p.PAD}x{p.PAD_W})"
        )
    if spec.c64 and p.KW == 1:
        # one step per chunk: no DMA-free step lets the lagging group's DMAs
        # land before the other group reads the buffer
        return "c64 needs at least 2 filter columns"
    if spec.unfold and (p.KW != 1 or p.W != 1):
        return "unfold needs a width-1 layer with a 3x1 filter"
    if p.KW == 1 and not spec.unfold and p.Wo <= 8:
        return f"output width {p.Wo} wastes most of the 32-column tile (unfold runs width 1)"
    if spec.unfold and 32 * 2 * p.H * p.total_c > I32_MAX:
        return "unfold: 32 images exceed the i32 buffer range"
    if spec.one_c:
        if p.cpg > _CK or p.cpg % 8:
            return f"one_c needs C <= {_CK}, a multiple of 8"
        if (p.cpg < _CK) != spec.c_tail or spec.odd_c:
            return "one_c: c_tail exactly when C < 64, no odd_c"
        nchk = 1
    elif spec.c_tail:
        if p.cpg % 8 or p.cpg % _CK == 0:
            return f"c_tail needs C a multiple of 8 but not of {_CK}"
        nchk = ceil_div(p.cpg, _CK)
    elif p.cpg % _CK:
        return f"C {p.cpg} must be a multiple of {_CK} (c_tail takes the rest)"
    else:
        nchk = p.cpg // _CK
    if spec.one_c:
        pass
    elif spec.odd_c:
        if nchk % 2 == 0 or nchk < 3:
            return f"odd_c needs an odd number (>= 3) of {_CK}-channel chunks"
    elif nchk % 2:
        return f"an odd number of {_CK}-channel chunks needs odd_c"
    if spec.kw16 not in KW16S:
        return f"kw16 must be one of {KW16S}"
    if p.total_k % spec.tk:
        return f"K {p.total_k} must be a multiple of {spec.tk}"
    for what, nbytes in (
        ("input image", 2 * p.H * p.W * p.total_c),
        ("output image", 2 * p.Ho * p.Wo * p.total_k),
        ("weights", fwd_rr_weight_bytes(p)),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    if spec.tiles() > I32_MAX:
        return "tile count exceeds i32"
    if spec.chunk:
        if not spec.k4w or p.KW != 3 or spec.unfold or spec.grp:
            return "chunk: k4w 3x3 tiles only (no unfold / grp)"
        if (
            2 * p.N * p.H * p.W * p.total_c > I32_MAX
            or 2 * p.N * p.Ho * p.Wo * p.total_k > I32_MAX
        ):
            return "chunk: the whole input / output must fit the i32 buffer range"
    if spec.xcd and arch not in _XCD_ARCHES:
        return f"xcd: the XCD-local walk is built for {sorted(_XCD_ARCHES)} only"
    return None


# ---------------------------------------------------------------------------
# Weight format
# ---------------------------------------------------------------------------


def _wfmt_source(
    o: int, C: int, kw16: int = 4, fw: int = 3, nwk: int = 2
) -> Tuple[int, int, int, int]:
    """KRSC source ``(k, r, s, c)`` of formatted 8-element granule ``o``.

    Layout: ``[K/tk][C/64][step 6][wave_k 2][r 3][k16 kw16][lane 64]`` granules
    (``tk = 32 * kw16``)
    of 8 elements, step = s * 2 + half; lane ``l`` holds channels
    ``half * 32 + (l / 16) * 8 ..`` of output channel ``_chan(l % 16)``.
    """
    lane = o % 64
    o //= 64
    k4 = o % kw16
    o //= kw16
    r = o % 3
    o //= 3
    wk = o % nwk
    o //= nwk
    st = o % (2 * fw)
    o //= 2 * fw
    nch = C // _CK
    ch = o % nch
    kb = o // nch
    s, half = divmod(st, 2)
    k = kb * 16 * nwk * kw16 + wk * 16 * kw16 + k4 * 16 + _chan(lane % 16)
    c = ch * _CK + half * 32 + (lane // 16) * 8
    return k, r, s, c


def fwd_rr_format_weights_reference(
    w, C: int, kw16: int = 4, fw: int = 3, nwk: int = 2
):
    """Host reference of the weight format: ``w`` is a KRSC numpy array;
    returns the formatted flat array."""
    import numpy as np

    K = w.shape[0]
    flat = np.asarray(w).reshape(K, 3, fw, C)
    out = []
    for o in range(K * 3 * fw * C // 8):
        k, r, s, c = _wfmt_source(o, C, kw16, fw, nwk)
        out.append(flat[k, r, s, c : c + 8])
    return np.concatenate(out)


@dataclass(frozen=True)
class FwdRrWfmtSpec:
    """The weight-format helper of :class:`FwdRrMfmaSpec` (C, K are kernargs)."""

    problem: DirectConvProblem
    name: str = "fwd_rr_wfmt"
    threads: int = 256
    kw16: int = 4  # the conv's FwdRrMfmaSpec.kw16
    fw: int = 3  # filter width (3x3 or 3x1)
    nwk: int = 2  # the conv's waves along K
    # read a dgrad's own KRSC filter (dgrad via forward): flip r, s and
    # swap C/K while formatting, so no separate filter transpose runs
    dg: bool = False

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            "" if self.kw16 == 4 else f"k{32 * self.kw16}",
            "" if self.fw == 3 else f"w{self.fw}",
            "" if self.nwk == 2 else f"wk{self.nwk}",
            "dg" if self.dg else "",
            flags={"bf16": self.problem.dtype == "bf16"},
        )

    def launch_geometry(self):
        granules = (
            self.problem.total_k
            * 3
            * self.fw
            * ceil_div(self.problem.cpg, _CK)
            * _CK
            // 8
        )
        return (ceil_div(granules, self.threads), 1, 1), (self.threads, 1, 1)


def build_fwd_rr_wfmt(spec: FwdRrWfmtSpec, *, arch: str = "gfx950") -> KernelDef:
    """One thread per 16-byte granule of the formatted weights.

    Kernargs follow :func:`conv_direct_transform_signature` with runtime
    channels: ``A`` (KRSC filter), ``D`` (formatted), their byte sizes, then
    ``p_cpg`` (C) and ``p_kpg`` (K).

    With ``dg`` the conv is a dgrad via forward: ``A`` is the dgrad's filter
    ``[C][3][fw][K]`` (its own K, C are this conv's C, K), read flipped, and
    each granule gathers its 8 channels one element at a time.
    """
    io_type = _io_type(spec.problem.dtype)
    dtype = spec.problem.dtype
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    src = b.param(
        "A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16
    )
    dst = b.param(
        "D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16
    )
    src_bytes = b.param("A_bytes", I32)
    dst_bytes = b.param("D_bytes", I32)
    C = b.param("p_cpg", I32)
    kpg = b.param("p_kpg", I32)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    o = b.add(b.mul(b.block_id_x(), ci(spec.threads)), b.thread_id_x())
    n_gran = b.div(dst_bytes, ci(16))
    rest = o
    lane = b.mod(rest, ci(64))
    rest = b.div(rest, ci(64))
    KW = spec.kw16
    k4 = b.mod(rest, ci(KW))
    rest = b.div(rest, ci(KW))
    r = b.mod(rest, ci(3))
    rest = b.div(rest, ci(3))
    NWKF = spec.nwk
    wk = b.mod(rest, ci(NWKF))
    rest = b.div(rest, ci(NWKF))
    FW = spec.fw
    st = b.mod(rest, ci(2 * FW))
    rest = b.div(rest, ci(2 * FW))
    nch = b.div(b.add(C, ci(_CK - 1)), ci(_CK))
    ch = b.mod(rest, nch)
    kb = b.div(rest, nch)
    s = b.div(st, ci(2))
    half = b.mod(st, ci(2))
    m = b.mod(lane, ci(16))
    g = b.div(m, ci(4))
    chan = b.add(
        b.add(b.mul(b.mod(g, ci(2)), ci(8)), b.mul(b.div(g, ci(2)), ci(4))),
        b.mod(m, ci(4)),
    )
    k = b.add(
        b.add(b.mul(kb, ci(16 * NWKF * KW)), b.mul(wk, ci(16 * KW))),
        b.add(b.mul(k4, ci(16)), chan),
    )
    c = b.add(
        b.add(b.mul(ch, ci(_CK)), b.mul(half, ci(32))),
        b.mul(b.div(lane, ci(16)), ci(8)),
    )
    ok = b.cmp_lt(o, n_gran)
    src_rsrc = b.buffer_rsrc(src, src_bytes)
    dst_rsrc = b.buffer_rsrc(dst, dst_bytes)
    oob = ci(I32_MAX)
    if spec.dg:
        K = kpg
        rf, sf = b.sub(ci(2), r), b.sub(ci(FW - 1), s)
        vals = []
        for j in range(8):
            cj = b.add(c, ci(j))
            el = b.add(
                b.mul(b.add(b.mul(b.add(b.mul(cj, ci(3)), rf), ci(FW)), sf), K), k
            )
            vals.append(
                _buf_load_half(
                    b,
                    dtype,
                    src_rsrc,
                    b.select(b.land(ok, b.cmp_lt(cj, C)), b.mul(el, ci(2)), oob),
                    ci(0),
                )
            )
        v = b.vec_pack(vals, io_type)
    else:
        src_el = b.add(b.mul(b.add(b.mul(b.add(b.mul(k, ci(3)), r), ci(FW)), s), C), c)
        c_ok = b.cmp_lt(c, C)  # zero-pad C to a whole chunk
        v = _buf_load_vN(
            b,
            dtype,
            src_rsrc,
            b.select(b.land(ok, c_ok), b.mul(src_el, ci(2)), oob),
            ci(0),
            4,
        )
    _buf_store_vN(b, dtype, dst_rsrc, b.select(ok, b.mul(o, ci(16)), oob), ci(0), v, 4)
    return b.kernel


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


def build_fwd_rr_mfma(spec: FwdRrMfmaSpec, *, arch: str = "gfx950") -> KernelDef:
    why = fwd_rr_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdRrMfmaSpec for {arch}: {why}")
    p = spec.problem
    dtype = p.dtype
    io_type = _io_type(dtype)
    KW, TK = spec.kw16, spec.tk
    PH, PW = p.PAD, p.PAD_W
    C64 = spec.c64
    FW = p.KW  # filter width: 3, or 1 (3x1)
    UNF = spec.unfold
    CHK = spec.chunk
    # staged columns (chunk: 4 slots x 8 columns, then their 4 x 2 halo columns)
    TC = 4 * 10 if CHK else _TQ + FW - 1
    SIDE_B = 1024 if CHK else 256  # bytes per side DMA (chunk: 16-byte lanes)
    NWK, TPV = spec.nwk, spec.tp
    TR = TPV + 2  # staged rows
    DMA_MAIN = ceil_div(TR, 2)  # row pairs (one per ping-pong group)
    DMA_N = DMA_MAIN + (ceil_div(TR, 8) if FW == 3 else 0)
    FSTEPS = 2 * FW  # formatted-weight steps per chunk (column x half)
    NSTEP = FW if C64 else FSTEPS  # steps per chunk
    # DMAs per carrying step: spread over all but the last step
    DPS = ceil_div(DMA_N, max(1, NSTEP - 1))
    STEP_B = _step_b(KW, NWK)
    CHUNK_B = FSTEPS * STEP_B
    STORES = _stores(KW)
    SLIM = RR_SLIM and not spec.unfold and not spec.odd_c
    WIMM = RR_WIMM and not spec.unfold and not spec.odd_c
    ST_AUX = 0 if KW == 3 else _ST_NT

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)
    p_N, p_Hi, p_Wi = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]
    sa_n, sa_h = params["p_A_stride_n"], params["p_A_stride_hi"]
    sd_n, sd_h, sd_w = (
        params["p_D_stride_n"],
        params["p_D_stride_ho"],
        params["p_D_stride_wo"],
    )

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    wp = b.mod(wave, c2)
    wq = b.mod(b.div(wave, c2), c2)
    grp = b.div(wave, ci(4))  # ping-pong group
    if spec.k4w:  # 1 x 2 x 4 grid: wave = wk * 2 + wq
        wp = c0
        wq = b.mod(wave, c2)
        wk = b.div(wave, c2)
    else:  # 2 x 2 x 2 grid: the channel half is the ping-pong group
        wk = grp

    def opaque_lane() -> Value:
        # An opaque copy of the lane id: lane-derived addressing is recomputed
        # where it is used instead of being hoisted into long-lived VGPRs
        # (which would spill at K128's register pressure).
        return b.inline_asm("", "=v,0", [lane], I32, sideeffect=True)

    # ---- tile walk: t -> (k block fastest, pixel tile, row tile, image) ------
    nkb = b.div(K, ci(TK))
    n_ht = b.div(b.add(p_Ho, ci(TPV - 1)), ci(TPV))
    if UNF:  # column tiles of 32 images
        n_wt = b.div(b.add(p_N, ci(_TQ - 1)), ci(_TQ))
        T = b.mul(b.mul(n_ht, n_wt), nkb)
    elif CHK:  # column tiles of 4 8-column chunks
        nqc = b.div(b.add(p_Wo, ci(7)), ci(8))
        nsl = b.mul(p_N, nqc)
        n_wt = b.div(b.add(nsl, ci(3)), ci(4))
        T = b.mul(b.mul(n_ht, n_wt), nkb)
    else:
        n_wt = b.div(b.add(p_Wo, ci(_TQ - 1)), ci(_TQ))
        T = b.mul(b.mul(b.mul(p_N, n_ht), n_wt), nkb)
    GRP = spec.grp
    CPG = b.div(C, params["p_groups"]) if GRP else C  # channels per group
    KPG = b.div(K, params["p_groups"]) if GRP else K
    nch = b.div(b.add(CPG, ci(_CK - 1)), ci(_CK))
    blk = b.block_id_x()
    rounds = b.div(b.add(b.sub(T, blk), ci(PERSIST - 1)), ci(PERSIST))
    img_a = b.mul(sa_n, c2)  # bytes per input / output image
    img_d = b.mul(sd_n, c2)
    a_all = b.mul(p_N, img_a) if CHK else None

    if spec.xcd:
        # slot v = blk + round * PERSIST of XCD x = v % NX, j = (v % PERSIST)
        # / NX -> tile x * per + min(x, rem) + round * (PERSIST / NX) + j
        # (per, rem = T / NX, T % NX): a bijection of [0, T) (the slots of
        # XCD x below T are exactly its per + (x < rem) tiles), so the
        # validity test t < T stays the slot's v < T.
        NX = _XCD_ARCHES[arch]
        assert PERSIST % NX == 0 and NX & (NX - 1) == 0
        x_per = b.div(T, ci(NX))
        x_rem = b.mod(T, ci(NX))

    def xcd_tile(v: Value) -> Value:
        x = b.land(v, ci(NX - 1))
        j = b.div(b.land(v, ci(PERSIST - 1)), ci(NX))
        r = b.div(v, ci(PERSIST))
        start = b.add(b.mul(x, x_per), b.smin(x, x_rem))
        return b.add(start, b.add(b.mul(r, ci(PERSIST // NX)), j))

    def decode(v: Value):
        t = xcd_tile(v) if spec.xcd else v
        kb = b.mod(t, nkb)
        rest = b.div(t, nkb)
        w0 = b.mul(b.mod(rest, n_wt), ci(_TQ))
        rest = b.div(rest, n_wt)
        h0 = b.mul(b.mod(rest, n_ht), ci(TPV))
        n = b.div(rest, n_ht)
        valid = b.cmp_lt(v, T)
        if GRP:  # the K block's group: its input channels start at g * cpg
            gc = b.mul(b.mul(b.div(b.mul(kb, ci(TK)), KPG), CPG), c2)
            return {"kb": kb, "w0": w0, "h0": h0, "n": n, "valid": valid, "gc": gc}
        if CHK:  # slot s: chunk 4 * column tile + s -> (image, first column)
            g0 = b.mul(b.div(w0, ci(_TQ)), ci(4))
            sl = [b.add(g0, ci(i)) for i in range(4)]
            return {
                "kb": kb,
                "w0": c0,
                "h0": h0,
                "n": c0,
                "valid": valid,
                "sn": [b.div(g, nqc) for g in sl],
                "sq": [b.mul(b.mod(g, nqc), ci(8)) for g in sl],
                "sv": [b.cmp_lt(g, nsl) for g in sl],
            }
        if UNF:  # w0 is the first image; columns are images
            ncol = b.select(valid, b.smin(b.sub(p_N, w0), ci(_TQ)), c0)
            return {"kb": kb, "w0": c0, "h0": h0, "n": w0, "valid": valid, "ncol": ncol}
        return {"kb": kb, "w0": w0, "h0": h0, "n": n, "valid": valid}

    def a_base(ctx):
        off = b.mul(b.zext(ctx["n"], I64), b.zext(img_a, I64))
        return b.global_ptr_offset(params["A"], off)

    def w_tile_off(ctx) -> Value:  # this wave's weights of the tile's chunk 0
        return b.mul(
            b.add(
                b.mul(b.mul(ctx["kb"], nch), ci(FSTEPS * NWK * 3 * KW)),
                b.mul(wk, ci(3 * KW)),
            ),
            ci(_TILE_B),
        )

    # ---- LDS ----------------------------------------------------------------
    elems = _BUF_B // 2
    bufs = [b.smem_alloc(io_type, [1, elems], name_hint=f"in{i}") for i in range(2)]
    addrs = [b.smem_addr_of(s) for s in bufs]

    def lds_at(i: int, byte_off: Value) -> Value:
        return b.smem_ptr_add(addrs[i], b.zext(byte_off, I64))

    # ---- DMA ------------------------------------------------------------------
    # main: lane l -> pixel column 8 * (wave % 4) + l / 8, slot l % 8, logical
    # granule (l % 8) ^ (l / 8); side: lane l -> column 32 + l / 32, dword
    # l % 32 of the pixel, slot (l % 32) / 4, granule slot ^ (l / 32).
    def sel4(vals, i: Value) -> Value:
        r = vals[0]
        for k in (1, 2, 3):
            r = b.select(b.cmp_eq(i, ci(k)), vals[k], r)
        return r

    def dma_lanes_chunk(ctx) -> Tuple[Value, Value]:
        # main: the wave's slot wave % 4, lane l -> its column l / 8; side:
        # lane l -> halo pixel l / 8 = slot (l / 16) column 8 + (l / 8) % 2;
        # both: logical granule (l % 8) ^ (l / 8); image in the lane offset
        ln = opaque_lane()
        l8 = b.div(ln, ci(8))
        gran = b.xor(b.mod(ln, ci(8)), l8)
        sm = b.mod(wave, ci(4))
        col = b.add(b.sub(sel4(ctx["sq"], sm), ci(PW)), l8)
        ok = b.land(sel4(ctx["sv"], sm), b.land(b.cmp_ge(col, c0), b.cmp_lt(col, p_Wi)))
        lt = b.add(
            b.mul(sel4(ctx["sn"], sm), img_a),
            b.mul(b.add(b.mul(col, C), b.mul(gran, ci(8))), c2),
        )
        hs = b.div(l8, c2)
        hcol = b.add(b.sub(sel4(ctx["sq"], hs), ci(PW - 8)), b.mod(l8, c2))
        sok = b.land(sel4(ctx["sv"], hs), b.cmp_lt(hcol, p_Wi))
        ls = b.add(
            b.mul(sel4(ctx["sn"], hs), img_a),
            b.mul(b.add(b.mul(hcol, C), b.mul(gran, ci(8))), c2),
        )
        return b.select(ok, lt, oob), b.select(sok, ls, oob)

    def dma_lanes(ctx) -> Tuple[Value, Value]:
        if CHK:
            return dma_lanes_chunk(ctx)
        ln = opaque_lane()
        l8 = b.div(ln, ci(8))
        l32 = b.div(ln, ci(32))
        # row soffset starts at w0 - sh (w0 is 0 or >= 32 > PW)
        sh = b.select(b.cmp_gt(ctx["w0"], c0), ci(PW), c0)
        col = b.add(b.mul(b.mod(wave, ci(4)), ci(8)), l8)
        gran = b.xor(b.mod(ln, ci(8)), l8)
        if UNF:  # column = image: past the tile's images the buffer range reads 0
            lt = b.add(b.mul(col, img_a), b.mul(gran, ci(16)))
            return lt, lt
        ok = b.land(
            b.cmp_ge(b.add(b.sub(ctx["w0"], ci(PW)), col), c0),
            b.cmp_lt(b.add(b.sub(ctx["w0"], ci(PW)), col), p_Wi),
        )
        lt = b.mul(
            b.add(b.mul(b.add(b.sub(col, ci(PW)), sh), C), b.mul(gran, ci(8))), c2
        )
        scol = b.add(ci(32), l32)
        sgran = b.xor(b.div(b.mod(ln, ci(32)), ci(4)), l32)
        sok = b.cmp_lt(b.add(b.sub(ctx["w0"], ci(PW)), scol), p_Wi)
        ls = b.add(
            b.mul(
                b.add(b.mul(b.add(b.sub(scol, ci(PW)), sh), C), b.mul(sgran, ci(8))), c2
            ),
            b.mul(b.mod(ln, ci(4)), ci(4)),
        )
        return b.select(ok, lt, oob), b.select(sok, ls, oob)

    def opaque_s(v: Value) -> Value:
        # a per-step copy: what is formed from it is formed where used, not
        # hoisted out of the chunk loop and held (spilled) across it
        return b.inline_asm("", "=s,0", [v], I32, sideeffect=True)

    def dma_scalars(ctx, ch: Value):
        """The per-step scalars of :func:`dma_u` (``RR_SLIM``): first halo row
        (far negative for a tile past the grid: every row check fails), the
        tile's column byte offset plus the chunk's, the row bytes, the wave's
        LDS offset of its main DMAs."""
        sh = b.select(b.cmp_gt(ctx["w0"], c0), ci(PW), c0)
        h0 = b.select(ctx["valid"], b.sub(ctx["h0"], ci(PH)), ci(-(1 << 20)))
        col_b = b.add(b.mul(b.mul(b.sub(ctx["w0"], sh), C), c2), b.mul(ch, ci(_PX_B)))
        if GRP:
            col_b = b.add(col_b, ctx["gc"])
        return {
            "h0": opaque_s(h0),
            "col": opaque_s(col_b),
            "row": opaque_s(b.mul(b.mul(p_Wi, C), c2)),
            "lds": opaque_s(
                b.add(b.mul(grp, ci(TC * _PX_B)), b.mul(b.mod(wave, ci(4)), ci(1024)))
            ),
        }

    def dma_u(u: int, ctx, ch: Value, buf: int, lt: Value, ls: Value, sq=None):
        """DMA ``u`` (0..11) of this wave for chunk ``ch`` of tile ``ctx``
        (``sq``: its step's :func:`dma_scalars`)."""
        if sq is not None:
            if u < DMA_MAIN:
                h = b.add(sq["h0"], b.add(grp, ci(2 * u)))
                dst = b.add(sq["lds"], ci(2 * u * TC * _PX_B))
                voff, dw, rok = lt, 4, b.cmp_ult(h, p_Hi)
            else:
                r = b.add(wave, ci(8 * (u - DMA_MAIN)))
                real = b.cmp_lt(r, ci(TR))
                dst = b.select(
                    real,
                    b.add(b.mul(r, ci(TC * _PX_B)), ci(32 * _PX_B)),
                    b.add(ci(TR * TC * _PX_B), b.mul(b.sub(r, ci(TR)), ci(SIDE_B))),
                )
                h = b.add(sq["h0"], r)
                voff, dw, rok = ls, 4 if CHK else 1, b.land(real, b.cmp_ult(h, p_Hi))
            # a row outside the image reads through a zero-size descriptor,
            # whatever its offset
            so = b.add(b.mul(h, sq["row"]), sq["col"])
            if spec.c_tail:
                voff = c_tail_voff(u, ch, voff)
            rng = b.mul(ctx["ncol"], img_a) if UNF else (a_all if CHK else img_a)
            rsrc = b.buffer_rsrc(a_base(ctx), b.select(rok, rng, c0))
            b.async_buffer_load_lds(rsrc, lds_at(buf, dst), voff, so, dw)
            return
        sh = b.select(b.cmp_gt(ctx["w0"], c0), ci(PW), c0)
        if u < DMA_MAIN:
            r = b.add(grp, ci(2 * u))  # < TR: TR is even
            dst = b.add(b.mul(r, ci(TC * _PX_B)), b.mul(b.mod(wave, ci(4)), ci(1024)))
            voff, dw = lt, 4
        else:
            r = b.add(wave, ci(8 * (u - DMA_MAIN)))
            real = b.cmp_lt(r, ci(TR))
            dst = b.select(
                real,
                b.add(b.mul(r, ci(TC * _PX_B)), ci(32 * _PX_B)),
                b.add(ci(TR * TC * _PX_B), b.mul(b.sub(r, ci(TR)), ci(SIDE_B))),
            )
            voff, dw = ls, 4 if CHK else 1
        h = b.add(b.sub(ctx["h0"], ci(PH)), r)
        rok = b.land(ctx["valid"], b.land(b.cmp_ge(h, c0), b.cmp_lt(h, p_Hi)))
        if u >= DMA_MAIN:
            rok = b.land(rok, real)
        row = b.mul(b.add(b.mul(h, p_Wi), b.sub(ctx["w0"], sh)), b.mul(C, c2))
        so = b.select(rok, b.add(row, b.mul(ch, ci(_PX_B))), c0)
        if GRP:
            so = b.add(so, ctx["gc"])
        if spec.c_tail:
            voff = c_tail_voff(u, ch, voff)
        rng = b.mul(ctx["ncol"], img_a) if UNF else (a_all if CHK else img_a)
        rsrc = b.buffer_rsrc(a_base(ctx), b.select(rok, rng, c0))
        b.async_buffer_load_lds(rsrc, lds_at(buf, dst), voff, so, dw)

    def c_tail_voff(u: int, ch: Value, voff: Value) -> Value:
        # lanes past C (the last chunk) read zeros
        ln = opaque_lane()
        if u < DMA_MAIN or CHK:
            lch = b.mul(b.xor(b.mod(ln, ci(8)), b.div(ln, ci(8))), ci(8))
        else:
            sg = b.xor(b.div(b.mod(ln, ci(32)), ci(4)), b.div(ln, ci(32)))
            lch = b.add(b.mul(sg, ci(8)), b.mul(b.mod(ln, ci(4)), c2))
        cok = b.cmp_lt(b.add(lch, b.mul(ch, ci(_CK))), CPG)
        return b.select(cok, voff, oob)

    # ---- weights ------------------------------------------------------------
    w_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    lane16 = b.mul(lane, ci(16))

    def load_w(soff: Value) -> List[List[Value]]:
        if WIMM:
            # tile i at soff + (i // 4) * 4 KiB plus (i % 4) KiB in the immediate
            sg = [soff] + [
                b.add(soff, ci(g * 4096)) for g in range(1, ceil_div(3 * KW, 4))
            ]
            return [
                [
                    _buf_load_vN(
                        b,
                        dtype,
                        w_rsrc,
                        b.add(lane16, ci((r * KW + k) % 4 * _TILE_B)),
                        sg[(r * KW + k) // 4],
                        4,
                    )
                    for k in range(KW)
                ]
                for r in range(3)
            ]
        return [
            [
                _buf_load_vN(
                    b, dtype, w_rsrc, lane16, b.add(soff, ci((r * KW + k) * _TILE_B)), 4
                )
                for k in range(KW)
            ]
            for r in range(3)
        ]

    # ---- LDS fragment reads: row wp * 8 + t, column wq * 16 + s + lane % 16 --
    def read_in(buf: int, s: int, half: int) -> List[Value]:
        ln = opaque_lane()
        l16, lhi = b.mod(ln, ci(16)), b.div(ln, ci(16))
        col = b.add(b.add(b.mul(wq, ci(16)), ci(s)), l16)
        if CHK and s > 0:  # past the slot's 8 columns: its halo columns
            j8 = b.mod(l16, ci(8))
            halo = b.add(
                b.add(b.mul(wq, ci(4)), b.mul(b.div(l16, ci(8)), c2)),
                b.add(j8, ci(32 + s - 8)),
            )
            col = b.select(b.cmp_lt(j8, ci(8 - s)), col, halo)
        gran = b.xor(b.add(ci(half * 4), lhi), b.mod(col, ci(8)))
        base = b.add(
            b.mul(b.add(b.mul(wp, ci(_WP * TC)), col), ci(_CK)), b.mul(gran, ci(8))
        )
        return [
            b.smem_load_vN(
                bufs[buf], c0, b.add(base, ci(t * TC * _CK)), dtype=io_type, n=8
            )
            for t in range(_NIN)
        ]

    # ---- one 64-channel chunk -------------------------------------------------
    def chunk(
        accs,
        wpre,
        cur: int,
        dctx,
        dch: Value,
        lt,
        ls,
        wsoff: Value,
        first: bool,
        epi=None,
    ):
        """Chunk on buffer ``cur``; DMAs chunk ``dch`` of tile ``dctx`` into the
        other buffer. ``first``: the tile's first chunk (weights preloaded in
        ``wpre``). ``epi``: the tile's last chunk; ``epi()`` sets up the
        epilogue and returns its per-row store emitter, interleaved into the
        last step (the stores are left in flight)."""
        accs = list(accs)
        HALVES = (0, 1) if C64 else (None,)
        for j in range(NSTEP):
            if j > 0:
                b.s_setprio(0)
                b.s_barrier_bare()
                b.sched_barrier(0)
            wf, fr = {}, {}
            for hv in HALVES:
                # (filter column, 32-channel half) and its formatted-weight step
                s, half = (j, hv) if C64 else divmod(j, 2)
                ws = 2 * s + half
                if first and ws == 0:
                    wf[hv] = wpre[0] if C64 else wpre
                elif first and C64 and ws == 1:
                    wf[hv] = wpre[1]
                else:
                    wf[hv] = load_w(b.add(wsoff, ci(ws * STEP_B)))
            nd = max(0, min(DPS, DMA_N - j * DPS))
            sq = dma_scalars(dctx, dch) if SLIM and nd else None
            for d in range(nd):
                dma_u(j * DPS + d, dctx, dch, 1 - cur, lt, ls, sq)
            for hv in HALVES:
                s, half = (j, hv) if C64 else divmod(j, 2)
                fr[hv] = read_in(cur, s, half)
            b.s_waitcnt(vmcnt=nd, lgkmcnt=0)
            b.sched_barrier(0)
            b.s_setprio(1)
            b.s_barrier_bare()
            b.sched_barrier(0)
            if epi is not None and j == NSTEP - 1:
                # rows finish one by one; row pr - 1's epilogue rides on row
                # pr's MFMAs (fenced: the scheduler would otherwise sink the
                # stores to the end of the step)
                store_row = epi()
                for pr in range(_WP):
                    for hv in HALVES:
                        for r in range(3):
                            for k in range(KW):
                                i = pr * KW + k
                                accs[i] = _mfma(
                                    b,
                                    dtype,
                                    "16x16x32",
                                    wf[hv][r][k],
                                    fr[hv][pr + r],
                                    accs[i],
                                )
                    if pr > 0:
                        store_row(accs, pr - 1)
                        b.sched_barrier(0)
                store_row(accs, _WP - 1)
                continue
            for hv in HALVES:
                for r in range(3):
                    for k in range(KW):
                        for pr in range(_WP):
                            i = pr * KW + k
                            accs[i] = _mfma(
                                b,
                                dtype,
                                "16x16x32",
                                wf[hv][r][k],
                                fr[hv][pr + r],
                                accs[i],
                            )
        b.sched_barrier(0)
        b.s_waitcnt(vmcnt=STORES if epi is not None else 0)
        b.s_setprio(0)
        b.s_barrier_bare()
        b.sched_barrier(0)
        return accs

    # ---- epilogue -------------------------------------------------------------
    i32x2 = VectorType(I32, 2)
    half8 = VectorType(BF16 if dtype == "bf16" else F16, 8)

    def epilogue(ctx):
        """Epilogue setup for tile ``ctx``; returns ``store_row(accs, pr)``."""
        if CHK:
            return epilogue_chunk(ctx)
        ln = opaque_lane()
        l16, lhi = b.mod(ln, ci(16)), b.div(ln, ci(16))
        # lane (px, g) stores granule 2 * (g >> 1) + (g & 1) (+ 4 for px >= 8)
        hi = b.cmp_ge(l16, ci(8))
        gran0 = b.add(b.mul(b.div(lhi, c2), c2), b.mod(lhi, c2))
        gran = b.add(gran0, b.select(hi, ci(4), c0))
        d_base = b.global_ptr_offset(
            params["D"], b.mul(b.zext(ctx["n"], I64), b.zext(img_d, I64))
        )
        d_rsrc = b.buffer_rsrc(d_base, b.mul(ctx["ncol"], img_d) if UNF else img_d)
        col0 = b.add(b.add(ctx["w0"], b.mul(wq, ci(16))), b.mod(l16, ci(8)))
        pix0 = b.add(
            b.mul(b.add(ctx["h0"], b.mul(wp, ci(_WP))), sd_h), b.mul(col0, dcol)
        )
        # A right-edge partial tile masks its columns past Wo (rows past Ho lie
        # past the image's bytes: the buffer range drops them; unfolded tiles
        # span images, so they mask rows too).
        ok0 = b.cmp_lt(col0, ncol_of(ctx))
        ok1 = b.cmp_lt(b.add(col0, ci(8)), ncol_of(ctx))
        if KW < 4:
            return epilogue_narrow(ctx, ln, l16, lhi, d_rsrc)
        kch = b.add(
            b.add(b.mul(ctx["kb"], ci(TK)), b.mul(wk, ci(64))), b.mul(gran, ci(8))
        )
        voff0 = b.mul(b.add(pix0, kch), c2)
        step8 = b.mul(b.mul(dcol, ci(8)), c2)

        def store_row(accs, pr: int):
            v = []
            for cc in range(2):
                da = b.vec_bitcast(_trunc(accs[pr * KW + 2 * cc]), i32x2)
                db = b.vec_bitcast(_trunc(accs[pr * KW + 2 * cc + 1]), i32x2)
                lo0, hi0 = b.permlane32_swap(b.vec_extract(da, 0), b.vec_extract(db, 0))
                lo1, hi1 = b.permlane32_swap(b.vec_extract(da, 1), b.vec_extract(db, 1))
                v.append([lo0, lo1, hi0, hi1])
            n0, n1 = [], []
            for d in range(4):
                r0 = b.mov_dpp(v[1][d], row_shr=8)
                r1 = b.mov_dpp(v[0][d], row_shl=8)
                n0.append(b.select(hi, r0, v[0][d]))
                n1.append(b.select(hi, v[1][d], r1))
            voff = b.add(voff0, b.mul(b.mul(sd_h, c2), ci(pr)))
            rok = row_ok(ctx, pr)
            for vals, off, ok in ((n0, voff, ok0), (n1, b.add(voff, step8), ok1)):
                if rok is not None:
                    ok = b.land(ok, rok)
                val = b.vec_bitcast(b.vec_pack(vals, I32), half8)
                _buf_store_vN(
                    b,
                    dtype,
                    d_rsrc,
                    b.select(ok, off, oob),
                    c0,
                    val,
                    4,
                    cachepolicy=ST_AUX,
                )

        return store_row

    def epilogue_narrow(ctx, ln, l16, lhi, d_rsrc):
        """K96 / K64 block (32 * kw16 channels per wave pair): lane (pixel l16,
        q = lhi) stores 8 channels of tiles 0-1 after one permlane32_swap
        (16 bytes) and, for kw16 3, the 4 channels of tile 2 it holds (8 bytes)."""
        col = b.add(b.add(ctx["w0"], b.mul(wq, ci(16))), l16)
        ok_c = b.cmp_lt(col, ncol_of(ctx))
        pix0 = b.add(
            b.mul(b.add(ctx["h0"], b.mul(wp, ci(_WP))), sd_h), b.mul(col, dcol)
        )
        kbase = b.add(b.mul(ctx["kb"], ci(TK)), b.mul(wk, ci(16 * KW)))
        q1, q2 = b.mod(lhi, c2), b.div(lhi, c2)
        k_pair = b.add(kbase, b.add(b.mul(q2, ci(16)), b.mul(q1, ci(8))))
        k_one = b.add(kbase, b.add(ci(32), b.add(b.mul(q1, ci(8)), b.mul(q2, ci(4)))))

        def store_row(accs, pr: int):
            row = b.add(pix0, b.mul(sd_h, ci(pr)))
            rok = row_ok(ctx, pr)
            ok = ok_c if rok is None else b.land(ok_c, rok)
            da = b.vec_bitcast(_trunc(accs[pr * KW]), i32x2)
            db = b.vec_bitcast(_trunc(accs[pr * KW + 1]), i32x2)
            lo0, hi0 = b.permlane32_swap(b.vec_extract(da, 0), b.vec_extract(db, 0))
            lo1, hi1 = b.permlane32_swap(b.vec_extract(da, 1), b.vec_extract(db, 1))
            val = b.vec_bitcast(b.vec_pack([lo0, lo1, hi0, hi1], I32), half8)
            off = b.mul(b.add(row, k_pair), c2)
            _buf_store_vN(
                b, dtype, d_rsrc, b.select(ok, off, oob), c0, val, 4, cachepolicy=ST_AUX
            )
            if KW == 3:
                off = b.mul(b.add(row, k_one), c2)
                _buf_store_vN(
                    b,
                    dtype,
                    d_rsrc,
                    b.select(ok, off, oob),
                    c0,
                    _trunc(accs[pr * KW + 2]),
                    2,
                    cachepolicy=ST_AUX,
                )

        return store_row

    def epilogue_chunk(ctx):
        """chunk: the wave's pixels 0-7 / 8-15 are slots 2 wq / 2 wq + 1 (any
        image; the whole output in one buffer range, rows past Ho masked)."""
        ln = opaque_lane()
        l16, lhi = b.mod(ln, ci(16)), b.div(ln, ci(16))
        j8 = b.mod(l16, ci(8))
        d_rsrc = b.buffer_rsrc(params["D"], b.mul(p_N, img_d))
        sa, sb = b.mul(wq, c2), b.add(b.mul(wq, c2), c1)

        def pix(si, sel=None):
            n_, q_, v_ = (sel4(ctx[k], si) for k in ("sn", "sq", "sv"))
            if sel is not None:
                n2, q2_, v2 = (sel4(ctx[k], sel[0]) for k in ("sn", "sq", "sv"))
                n_, q_, v_ = (
                    b.select(sel[1], x2, x1)
                    for x1, x2 in ((n_, n2), (q_, q2_), (v_, v2))
                )
            col = b.add(q_, j8)
            ok = b.land(v_, b.cmp_lt(col, p_Wo))
            off = b.add(
                b.add(b.mul(n_, sd_n), b.mul(ctx["h0"], sd_h)), b.mul(col, sd_w)
            )
            return off, ok

        def rowok(pr: int):
            return b.cmp_lt(b.add(ctx["h0"], ci(pr)), p_Ho)

        if KW < 4:
            pix0, ok_c = pix(sa, (sb, b.cmp_ge(l16, ci(8))))
            kbase = b.add(b.mul(ctx["kb"], ci(TK)), b.mul(wk, ci(16 * KW)))
            q1, q2 = b.mod(lhi, c2), b.div(lhi, c2)
            k_pair = b.add(kbase, b.add(b.mul(q2, ci(16)), b.mul(q1, ci(8))))
            k_one = b.add(
                kbase, b.add(ci(32), b.add(b.mul(q1, ci(8)), b.mul(q2, ci(4))))
            )

            def store_row_n(accs, pr: int):
                row = b.add(pix0, b.mul(sd_h, ci(pr)))
                ok = b.land(ok_c, rowok(pr))
                da = b.vec_bitcast(_trunc(accs[pr * KW]), i32x2)
                db = b.vec_bitcast(_trunc(accs[pr * KW + 1]), i32x2)
                lo0, hi0 = b.permlane32_swap(b.vec_extract(da, 0), b.vec_extract(db, 0))
                lo1, hi1 = b.permlane32_swap(b.vec_extract(da, 1), b.vec_extract(db, 1))
                val = b.vec_bitcast(b.vec_pack([lo0, lo1, hi0, hi1], I32), half8)
                off = b.mul(b.add(row, k_pair), c2)
                _buf_store_vN(
                    b,
                    dtype,
                    d_rsrc,
                    b.select(ok, off, oob),
                    c0,
                    val,
                    4,
                    cachepolicy=ST_AUX,
                )
                if KW == 3:
                    off = b.mul(b.add(row, k_one), c2)
                    _buf_store_vN(
                        b,
                        dtype,
                        d_rsrc,
                        b.select(ok, off, oob),
                        c0,
                        _trunc(accs[pr * KW + 2]),
                        2,
                        cachepolicy=ST_AUX,
                    )

            return store_row_n
        hi = b.cmp_ge(l16, ci(8))
        gran0 = b.add(b.mul(b.div(lhi, c2), c2), b.mod(lhi, c2))
        gran = b.add(gran0, b.select(hi, ci(4), c0))
        kch = b.add(
            b.add(b.mul(ctx["kb"], ci(TK)), b.mul(wk, ci(64))), b.mul(gran, ci(8))
        )
        pa, oka = pix(sa)
        pb, okb = pix(sb)
        va0 = b.mul(b.add(pa, kch), c2)
        vb0 = b.mul(b.add(pb, kch), c2)

        def store_row(accs, pr: int):
            v = []
            for cc in range(2):
                da = b.vec_bitcast(_trunc(accs[pr * KW + 2 * cc]), i32x2)
                db = b.vec_bitcast(_trunc(accs[pr * KW + 2 * cc + 1]), i32x2)
                lo0, hi0 = b.permlane32_swap(b.vec_extract(da, 0), b.vec_extract(db, 0))
                lo1, hi1 = b.permlane32_swap(b.vec_extract(da, 1), b.vec_extract(db, 1))
                v.append([lo0, lo1, hi0, hi1])
            n0, n1 = [], []
            for d in range(4):
                r0 = b.mov_dpp(v[1][d], row_shr=8)
                r1 = b.mov_dpp(v[0][d], row_shl=8)
                n0.append(b.select(hi, r0, v[0][d]))
                n1.append(b.select(hi, v[1][d], r1))
            dr = b.mul(b.mul(sd_h, c2), ci(pr))
            rok = rowok(pr)
            for vals, off, ok in ((n0, b.add(va0, dr), oka), (n1, b.add(vb0, dr), okb)):
                val = b.vec_bitcast(b.vec_pack(vals, I32), half8)
                _buf_store_vN(
                    b,
                    dtype,
                    d_rsrc,
                    b.select(b.land(ok, rok), off, oob),
                    c0,
                    val,
                    4,
                    cachepolicy=ST_AUX,
                )

        return store_row

    dcol = sd_n if UNF else sd_w  # output column stride

    def ncol_of(ctx) -> Value:
        return ctx["ncol"] if UNF else p_Wo

    def row_ok(ctx, pr: int):
        if not UNF:
            return None
        return b.cmp_lt(b.add(ctx["h0"], b.add(b.mul(wp, ci(_WP)), ci(pr))), p_Ho)

    def _trunc(acc: Value) -> Value:
        if dtype == "bf16":
            return b.vec_trunc_f32_to_bf16(acc)
        return b.vec_trunc_f32_to_f16(acc)

    # ---- prologue -------------------------------------------------------------
    ctx0 = decode(blk)
    lt0, ls0 = dma_lanes(ctx0)
    for u in range(DMA_N):
        dma_u(u, ctx0, c0, 0, lt0, ls0)
    w0 = load_w(w_tile_off(ctx0))
    if C64:
        w0b = load_w(b.add(w_tile_off(ctx0), ci(STEP_B)))
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    with b.scf_if(b.cmp_eq(grp, c1)):  # group 1 runs one barrier behind
        b.s_barrier_bare()
    b.sched_barrier(0)

    zero = b.zero_vec_f32(4)
    w_flat = [w0[r][k] for r in range(3) for k in range(KW)]
    if C64:
        w_flat += [w0b[r][k] for r in range(3) for k in range(KW)]
    NW = len(w_flat)
    ODD = spec.odd_c
    ONE = spec.one_c

    def tile_body(t: Value, carried, p: int):
        """One tile starting on LDS buffer ``p``; returns the next tile's
        carried state (first weights, DMA lane offsets)."""

        def unflat(v):
            return [list(v[r * KW : r * KW + KW]) for r in range(3)]

        wpre = unflat(carried[: 3 * KW])
        if C64:
            wpre = [wpre, unflat(carried[3 * KW : 6 * KW])]
        lt, ls = carried[NW], carried[NW + 1]
        ctx = decode(t)
        cn = decode(b.add(t, ci(PERSIST)))
        wsoff = w_tile_off(ctx)
        accs = [zero] * (_WP * KW)
        if ONE:  # the tile's only chunk DMAs the next tile's into buffer 1 - p
            lt_n, ls_n = dma_lanes(cn)
            chunk(
                accs,
                wpre,
                p,
                cn,
                c0,
                lt_n,
                ls_n,
                wsoff,
                True,
                epi=lambda: epilogue(ctx),
            )
            wn = load_w(w_tile_off(cn))
            nxt = [wn[r][k] for r in range(3) for k in range(KW)]
            if C64:
                wn = load_w(b.add(w_tile_off(cn), ci(STEP_B)))
                nxt += [wn[r][k] for r in range(3) for k in range(KW)]
            b.sched_barrier(0)
            return nxt + [lt_n, ls_n]
        accs = chunk(accs, wpre, p, ctx, c1, lt, ls, wsoff, True)
        mids = b.div(b.sub(nch, ci(3) if ODD else c2), c2)
        mid = b.scf_for_iter(
            c0,
            mids,
            c1,
            [(f"acc{i}", a) for i, a in enumerate(accs)],
            iv_name="pair",
            elide_trailing_barrier=False,
        )
        with mid as (pi, macc):
            ch = b.add(c1, b.mul(pi, c2))
            m = chunk(
                macc,
                None,
                1 - p,
                ctx,
                b.add(ch, c1),
                lt,
                ls,
                b.add(wsoff, b.mul(ch, ci(CHUNK_B))),
                False,
            )
            m = chunk(
                m,
                None,
                p,
                ctx,
                b.add(ch, c2),
                lt,
                ls,
                b.add(wsoff, b.mul(b.add(ch, c1), ci(CHUNK_B))),
                False,
            )
            b.scf_yield(*m)
        accs = list(mid.results)
        if ODD:  # one more chunk: the last one then lands on buffer p
            accs = chunk(
                accs,
                None,
                1 - p,
                ctx,
                b.sub(nch, c1),
                lt,
                ls,
                b.add(wsoff, b.mul(b.sub(nch, c2), ci(CHUNK_B))),
                False,
            )
        lt_n, ls_n = dma_lanes(cn)
        last = b.sub(nch, c1)
        accs = chunk(
            accs,
            None,
            p if ODD else 1 - p,
            cn,
            c0,
            lt_n,
            ls_n,
            b.add(wsoff, b.mul(last, ci(CHUNK_B))),
            False,
            epi=lambda: epilogue(ctx),
        )
        # the next tile's first weights (its first wait also drains the stores)
        wn = load_w(w_tile_off(cn))
        nxt = [wn[r][k] for r in range(3) for k in range(KW)]
        if C64:
            wn = load_w(b.add(w_tile_off(cn), ci(STEP_B)))
            nxt += [wn[r][k] for r in range(3) for k in range(KW)]
        b.sched_barrier(0)
        return nxt + [lt_n, ls_n]

    init = [(f"wp{i}", v) for i, v in enumerate(w_flat)] + [("lt", lt0), ("ls", ls0)]
    if not (ODD or ONE):
        tiles = b.scf_for_iter(
            c0, rounds, c1, init, iv_name="round", elide_trailing_barrier=False
        )
        with tiles as (rnd, carried):
            t = b.add(blk, b.mul(rnd, ci(PERSIST)))
            b.scf_yield(*tile_body(t, carried, 0))
    else:
        # An odd chunk count flips the starting buffer every tile: two tiles
        # per iteration (buffers 0 then 1), the second one guarded.
        twos = b.div(b.add(rounds, c1), c2)
        tiles = b.scf_for_iter(
            c0, twos, c1, init, iv_name="round2", elide_trailing_barrier=False
        )
        with tiles as (r2, carried):
            rnd = b.mul(r2, c2)
            t = b.add(blk, b.mul(rnd, ci(PERSIST)))
            nxt = tile_body(t, carried, 0)
            second = b.scf_if_else(
                b.cmp_lt(b.add(rnd, c1), rounds), [v.type for v in nxt]
            )
            with second as (then_ctx, else_ctx):
                with then_ctx:
                    b.scf_yield(*tile_body(b.add(t, ci(PERSIST)), nxt, 1))
                with else_ctx:
                    b.scf_yield(*nxt)
            b.scf_yield(*second.results)
    b.s_setprio(0)
    with b.scf_if(b.cmp_eq(grp, c0)):  # balance the barrier count
        b.s_barrier_bare()
    return b.kernel
