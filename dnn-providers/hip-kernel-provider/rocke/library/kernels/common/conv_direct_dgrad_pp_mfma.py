# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ping-pong direct dgrad on wave64 MFMA with LDS DMA (gfx950).

A dense NHWC KxK stride-1 dgrad, computed as the forward pass over the
transposed problem -- dY in, dX out, flipped padding -- by a channel-chunked
direct kernel that reads the original KRSC filter itself, flipped and
transposed, so there is no weight pre-pass. Throughout this module "C" and "K"
are the pass's: C is the reduction (dY channels), K the outputs (dX channels).

* **LDS DMA.** Every operand byte reaches LDS through ``buffer_load ... lds``
  (16 bytes per lane; 4 for the narrow tail of an input row), never through
  VGPRs and ``ds_write``. A DMA instruction writes 1 KiB lane-contiguously, so
  the bank-conflict-free layout is chosen on the *global* side. An input row of
  64 channels is 8 granules of 16 bytes, granule ``g`` of column ``x`` stored
  at slot ``g ^ (x % 8)``: a fragment read (16 columns x one granule) hits 8
  distinct slots per 8 lanes.
* **In-kernel transposed weights.** The filter is ``W[C][R][S][K]`` in this
  pass's terms, contiguous along the outputs. The weights of one tap are
  staged as ``[64 reduction rows][block_k outputs]`` and the A fragments are
  read with two ``ds_read_b64_tr_b16`` per 16x16x32 operand. Granule ``g`` of
  row ``r`` sits at slot ``g ^ 2 * (r % 4 + 4 * ((r / 8) % 2))``, which puts
  the 32 lanes of a transposing-read phase on distinct 8-byte slots of the
  256-byte bank period (rows of at least 256 bytes: block_k >= 128).
* **Two wave ranks.** The 8 waves of a block form two ranks of 4. Rank 0
  streams the weights of the *next filter tap* into a double buffer; rank 1
  streams the *next channel chunk's* input tile (the ``(TH + KH - 1) x (TW +
  KW - 1)`` halo, 64 channels) into the other input buffer, a few 1 KiB
  pieces per step, so the loads trickle under the MFMAs instead of arriving
  as one burst.
* **Ping-pong schedule.** Rank 1 starts one barrier late. Every step (one tap
  and one 32-channel half of the chunk) is ``reads + DMA issue; barrier;
  setprio 1; 16 MFMAs; setprio 0; barrier``, so while one rank's MFMAs run at
  raised priority the other rank reads its fragments and issues its DMA.

The DMA completion protocol is explicit (``s_waitcnt vmcnt``): rank 0 waits
for the next tap's weights after the second half of the current tap, rank 1
for the next chunk's input before the last step of the chunk. Every buffer is
its own LDS allocation with its own alias scope, and the chunk loop is
unrolled twice so each read names its buffer statically: the waitcnt pass then
only waits for DMA into the buffer a read actually touches.

Wave tile: rank-agnostic ``wave_j = wave % 2`` owns half the pixel rows of the
16x16 output tile (8 rows of 16 pixels: 8 B fragments), ``wave_k = wave / 2``
a quarter of the block's output channels (``block_k / 64`` A fragments); 16
``mfma_f32_16x16x32`` per step and A fragment.

AOT: batch, extents, C and K are kernargs (the forward direct-conv kernarg
block over the transposed problem); the filter, padding, dtype and tile are
baked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.arch import ArchTarget
from rocke.core.ir import F32, I64, IRBuilder, KernelDef, Value, VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    I32_MAX,
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    _mfma,
    _trunc_f32,
    dgrad_4c_transposed_problem,
    emit_direct_params,
    one_padding_reason,
    problem_dims_reason,
)

__all__ = [
    "DgradPpMfmaSpec",
    "build_dgrad_pp_mfma",
    "dgrad_pp_mfma_problem",
    "dgrad_pp_mfma_reason",
    "dgrad_pp_mfma_tile_efficiency",
    "is_valid_dgrad_pp_mfma_spec",
]

_WAVE = 64
_CK = 64  # channels per chunk
_GRAN = 8  # halves per 16-byte granule
_TH = 16  # output rows per block
_TW = 16  # output columns per block
_WAVES = 8
_RANK_WAVES = 4
_ATOM = 16
_PARTS_C = _CK // 32  # 32-channel MFMA steps per chunk and tap
BLOCK_K = (128, 256)


# The forward problem the dgrad of a problem runs as: the 4c dgrad's
# transposition (dY in, dX out, PAD and an explicit pad_w flipped; the stride
# is kept, so the validator rejects a strided dgrad).
dgrad_pp_mfma_problem = dgrad_4c_transposed_problem


@dataclass(frozen=True)
class DgradPpMfmaSpec:
    """One configuration of :func:`build_dgrad_pp_mfma`.

    ``problem`` is the forward pass the dgrad runs as
    (:func:`dgrad_pp_mfma_problem`): cpg = the dgrad's K, kpg = its C.
    """

    problem: DirectConvProblem
    name: str = "dgrad_pp_mfma"
    block_k: int = 128  # output channels per block (4 waves x wave_k)
    wave_size: int = 64

    @property
    def wave_k(self) -> int:
        return self.block_k // 4

    @property
    def k_frags(self) -> int:  # A fragments (16 channels) per wave
        return self.wave_k // _ATOM

    @property
    def in_h(self) -> int:
        return _TH + self.problem.KH - 1

    @property
    def in_w(self) -> int:
        return _TW + self.problem.KW - 1

    @property
    def in_rows(self) -> int:  # staged rows, rounded up to whole DMA groups
        return ceil_div(self.in_h, _RANK_WAVES) * _RANK_WAVES

    @property
    def in_elems(self) -> int:  # one input buffer
        return self.in_rows * self.in_w * _CK

    @property
    def w_elems(self) -> int:  # one weight buffer
        return self.block_k * _CK

    def lds_bytes(self) -> int:
        return 2 * (2 * self.in_elems + 2 * self.w_elems)

    @property
    def threads_per_block(self) -> int:
        return _WAVES * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"bk{self.block_k}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        n = (
            p.N
            * ceil_div(p.Ho, _TH)
            * ceil_div(p.Wo, _TW)
            * ceil_div(p.total_k, self.block_k)
        )
        return (n, 1, 1)

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)


def dgrad_pp_mfma_tile_efficiency(spec: DgradPpMfmaSpec) -> float:
    """Real work over the padded tile grid (pixels x output channels)."""
    p = spec.problem
    return (p.Ho * p.Wo * p.total_k) / (
        ceil_div(p.Ho, _TH)
        * _TH
        * ceil_div(p.Wo, _TW)
        * _TW
        * ceil_div(p.total_k, spec.block_k)
        * spec.block_k
    )


def dgrad_pp_mfma_reason(spec: DgradPpMfmaSpec, arch: str = "gfx950") -> "str | None":
    p = spec.problem
    target = ArchTarget.from_gfx(arch)
    if target.wave_size != 64 or spec.wave_size != 64:
        return "dgrad_pp_mfma is a wave64 MFMA kernel"
    if not target.memory.has_ds_read_tr:
        return f"dgrad_pp_mfma needs ds_read_b64_tr_b16 (absent on {arch})"
    if target.async_lds_max_dwords < 4:
        return f"dgrad_pp_mfma needs the 16-byte LDS DMA (absent on {arch})"
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
    if p.groups != 1:
        return "groups must be 1"
    if p.stride != 1:
        return "stride must be 1"
    if p.dil_h != 1 or p.dil_w != 1:
        return "dilation must be 1"
    if p.KH != p.KW or p.KH < 3 or p.KH % 2 == 0:
        return "square odd filters >= 3 only"
    if p.PAD != (p.KH - 1) // 2:
        return f"'same' padding only: PAD must be {(p.KH - 1) // 2} (got {p.PAD})"
    why = one_padding_reason(p)
    if why is not None:
        return why
    if p.cpg % _GRAN:
        return f"C {p.cpg} must be a multiple of {_GRAN} (16-byte granules)"
    if p.kpg % 4:
        return f"K {p.kpg} must be a multiple of 4 (8-byte output writes)"
    if spec.block_k not in BLOCK_K:
        return "block_k must be 128 or 256 (256-byte weight rows for the swizzle)"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    return _dgrad_pp_mfma_limits_reason(spec)


def _dgrad_pp_mfma_limits_reason(spec: DgradPpMfmaSpec) -> "str | None":
    """The i32 limits: the buffer descriptors span one input image, one
    output image and the whole filter, and a masked lane reads at byte offset
    ``I32_MAX``, which is out of range only while each of them is
    at most ``I32_MAX`` bytes; the flat block id is an i32 too."""
    p = spec.problem
    for what, nbytes in (
        ("input image", 2 * p.H * p.W * p.total_c),
        ("output image", 2 * p.Ho * p.Wo * p.total_k),
        ("filter", 2 * p.total_k * p.KH * p.KW * p.cpg),
    ):
        if nbytes > I32_MAX:
            return f"{what} exceeds {I32_MAX} B, the i32 buffer range"
    blocks = spec.grid()[0]
    if blocks > I32_MAX:
        return f"{blocks} workgroups exceed the i32 block id"
    return None


def is_valid_dgrad_pp_mfma_spec(
    spec: DgradPpMfmaSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    why = dgrad_pp_mfma_reason(spec, arch)
    return (why is None, why or "ok")


def _input_pieces(in_rows: int, in_w: int) -> List[List[Tuple[int, int, int]]]:
    """The DMA pieces of one input buffer, grouped 4 per step slot.

    A piece is ``(row, first column, dwords per lane)``: 8 columns (1 KiB) per
    16-byte piece, and the row's last ``in_w % 8`` (even) columns as 2-column
    4-byte pieces. The 4 pieces of a group share the column and width and take
    consecutive rows, one per rank-1 wave, so a wave's piece is ``group row +
    wave``; ``in_rows`` is a whole number of groups (the rows past the halo
    are never read).
    """
    cols = [(c0, 4) for c0 in range(0, in_w - 7, 8)]
    cols += [(c0, 1) for c0 in range(in_w - in_w % 8, in_w, 2)]
    groups = []
    for c0, dw in cols:
        for r0 in range(0, in_rows, _RANK_WAVES):
            groups.append([(r0 + g, c0, dw) for g in range(_RANK_WAVES)])
    return groups


def _tr_swz(b: IRBuilder, row: Value) -> Value:
    """Granule XOR of weight row ``row``, 2 * (row % 4 + 4 * ((row // 8) % 2)):
    the 32 lanes of a transposing-read phase (rows 8b + i, b in 0..1, i in
    0..3, column pairs 4j) land on 32 distinct 8-byte slots of a 256-byte bank
    period. Not an entry of ``rocke.helpers.layouts.XOR_SWIZZLE_TABLE``, whose
    byte-offset swizzles cover the listed MFMA tile shapes but not this
    64-row tile read across rows by ``ds_read_b64_tr_b16``; the C++ mirror is
    ``tr_swz`` in ``conv_direct_dgrad_pp_mfma.cpp``."""
    lo = b.mod(row, b.const_i32(4))
    hi = b.mod(b.div(row, b.const_i32(8)), b.const_i32(2))
    return b.mul(b.add(lo, b.mul(hi, b.const_i32(4))), b.const_i32(2))


def build_dgrad_pp_mfma(spec: DgradPpMfmaSpec, arch: str = "gfx950") -> KernelDef:
    why = dgrad_pp_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DgradPpMfmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    dtype = p.dtype
    io_type = _io_type(dtype)
    BK = spec.block_k
    KF = spec.k_frags
    IN_H, IN_W = spec.in_h, spec.in_w
    ROW_E = IN_W * _CK  # elements per staged input row
    TAPS = KH * KW
    STEPS = TAPS * _PARTS_C  # MFMA steps per chunk
    W_PIECES = BK * _CK * 2 // 1024  # 1 KiB weight pieces per tap
    W_PER_WAVE = W_PIECES // _RANK_WAVES
    GPR = BK // _GRAN  # weight granules per row
    RPI = 1024 // (BK * 2)  # weight rows per 1 KiB piece
    in_groups = _input_pieces(spec.in_rows, IN_W)
    # Spread the input groups over the first steps of a chunk; the last steps
    # leave the DMA time to land before the chunk-end wait.
    in_sched: List[List[int]] = [[] for _ in range(STEPS)]
    span = max(1, STEPS - 4)
    for i in range(len(in_groups)):
        in_sched[i * span // len(in_groups)].append(i)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block
    b.kernel.attrs["lds_alias_scopes"] = True
    b.kernel.attrs["waves_per_eu"] = 2
    params = emit_direct_params(b, io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci(I32_MAX)

    p_Hi, p_Wi = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    C, K = params["p_total_c"], params["p_total_k"]
    sa_n, sa_h, sa_w = (
        params["p_A_stride_n"],
        params["p_A_stride_hi"],
        params["p_A_stride_wi"],
    )
    sd_n, sd_h, sd_w = (
        params["p_D_stride_n"],
        params["p_D_stride_ho"],
        params["p_D_stride_wo"],
    )

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    wave_j = b.mod(wave, c2)
    wave_k = b.div(wave, c2)
    wave_g = b.mod(wave, ci(_RANK_WAVES))
    rank = b.div(wave, ci(_RANK_WAVES))

    # ---- grid decode: block -> (k tile fastest, w tile, h tile, image) ------
    n_kt = b.div(b.add(K, ci(BK - 1)), ci(BK))
    n_wt = b.div(b.add(p_Wo, ci(_TW - 1)), ci(_TW))
    n_ht = b.div(b.add(p_Ho, ci(_TH - 1)), ci(_TH))
    blk = b.block_id_x()
    k_tile = b.mod(blk, n_kt)
    rest = b.div(blk, n_kt)
    w_tile = b.mod(rest, n_wt)
    rest = b.div(rest, n_wt)
    h_tile = b.mod(rest, n_ht)
    n_img = b.div(rest, n_ht)
    oh0 = b.mul(h_tile, ci(_TH))
    ow0 = b.mul(w_tile, ci(_TW))
    k0 = b.mul(k_tile, ci(BK))

    # Per-image buffer bases keep every offset inside 32 bits.
    a_img = b.global_ptr_offset(
        params["A"], b.mul(b.zext(n_img, I64), b.zext(b.mul(sa_n, c2), I64))
    )
    d_img = b.global_ptr_offset(
        params["D"], b.mul(b.zext(n_img, I64), b.zext(b.mul(sd_n, c2), I64))
    )
    a_rsrc = b.buffer_rsrc(a_img, b.mul(sa_n, c2))
    w_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    d_rsrc = b.buffer_rsrc(d_img, b.mul(sd_n, c2))

    # ---- LDS ----------------------------------------------------------------
    in_bufs = [
        b.smem_alloc(io_type, [1, spec.in_elems], name_hint=f"in{i}") for i in range(2)
    ]
    w_bufs = [
        b.smem_alloc(io_type, [1, spec.w_elems], name_hint=f"w{i}") for i in range(2)
    ]
    in_addr = [b.smem_addr_of(s) for s in in_bufs]
    w_addr = [b.smem_addr_of(s) for s in w_bufs]

    def lds_at(base: Value, byte_off: Value) -> Value:
        return b.smem_ptr_add(base, b.zext(byte_off, I64))

    # ---- rank 0: weight DMA geometry ----------------------------------------
    # [64 reduction rows][BK outputs] per tap, 16-byte granule g of row r
    # stored at slot g ^ _tr_swz(r). Piece q covers rows q * RPI ..; lane l
    # writes row q * RPI + l // GPR, slot l % GPR.
    row_len = b.mul(K, ci(TAPS))  # filter row: TAPS x this pass's outputs
    w_pieces = []
    for i in range(W_PER_WAVE):
        q = b.add(b.mul(wave_g, ci(W_PER_WAVE)), ci(i))
        row = b.add(b.mul(q, ci(RPI)), b.div(lane, ci(GPR)))
        gran = b.xor(b.mod(lane, ci(GPR)), _tr_swz(b, row))
        cc = b.add(k0, b.mul(gran, ci(_GRAN)))
        w_pieces.append(
            (
                row,
                b.add(b.mul(row, row_len), cc),
                b.cmp_lt(cc, K),
                b.mul(q, ci(1024)),
            )
        )

    def issue_weights(buf: int, tap: int, c_base: Value):
        """DMA the weights of ``tap`` for reduction rows ``c_base ..`` into
        buffer ``buf``: the flipped filter tap, rows past C zero-filled."""
        r, s_ = divmod(tap, KW)
        tap_f = (KH - 1 - r) * KW + (KW - 1 - s_)
        base = b.add(b.mul(c_base, row_len), b.mul(ci(tap_f), K))
        for row, off, k_ok, lds_off in w_pieces:
            ok = b.land(k_ok, b.cmp_lt(b.add(c_base, row), C))
            voff = b.select(ok, b.mul(b.add(off, base), c2), oob)
            b.async_buffer_load_lds(w_rsrc, lds_at(w_addr[buf], lds_off), voff, c0, 4)

    # ---- rank 1: input DMA geometry -----------------------------------------
    # A 16-byte piece covers 8 columns: lane l -> column c0 + l / 8, slot l % 8,
    # logical granule (l % 8) ^ ((c0 + l / 8) % 8) = (l % 8) ^ (l / 8).
    # A 4-byte piece covers 2 columns: lane l -> column c0 + l / 32, element
    # (l % 32) * 2 of the column, slot (l % 32) / 4, logical granule
    # slot ^ ((c0 + l / 32) % 8).
    l_row = b.div(lane, ci(8))
    l_gran = b.xor(b.mod(lane, ci(8)), l_row)
    ih0 = b.sub(oh0, ci(PAD))
    iw0 = b.sub(ow0, ci(PAD))
    lane_geo = {}
    for c_first, dw in sorted({(g[0][1], g[0][2]) for g in in_groups}):
        if dw == 4:
            col = b.add(ci(c_first), l_row)
            ch = b.mul(l_gran, ci(_GRAN))
        else:
            col = b.add(ci(c_first), b.div(lane, ci(32)))
            e = b.mul(b.mod(lane, ci(32)), c2)
            gran = b.xor(b.div(e, ci(_GRAN)), b.mod(col, ci(8)))
            ch = b.add(b.mul(gran, ci(_GRAN)), b.mod(e, ci(_GRAN)))
        iw = b.add(iw0, col)
        ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, p_Wi))
        lane_geo[(c_first, dw)] = (b.add(b.mul(iw, sa_w), ch), ch, ok)

    def issue_input_group(buf: int, gi: int, c_base: Value):
        """DMA input piece group ``gi`` of the chunk at ``c_base`` into ``buf``."""
        r_first, c_first, dw = in_groups[gi][0]
        off, ch, ok = lane_geo[(c_first, dw)]
        row = b.add(ci(r_first), wave_g)
        ih = b.add(ih0, row)
        row_ok = b.land(
            b.cmp_lt(row, ci(IN_H)), b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, p_Hi))
        )
        c_ok = b.cmp_lt(b.add(c_base, ch), C)
        g_off = b.add(b.add(off, b.mul(ih, sa_h)), c_base)
        voff = b.select(b.land(b.land(ok, row_ok), c_ok), b.mul(g_off, c2), oob)
        lds_off = b.add(b.mul(row, ci(ROW_E * 2)), ci(c_first * _CK * 2))
        b.async_buffer_load_lds(a_rsrc, lds_at(in_addr[buf], lds_off), voff, c0, dw)

    # ---- fragment reads -----------------------------------------------------
    # A (weights), transposing reads: lane (tb, ti, tj) = (l // 16, (l // 4) %
    # 4, l % 4) reads outputs m0 + 4 tj .. +3 of reduction row 8 tb + ti (+ 4
    # for the second read); the hardware hands lane l the reduction values
    # 8 tb .. 8 tb + 7 of output m0 + l % 16 -- the A operand.
    tb = b.div(lane, ci(16))
    ti = b.mod(b.div(lane, ci(4)), ci(4))
    tj = b.mod(lane, ci(4))
    a_idx = {}
    for kf in range(KF):
        col = b.add(
            b.mul(wave_k, ci(spec.wave_k)), b.add(ci(kf * _ATOM), b.mul(tj, ci(4)))
        )
        for part in range(_PARTS_C):
            for rd in range(2):
                row = b.add(b.add(ci(part * 32 + 4 * rd), b.mul(tb, ci(8))), ti)
                slot = b.xor(b.div(col, ci(_GRAN)), _tr_swz(b, row))
                a_idx[(kf, part, rd)] = b.add(
                    b.mul(row, ci(BK)),
                    b.add(b.mul(slot, ci(_GRAN)), b.mod(col, ci(_GRAN))),
                )
    # B (input): pixel row h = wave_j * 8 + jb, column w = lane % 16; tap (r, s)
    # reads staged row h + r, column w + s, slot granule ^ ((w + s) % 8).
    l16 = b.mod(lane, ci(16))
    b_lane = {}
    for s in range(KW):
        col = b.add(l16, ci(s))
        for part in range(_PARTS_C):
            g = b.add(ci(part * 4), tb)
            slot = b.xor(g, b.mod(col, ci(8)))
            b_lane[(s, part)] = b.add(b.mul(col, ci(_CK)), b.mul(slot, ci(_GRAN)))
    row0 = b.mul(wave_j, ci(8 * ROW_E))

    def read_a(buf: int, kf: int, part: int) -> Value:
        lo = b.ds_read_tr16_b64(w_bufs[buf], c0, a_idx[(kf, part, 0)], dtype=io_type)
        hi = b.ds_read_tr16_b64(w_bufs[buf], c0, a_idx[(kf, part, 1)], dtype=io_type)
        return b.vec_concat(lo, hi)

    def read_b(buf: int, jb: int, r: int, s: int, part: int) -> Value:
        idx = b.add(b_lane[(s, part)], b.add(row0, ci((jb + r) * ROW_E)))
        return b.smem_load_vN(in_bufs[buf], c0, idx, dtype=io_type, n=_GRAN)

    # ---- one chunk ----------------------------------------------------------
    n_acc = KF * 8

    def chunk(accs: List[Value], par: int, c_base: Value, my_rank: int) -> List[Value]:
        """One 64-channel chunk for wave rank ``my_rank``; ``par`` = chunk parity.

        Input buffer ``par`` holds this chunk; weight buffer of tap t is
        ``(par * TAPS + t) % 2`` (taps alternate buffers across chunks).
        """
        accs = list(accs)
        c_next = b.add(c_base, ci(_CK))
        for step in range(STEPS):
            tap, part = divmod(step, _PARTS_C)
            r, s = divmod(tap, KW)
            wbuf = (par * TAPS + tap) % 2
            a_fr = [read_a(wbuf, kf, part) for kf in range(KF)]
            b_fr = [read_b(par, jb, r, s, part) for jb in range(8)]
            if my_rank == 0 and part == 0:
                # Next tap's weights (tap 0 of the next chunk after the last).
                if tap + 1 < TAPS:
                    issue_weights(1 - wbuf, tap + 1, c_base)
                else:
                    issue_weights(1 - wbuf, 0, c_next)
            if my_rank == 1:
                for gi in in_sched[step]:
                    issue_input_group(1 - par, gi, c_next)
                if step == STEPS - 1:
                    # The next chunk's input, before rank 0 reads it.
                    b.s_waitcnt(vmcnt=0)
            b.sched_barrier(0)
            b.s_barrier_bare()
            b.sched_barrier(0)
            b.s_setprio(1)
            order = range(8) if step % 2 == 0 else range(7, -1, -1)
            for kf in range(KF):
                for jb in order:
                    i = kf * 8 + jb
                    accs[i] = _mfma(b, dtype, "16x16x32", a_fr[kf], b_fr[jb], accs[i])
            b.s_setprio(0)
            b.sched_barrier(0)
            if my_rank == 0 and part == _PARTS_C - 1:
                # The next tap's weights, before either rank reads them.
                b.s_waitcnt(vmcnt=0)
            b.s_barrier_bare()
            b.sched_barrier(0)
        return accs

    n_chunks = b.div(b.add(C, ci(_CK - 1)), ci(_CK))
    n_pairs = b.div(n_chunks, c2)
    zero = b.zero_vec_f32(4)
    acc_t = VectorType(F32, 4)

    def rank_body(my_rank: int) -> List[Value]:
        # Prologue: tap 0 weights (rank 0) and chunk 0's input (rank 1).
        if my_rank == 0:
            issue_weights(0, 0, c0)
        else:
            for gi in range(len(in_groups)):
                issue_input_group(0, gi, c0)
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        if my_rank == 1:
            b.s_barrier_bare()  # rank 1 runs one barrier behind
        b.sched_barrier(0)
        loop = b.scf_for_iter(
            c0,
            n_pairs,
            c1,
            [(f"acc{my_rank}_{i}", zero) for i in range(n_acc)],
            iv_name=f"pair{my_rank}",
            elide_trailing_barrier=False,
        )
        with loop as (pi, carried):
            c_even = b.mul(pi, ci(2 * _CK))
            accs = chunk(list(carried), 0, c_even, my_rank)
            accs = chunk(accs, 1, b.add(c_even, ci(_CK)), my_rank)
            b.scf_yield(*accs)
        accs = list(loop.results)
        odd = b.cmp_eq(b.mod(n_chunks, c2), c1)
        tail = b.scf_if_else(odd, result_types=[acc_t] * n_acc)
        with tail as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(*chunk(accs, 0, b.mul(n_pairs, ci(2 * _CK)), my_rank))
            with else_ctx:
                b.scf_yield(*accs)
        accs = list(tail.results)
        if my_rank == 0:
            b.s_barrier_bare()  # pair rank 1's last barrier
        return accs

    branch = b.scf_if_else(b.cmp_eq(rank, c0), result_types=[acc_t] * n_acc)
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(*rank_body(0))
        with else_ctx:
            b.scf_yield(*rank_body(1))
    accs = list(branch.results)

    # ---- epilogue: lane holds channels tb*4 .. +3 of pixel (h, lane % 16) ---
    k_lane = b.add(b.add(k0, b.mul(wave_k, ci(spec.wave_k))), b.mul(tb, ci(4)))
    ow = b.add(ow0, l16)
    w_ok = b.cmp_lt(ow, p_Wo)
    for jb in range(8):
        oh = b.add(oh0, b.add(b.mul(wave_j, ci(8)), ci(jb)))
        hw_ok = b.land(w_ok, b.cmp_lt(oh, p_Ho))
        pix = b.add(b.mul(oh, sd_h), b.mul(ow, sd_w))
        for kf in range(KF):
            k = b.add(k_lane, ci(kf * _ATOM))
            ok = b.land(hw_ok, b.cmp_lt(k, K))
            voff = b.select(ok, b.mul(b.add(pix, k), c2), oob)
            val = _trunc_f32(b, dtype, accs[kf * 8 + jb])
            _buf_store_vN(b, dtype, d_rsrc, voff, c0, val, 2)
    return b.kernel
