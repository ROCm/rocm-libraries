# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Forward convolutions with a tiny channel count on wave32 WMMA (gfx1250).

Two direct forward kernels for the layers where implicit GEMM leaves most of
every MMA tile empty:

``build_fwd_smallc_wmma`` -- few *input* channels (stem convolutions, C <= 4).
    The GEMM is ``Y[px, k] = sum_j W[k, j] * im2col(X)[j, px]`` with the
    reduction ``j = (r, s, c)`` flattened over the whole filter, so a 3x3x3
    stem is a single 16x16x32 WMMA step. For a small C an input row is a short
    contiguous run of ``W * C`` elements in NHWC: a block stages its output
    tile's input window (rows x columns x C) with one 2-D TDM transfer into
    LDS, and each lane gathers its im2col fragment straight from that window.
    The weights of the block's output channels stay in registers for every
    tile it serves (``tiles_per_block`` tiles with a double-buffered window).

    Fragment layout: lane half ``h`` of a WMMA operand holds reduction
    indices ``h*16 .. h*16+15``. Each filter row's ``KW*C`` contiguous window
    elements are split in two runs of ``E0 = ceil(KW*C / 2)``: half 0 takes
    the first run of every filter row, half 1 the second run, which is the
    same window offsets shifted by ``E0``. So a lane's fragment is read at
    compile-time offsets from one lane base. When ``KW*C`` is odd the last
    slot of half 1 reads one element past the filter row; its weight is zero.

    Image edges: the window's first column lands at a runtime LDS offset that
    keeps the TDM destination 16-byte aligned; the transfer is clipped to the
    window so the cells left / above the image (zeroed once per tile) and
    right / below it (zero-filled by the hardware) read as padding.

``build_fwd_smallk_wmma`` -- few *output* channels (K <= 8). The forward
    mirror of the small-C dgrad: ``Z[q, (r, s, k)] = sum_c X[q, c] W[k, r, s, c]``
    for every input pixel ``q`` of the output tile's window (a 1x1 GEMM with
    the taps and output channels as its M dimension), then the overlap-add
    ``Y[p, k] = sum_{r,s} Z[p*S - PAD + (r, s), (r, s, k)]`` in fp32 LDS.
    Input channel chunks stream through a TDM ring.

AOT: batch, extents and the *other* channel count are kernargs; filter,
stride, padding, the small channel count and the tile geometry are baked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, IRBuilder, KernelDef, Value

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)

WMMA_K = 32
CH_PAD = 8
SMALL_WMMA_VGPR_BUDGET = 448
SMALLC_MAX_C = 4
SMALLK_MAX_K = 8
_I32_MAX = (1 << 31) - 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _tdm_descriptor_3d(
    b: IRBuilder,
    global_ptr: Value,
    lds_addr: Value,
    *,
    dims: Sequence[Value],
    strides: Sequence[Value],
    tile0: int,
    tile1: int,
    tile2: Value,
) -> tuple[Value, Value, Value, Value, Value]:
    """The five D# groups of a rank-3 TDM tile transfer (2-byte elements).

    Dimension 0 is contiguous; ``strides`` are the element strides of
    dimensions 1 and 2, ``dims`` the extents left from ``global_ptr`` (the
    hardware zero-fills the tile past them). The tile lands packed
    ``[tile2][tile1][tile0]`` at the i32 LDS byte address ``lds_addr``.
    ``tile2`` may be a runtime value.
    """
    c16 = b.const_i32(16)
    d0, d1, d2 = dims
    s1, s2 = strides
    gaddr = b.global_ptr_to_i64(global_ptr)
    g_hi = b.trunc(b.lshr(gaddr, b.const_i64(32)), I32)
    group0 = (
        b.const_i32(1),
        lds_addr,
        b.trunc(gaddr, I32),
        b.lor(b.land(g_hi, b.const_i32(0x1FFFFFF)), b.const_i32(-(1 << 31))),
    )
    group1 = (
        b.const_i32(1 << 16),  # data_size = 2 bytes
        b.shl(d0, c16),
        b.lor(b.lshr(d0, c16), b.shl(d1, c16)),
        b.lor(b.lshr(d1, c16), b.const_i32(tile0 << 16)),
        b.lor(b.const_i32(tile1), b.shl(tile2, c16)),
        s1,
        b.shl(s2, c16),
        b.lshr(s2, c16),
    )
    group2 = (d2, b.const_i32(0), b.const_i32(0), b.const_i32(0))

    def pk(words):
        return b.vec_pack([b.readfirstlane(v) for v in words], I32)

    return pk(group0), pk(group1), pk(group2), b.zero_vec(I32, 4), b.zero_vec(I32, 8)


def _wmma_op(p: DirectConvProblem, arch: str):
    ab = "bf16" if p.dtype == "bf16" else "f16"
    return ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )


def _common_reason(p: DirectConvProblem, arch: str, name: str) -> str | None:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    if target.wave_size != 32:
        return f"{name} is a wave32 kernel"
    if not target.memory.has_tdm:
        return f"no TDM on {arch}"
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing wmma_f32_16x16x32_{ab} on {arch}"
    if p.groups != 1 or p.is_dilated:
        return "ungrouped, undilated convolutions only"
    if p.KH != p.KW:
        return "square filters only"
    if not 0 <= p.PAD <= p.KH - 1:
        return f"PAD {p.PAD} outside 0..K-1"
    tensor_bytes = (
        2 * p.N * p.H * p.W * p.total_c,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.Ho * p.Wo * p.total_k,
    )
    if max(tensor_bytes) > _I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
    return None


# --------------------------------------------------------------------------
# Small C: im2col from an LDS input window, one GEMM over (r, s, c).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FwdSmallcWmmaSpec:
    """One configuration of :func:`build_fwd_smallc_wmma`."""

    problem: DirectConvProblem
    name: str = "fwd_smallc_wmma"
    waves_k: int = 1  # waves along output channels
    waves_p: int = 4  # waves along output pixels
    tiles_k: int = 2  # 16-channel tiles per wave
    tiles_p: int = 4  # 16-pixel tiles per wave
    out_h: int = 8  # output rows per tile
    out_w: int = 32  # output columns per tile
    tiles_per_block: int = 1  # output tiles a block walks (weights stay resident)
    tdm_store: bool = False  # stage the output tile in LDS, one TDM store
    sched: str | None = None
    wave_size: int = 32

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def run(self) -> int:  # E0: window elements per filter row and lane half
        p = self.problem
        return _ceil_div(p.KW * p.cpg, 2)

    @property
    def steps(self) -> int:
        return _ceil_div(self.problem.KH * self.run, 16)

    @property
    def win_h(self) -> int:
        return (self.out_h - 1) * self.S + self.problem.KH

    @property
    def win_w(self) -> int:  # + one column for the odd-run slot
        return (self.out_w - 1) * self.S + self.problem.KW + 1

    @property
    def row_pitch(self) -> int:
        # A row lands at most 7 elements right of its first column (the
        # aligned landing offset) and must hold the whole window row.
        return _round_up(self.win_w * self.problem.cpg, 8) + 8

    @property
    def buf_elems(self) -> int:
        # The tile is ``win_h`` rows from the landing row (up to PAD rows
        # down) plus the last row's wrap of up to one landing offset.
        return (self.win_h + self.problem.PAD + 1) * self.row_pitch

    @property
    def nbuf(self) -> int:
        return 2 if self.tiles_per_block > 1 else 1

    @property
    def block_k(self) -> int:
        return self.waves_k * self.tiles_k * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def threads_per_block(self) -> int:
        return self.waves_k * self.waves_p * self.wave_size

    @property
    def out_elems(self) -> int:
        return self.block_p * self.block_k if self.tdm_store else 0

    def lds_bytes(self) -> int:
        return 2 * (self.nbuf * self.buf_elems + self.out_elems)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"s{p.stride}",
            f"p{p.PAD}",
            f"c{p.cpg}",
            f"wk{self.waves_k}",
            f"wp{self.waves_p}",
            f"tk{self.tiles_k}",
            f"tp{self.tiles_p}",
            f"oh{self.out_h}",
            f"ow{self.out_w}",
            f"tb{self.tiles_per_block}",
            "ts" if self.tdm_store else "",
            f"s{self.sched}" if self.sched else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def n_tiles(self) -> int:
        p = self.problem
        return p.N * _ceil_div(p.Ho, self.out_h) * _ceil_div(p.Wo, self.out_w)

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (
            _ceil_div(self.n_tiles(), self.tiles_per_block),
            _ceil_div(p.total_k, self.block_k),
            1,
        )

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def fwd_smallc_tile_efficiency(spec: FwdSmallcWmmaSpec) -> float:
    p = spec.problem
    oh, ow = spec.out_h, spec.out_w
    return (p.Ho * p.Wo) / (_ceil_div(p.Ho, oh) * oh * _ceil_div(p.Wo, ow) * ow)


def fwd_smallc_reason(spec: FwdSmallcWmmaSpec, arch: str = "gfx1250") -> str | None:
    p = spec.problem
    why = _common_reason(p, arch, "fwd_smallc_wmma")
    if why:
        return why
    target = ArchTarget.from_gfx(arch)
    if spec.wave_size != 32:
        return "fwd_smallc_wmma is a wave32 kernel"
    if p.cpg > SMALLC_MAX_C:
        return f"C {p.cpg} > {SMALLC_MAX_C}"
    if p.kpg % 8:
        return f"K {p.kpg} must be a multiple of 8 (16-byte output rows)"
    if min(spec.waves_k, spec.waves_p, spec.tiles_k, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.tiles_per_block < 1:
        return "tiles_per_block must be >= 1"
    if spec.out_h * spec.out_w != spec.block_p:
        return (
            f"out tile {spec.out_h}x{spec.out_w} must hold exactly the "
            f"{spec.block_p} GEMM pixels"
        )
    if spec.row_pitch > 0xFFFF or spec.win_h > 0xFFFF:
        return "window too large for one TDM tile"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 8 * spec.tiles_k * (spec.tiles_p + spec.steps) + 8 * spec.tiles_p
    if vgprs > SMALL_WMMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    return None


def is_valid_fwd_smallc_spec(
    spec: FwdSmallcWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = fwd_smallc_reason(spec, arch)
    return (why is None, why or "ok")


def build_fwd_smallc_wmma(spec: FwdSmallcWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    why = fwd_smallc_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdSmallcWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, S, PAD, C = p.KH, p.KW, p.stride, p.PAD, p.cpg
    NC = KH * KW * C
    RUN = KW * C  # window elements of one filter row
    E0 = spec.run
    STEPS = spec.steps
    io_type = _io_type(p.dtype)
    wmma_op = _wmma_op(p, arch)

    WAVE = spec.wave_size
    WK = spec.waves_k
    TM, TN = spec.tiles_k, spec.tiles_p
    THREADS = spec.threads_per_block
    BK = spec.block_k
    TOH, TOW = spec.out_h, spec.out_w
    WIN_H = spec.win_h
    WP = spec.row_pitch
    BUF = spec.buf_elems
    NB = spec.nbuf
    T = spec.tiles_per_block
    OUT0 = NB * BUF  # output staging tile (tdm_store), in 2-byte elements

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_N = params["p_N"]
    c_K = params["p_total_k"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]
    c_C = b.const_i32(C)

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_k = b.mod(wave_id, b.const_i32(WK))
    wave_p = b.div(wave_id, b.const_i32(WK))

    n_ht = b.div(b.add(c_Ho, b.const_i32(TOH - 1)), b.const_i32(TOH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(TOW - 1)), b.const_i32(TOW))
    n_tiles = b.mul(b.mul(c_N, n_ht), n_wt)
    k_blk = b.mul(b.block_id_y(), b.const_i32(BK))

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="sc_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)

    # ---- weights: A fragments of every (m tile, step), resident.
    zero_e = b.vec_extract(b.zero_vec(io_type, 2), 0)
    a_frags = []
    for m in range(TM):
        k = b.add(
            k_blk,
            b.add(
                b.mul(b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16), l16
            ),
        )
        k_ok = b.cmp_lt(k, c_K)
        k_row = b.mul(k, b.const_i32(NC))
        per_step = []
        for t in range(STEPS):
            elems = []
            for i in range(16):
                g = t * 16 + i
                if g >= KH * E0:
                    elems.append(zero_e)
                    continue
                r, e = divmod(g, E0)
                ok0 = e < RUN
                ok1 = e + E0 < RUN
                if not ok0 and not ok1:
                    elems.append(zero_e)
                    continue
                j = b.add(b.const_i32(r * RUN + e), b.mul(lhalf, b.const_i32(E0)))
                ok = k_ok
                if not ok1:
                    ok = b.land(ok, b.cmp_eq(lhalf, c0))
                elems.append(
                    b.masked_global_load(
                        Bp, b.add(k_row, j), ok, zero_e, io_type, align=2
                    )
                )
            per_step.append(b.vec_pack(elems, io_type))
        a_frags.append(per_step)

    # ---- per-lane pixel geometry (tile independent).
    lane_px = []  # (ty, tx) of each pixel tile's lane pixel
    for tn in range(TN):
        q = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        lane_px.append((b.div(q, b.const_i32(TOW)), b.mod(q, b.const_i32(TOW))))
    lane_off = []  # LDS element offset of tap (0, 0), half run included
    for ty, tx in lane_px:
        lane_off.append(
            b.add(
                b.add(
                    b.mul(b.mul(ty, b.const_i32(S)), b.const_i32(WP)),
                    b.mul(tx, b.const_i32(S * C)),
                ),
                b.mul(lhalf, b.const_i32(E0)),
            )
        )

    def _tile_geom(tile: Value):
        tw = b.mod(tile, n_wt)
        rest = b.div(tile, n_wt)
        th = b.mod(rest, n_ht)
        n_img = b.div(rest, n_ht)
        oh0 = b.mul(th, b.const_i32(TOH))
        ow0 = b.mul(tw, b.const_i32(TOW))
        h_start = b.sub(b.mul(oh0, b.const_i32(S)), b.const_i32(PAD))
        w_start = b.sub(b.mul(ow0, b.const_i32(S)), b.const_i32(PAD))
        hs = b.smax(b.sub(c0, h_start), c0)
        sw = b.smax(b.sub(c0, w_start), c0)
        swc = b.mul(sw, c_C)
        land = b.mul(b.div(b.add(swc, b.const_i32(7)), c8), c8)
        o0 = b.sub(land, swc)  # LDS position of virtual column w_start
        return dict(
            n=n_img,
            oh0=oh0,
            ow0=ow0,
            h_start=h_start,
            w_start=w_start,
            hs=hs,
            sw=sw,
            land=land,
            o0=o0,
        )

    def _stage(tile: Value, buf: int) -> None:
        """Zero the padding prefix, then TDM the input window into ``buf``."""
        g = _tile_geom(tile)
        base = buf * BUF
        prefix = b.add(b.mul(g["hs"], b.const_i32(WP)), g["land"])
        zero_v = b.zero_vec(io_type, 8)
        max_prefix = PAD * WP + _round_up(PAD * C, 8)
        if max_prefix:
            nvec = b.div(prefix, c8)
            for j in range(0, _ceil_div(max_prefix, 8), THREADS):
                v = b.add(tid, b.const_i32(j))
                with b.scf_if(b.cmp_lt(v, nvec)):
                    b.smem_store_vN(
                        lds, [c0, b.add(b.mul(v, c8), b.const_i32(base))], zero_v, n=8
                    )
        hr0 = b.add(g["h_start"], g["hs"])
        wc0 = b.add(g["w_start"], g["sw"])
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            off = b.add(
                b.add(
                    b.mul(g["n"], params["p_A_stride_n"]),
                    b.mul(hr0, params["p_A_stride_hi"]),
                ),
                b.mul(wc0, params["p_A_stride_wi"]),
            )
            src = b.global_ptr_add(A, b.mul(off, c2))
            dst = b.add(lds_addr, b.mul(b.add(prefix, b.const_i32(base)), c2))
            d0 = b.smax(
                b.smin(b.mul(b.sub(c_W, wc0), c_C), b.sub(b.const_i32(WP), g["land"])),
                c0,
            )
            d1 = b.smax(b.smin(b.sub(c_H, hr0), b.sub(b.const_i32(WIN_H), g["hs"])), c0)
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(d0, d1, c1),
                    strides=(params["p_A_stride_hi"], params["p_A_stride_n"]),
                    tile0=WP,
                    tile1=WIN_H,
                    tile2=c1,
                )
            )

    def _compute(tile: Value, buf: int) -> None:
        g = _tile_geom(tile)
        base = b.add(g["o0"], b.const_i32(buf * BUF))
        accs = [b.zero_vec_f32(8) for _ in range(TM * TN)]
        for tn in range(TN):
            lb = b.add(base, lane_off[tn])
            for t in range(STEPS):
                elems = []
                for i in range(16):
                    gs = t * 16 + i
                    if gs >= KH * E0:
                        elems.append(zero_e)
                        continue
                    r, e = divmod(gs, E0)
                    if e >= RUN:  # neither half is a real tap
                        elems.append(zero_e)
                        continue
                    v = b.smem_load_vN(
                        lds, c0, b.add(lb, b.const_i32(r * WP + e)), dtype=io_type, n=1
                    )
                    elems.append(b.vec_extract(v, 0))
                bf = b.vec_pack(elems, io_type)
                for m in range(TM):
                    i = m * TN + tn
                    accs[i] = b.mma(wmma_op, a_frags[m][t], bf, accs[i])
        cvt = b.vec_trunc_f32_to_bf16 if p.dtype == "bf16" else b.vec_trunc_f32_to_f16
        if spec.tdm_store:
            # ---- epilogue: the [out_h][out_w][block_k] tile to LDS, then one
            # TDM store clipped to the image and to K.
            for tn in range(TN):
                q = b.add(
                    b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16),
                    l16,
                )
                for m in range(TM):
                    kl = b.add(
                        b.mul(
                            b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16
                        ),
                        b.mul(lhalf, c8),
                    )
                    oi = b.add(b.add(b.mul(q, b.const_i32(BK)), kl), b.const_i32(OUT0))
                    b.smem_store_vN(lds, [c0, oi], cvt(accs[m * TN + tn]), n=8)
            b.sync_lds_only()
            with b.scf_if(b.cmp_eq(wave_id, c0)):
                off = b.add(
                    b.add(
                        b.add(
                            b.mul(g["n"], params["p_D_stride_n"]),
                            b.mul(g["oh0"], params["p_D_stride_ho"]),
                        ),
                        b.mul(g["ow0"], params["p_D_stride_wo"]),
                    ),
                    k_blk,
                )
                b.tensor_store_from_lds(
                    *_tdm_descriptor_3d(
                        b,
                        b.global_ptr_add(D, b.mul(off, c2)),
                        b.add(lds_addr, b.const_i32(OUT0 * 2)),
                        dims=(
                            b.smax(b.smin(b.sub(c_K, k_blk), b.const_i32(BK)), c0),
                            b.smax(b.sub(c_Wo, g["ow0"]), c0),
                            b.smax(b.sub(c_Ho, g["oh0"]), c0),
                        ),
                        strides=(params["p_D_stride_wo"], params["p_D_stride_ho"]),
                        tile0=BK,
                        tile1=TOW,
                        tile2=b.const_i32(TOH),
                    )
                )
            return
        # ---- epilogue: 8 consecutive output channels per lane, 16 B stores.
        for tn in range(TN):
            ty, tx = lane_px[tn]
            oh = b.add(g["oh0"], ty)
            ow = b.add(g["ow0"], tx)
            px_ok = b.land(b.cmp_lt(oh, c_Ho), b.cmp_lt(ow, c_Wo))
            px_off = b.add(
                b.add(
                    b.mul(g["n"], params["p_D_stride_n"]),
                    b.mul(oh, params["p_D_stride_ho"]),
                ),
                b.mul(ow, params["p_D_stride_wo"]),
            )
            for m in range(TM):
                k = b.add(
                    k_blk,
                    b.add(
                        b.mul(
                            b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16
                        ),
                        b.mul(lhalf, c8),
                    ),
                )
                with b.scf_if(b.land(px_ok, b.cmp_lt(k, c_K))):
                    b.global_store_vN(
                        D, b.add(px_off, k), cvt(accs[m * TN + tn]), 8, align=16
                    )

    first = b.mul(b.block_id_x(), b.const_i32(T))
    if T == 1:
        _stage(first, 0)
        b.s_wait_tensorcnt(0)
        b.sync_lds_only()
        _compute(first, 0)
        if spec.tdm_store:
            with b.scf_if(b.cmp_eq(wave_id, c0)):
                b.s_wait_tensorcnt(0)
        return b.kernel

    with b.scf_if(b.cmp_lt(first, n_tiles)):
        _stage(first, 0)
    # Unrolled over the block's tiles: the window buffer alternates.
    for t in range(T):
        tile = b.add(first, b.const_i32(t))
        b.s_wait_tensorcnt(0)
        b.sync_lds_only()
        if t + 1 < T:
            nxt = b.add(tile, c1)
            with b.scf_if(b.cmp_lt(nxt, n_tiles)):
                _stage(nxt, (t + 1) % NB)
        with b.scf_if(b.cmp_lt(tile, n_tiles)):
            _compute(tile, t % NB)
    if spec.tdm_store:
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            b.s_wait_tensorcnt(0)
    return b.kernel


# --------------------------------------------------------------------------
# Small K: Z = X * W[(r, s, k), c] over the window, then overlap-add.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FwdSmallkWmmaSpec:
    """One configuration of :func:`build_fwd_smallk_wmma`."""

    problem: DirectConvProblem
    name: str = "fwd_smallk_wmma"
    waves_c: int = 1  # waves along GEMM columns (r, s, k)
    waves_p: int = 4  # waves along window pixels
    tiles_c: int = 3
    tiles_p: int = 6
    out_h: int = 16  # output rows per block
    out_w: int = 16  # output columns per block
    ck: int = 32
    stages: int = 2
    prefetch: bool = False
    sched: str | None = None
    wave_size: int = 32

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def ncols(self) -> int:  # real GEMM columns: R*S*K
        p = self.problem
        return p.KH * p.KW * p.kpg

    @property
    def win_h(self) -> int:
        return (self.out_h - 1) * self.S + self.problem.KH

    @property
    def win_w(self) -> int:
        return (self.out_w - 1) * self.S + self.problem.KW

    @property
    def block_c(self) -> int:
        return self.waves_c * self.tiles_c * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def threads_per_block(self) -> int:
        return self.waves_c * self.waves_p * self.wave_size

    @property
    def x_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def w_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def z_pitch(self) -> int:
        return self.block_c + 8

    @property
    def x_elems(self) -> int:
        # The GEMM pixels plus the last window row's wrap past the window.
        return (self.block_p + self.problem.PAD + 1) * self.x_pitch

    @property
    def wei_elems(self) -> int:
        return self.block_c * self.w_pitch

    def lds_bytes(self) -> int:
        ring = 2 * self.stages * (self.x_elems + self.wei_elems)
        z = 4 * self.block_p * self.z_pitch
        return max(ring, z)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"s{p.stride}",
            f"p{p.PAD}",
            f"k{p.kpg}",
            f"wc{self.waves_c}",
            f"wp{self.waves_p}",
            f"tc{self.tiles_c}",
            f"tp{self.tiles_p}",
            f"oh{self.out_h}",
            f"ow{self.out_w}",
            f"ck{self.ck}",
            f"st{self.stages}",
            "pf" if self.prefetch else "",
            f"s{self.sched}" if self.sched else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (
            p.N * _ceil_div(p.Ho, self.out_h) * _ceil_div(p.Wo, self.out_w),
            1,
            1,
        )

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def fwd_smallk_tile_efficiency(spec: FwdSmallkWmmaSpec) -> float:
    p = spec.problem
    oh, ow = spec.out_h, spec.out_w
    return (p.Ho * p.Wo) / (_ceil_div(p.Ho, oh) * oh * _ceil_div(p.Wo, ow) * ow)


def fwd_smallk_reason(spec: FwdSmallkWmmaSpec, arch: str = "gfx1250") -> str | None:
    p = spec.problem
    why = _common_reason(p, arch, "fwd_smallk_wmma")
    if why:
        return why
    target = ArchTarget.from_gfx(arch)
    if spec.wave_size != 32:
        return "fwd_smallk_wmma is a wave32 kernel"
    if p.kpg > SMALLK_MAX_K:
        return f"K {p.kpg} > {SMALLK_MAX_K}"
    if p.kpg not in (1, 2, 4, 8):
        return f"K {p.kpg} must be 1, 2, 4 or 8 (vector Z reads)"
    if p.cpg % 8:
        return f"C {p.cpg} must be a multiple of 8 (16-byte input rows)"
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_c, spec.waves_p, spec.tiles_c, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.block_c < spec.ncols or spec.block_c - spec.ncols >= 16:
        return f"block_c {spec.block_c} must be R*S*K = {spec.ncols} rounded up to 16"
    if spec.win_h * spec.win_w > spec.block_p:
        return f"input window {spec.win_h}x{spec.win_w} > {spec.block_p} GEMM pixels"
    if spec.stages < 2:
        return "stages must be >= 2"
    if spec.waves_c * spec.waves_p < 2:
        return "needs at least two waves"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 8 * spec.tiles_c * spec.tiles_p + 8 * (spec.tiles_c + spec.tiles_p) * (
        2 if spec.prefetch else 1
    )
    if vgprs > SMALL_WMMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    return None


def is_valid_fwd_smallk_spec(
    spec: FwdSmallkWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = fwd_smallk_reason(spec, arch)
    return (why is None, why or "ok")


def build_fwd_smallk_wmma(spec: FwdSmallkWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    why = fwd_smallk_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdSmallkWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, S, PAD, K = p.KH, p.KW, p.stride, p.PAD, p.kpg
    RS = KH * KW
    io_type = _io_type(p.dtype)
    wmma_op = _wmma_op(p, arch)

    WAVE = spec.wave_size
    WC = spec.waves_c
    TM, TN = spec.tiles_c, spec.tiles_p
    THREADS = spec.threads_per_block
    BP = spec.block_p
    OH, OW = spec.out_h, spec.out_w
    DH, DW = spec.win_h, spec.win_w
    CK, NS = spec.ck, spec.stages
    XP, WPITCH, ZP = spec.x_pitch, spec.w_pitch, spec.z_pitch
    XE, WEI = spec.x_elems, spec.wei_elems

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_C = params["p_total_c"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_c = b.mod(wave_id, b.const_i32(WC))
    wave_p = b.div(wave_id, b.const_i32(WC))

    n_ht = b.div(b.add(c_Ho, b.const_i32(OH - 1)), b.const_i32(OH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(OW - 1)), b.const_i32(OW))
    t = b.block_id_x()
    wt = b.mod(t, n_wt)
    t = b.div(t, n_wt)
    ht = b.mod(t, n_ht)
    n_img = b.div(t, n_ht)
    oh0 = b.mul(ht, b.const_i32(OH))
    ow0 = b.mul(wt, b.const_i32(OW))
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_C, b.const_i32(CK - 1)), c_CK)

    # The input window starts at (oh0*S - PAD, ow0*S - PAD); a negative start
    # is clipped and the transfer lands shifted (the cells before it zeroed).
    i_lo = b.sub(b.mul(oh0, b.const_i32(S)), b.const_i32(PAD))
    j_lo = b.sub(b.mul(ow0, b.const_i32(S)), b.const_i32(PAD))
    hs = b.smax(b.sub(c0, i_lo), c0)
    sw = b.smax(b.sub(c0, j_lo), c0)
    i_start = b.add(i_lo, hs)
    j_start = b.add(j_lo, sw)
    z_elems = b.mul(b.add(b.mul(hs, b.const_i32(DW)), sw), b.const_i32(XP))
    x_w_ext = b.smax(b.smin(b.sub(c_W, j_start), b.sub(b.const_i32(DW), sw)), c0)
    x_h_ext = b.smax(b.sub(c_H, i_start), c0)
    x_tile2 = b.sub(b.const_i32(DH), hs)
    x_base = b.add(
        b.add(
            b.mul(n_img, params["p_A_stride_n"]),
            b.mul(i_start, params["p_A_stride_hi"]),
        ),
        b.mul(j_start, params["p_A_stride_wi"]),
    )
    c_RSC = b.mul(b.const_i32(RS), c_C)

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="sk_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)
    WEI0 = NS * XE
    wei_addr = b.add(lds_addr, b.const_i32(WEI0 * 2))

    max_z = (PAD * DW + PAD) * XP
    if max_z:
        zero_v = b.zero_vec(io_type, 8)
        z_vecs = b.div(z_elems, c8)
        for j in range(0, _ceil_div(max_z, 8), THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, z_vecs)):
                flat = b.mul(v, c8)
                for st in range(NS):
                    b.smem_store_vN(
                        lds, [c0, b.add(flat, b.const_i32(st * XE))], zero_v, n=8
                    )

    def _issue(chunk: Value) -> None:
        stage = b.mod(chunk, b.const_i32(NS))
        cc0 = b.mul(chunk, c_CK)
        c_ext = b.smax(b.smin(b.sub(c_C, cc0), c_CK), c0)
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            src = b.global_ptr_add(A, b.mul(b.add(x_base, cc0), c2))
            dst = b.add(
                lds_addr,
                b.mul(b.add(b.mul(stage, b.const_i32(XE)), z_elems), c2),
            )
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(c_ext, x_w_ext, x_h_ext),
                    strides=(params["p_A_stride_wi"], params["p_A_stride_hi"]),
                    tile0=XP,
                    tile1=DW,
                    tile2=x_tile2,
                )
            )
        with b.scf_if(b.cmp_eq(wave_id, c1)):
            # Rows (r, s, k) of the chunk: [rs][k][c], c contiguous.
            src = b.global_ptr_add(Bp, b.mul(cc0, c2))
            dst = b.add(wei_addr, b.mul(stage, b.const_i32(WEI * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(c_ext, b.const_i32(K), b.const_i32(RS)),
                    strides=(c_RSC, c_C),
                    tile0=WPITCH,
                    tile1=K,
                    tile2=b.const_i32(RS),
                )
            )

    px_off = []
    for tn in range(TN):
        flat = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        px_off.append(b.add(b.mul(flat, b.const_i32(XP)), b.mul(lhalf, c16)))
    wei_off = []
    for m in range(TM):
        row = b.add(
            b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16), l16
        )
        wei_off.append(
            b.add(
                b.add(b.mul(row, b.const_i32(WPITCH)), b.mul(lhalf, c16)),
                b.const_i32(WEI0),
            )
        )

    def _read(base: Value, off: Value, k_off: int) -> Value:
        addr = b.add(base, b.add(off, b.const_i32(k_off)))
        lo = b.smem_load_vN(lds, c0, addr, dtype=io_type, n=8)
        hi = b.smem_load_vN(lds, c0, b.add(addr, c8), dtype=io_type, n=8)
        return b.vec_concat(lo, hi)

    steps = list(range(CK // WMMA_K))

    def _chunk(kc: Value, accs):
        if NS == 2:
            b.s_wait_tensorcnt(0)
        else:
            more = b.cmp_lt(b.add(kc, b.const_i32(NS - 2)), n_chunks)
            with b.scf_if_else(more) as (then_ctx, else_ctx):
                with then_ctx:
                    b.s_wait_tensorcnt(NS - 2)
                with else_ctx:
                    b.s_wait_tensorcnt(0)
        b.sync_lds_only()
        nxt = b.add(kc, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_chunks)):
            _issue(nxt)
        stage = b.mod(kc, b.const_i32(NS))
        xb = b.mul(stage, b.const_i32(XE))
        wb = b.mul(stage, b.const_i32(WEI))

        def _frags(ks):
            bf = [_read(xb, px_off[tn], ks * WMMA_K) for tn in range(TN)]
            af = [_read(wb, wei_off[m], ks * WMMA_K) for m in range(TM)]
            return af, bf

        accs = list(accs)
        b.s_setprio(1)
        nf = _frags(steps[0]) if spec.prefetch else None
        for j, ks in enumerate(steps):
            if spec.prefetch:
                af, bf = nf
                if j + 1 < len(steps):
                    nf = _frags(steps[j + 1])
            else:
                af, bf = _frags(ks)
            for m in range(TM):
                for tn in range(TN):
                    i = m * TN + tn
                    accs[i] = b.mma(wmma_op, af[m], bf[tn], accs[i])
        b.s_setprio(0)
        return accs

    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_chunks)):
            _issue(b.const_i32(st))
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0, n_chunks, c1, [(f"acc{i}", zero_acc) for i in range(TM * TN)], iv_name="kc"
    )
    with loop as (kc, carried):
        b.scf_yield(*_chunk(kc, list(carried)))
    accs = list(loop.results)

    # ---- epilogue 1: Z (fp32) to LDS as [window pixel][(r, s, k)].
    b.sync_lds_only()
    for m in range(TM):
        col = b.add(
            b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16),
            b.mul(lhalf, c8),
        )
        for tn in range(TN):
            px = b.add(
                b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
            )
            # f32 element i of the shared 2-byte allocation sits at index 2*i.
            zi = b.add(b.mul(px, b.const_i32(ZP)), col)
            b.smem_store_vN(lds, [c0, b.mul(zi, c2)], accs[m * TN + tn], n=8)
    b.sync_lds_only()

    # ---- epilogue 2: overlap-add, one output pixel (all K) per thread.
    cvt = b.vec_trunc_f32_to_bf16 if p.dtype == "bf16" else b.vec_trunc_f32_to_f16
    E = OH * OW
    for it in range(_ceil_div(E, THREADS)):
        e = b.add(tid, b.const_i32(it * THREADS))
        ox = b.mod(e, b.const_i32(OW))
        oy = b.div(e, b.const_i32(OW))
        oh = b.add(oh0, oy)
        ow = b.add(ow0, ox)
        ok = b.land(
            b.cmp_lt(e, b.const_i32(E)),
            b.land(b.cmp_lt(oh, c_Ho), b.cmp_lt(ow, c_Wo)),
        )
        with b.scf_if(ok):
            base = b.mul(
                b.add(
                    b.mul(b.mul(oy, b.const_i32(S)), b.const_i32(DW)),
                    b.mul(ox, b.const_i32(S)),
                ),
                b.const_i32(ZP),
            )
            acc = None
            for r in range(KH):
                for s in range(KW):
                    off = (r * DW + s) * ZP + (r * KW + s) * K
                    z = b.smem_load_vN(
                        lds,
                        c0,
                        b.mul(b.add(base, b.const_i32(off)), c2),
                        dtype=F32,
                        n=K,
                    )
                    acc = z if acc is None else b.vector_add(acc, z)
            out = b.add(
                b.add(
                    b.mul(n_img, params["p_D_stride_n"]),
                    b.mul(oh, params["p_D_stride_ho"]),
                ),
                b.mul(ow, params["p_D_stride_wo"]),
            )
            b.global_store_vN(D, out, cvt(acc), K, align=2 * K)
    return b.kernel
