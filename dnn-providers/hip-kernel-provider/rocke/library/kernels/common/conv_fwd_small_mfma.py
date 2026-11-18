# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Forward convolutions with a tiny channel count on wave64 MFMA (gfx950).

The MFMA ports of the two kernels of :mod:`kernels.common.conv_fwd_small_wmma`:
``build_fwd_smallc_mfma`` (few input channels) below, and
``build_fwd_smallk_mfma`` (few output channels, see :class:`FwdSmallkMfmaSpec`).

``build_fwd_smallc_mfma`` serves the stem layers (C <= 4) where implicit GEMM
gathers its im2col operand one scalar per K element straight from global
memory. The GEMM is

    Y[px, k] = sum_j W[k, j] * im2col(X)[j, px],   j = (r, s, c) over the filter

with the reduction flattened over the whole filter, so a 3x3x3 stem is a single
16x16x32 MFMA step. In NHWC an input row of a small-C layer is a short
contiguous run of ``W * C`` elements: a block stages its output tile's input
window (``win_h`` rows x ``win_w * C`` elements, zero outside the image) into
LDS and each lane gathers its im2col fragment from that window. There is no
TDM on gfx950: a window row is read as 16-byte buffer loads from the 8-element
boundary at or below its first element (a load wholly outside the tensor reads
zero through the buffer bounds), each element is masked to the image, and the
elements land one by one at their window position; each LDS row keeps 8
elements of slack on both sides for the loads' overhang (an element further
past the row lands on the last slack element, so every store stays in the
row). The buffer bounds are checked per 4-byte component, so the X buffer is
rounded up to whole dwords: the last element of an odd-length X shares its
dword with the halfword past the tensor, which the column mask zeroes.

Fragment layout (``mfma_f32_16x16x32``): lane ``l`` feeds row / column
``l % 16`` and the 8 reduction indices ``(l / 16) * 8 .. + 7`` of each step.
The reduction index decides the tap and channel, so the window offset of each
of a lane's fragment elements is fixed for the kernel: the ``8 * steps``
offsets are computed once per lane, before any tile, and each tile adds only
its pixel base. Indices past the filter (``j >= KH * KW * C``) read the
pixel's own first element -- a finite value -- against a zero weight.

The weights of the block's output channels stay in registers (A fragments)
for the ``tiles_per_block`` output tiles a block walks; the accumulator of a
16x16 tile holds one pixel and 4 consecutive output channels per lane, stored
as one 8-byte write. With ``prefetch`` the next tile's window loads are issued
before the current tile's compute and stores, so they are in flight meanwhile.

AOT: batch, extents and the *other* channel count are kernargs; filter,
stride, padding, the small channel count and the tile geometry are baked.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.ir import IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    _mfma,
    _trunc_f32,
    emit_direct_params,
    one_padding_reason,
    pad_w_reason,
    problem_dims_reason,
)
from kernels.common.conv_fwd_small_wmma import _common_shape_reason

SMALLC_MFMA_MAX_C = 4
# Accumulators + resident weights + per-lane window offsets, in VGPRs.
SMALLC_MFMA_VGPR_BUDGET = 400
_MFMA_K = 32  # reduction depth of one 16x16x32 step
# The byte offset of a masked-out buffer load: 2**31, at or past the end of
# every buffer (tensors are below 2 GiB, and the small-C X buffer rounded up
# to whole dwords is at most 2**31 bytes), so the load reads zero.
_OOB_OFFSET = -0x80000000


def _mfma_common_reason(
    p: DirectConvProblem, arch: str, name: str, wave_size: int, *, square: bool
) -> tuple[str | None, ArchTarget | None]:
    """The rules small C and small K share: a wave64 target with the
    16x16x32 MFMA of the dtype, every problem extent at least 1, the padding
    (``square=False``, small K: rectangular filters with their own column
    padding), then the shapes the WMMA forms serve too. Returns
    ``(why, target)``; the target is ``None`` only when ``arch`` is unknown."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e), None
    if target.wave_size != 64 or wave_size != 64:
        return f"{name} is a wave64 MFMA kernel", target
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}", target
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=_MFMA_K
    ):
        return f"missing mfma_f32_16x16x32_{ab} on {arch}", target
    why = problem_dims_reason(p)
    if why is None:
        why = one_padding_reason(p) if square else pad_w_reason(p)
    if why is None:
        why = _common_shape_reason(p, square=square)
    return why, target


@dataclass(frozen=True)
class FwdSmallcMfmaSpec:
    """One configuration of :func:`build_fwd_smallc_mfma`."""

    problem: DirectConvProblem
    name: str = "fwd_smallc_mfma"
    waves_k: int = 1  # waves along output channels
    waves_p: int = 4  # waves along output pixels
    tiles_k: int = 4  # 16-channel tiles per wave
    tiles_p: int = 2  # 16-pixel tiles per wave
    out_h: int = 4  # output rows per tile
    out_w: int = 32  # output columns per tile
    tiles_per_block: int = 1  # output tiles a block walks (weights stay resident)
    # With tiles_per_block > 1: issue the next tile's window loads into
    # registers before the current tile's compute and stores, so they are in
    # flight meanwhile (the window load otherwise stalls each tile in turn).
    prefetch: bool = False
    wave_size: int = 64

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def nc(self) -> int:  # real reduction length KH * KW * C
        p = self.problem
        return p.KH * p.KW * p.cpg

    @property
    def steps(self) -> int:
        return ceil_div(self.nc, _MFMA_K)

    @property
    def win_h(self) -> int:
        return (self.out_h - 1) * self.S + self.problem.KH

    @property
    def win_w(self) -> int:
        return (self.out_w - 1) * self.S + self.problem.KW

    @property
    def row_pitch(self) -> int:  # LDS elements per window row
        return self.win_w * self.problem.cpg

    @property
    def lds_pitch(self) -> int:  # LDS row: the window row plus 8 of slack each side
        return self.row_pitch + 16

    @property
    def row_vecs(self) -> int:  # 16-byte loads covering a window row from below
        return ceil_div(self.row_pitch + 7, 8)

    @property
    def win_elems(self) -> int:
        return self.win_h * self.lds_pitch

    @property
    def block_k(self) -> int:
        return self.waves_k * self.tiles_k * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def threads_per_block(self) -> int:
        return self.waves_k * self.waves_p * self.wave_size

    def lds_bytes(self) -> int:
        return 2 * self.win_elems

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
            "pf" if self.prefetch else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def n_tiles(self) -> int:
        p = self.problem
        return p.N * ceil_div(p.Ho, self.out_h) * ceil_div(p.Wo, self.out_w)

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (
            ceil_div(self.n_tiles(), self.tiles_per_block),
            ceil_div(p.total_k, self.block_k),
            1,
        )

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def fwd_smallc_mfma_tile_efficiency(spec: FwdSmallcMfmaSpec) -> float:
    p = spec.problem
    oh, ow = spec.out_h, spec.out_w
    return (p.Ho * p.Wo) / (ceil_div(p.Ho, oh) * oh * ceil_div(p.Wo, ow) * ow)


def fwd_smallc_mfma_reason(spec: FwdSmallcMfmaSpec, arch: str = "gfx950") -> str | None:
    p = spec.problem
    why, target = _mfma_common_reason(
        p, arch, "fwd_smallc_mfma", spec.wave_size, square=True
    )
    if why:
        return why
    if p.cpg > SMALLC_MFMA_MAX_C:
        return f"C {p.cpg} > {SMALLC_MFMA_MAX_C}"
    if p.kpg % 4:
        return f"K {p.kpg} must be a multiple of 4 (8-byte output writes)"
    if min(spec.waves_k, spec.waves_p, spec.tiles_k, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.tiles_per_block < 1:
        return "tiles_per_block must be >= 1"
    if spec.prefetch and spec.tiles_per_block < 2:
        return "prefetch needs tiles_per_block >= 2 (a next tile to prefetch)"
    if spec.out_h < 1 or spec.out_w < 1:
        return f"out tile {spec.out_h}x{spec.out_w} must be at least 1x1"
    if spec.out_h * spec.out_w != spec.block_p:
        return (
            f"out tile {spec.out_h}x{spec.out_w} must hold exactly the "
            f"{spec.block_p} GEMM pixels"
        )
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = (
        4 * spec.tiles_k * spec.tiles_p  # accumulators
        + 4 * spec.tiles_k * spec.steps  # resident weight fragments
        + 8 * spec.steps  # window offsets
        + 4 * spec.steps  # one B fragment in flight
    )
    if vgprs > SMALLC_MFMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators, weights and offsets"
    return None


def is_valid_fwd_smallc_mfma_spec(
    spec: FwdSmallcMfmaSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    why = fwd_smallc_mfma_reason(spec, arch)
    return (why is None, why or "ok")


def build_fwd_smallc_mfma(spec: FwdSmallcMfmaSpec, arch: str = "gfx950") -> KernelDef:
    why = fwd_smallc_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdSmallcMfmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, S, PAD, C = p.KH, p.KW, p.stride, p.PAD, p.cpg
    NC = spec.nc
    STEPS = spec.steps
    io_type = _io_type(p.dtype)

    WAVE = spec.wave_size
    WK = spec.waves_k
    TM, TN = spec.tiles_k, spec.tiles_p
    THREADS = spec.threads_per_block
    BK = spec.block_k
    TOH, TOW = spec.out_h, spec.out_w
    WIN_H = spec.win_h
    WP = spec.row_pitch
    WPS = spec.lds_pitch
    NV = spec.row_vecs
    WIN = spec.win_elems
    T = spec.tiles_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c4 = b.const_i32(4)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_K = params["p_total_k"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lgrp = b.div(lane, c16)  # 0..3: the lane's 8-deep slice of a step
    wave_k = b.mod(wave_id, b.const_i32(WK))
    wave_p = b.div(wave_id, b.const_i32(WK))

    n_ht = b.div(b.add(c_Ho, b.const_i32(TOH - 1)), b.const_i32(TOH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(TOW - 1)), b.const_i32(TOW))
    n_tiles = b.mul(b.mul(params["p_N"], n_ht), n_wt)
    k_blk = b.mul(b.block_id_y(), b.const_i32(BK))

    lds = b.smem_alloc(io_type, [1, WIN], name_hint="scm_lds")

    # ---- the lane's reduction indices: j = t*32 + lgrp*8 + i.
    lane_j0 = b.mul(lgrp, c8)
    zero_e = b.vec_extract(b.zero_vec(io_type, 2), 0)

    # ---- weights: A fragments of every (m tile, step), resident.
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
            for i in range(8):
                j = b.add(lane_j0, b.const_i32(t * _MFMA_K + i))
                ok = b.land(k_ok, b.cmp_lt(j, b.const_i32(NC)))
                elems.append(
                    b.masked_global_load(
                        Bp, b.add(k_row, j), ok, zero_e, io_type, align=2
                    )
                )
            per_step.append(b.vec_pack(elems, io_type))
        a_frags.append(per_step)

    # ---- the window offset of each fragment element, relative to the pixel's
    # (tap 0, channel 0) element; past-the-filter indices read offset 0.
    c_kwc = b.const_i32(KW * C)
    c_C = b.const_i32(C)
    win_off = []
    for t in range(STEPS):
        per = []
        for i in range(8):
            j = b.add(lane_j0, b.const_i32(t * _MFMA_K + i))
            r = b.div(j, c_kwc)
            sc = b.mod(j, c_kwc)  # s*C + c: contiguous within the window row
            off = b.add(b.mul(r, b.const_i32(WPS)), sc)
            per.append(b.select(b.cmp_lt(j, b.const_i32(NC)), off, c0))
        win_off.append(per)

    # ---- per-lane pixel geometry (tile independent).
    lane_px = []  # (ty, tx) of each pixel tile's lane pixel
    lane_base = []  # window element of the pixel's tap (0, 0)
    for tn in range(TN):
        q = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        ty = b.div(q, b.const_i32(TOW))
        tx = b.mod(q, b.const_i32(TOW))
        lane_px.append((ty, tx))
        lane_base.append(
            b.add(
                b.add(
                    b.mul(b.mul(ty, b.const_i32(S)), b.const_i32(WPS)),
                    b.mul(tx, b.const_i32(S * C)),
                ),
                c8,  # the row's left slack
            )
        )

    # X rounded up to whole dwords (the bounds are checked per dword): the
    # last element of an odd-length X is read, the halfword past it masked.
    a_rsrc = b.buffer_rsrc(
        A, b.land(b.add_nuw(params["A_bytes"], b.const_i32(3)), b.const_i32(-4))
    )
    c_sent = b.const_i32(_OOB_OFFSET)

    def _geom(tile: Value) -> dict:
        tw = b.mod(tile, n_wt)
        rest = b.div(tile, n_wt)
        th = b.mod(rest, n_ht)
        n_img = b.div(rest, n_ht)
        oh0 = b.mul(th, b.const_i32(TOH))
        ow0 = b.mul(tw, b.const_i32(TOW))
        h_start = b.sub(b.mul(oh0, b.const_i32(S)), b.const_i32(PAD))
        w_start = b.sub(b.mul(ow0, b.const_i32(S)), b.const_i32(PAD))
        return dict(n_img=n_img, oh0=oh0, ow0=ow0, h_start=h_start, w_start=w_start)

    def _load_window(g: dict) -> list:
        """Issue the window's 16-byte loads (row ``row`` is input row
        h_start + row; loads start at the 8-element boundary at or below the
        row's first element, ``land`` elements early). A tile past the grid
        reads past the tensor, which the buffer bounds turn into zeros."""
        h_start, w_start = g["h_start"], g["w_start"]
        img_base = b.mul(g["n_img"], params["p_A_stride_n"])
        wcC = b.mul(w_start, c_C)
        out = []
        for it in range(ceil_div(WIN_H * NV, THREADS)):
            v_lin = b.add(tid, b.const_i32(it * THREADS))
            in_win = b.cmp_lt(v_lin, b.const_i32(WIN_H * NV))
            row = b.div(v_lin, b.const_i32(NV))
            v = b.mod(v_lin, b.const_i32(NV))
            hr = b.add(h_start, row)
            row_ok = b.land(in_win, b.land(b.cmp_ge(hr, c0), b.cmp_lt(hr, c_H)))
            gs = b.add(b.add(img_base, b.mul(hr, params["p_A_stride_hi"])), wcC)
            land = b.land(gs, b.const_i32(7))  # gs mod 8, also for gs < 0
            g0 = b.add(b.sub(gs, land), b.mul(v, c8))
            ok = b.land(row_ok, b.cmp_ge(g0, c0))
            voff = b.select(ok, b.mul(g0, b.const_i32(2)), c_sent)
            vec = (
                b.buffer_load_vN_bf16(a_rsrc, voff, c0, 4)
                if p.dtype == "bf16"
                else b.buffer_load_vN_f16(a_rsrc, voff, c0, 4)
            )
            out.append((vec, b.sub(b.mul(v, c8), land), row, in_win))
        return out

    def _commit_window(g: dict, loads: list) -> None:
        """Mask each loaded element to the image and store it at LDS
        row*WPS + 8 + e (element e is input column w_start + e / C); the
        elements past the row's right slack land on its last slack element."""
        w_start = g["w_start"]
        for vec, e0, row, in_win in loads:
            lds_row = b.add(b.mul(row, b.const_i32(WPS)), c8)
            with b.scf_if(in_win):
                for i in range(8):
                    e = b.add(e0, b.const_i32(i))
                    e_pos = b.smax(e, c0)
                    col = b.add(w_start, b.div(e_pos, c_C))
                    valid = b.land(
                        b.land(b.cmp_ge(e, c0), b.cmp_lt(e, b.const_i32(WP))),
                        b.land(b.cmp_ge(col, c0), b.cmp_lt(col, c_W)),
                    )
                    val = b.select(valid, b.vec_extract(vec, i), zero_e)
                    at = b.add(lds_row, b.smin(e, b.const_i32(WP + 7)))
                    b.smem_store_vN(lds, [c0, at], val, n=1)

    def _tile(tile: Value) -> None:
        g = _geom(tile)
        _commit_window(g, _load_window(g))
        b.sync_lds_only()
        _compute_store(g)

    def _compute_store(g: dict) -> None:
        n_img, oh0, ow0 = g["n_img"], g["oh0"], g["ow0"]
        # ---- compute: one B fragment per (pixel tile, step), gathered from LDS.
        accs = [b.zero_vec_f32(4) for _ in range(TM * TN)]
        for tn in range(TN):
            for t in range(STEPS):
                elems = []
                for i in range(8):
                    lv = b.smem_load_vN(
                        lds, c0, b.add(lane_base[tn], win_off[t][i]), dtype=io_type, n=1
                    )
                    elems.append(b.vec_extract(lv, 0))
                bf = b.vec_pack(elems, io_type)
                for m in range(TM):
                    idx = m * TN + tn
                    accs[idx] = _mfma(
                        b, p.dtype, "16x16x32", a_frags[m][t], bf, accs[idx]
                    )

        # ---- epilogue: 4 consecutive output channels per lane, 8-byte writes.
        for tn in range(TN):
            ty, tx = lane_px[tn]
            oh = b.add(oh0, ty)
            ow = b.add(ow0, tx)
            px_ok = b.land(b.cmp_lt(oh, c_Ho), b.cmp_lt(ow, c_Wo))
            px_off = b.add(
                b.add(
                    b.mul(n_img, params["p_D_stride_n"]),
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
                        b.mul(lgrp, c4),
                    ),
                )
                with b.scf_if(b.land(px_ok, b.cmp_lt(k, c_K))):
                    b.global_store_vN(
                        D,
                        b.add(px_off, k),
                        _trunc_f32(b, p.dtype, accs[m * TN + tn]),
                        4,
                        align=8,
                    )

    first = b.mul(b.block_id_x(), b.const_i32(T))
    if spec.prefetch:
        # Tile t+1's loads are issued before tile t's compute and stores; its
        # commit waits for them after the barrier that retires tile t's reads.
        tiles = [b.add(first, b.const_i32(t)) if t else first for t in range(T)]
        geoms = [_geom(tl) for tl in tiles]
        loads = _load_window(geoms[0])
        for t in range(T):
            if t:
                b.sync_lds_only()
            _commit_window(geoms[t], loads)
            b.sync_lds_only()
            if t + 1 < T:
                loads = _load_window(geoms[t + 1])
            with b.scf_if(b.cmp_lt(tiles[t], n_tiles)):
                _compute_store(geoms[t])
        return b.kernel
    for t in range(T):
        tile = b.add(first, b.const_i32(t)) if t else first
        if t:
            # The previous tile's gathers before the restage.
            b.sync_lds_only()
        with b.scf_if(b.cmp_lt(tile, n_tiles)):
            _tile(tile)
    return b.kernel


# --------------------------------------------------------------------------
# Small K: Z = X * W[(r, s, k), c] over the input window, then overlap-add.
# --------------------------------------------------------------------------

SMALLK_MFMA_MAX_K = 8
SMALLK_MFMA_VGPR_BUDGET = 400


@dataclass(frozen=True)
class FwdSmallkMfmaSpec:
    """One configuration of :func:`build_fwd_smallk_mfma`.

    The MFMA port of :func:`kernels.common.conv_fwd_small_wmma.build_fwd_smallk_wmma`
    for layers with few *output* channels (K <= 8), where implicit GEMM fills a
    16-wide output tile with K real columns and reads every input element once
    per tap. Here a block computes, for every input pixel ``q`` of its output
    tile's window, ``Z[q, (r, s, k)] = sum_c X[q, c] * W[k, r, s, c]`` -- a GEMM
    whose N is the ``KH*KW*K`` (tap, output channel) pairs and whose reduction
    is C -- then ``Y[p, k] = sum_{r,s} Z[p*S - PAD + (r, s), (r, s, k)]`` from
    an fp32 copy of Z in LDS. Each input element is read once per block,
    straight from global memory into its MFMA fragment (8 channels, one 16-byte
    load); the weights' fragments are the same for every block and stay in L2.
    """

    problem: DirectConvProblem
    name: str = "fwd_smallk_mfma"
    waves: int = 4  # waves along window-pixel tiles
    out_h: int = 16  # output rows per block
    out_w: int = 16  # output columns per block
    wave_size: int = 64

    @property
    def S(self) -> int:
        return self.problem.stride

    @property
    def ncols(self) -> int:  # real GEMM columns: KH*KW*K
        p = self.problem
        return p.KH * p.KW * p.kpg

    @property
    def n_tiles(self) -> int:  # 16-column GEMM tiles
        return ceil_div(self.ncols, 16)

    @property
    def win_h(self) -> int:
        return (self.out_h - 1) * self.S + self.problem.KH

    @property
    def win_w(self) -> int:
        return (self.out_w - 1) * self.S + self.problem.KW

    @property
    def m_tiles(self) -> int:  # 16-pixel tiles of the window, per wave
        # (max: a rejected waves < 1 still reports its LDS size)
        return ceil_div(ceil_div(self.win_h * self.win_w, 16), max(self.waves, 1))

    @property
    def z_pitch(self) -> int:  # fp32 Z row: the GEMM columns, padded off banks
        return self.n_tiles * 16 + 4

    @property
    def threads_per_block(self) -> int:
        return self.waves * self.wave_size

    def lds_bytes(self) -> int:
        return 4 * self.waves * self.m_tiles * 16 * self.z_pitch

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"s{p.stride}",
            f"p{p.PAD}",
            f"pw{p.PAD_W}" if p.PAD_W != p.PAD else "",
            f"k{p.kpg}",
            f"w{self.waves}",
            f"oh{self.out_h}",
            f"ow{self.out_w}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (p.N * ceil_div(p.Ho, self.out_h) * ceil_div(p.Wo, self.out_w), 1, 1)

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def fwd_smallk_mfma_tile_efficiency(spec: FwdSmallkMfmaSpec) -> float:
    p = spec.problem
    oh, ow = spec.out_h, spec.out_w
    return (p.Ho * p.Wo) / (ceil_div(p.Ho, oh) * oh * ceil_div(p.Wo, ow) * ow)


def fwd_smallk_mfma_reason(spec: FwdSmallkMfmaSpec, arch: str = "gfx950") -> str | None:
    p = spec.problem
    why, target = _mfma_common_reason(
        p, arch, "fwd_smallk_mfma", spec.wave_size, square=False
    )
    if why:
        return why
    if p.kpg > SMALLK_MFMA_MAX_K:
        return f"K {p.kpg} > {SMALLK_MFMA_MAX_K}"
    if p.cpg % 8:
        return f"C {p.cpg} must be a multiple of 8 (16-byte input fragments)"
    if spec.waves < 1 or spec.out_h < 1 or spec.out_w < 1:
        return "waves and the output tile must be >= 1"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 4 * spec.m_tiles * spec.n_tiles + 4 * (spec.m_tiles + spec.n_tiles)
    if vgprs > SMALLK_MFMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    return None


def is_valid_fwd_smallk_mfma_spec(
    spec: FwdSmallkMfmaSpec, arch: str = "gfx950"
) -> tuple[bool, str]:
    why = fwd_smallk_mfma_reason(spec, arch)
    return (why is None, why or "ok")


def build_fwd_smallk_mfma(spec: FwdSmallkMfmaSpec, arch: str = "gfx950") -> KernelDef:
    why = fwd_smallk_mfma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdSmallkMfmaSpec for {arch}: {why}")

    from rocke.core.ir import F32

    p = spec.problem
    KH, KW, S, PAD, PAD_W, K = p.KH, p.KW, p.stride, p.PAD, p.PAD_W, p.kpg
    RS = KH * KW
    NCOL = spec.ncols
    NT = spec.n_tiles
    MT = spec.m_tiles
    io_type = _io_type(p.dtype)

    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    OH, OW = spec.out_h, spec.out_w
    DH, DW = spec.win_h, spec.win_w
    NPIX = DH * DW
    ZP = spec.z_pitch

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_C = params["p_total_c"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]
    c_sent = b.const_i32(_OOB_OFFSET)

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lgrp = b.div(lane, c16)  # 0..3: the lane's 8 channels of a 32-deep step
    c_lane = b.mul(lgrp, c8)

    n_ht = b.div(b.add(c_Ho, b.const_i32(OH - 1)), b.const_i32(OH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(OW - 1)), b.const_i32(OW))
    tile = b.block_id_x()
    tw = b.mod(tile, n_wt)
    rest = b.div(tile, n_wt)
    th = b.mod(rest, n_ht)
    n_img = b.div(rest, n_ht)
    oh0 = b.mul(th, b.const_i32(OH))
    ow0 = b.mul(tw, b.const_i32(OW))
    h_start = b.sub(b.mul(oh0, b.const_i32(S)), b.const_i32(PAD))
    w_start = b.sub(b.mul(ow0, b.const_i32(S)), b.const_i32(PAD_W))

    a_rsrc = b.buffer_rsrc(A, params["A_bytes"])
    b_rsrc = b.buffer_rsrc(Bp, params["B_bytes"])

    # ---- A rows: the window pixel of each of this wave's pixel tiles (lane
    # row l16), as the byte offset of its channel 0 -- or the sentinel.
    img_off = b.mul(n_img, params["p_A_stride_n"])
    a_base = []
    for mt in range(MT):
        q = b.add(
            b.mul(b.add(b.mul(wave_id, b.const_i32(MT)), b.const_i32(mt)), c16), l16
        )
        hr = b.add(h_start, b.div(q, b.const_i32(DW)))
        wc = b.add(w_start, b.mod(q, b.const_i32(DW)))
        ok = b.land(
            b.cmp_lt(q, b.const_i32(NPIX)),
            b.land(
                b.land(b.cmp_ge(hr, c0), b.cmp_lt(hr, c_H)),
                b.land(b.cmp_ge(wc, c0), b.cmp_lt(wc, c_W)),
            ),
        )
        off = b.add(
            img_off,
            b.add(
                b.mul(hr, params["p_A_stride_hi"]), b.mul(wc, params["p_A_stride_wi"])
            ),
        )
        a_base.append((ok, b.mul(off, c2)))
    # ---- B columns: GEMM column n = (r*KW + s)*K + k reads W[k, r, s, :].
    b_base = []
    for nt in range(NT):
        n = b.add(b.const_i32(nt * 16), l16)
        rs = b.div(n, b.const_i32(K))
        kk = b.mod(n, b.const_i32(K))
        ok = b.cmp_lt(n, b.const_i32(NCOL))
        off = b.mul(b.add(b.mul(kk, b.const_i32(RS)), rs), c_C)
        b_base.append((ok, b.mul(off, c2)))

    def _load(rsrc, ok_base, c_off):
        ok, base = ok_base
        cc = b.add(c_off, c_lane)
        valid = b.land(ok, b.cmp_lt(cc, c_C))
        voff = b.select(valid, b.add(base, b.mul(cc, c2)), c_sent)
        if p.dtype == "bf16":
            return b.buffer_load_vN_bf16(rsrc, voff, c0, 4)
        return b.buffer_load_vN_f16(rsrc, voff, c0, 4)

    # ---- Z = X * W^T over C, 32 channels per step.
    accs = [b.zero_vec_f32(4) for _ in range(MT * NT)]
    for_op = b.scf_for_iter(
        c0,
        c_C,
        b.const_i32(32),
        [(f"zacc{i}", a) for i, a in enumerate(accs)],
        iv_name="cs",
    )
    with for_op as (cs, iter_accs):
        bf = [_load(b_rsrc, b_base[nt], cs) for nt in range(NT)]
        af = [_load(a_rsrc, a_base[mt], cs) for mt in range(MT)]
        new = list(iter_accs)
        for mt in range(MT):
            for nt in range(NT):
                i = mt * NT + nt
                new[i] = _mfma(b, p.dtype, "16x16x32", af[mt], bf[nt], new[i])
        b.scf_yield(*new)
    accs = list(for_op.results)

    # ---- Z to LDS (fp32): lane holds column l16 and pixel rows lgrp*4 .. +3.
    zl = b.smem_alloc(F32, [1, spec.lds_bytes() // 4], name_hint="z_lds")
    for mt in range(MT):
        q0 = b.add(
            b.mul(b.add(b.mul(wave_id, b.const_i32(MT)), b.const_i32(mt)), c16),
            b.mul(lgrp, b.const_i32(4)),
        )
        for nt in range(NT):
            col = b.add(b.const_i32(nt * 16), l16)
            acc = accs[mt * NT + nt]
            for i in range(4):
                zi = b.add(b.mul(b.add(q0, b.const_i32(i)), b.const_i32(ZP)), col)
                b.smem_store_vN(zl, [c0, zi], b.vec_extract(acc, i), n=1)
    b.sync_lds_only()

    # ---- overlap-add: one output pixel (all K channels) per thread and pass.
    for it in range(ceil_div(OH * OW, THREADS)):
        px = b.add(tid, b.const_i32(it * THREADS))
        # The last pass's threads past the tile read the tile's last pixel, so
        # their Z reads stay in the LDS copy; the store below skips them.
        pxr = (
            b.smin(px, b.const_i32(OH * OW - 1)) if (it + 1) * THREADS > OH * OW else px
        )
        oy = b.div(pxr, b.const_i32(OW))
        ox = b.mod(pxr, b.const_i32(OW))
        oh = b.add(oh0, oy)
        ow = b.add(ow0, ox)
        ok = b.land(
            b.cmp_lt(px, b.const_i32(OH * OW)),
            b.land(b.cmp_lt(oh, c_Ho), b.cmp_lt(ow, c_Wo)),
        )
        q0 = b.add(
            b.mul(b.mul(oy, b.const_i32(S)), b.const_i32(DW)), b.mul(ox, b.const_i32(S))
        )
        sums = []
        for k in range(K):
            acc = None
            for r in range(KH):
                for s in range(KW):
                    zi = b.add(
                        b.mul(b.add(q0, b.const_i32(r * DW + s)), b.const_i32(ZP)),
                        b.const_i32((r * KW + s) * K + k),
                    )
                    v = b.vec_extract(b.smem_load_vN(zl, c0, zi, dtype=F32, n=1), 0)
                    acc = v if acc is None else b.fadd(acc, v)
            sums.append(acc)
        y_off = b.add(
            b.add(
                b.mul(n_img, params["p_D_stride_n"]),
                b.mul(oh, params["p_D_stride_ho"]),
            ),
            b.mul(ow, params["p_D_stride_wo"]),
        )
        with b.scf_if(ok):
            if K in (2, 4, 8):
                b.global_store_vN(
                    D,
                    y_off,
                    _trunc_f32(b, p.dtype, b.vec_pack(sums, F32)),
                    K,
                    align=2 * K,
                )
            else:
                for k in range(K):
                    b.global_store(
                        D,
                        b.add(y_off, b.const_i32(k)),
                        (
                            b.trunc_f32_to_bf16(sums[k])
                            if p.dtype == "bf16"
                            else b.trunc_f32_to_f16(sums[k])
                        ),
                        align=2,
                    )
    return b.kernel
