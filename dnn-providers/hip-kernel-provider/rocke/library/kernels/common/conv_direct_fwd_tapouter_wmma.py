# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tap-outer direct forward convolution on wave32 WMMA (gfx1250).

Why a separate family
---------------------
:mod:`conv_direct_fwd_wmma` stages one LDS halo per input-channel chunk and
reads every tap from it. The halo spans ``(K - 1) * dilation`` extra rows and
columns, so at large dilations (atrous / ASPP layers, dilation 12..36) it
outgrows LDS. Implicit GEMM serves those layers, but its im2col gather of a
dilated tap is cache-unfriendly. This kernel instead runs the ``KH * KW`` taps
as shifted 1x1 GEMMs accumulating into one output tile::

    Y[n, p, q, k] = sum_{r, s, c} X[n, p + r*dh - PAD, q + s*dw - PAD, c] W[k, r, s, c]

Per workgroup, with M = output channels and N = output pixels::

    block_k                     output channels   (waves_k * tiles_k * 16)
    tile_h x tile_w             output pixels     (waves_p * tiles_p * 16)
    ck                          input channels reduced per step

    for chunk in channel chunks:                      # one runtime step loop
        for tap (r, s) whose window meets the image:  #   over (chunk, tap)
            TDM X[n, h0 + r*dh : +tile_h, w0 + s*dw : +tile_w, chunk]
                -> LDS window [row][col][c]            (zero-filled past the image)
            TDM W[k0 : k0 + block_k, r, s, chunk] -> LDS [k][c]
            acc[k-tile, px-tile] += W fragment x X fragment   (WMMA 16x16x32)

The window of a tap is a plain rectangle of input pixels, the same for every
lane (pixel ``i`` of the tile reads window cell ``i``), so the B reads are the
ones of a 1x1 GEMM and no halo is held. A tap whose window lies wholly in the
padding contributes nothing and is skipped: the taps of a block that meet the
image form a rectangle ``[r_lo, r_lo + nr) x [s_lo, s_lo + ns)`` (counted per
block from its tile origin), and the step loop runs over ``chunks x nr x ns``.
At dilation 24..36 on 65..128-pixel maps that skips a large share of the taps
an implicit GEMM would run.

Channel chunks are the outer loop and taps the inner one, so the blocks in
flight read the same channel slice of their (overlapping) windows at about the
same time and the input is fetched from DRAM about once, not once per tap.

Edges: a transfer cannot start at a negative coordinate, so a window that
starts above / left of the image is read from row / column 0 and lands
shifted down / right by ``hs`` rows / ``ws`` columns. The ``hs * tile_w + ws``
leading cells get zeros from LDS stores issued with the transfer; the leading
columns of the later rows get the zero fill of the previous row's clipped
columns (the transfer's row is ``tile_w`` wide, its extent ``tile_w - ws``), so
each window slot keeps ``tile_w`` cells of slack at its end. The bottom and
right edges are the transfer's own zero fill.

The slots form a ring of ``stages`` (window + weights each): one barrier per
step publishes it, after which the slot the previous step read is refilled
``stages - 1`` steps ahead. Wave 0 issues the window transfer, wave 1 the
weights. The epilogue stages the Y tile in LDS and stores it with one TDM
transfer that clips it to the image and the group's channels.

AOT
---
Batch, spatial extents, the group count and both channel counts are kernargs
(the direct-conv forward block of
:func:`kernels.common.conv_abi.conv_direct_arg_names`).
The filter, padding and dilation are build-time capabilities.

C++ twin: ``library/cpp/instances/common/conv_direct_fwd_tapouter_wmma.cpp``
(``instance_conv_direct_fwd_tapouter_wmma.h``).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
)
from rocke.core.ir import I32, IRBuilder, KernelDef, Value
from rocke.helpers.spec import align_up, ceil_div

from kernels.common.conv_direct_fwd_wmma import (
    _lds_frag16,
    _tdm_descriptor_3d,
    _tdm_ring_wait,
)
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)
from kernels.common.conv_fwd_wmma_gate import (
    CH_PAD,
    WMMA_K,
    wmma_fwd_gate,
    wmma_fwd_limits_reason,
    wmma_fwd_op,
)

TAPOUTER_VGPR_BUDGET = 512


@dataclass(frozen=True)
class DirectFwdTapOuterSpec:
    """One configuration of :func:`build_direct_fwd_tapouter_wmma`."""

    problem: DirectConvProblem
    name: str = "direct_fwd_tapouter_wmma"
    waves_k: int = 2  # waves along output channels
    waves_p: int = 4  # waves along output pixels
    tiles_k: int = 4  # 16-channel tiles per wave
    tiles_p: int = 2  # 16-pixel tiles per wave
    tile_w: int = 16  # output columns per block; tile_h follows from the pixels
    ck: int = 64  # input channels per step (multiple of 32)
    stages: int = 3  # LDS ring depth
    # Read the next k-step's fragments before issuing this one's WMMAs.
    prefetch: bool = False
    # AMDGPU machine-scheduler strategy (rocke.core.codegen_policy), or None.
    sched: str | None = None
    wave_size: int = 32
    # Transfers per operand and step: the window rows and the weight rows are
    # cut in tdm_split slabs, slab j issued by waves 2j (X) and 2j + 1 (W).
    tdm_split: int = 1

    @property
    def block_k(self) -> int:
        return self.waves_k * self.tiles_k * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def tile_h(self) -> int:
        return self.block_p // self.tile_w if self.tile_w > 0 else 0

    @property
    def threads_per_block(self) -> int:
        return self.waves_k * self.waves_p * self.wave_size

    @property
    def x_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def epi_pitch(self) -> int:
        return self.block_k + CH_PAD

    @property
    def win_elems(self) -> int:
        # The window and the tile_w cells its shifted last row wraps into.
        return (self.block_p + self.tile_w) * self.x_pitch

    @property
    def wei_elems(self) -> int:
        return self.block_k * self.x_pitch

    @property
    def stage_elems(self) -> int:
        return self.win_elems + self.wei_elems

    def lds_bytes(self) -> int:
        ring = self.stages * self.stage_elems
        epi = self.block_p * self.epi_pitch
        return 2 * max(ring, epi)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"dil{p.dil_h}x{p.dil_w}",
            f"wk{self.waves_k}",
            f"wp{self.waves_p}",
            f"tk{self.tiles_k}",
            f"tp{self.tiles_p}",
            f"tw{self.tile_w}",
            f"ck{self.ck}",
            f"st{self.stages}",
            "pf" if self.prefetch else "",
            f"s{self.sched}" if self.sched else "",
            f"ts{self.tdm_split}" if self.tdm_split > 1 else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        return (
            p.N
            * p.groups
            * ceil_div(p.Ho, self.tile_h)
            * ceil_div(p.Wo, self.tile_w)
            * ceil_div(p.kpg, self.block_k),
            1,
            1,
        )

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def tapouter_tile_efficiency(spec: DirectFwdTapOuterSpec) -> float:
    """Share of ``spec``'s padded tile grid that is real work on its problem."""
    p = spec.problem
    tiles = (
        align_up(p.Ho, spec.tile_h),
        align_up(p.Wo, spec.tile_w),
        align_up(p.kpg, spec.block_k),
    )
    return (p.Ho * p.Wo * p.kpg) / (tiles[0] * tiles[1] * tiles[2])


def tapouter_reason(spec: DirectFwdTapOuterSpec, arch: str = "gfx1250") -> str | None:
    """Why ``spec`` cannot run on ``arch`` (None when it can)."""
    p = spec.problem
    why, target = wmma_fwd_gate(p, arch, "direct_fwd_tapouter_wmma", spec.wave_size)
    if why:
        return why
    if p.stride != 1:
        return f"stride {p.stride}: the tap windows are dense (stride 1 only)"
    if p.dil_h < 1 or p.dil_w < 1:
        return f"dilation {p.dil_h}x{p.dil_w} < 1"
    if p.PAD < 0:
        return f"PAD {p.PAD} < 0"
    if p.Ho < 1 or p.Wo < 1:
        return f"empty output {p.Ho}x{p.Wo}"
    if p.cpg % 8 or p.kpg % 8:
        return f"cpg {p.cpg} and kpg {p.kpg} must be multiples of 8 (16-byte rows)"
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_k, spec.waves_p, spec.tiles_k, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.waves_k * spec.waves_p < 2:
        return "needs at least two waves (one TDM issuer per operand)"
    if spec.stages < 2:
        return "stages must be >= 2"
    ts = spec.tdm_split
    if ts < 1 or 2 * ts > spec.waves_k * spec.waves_p:
        return f"tdm_split {ts} needs 2 * {ts} waves"
    if spec.tile_h % ts or spec.block_k % ts:
        return f"tdm_split {ts} must divide tile_h {spec.tile_h} and block_k"
    if ts > 1 and spec.stages > 2:
        # A slab in the top padding issues no transfer, so a wave's tensor
        # count no longer marks its steps: only a full wait is safe.
        return f"tdm_split {ts} needs 2 stages (a skipped slab breaks the wait)"
    if spec.tile_w < 1 or spec.block_p % spec.tile_w:
        return f"tile_w {spec.tile_w} must divide {spec.block_p}"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if max(spec.x_pitch, spec.epi_pitch, spec.tile_w, spec.block_k) > 0xFFFF:
        return "tile too wide for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    vgprs = 8 * spec.tiles_k * spec.tiles_p + 8 * (spec.tiles_k + spec.tiles_p) * (
        2 if spec.prefetch else 1
    )
    if vgprs > TAPOUTER_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    return wmma_fwd_limits_reason(p, spec.sched, spec.grid()[0])


def is_valid_tapouter_spec(
    spec: DirectFwdTapOuterSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = tapouter_reason(spec, arch)
    return (why is None, why or "ok")


def build_direct_fwd_tapouter_wmma(
    spec: DirectFwdTapOuterSpec, arch: str = "gfx1250"
) -> KernelDef:
    why = tapouter_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DirectFwdTapOuterSpec for {arch}: {why}")

    p = spec.problem
    KH, KW = p.KH, p.KW
    DH, DW = p.dil_h, p.dil_w
    PAD = p.PAD
    io_type = _io_type(p.dtype)
    wmma_op = wmma_fwd_op(p.dtype, arch)

    WAVE = spec.wave_size
    WK, WP = spec.waves_k, spec.waves_p
    TM, TN = spec.tiles_k, spec.tiles_p
    THREADS = spec.threads_per_block
    BK, BP = spec.block_k, spec.block_p
    TH, TW = spec.tile_h, spec.tile_w
    CK = spec.ck
    NS = spec.stages
    XP, EPITCH = spec.x_pitch, spec.epi_pitch
    WIN, STAGE = spec.win_elems, spec.stage_elems
    TAPS = KH * KW
    SPLIT = spec.tdm_split
    XR, WR = TH // SPLIT, BK // SPLIT

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    Xp, Wp, Yp = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    c_groups = params["p_groups"]
    c_cpg = b.div(params["p_total_c"], c_groups)
    c_kpg = b.div(params["p_total_k"], c_groups)
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]
    xs_n, xs_h, xs_w = (
        params["p_A_stride_n"],
        params["p_A_stride_hi"],
        params["p_A_stride_wi"],
    )
    ys_n, ys_h, ys_w = (
        params["p_D_stride_n"],
        params["p_D_stride_ho"],
        params["p_D_stride_wo"],
    )

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_k = b.mod(wave_id, b.const_i32(WK))
    wave_p = b.div(wave_id, b.const_i32(WK))

    # ---- tile of this block: bx = (((n * G + g) * n_ht + ht) * n_wt + wt) * n_kt + kt.
    n_kt = b.div(b.add(c_kpg, b.const_i32(BK - 1)), b.const_i32(BK))
    n_ht = b.div(b.add(c_Ho, b.const_i32(TH - 1)), b.const_i32(TH))
    n_wt = b.div(b.add(c_Wo, b.const_i32(TW - 1)), b.const_i32(TW))
    t = b.block_id_x()
    kt = b.mod(t, n_kt)
    t = b.div(t, n_kt)
    wt = b.mod(t, n_wt)
    t = b.div(t, n_wt)
    ht = b.mod(t, n_ht)
    t = b.div(t, n_ht)
    g = b.mod(t, c_groups)
    n_img = b.div(t, c_groups)
    k_base = b.mul(kt, b.const_i32(BK))
    h_tile = b.mul(ht, b.const_i32(TH))
    w_tile = b.mul(wt, b.const_i32(TW))
    k_grp = b.mul(g, c_kpg)
    c_grp = b.mul(g, c_cpg)
    w_row = b.mul(c_cpg, b.const_i32(TAPS))  # element stride between k rows
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_cpg, b.const_i32(CK - 1)), c_CK)
    k_ext = b.smax(b.smin(b.sub(c_kpg, k_base), b.const_i32(BK)), c0)

    # ---- taps whose window meets the image. Window rows of filter row r:
    # [oh0 + r*DH, oh0 + r*DH + th_eff), th_eff = the tile's rows inside Ho.
    oh0 = b.sub(h_tile, b.const_i32(PAD))
    ow0 = b.sub(w_tile, b.const_i32(PAD))
    th_eff = b.smin(b.sub(c_Ho, h_tile), b.const_i32(TH))
    tw_eff = b.smin(b.sub(c_Wo, w_tile), b.const_i32(TW))

    def _tap_range(o0: Value, ext: Value, lim: Value, nk: int, dil: int):
        """(first tap, taps) of one axis meeting [0, lim)."""
        before = c0
        after = c0
        for r in range(nk):
            lo = b.add(o0, b.const_i32(r * dil)) if r else o0
            before = b.add(before, b.select(b.cmp_le(b.add(lo, ext), c0), c1, c0))
            after = b.add(after, b.select(b.cmp_ge(lo, lim), c1, c0))
        return before, b.smax(b.sub(b.const_i32(nk), b.add(before, after)), c0)

    r_lo, n_r = _tap_range(oh0, th_eff, c_H, KH, DH)
    s_lo, n_s = _tap_range(ow0, tw_eff, c_W, KW, DW)
    n_taps = b.mul(n_r, n_s)
    n_steps = b.mul(n_chunks, n_taps)
    n_taps1 = b.smax(n_taps, c1)
    n_s1 = b.smax(n_s, c1)

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="ft_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)
    x_img = b.add(b.mul(n_img, xs_n), c_grp)
    w_blk = b.mul(b.add(k_grp, k_base), w_row)
    max_z = ((TH - 1) * TW + TW - 1) * XP  # an all-but-one-cell shifted window
    zero_v = b.zero_vec(io_type, 8)

    def _issue(u: Value) -> None:
        """Transfers of step ``u`` (chunk, tap) into ring slot u % NS."""
        chunk = b.div(u, n_taps1)
        ti = b.sub(u, b.mul(chunk, n_taps1))
        rr = b.div(ti, n_s1)
        r = b.add(r_lo, rr)
        s = b.add(s_lo, b.sub(ti, b.mul(rr, n_s1)))
        k0 = b.mul(chunk, c_CK)
        c_ext = b.smax(b.smin(b.sub(c_cpg, k0), c_CK), c0)
        slot = b.mul(b.mod(u, b.const_i32(NS)), b.const_i32(STAGE))
        ih0 = b.add(oh0, b.mul(r, b.const_i32(DH)))
        iw0 = b.add(ow0, b.mul(s, b.const_i32(DW)))
        hs = b.smin(b.smax(b.sub(c0, ih0), c0), b.const_i32(TH))
        ws = b.smin(b.smax(b.sub(c0, iw0), c0), b.const_i32(TW))
        h_start = b.smax(ih0, c0)
        w_start = b.smax(iw0, c0)
        h_ext = b.smax(b.sub(c_H, h_start), c0)
        w_ext = b.smax(b.smin(b.sub(c_W, w_start), b.sub(b.const_i32(TW), ws)), c0)
        z_elems = b.mul(b.add(b.mul(hs, b.const_i32(TW)), ws), b.const_i32(XP))
        # The leading cells of a window shifted by the top / left edge.
        if max_z:
            z_vecs = b.div(z_elems, c8)
            with b.scf_if(b.cmp_lt(c0, z_vecs)):
                for j in range(0, ceil_div(max_z, 8), THREADS):
                    v = b.add(tid, b.const_i32(j))
                    with b.scf_if(b.cmp_lt(v, z_vecs)):
                        b.smem_store_vN(
                            lds, [c0, b.add(slot, b.mul(v, c8))], zero_v, n=8
                        )
        x_row0 = b.add(b.add(x_img, k0), b.mul(w_start, xs_w))
        tap = b.add(b.mul(r, b.const_i32(KW)), s)
        w_off0 = b.add(w_blk, b.add(b.mul(tap, c_cpg), k0))
        for j in range(SPLIT):
            # Window rows [a, a + XR) from wave 2j (the rows inside the top
            # padding are zeroed above); weight rows [j*WR, +WR) from wave 2j+1.
            a = j * XR
            start = b.smax(b.const_i32(a), hs)
            rows = b.sub(b.const_i32(a + XR), start)
            ok = b.land(b.cmp_eq(wave_id, b.const_i32(2 * j)), b.cmp_lt(c0, rows))
            with b.scf_if(ok):
                ih = b.add(ih0, start)
                src = b.global_ptr_add(Xp, b.mul(b.add(x_row0, b.mul(ih, xs_h)), c2))
                if SPLIT > 1:
                    # A slab below the image transfers zeros from a valid address.
                    src = b.select(b.cmp_lt(ih, c_H), src, b.global_ptr_add(Xp, c0))
                cell = b.add(b.mul(start, b.const_i32(TW)), ws)
                dst = b.add(
                    lds_addr, b.mul(b.add(slot, b.mul(cell, b.const_i32(XP))), c2)
                )
                b.tensor_load_to_lds(
                    *_tdm_descriptor_3d(
                        b,
                        src,
                        dst,
                        dims=(c_ext, w_ext, b.smax(b.sub(c_H, ih), c0)),
                        strides=(xs_w, xs_h),
                        tile0=XP,
                        tile1=TW,
                        tile2=rows,
                    )
                )
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(2 * j + 1))):
                off = b.add(w_off0, b.mul(b.const_i32(j * WR), w_row))
                src = b.global_ptr_add(Wp, b.mul(off, c2))
                dst = b.add(
                    lds_addr,
                    b.mul(b.add(slot, b.const_i32(WIN + j * WR * XP)), c2),
                )
                k_ext_j = b.smax(
                    b.smin(b.sub(k_ext, b.const_i32(j * WR)), b.const_i32(WR)), c0
                )
                b.tensor_load_to_lds(
                    *_tdm_descriptor_3d(
                        b,
                        src,
                        dst,
                        dims=(c_ext, k_ext_j, c1),
                        strides=(w_row, w_row),
                        tile0=XP,
                        tile1=WR,
                        tile2=c1,
                    )
                )

    # ---- per-lane LDS read offsets (within a slot).
    # B (X): lane l reads pixel l % 16 of its tile, c (l // 16) * 16 .. +15.
    px_off = []
    for tn in range(TN):
        px = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        px_off.append(b.add(b.mul(px, b.const_i32(XP)), b.mul(lhalf, c16)))
    # A (W): lane l reads output channel l % 16 of its tile.
    wei_off = []
    for m in range(TM):
        row = b.add(
            b.mul(b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16), l16
        )
        wei_off.append(
            b.add(
                b.mul(row, b.const_i32(XP)),
                b.add(b.mul(lhalf, c16), b.const_i32(WIN)),
            )
        )

    def _read16(base: Value, off: Value, const_off: int) -> Value:
        return _lds_frag16(b, lds, base, off, const_off, c0=c0, c8=c8, io_type=io_type)

    def _frags(base: Value, ks: int):
        af = [_read16(base, wei_off[m], ks * WMMA_K) for m in range(TM)]
        bf = [_read16(base, px_off[tn], ks * WMMA_K) for tn in range(TN)]
        return af, bf

    KSTEPS = CK // WMMA_K

    def _step(u: Value, accs: list[Value]) -> list[Value]:
        _tdm_ring_wait(b, u, NS, n_steps)
        b.sync_lds_only()
        nxt = b.add(u, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_steps)):
            _issue(nxt)
        base = b.mul(b.mod(u, b.const_i32(NS)), b.const_i32(STAGE))
        accs = list(accs)
        b.s_setprio(1)
        nxt_frags = _frags(base, 0) if spec.prefetch else None
        for ks in range(KSTEPS):
            if spec.prefetch:
                af, bf = nxt_frags
                if ks + 1 < KSTEPS:
                    nxt_frags = _frags(base, ks + 1)
            else:
                af, bf = _frags(base, ks)
            for m in range(TM):
                for tn in range(TN):
                    i = m * TN + tn
                    accs[i] = b.mma(wmma_op, af[m], bf[tn], accs[i])
        b.s_setprio(0)
        return accs

    k_out = b.add(k_grp, k_base)

    def _epilogue(accs: list[Value]) -> None:
        """Stage the Y tile in LDS as [row][col][channel] and store it with one
        TDM transfer, which clips it to the image and the group's channels."""
        b.sync_lds_only()
        for m in range(TM):
            col = b.add(
                b.mul(b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16),
                b.mul(lhalf, c8),
            )
            for tn in range(TN):
                px = b.add(
                    b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16),
                    l16,
                )
                val = b.vec_cast_f32_to(accs[m * TN + tn], io_type)
                b.smem_store_vN(
                    lds, [c0, b.add(b.mul(px, b.const_i32(EPITCH)), col)], val, n=8
                )
        b.sync_lds_only()
        with b.scf_if(b.cmp_eq(wave_id, c0)):
            off = b.add(
                b.add(b.mul(n_img, ys_n), b.mul(h_tile, ys_h)),
                b.add(b.mul(w_tile, ys_w), k_out),
            )
            dst = b.global_ptr_add(Yp, b.mul(off, c2))
            b.tensor_store_from_lds(
                *_tdm_descriptor_3d(
                    b,
                    dst,
                    lds_addr,
                    dims=(
                        k_ext,
                        b.smax(b.sub(c_Wo, w_tile), c0),
                        b.smax(b.sub(c_Ho, h_tile), c0),
                    ),
                    strides=(ys_w, ys_h),
                    tile0=EPITCH,
                    tile1=TW,
                    tile2=b.const_i32(TH),
                )
            )
            b.s_wait_tensorcnt(0)

    # ---- main loop over (chunk, tap) steps.
    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_steps)):
            _issue(b.const_i32(st))
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0,
        n_steps,
        c1,
        [(f"acc{i}", zero_acc) for i in range(TM * TN)],
        iv_name="u",
    )
    with loop as (u, carried):
        b.scf_yield(*_step(u, list(carried)))
    _epilogue(list(loop.results))
    return b.kernel
