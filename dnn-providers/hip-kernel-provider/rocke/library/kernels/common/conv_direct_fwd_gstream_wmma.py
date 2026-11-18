# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped forward convolution streamed down the image, on wave32 WMMA (gfx1250).

Why a separate kernel
---------------------
``direct_fwd_wmma`` reduces over input channels in LDS chunks of 32. A grouped
layer with 8-32 channels per group (ResNeXt, RegNet) has a single chunk: each
block loads one halo, multiplies it once and stores, so the transfer latency
is never hidden behind work (the ring of stages has nothing to prefetch), and
the packed (block-diagonal) form multiplies half or three quarters zeros.

This kernel keeps a block on one channel slice and one column strip, and walks
``walk`` consecutive row bands (in flattened (image, band) order) through a
ring of ``stages`` halo slots: band ``i + stages - 1``'s halo is in flight
while band ``i`` is multiplied, so the work of one band hides the next one's
transfer::

    weights of the block's output channels -> VGPRs (A fragments), once
    for i in range(walk):                      # runtime scf.for
        TDM X[n, S*h0 - PAD : ..., S*w0 - PAD : ..., c0 : c0 + CH]
            -> LDS slot (i + stages - 1) % stages     (zero-filled at edges)
        for k-step:                            # Python-unrolled
            B = X fragment of the tap(s) of the k-step
            acc = wmma_16x16x32(A[k-step], B, acc)
        store Y (16 bytes per lane, masked by a buffer descriptor)

The WMMA K of 32 is filled without zero blocks:

- 8 or 16 channels per group (``cpg <= 16``): an M tile of 16 output channels
  reads 16 input channels (one group of 16, or two groups of 8), and each half
  of the WMMA K takes a different filter tap: lanes 0-15 hold tap ``2j``,
  lanes 16-31 tap ``2j + 1``. A 3x3 filter is 5 k-steps (the last tap pairs
  with a zero-weight copy of itself). With 8 channels per group the two groups
  of an M tile are block-diagonal (half the A zeros), against three quarters
  in a 32-channel packed super-group.
- 32 or more channels per group: plain 32-channel k-steps per tap; the block
  covers whole groups or part of one.

A operands stay in registers for the whole walk. M tiles of different groups
read different B (X) fragments; the M tiles of a wave inside one group share
theirs. Blocks of 64 or more input channels keep the halo transfer's
contiguous runs long.

AOT
---
Batch, spatial extents and the group count are kernargs. The filter, stride,
padding and the per-group channel counts (``cpg == kpg``) are baked.

C++ twin: ``library/cpp/instances/common/conv_direct_fwd_gstream_wmma.cpp``
(``instance_conv_direct_fwd_gstream_wmma.h``).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
)
from rocke.core.ir import I32, IRBuilder, KernelDef
from rocke.helpers.spec import align_up, ceil_div

from kernels.common.conv_direct_fwd_wmma import (
    _lds_frag16,
    _tdm_descriptor_3d,
    _tdm_ring_wait,
)
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _buf_store_vN,
    _io_type,
    emit_direct_params,
)
from kernels.common.conv_fwd_wmma_gate import (
    CH_PAD,
    WMMA_K,
    wmma_fwd_gate,
    wmma_fwd_halo_cols,
    wmma_fwd_halo_extent,
    wmma_fwd_lead_cols,
    wmma_fwd_limits_reason,
    wmma_fwd_op,
)

GSTREAM_VGPR_BUDGET = 448


@dataclass(frozen=True)
class FwdGstreamWmmaSpec:
    """One configuration of :func:`build_fwd_gstream_wmma`."""

    problem: DirectConvProblem
    name: str = "direct_fwd_gstream_wmma"
    waves_k: int = 1  # waves along output channels
    waves_p: int = 4  # waves along output pixels
    tiles_k: int = 1  # 16-channel tiles per wave
    tiles_p: int = 2  # 16-pixel tiles per wave
    tile_w: int = 16  # output columns per band; the band's rows follow
    walk: int = 8  # row bands per block
    stages: int = 2  # halo ring depth
    # Stride 2: even and odd input columns in two sub-halos (see direct_fwd_wmma).
    phase_split: bool = False
    # Read the next k-step's B fragments before this k-step's WMMAs
    # (``pf_depth`` k-steps ahead).
    prefetch: bool = True
    pf_depth: int = 1
    sched: str | None = None
    wave_size: int = 32

    @property
    def span16(self) -> bool:
        return self.problem.cpg <= 16

    @property
    def kpad(self) -> int:
        """Output channels of a group as the block sees them: kpg rounded up
        to whole 16-row M tiles (the padding rows have zero weights and are
        not stored)."""
        kpg = self.problem.kpg
        return kpg if self.span16 else align_up(kpg, 16)

    @property
    def block_k(self) -> int:
        return self.waves_k * self.tiles_k * 16

    @property
    def block_c(self) -> int:
        """Input channels of a block (one LDS row): its whole groups' channels,
        or one group's when the block is part of a group."""
        p = self.problem
        return max(self.block_k, self.kpad) * p.cpg // self.kpad

    @property
    def share_b(self) -> bool:
        """All M tiles of a wave lie in one group (they read the same X)."""
        return not self.span16 and self.kpad % (self.tiles_k * 16) == 0

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def tile_h(self) -> int:
        return self.block_p // self.tile_w if self.tile_w > 0 else 0

    @property
    def halo_h(self) -> int:
        p = self.problem
        return wmma_fwd_halo_extent(p.stride, self.tile_h, p.KH, p.dil_h)

    @property
    def halo_w(self) -> int:
        p = self.problem
        return wmma_fwd_halo_extent(p.stride, self.tile_w, p.KW, p.dil_w)

    @property
    def phases_w(self) -> int:
        return self.problem.stride if self.phase_split else 1

    @property
    def halo_cols(self) -> int:
        return wmma_fwd_halo_cols(self.problem, self.tile_w, self.phase_split)

    @property
    def lead_cols(self) -> int:
        p = self.problem
        return wmma_fwd_lead_cols(p.PAD, p.stride, self.phase_split)

    @property
    def x_pitch(self) -> int:
        return self.block_c + CH_PAD

    @property
    def phase_elems(self) -> int:
        return (
            self.halo_h * self.halo_cols * self.x_pitch + self.lead_cols * self.x_pitch
        )

    @property
    def slot_elems(self) -> int:
        """One ring slot: the (sub-)halos and their wrap slack."""
        if self.phase_split:
            return self.phases_w * self.phase_elems
        return (self.halo_h * self.halo_w + self.problem.PAD) * self.x_pitch

    def lds_bytes(self) -> int:
        return 2 * self.stages * self.slot_elems

    @property
    def taps(self) -> int:
        return self.problem.KH * self.problem.KW

    def k_steps(self) -> list[tuple[int, int, int]]:
        """``(tap of lanes 0-15, tap of lanes 16-31, 32-channel chunk)`` per
        k-step; tap -1 is a zero-weight half (it reads the previous tap's X)."""
        T = self.taps
        if self.span16:
            out = []
            for j in range(0, T, 2):
                out.append((j, j + 1 if j + 1 < T else -1, 0))
            return out
        return [
            (t, t, c)
            for t in range(T)
            for c in range(ceil_div(self.problem.cpg, WMMA_K))
        ]

    @property
    def threads_per_block(self) -> int:
        return self.waves_k * self.waves_p * self.wave_size

    def n_bands(self) -> int:
        return ceil_div(self.problem.Ho, self.tile_h)

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        n_kb = p.groups * self.kpad // self.block_k
        n_wt = ceil_div(p.Wo, self.tile_w)
        walkers = ceil_div(p.N * self.n_bands(), self.walk)
        return (n_kb * n_wt * walkers, 1, 1)

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"str{p.stride}" if p.stride > 1 else "",
            f"c{p.cpg}k{p.kpg}",
            f"wk{self.waves_k}",
            f"wp{self.waves_p}",
            f"tk{self.tiles_k}",
            f"tp{self.tiles_p}",
            f"tw{self.tile_w}",
            f"wa{self.walk}",
            f"st{self.stages}",
            "ps" if self.phase_split else "",
            "pf" if self.prefetch else "",
            f"pd{self.pf_depth}" if self.prefetch and self.pf_depth > 1 else "",
            f"s{self.sched}" if self.sched else "",
            flags={"bf16": p.dtype == "bf16"},
        )


def fwd_gstream_tile_efficiency(spec: FwdGstreamWmmaSpec) -> float:
    """Share of ``spec``'s padded pixel grid that is real work on its problem."""
    p = spec.problem
    hp = spec.n_bands() * spec.tile_h
    wp = align_up(p.Wo, spec.tile_w)
    return (p.Ho * p.Wo) / (hp * wp)


def fwd_gstream_reason(spec: FwdGstreamWmmaSpec, arch: str = "gfx1250") -> str | None:
    """Why ``spec`` cannot run on ``arch`` (None when it can)."""
    p = spec.problem
    why, target = wmma_fwd_gate(p, arch, "direct_fwd_gstream_wmma", spec.wave_size)
    if why:
        return why
    if p.stride not in (1, 2):
        return f"stride {p.stride} not in (1, 2)"
    if (p.dil_h, p.dil_w) != (1, 1):
        return "dilation is not served"
    if p.KH != p.KW:
        return f"filter {p.KH}x{p.KW} is not square"
    if p.PAD < 0 or p.PAD > p.KH - 1:
        return f"PAD {p.PAD} outside 0..K-1"
    if p.Ho < 1 or p.Wo < 1:
        return f"empty output {p.Ho}x{p.Wo}"
    if p.cpg != p.kpg:
        return f"cpg {p.cpg} != kpg {p.kpg}"
    if p.cpg not in (8, 16) and (p.cpg < WMMA_K or p.cpg % 8):
        return f"cpg {p.cpg} not 8, 16 or a multiple of 8 from 32"
    if p.cpg > 16 and p.cpg % WMMA_K and align_up(p.cpg, WMMA_K) - p.cpg > CH_PAD:
        return f"cpg {p.cpg}: its last k-step reads past the LDS row skew"
    if min(spec.waves_k, spec.waves_p, spec.tiles_k, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.block_k % spec.kpad and spec.kpad % spec.block_k:
        return f"block_k {spec.block_k} is neither whole groups nor part of one"
    if (p.groups * spec.kpad) % spec.block_k:
        return f"K {p.total_k} not a multiple of block_k {spec.block_k}"
    if spec.pf_depth < 1 or (spec.pf_depth > 1 and not spec.prefetch):
        return f"pf_depth {spec.pf_depth}: >= 1, and > 1 only with prefetch"
    if spec.stages < 2 or spec.walk < 1:
        return "stages must be >= 2 and walk >= 1"
    tw = spec.tile_w
    if tw < 1 or spec.block_p % tw:
        return f"tile_w {tw} must divide the {spec.block_p} pixels of a band"
    if spec.phase_split:
        if p.stride != 2:
            return "phase_split serves stride 2 only"
        if spec.waves_k * spec.waves_p < 2:
            return "phase_split needs two waves (one transfer each)"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if max(spec.x_pitch, spec.halo_w, spec.halo_h) > 0xFFFF:
        return "tile too wide for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    TM, TN = spec.tiles_k, spec.tiles_p
    nks = len(spec.k_steps())
    b_sets = TN if spec.share_b else TM * TN
    vgprs = (
        8 * TM * TN
        + 8 * TM * nks
        + 8 * b_sets * (1 + spec.pf_depth if spec.prefetch else 1)
    )
    if vgprs > GSTREAM_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of weights, accumulators and fragments"
    return wmma_fwd_limits_reason(p, spec.sched, spec.grid()[0])


def is_valid_fwd_gstream_spec(
    spec: FwdGstreamWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = fwd_gstream_reason(spec, arch)
    return (why is None, why or "ok")


def build_fwd_gstream_wmma(
    spec: FwdGstreamWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    why = fwd_gstream_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid FwdGstreamWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW = p.KH, p.KW
    S, PAD = p.stride, p.PAD
    CPG = p.cpg
    io_type = _io_type(p.dtype)
    wmma_op = wmma_fwd_op(p.dtype, arch)

    WAVE = spec.wave_size
    WK = spec.waves_k
    TM, TN = spec.tiles_k, spec.tiles_p
    THREADS = spec.threads_per_block
    BK, CH = spec.block_k, spec.block_c
    TH, TW = spec.tile_h, spec.tile_w
    HALO_H, HALO_W = spec.halo_h, spec.halo_w
    NS = spec.stages
    XP = spec.x_pitch
    SLOT = spec.slot_elems
    TAPS = spec.taps
    PS = spec.phase_split
    NPW = spec.phases_w
    HC = spec.halo_cols
    PSTR = spec.phase_elems if PS else 0
    SPAN16 = spec.span16
    SHARE_B = spec.share_b
    KSTEPS = spec.k_steps()
    KPAD = spec.kpad  # padded group rows (== kpg unless kpg % 16)
    PADG = KPAD != p.kpg or CPG % WMMA_K and not SPAN16  # masked group edges

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
    c_N = params["p_N"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]
    c_K = params["p_total_k"]
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
    wave_id = b.div(tid, b.const_i32(WAVE))
    lane = b.mod(tid, b.const_i32(WAVE))
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_k = b.mod(wave_id, b.const_i32(WK))
    wave_p = b.div(wave_id, b.const_i32(WK))

    # ---- block: bx = (walker * n_wt + wt) * n_kb + kb.
    if KPAD != p.kpg:
        n_kb = b.div(
            b.mul(b.div(c_K, b.const_i32(p.kpg)), b.const_i32(KPAD)), b.const_i32(BK)
        )
    else:
        n_kb = b.div(c_K, b.const_i32(BK))
    n_wt = b.div(b.add(c_Wo, b.const_i32(TW - 1)), b.const_i32(TW))
    n_bands = b.div(b.add(c_Ho, b.const_i32(TH - 1)), b.const_i32(TH))
    t = b.block_id_x()
    kb = b.mod(t, n_kb)
    t = b.div(t, n_kb)
    wt = b.mod(t, n_wt)
    walker = b.div(t, n_wt)
    k_base = b.mul(kb, b.const_i32(BK))  # first output channel of the block
    # First input channel: the block's first group's (whole groups), or its
    # group's (part of one).
    c_base = b.mul(b.div(k_base, b.const_i32(KPAD)), b.const_i32(CPG))
    w_tile = b.mul(wt, b.const_i32(TW))
    task0 = b.mul(walker, b.const_i32(spec.walk))
    n_tasks = b.mul(c_N, n_bands)
    n_it = b.smin(b.const_i32(spec.walk), b.sub(n_tasks, task0))

    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="gs_lds")
    lds_addr = b.trunc(b.smem_addr_of(lds), I32)

    # ---- per (phase) sub-halo column geometry: fixed for the whole walk.
    c_S = b.const_i32(S)
    w0 = b.sub(w_tile if S == 1 else b.mul(w_tile, c_S), b.const_i32(PAD))
    sub_cols = []
    for a in range(NPW):
        wa = b.add(w0, b.const_i32(a)) if a else w0
        if PS:
            sw = b.smax(b.div(b.sub(b.const_i32(S - 1), wa), c_S), c0)
            sw = b.smin(sw, b.const_i32(HC))
            w_start = b.smax(b.add(wa, b.mul(sw, c_S)), c0)
            w_ext = b.div(b.add(b.sub(c_W, w_start), b.const_i32(S - 1)), c_S)
        else:
            sw = b.smin(b.smax(b.sub(c0, wa), c0), b.const_i32(HC))
            w_start = b.smax(b.add(wa, sw), c0)
            w_ext = b.sub(c_W, w_start)
        w_ext = b.smax(b.smin(w_ext, b.sub(b.const_i32(HC), sw)), c0)
        sub_cols.append((sw, w_start, w_ext))
    xs_col = b.mul(xs_w, c_S) if PS else xs_w
    max_z = (min(PAD, HALO_H) * HC + min(spec.lead_cols, HC)) * XP
    zero_v = b.zero_vec(io_type, 8)

    def _issue(task, slot) -> None:
        """Band ``task``'s halo into ring slot ``slot`` (wave a: phase a)."""
        n = b.div(task, n_bands)
        band = b.mod(task, n_bands)
        h0 = b.sub(b.mul(band, b.const_i32(TH * S)), b.const_i32(PAD))
        hs = b.smin(b.smax(b.sub(c0, h0), c0), b.const_i32(HALO_H))
        h_start = b.smax(b.add(h0, hs), c0)
        h_ext = b.smax(b.sub(c_H, h_start), c0)
        tile2 = b.sub(b.const_i32(HALO_H), hs)
        slot_off = b.mul(slot, b.const_i32(SLOT))
        for a, (sw, w_start, w_ext) in enumerate(sub_cols):
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(a))):
                z_elems = b.mul(b.add(b.mul(hs, b.const_i32(HC)), sw), b.const_i32(XP))
                region = b.add(slot_off, b.const_i32(a * PSTR))
                if max_z:
                    # Cells above / left of the image: no transfer writes them.
                    z_vecs = b.div(z_elems, c8)
                    for j in range(0, ceil_div(max_z, 8), WAVE):
                        v = b.add(lane, b.const_i32(j))
                        with b.scf_if(b.cmp_lt(v, z_vecs)):
                            b.smem_store_vN(
                                lds, [c0, b.add(region, b.mul(v, c8))], zero_v, n=8
                            )
                base = b.add(
                    b.add(b.mul(n, xs_n), b.mul(h_start, xs_h)),
                    b.add(b.mul(w_start, xs_w), c_base),
                )
                src = b.global_ptr_add(Xp, b.mul(base, c2))
                dst = b.add(lds_addr, b.mul(b.add(region, z_elems), c2))
                b.tensor_load_to_lds(
                    *_tdm_descriptor_3d(
                        b,
                        src,
                        dst,
                        dims=(b.const_i32(CH), w_ext, h_ext),
                        strides=(xs_col, xs_h),
                        tile0=XP,
                        tile1=HC,
                        tile2=tile2,
                    )
                )

    # ---- A fragments (weights) in registers for the whole walk.
    # Lane l of M tile m: output channel row l % 16, K half l // 16.
    ks_tap = []  # per k-step: this lane's tap (lane-varying when the halves differ)
    for t0, t1, _ in KSTEPS:
        if t0 == t1:
            ks_tap.append(b.const_i32(t0))
        else:
            # A zero half (tap -1) reads the previous tap's X (finite values).
            tb = t1 if t1 >= 0 else t0
            ks_tap.append(
                b.select(b.cmp_eq(lhalf, c0), b.const_i32(t0), b.const_i32(tb))
            )
    zero8 = b.zero_vec(io_type, 8)
    afrag: list[list] = []
    for m in range(TM):
        row = b.add(
            b.add(
                k_base,
                b.mul(b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16),
            ),
            l16,
        )
        row_ok = None
        if KPAD != p.kpg:
            # padded row -> real output channel; padding rows load zeros
            r_in = b.mod(row, b.const_i32(KPAD))
            row_ok = b.cmp_lt(r_in, b.const_i32(p.kpg))
            row = b.select(
                row_ok,
                b.add(b.mul(b.div(row, b.const_i32(KPAD)), b.const_i32(p.kpg)), r_in),
                c0,
            )
        row_base = b.mul(row, b.const_i32(TAPS))
        frags = []
        for ks, (t0, t1, cc) in enumerate(KSTEPS):
            gidx = b.mul(b.add(row_base, ks_tap[ks]), b.const_i32(CPG))
            if SPAN16 and CPG == 8:
                w8 = b.global_load_vN(Wp, gidx, io_type, 8)
                first = b.cmp_eq(b.div(l16, c8), c0)  # the M tile's first group
                lo, hi = b.select(first, w8, zero8), b.select(first, zero8, w8)
            else:
                if not SPAN16:
                    gidx = b.add(
                        gidx, b.add(b.const_i32(cc * WMMA_K), b.mul(lhalf, c16))
                    )
                if PADG:
                    # channels past cpg (the last k-step) and padding rows: zero
                    ch = b.add(b.const_i32(cc * WMMA_K), b.mul(lhalf, c16))
                    ok_lo = b.cmp_lt(ch, b.const_i32(CPG))
                    ok_hi = b.cmp_lt(b.add(ch, c8), b.const_i32(CPG))
                    if row_ok is not None:
                        ok_lo, ok_hi = b.land(ok_lo, row_ok), b.land(ok_hi, row_ok)
                    lo = b.global_load_vN(Wp, b.select(ok_lo, gidx, c0), io_type, 8)
                    hi = b.global_load_vN(
                        Wp, b.select(ok_hi, b.add(gidx, c8), c0), io_type, 8
                    )
                    lo, hi = b.select(ok_lo, lo, zero8), b.select(ok_hi, hi, zero8)
                else:
                    lo = b.global_load_vN(Wp, gidx, io_type, 8)
                    hi = b.global_load_vN(Wp, b.add(gidx, c8), io_type, 8)
            if t1 < 0:
                keep = b.cmp_eq(lhalf, c0)
                lo, hi = b.select(keep, lo, zero8), b.select(keep, hi, zero8)
            frags.append(b.vec_concat(lo, hi))
        afrag.append(frags)

    # ---- per-lane B (X) read offsets.
    # Lane l reads pixel l % 16 of its tile; its K half's tap / channels.
    px_off = []
    for tn in range(TN):
        flat = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        th = b.div(flat, b.const_i32(TW))
        tw = b.mod(flat, b.const_i32(TW))
        if S > 1:
            th = b.mul(th, c_S)
            if not PS:
                tw = b.mul(tw, c_S)
        off = b.mul(b.add(b.mul(th, b.const_i32(HC)), tw), b.const_i32(XP))
        if not SPAN16:
            off = b.add(off, b.mul(lhalf, c16))
        px_off.append(off)

    def _tap_off(tap: int) -> int:
        r, s = divmod(tap, KW)
        if PS:
            a, j = s % S, s // S
            return a * PSTR + (r * HC + j) * XP
        return (r * HALO_W + s) * XP

    ks_off = []  # (static offset, lane-varying offset or None) per k-step
    for t0, t1, cc in KSTEPS:
        tb = t1 if t1 >= 0 else t0
        if _tap_off(t0) == _tap_off(tb):
            ks_off.append((_tap_off(t0) + cc * WMMA_K, None))
        else:
            ks_off.append(
                (
                    _tap_off(t0),
                    b.select(
                        b.cmp_eq(lhalf, c0),
                        c0,
                        b.const_i32(_tap_off(tb) - _tap_off(t0)),
                    ),
                )
            )
    m_glob = [b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)) for m in range(TM)]
    # X channel offset of M tile m inside the block: its group's first channel
    # (relative to the block's), 0 when the block is part of one group.
    if BK > KPAD:
        m_choff = [
            b.mul(b.div(b.mul(mg, c16), b.const_i32(KPAD)), b.const_i32(CPG))
            for mg in m_glob
        ]
    else:
        m_choff = [c0] * TM

    def _bfrags(slot_base, ks):
        static, var = ks_off[ks]
        out = []
        sets = (
            [(0, tn) for tn in range(TN)]
            if SHARE_B
            else [(m, tn) for m in range(TM) for tn in range(TN)]
        )
        for m, tn in sets:
            addr = b.add(slot_base, px_off[tn])
            if var is not None:
                addr = b.add(addr, var)
            if BK > KPAD:
                addr = b.add(addr, m_choff[m])
            addr = b.add(addr, b.const_i32(static))
            out.append(_lds_frag16(b, lds, addr, c0=c0, c8=c8, io_type=io_type))
        return out

    # ---- epilogue: 16 bytes per lane, masked through the buffer bounds.
    y_rsrc = b.buffer_rsrc(Yp, params["D_bytes"])
    ep_px = []
    for tn in range(TN):
        flat = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        ep_px.append((b.div(flat, b.const_i32(TW)), b.mod(flat, b.const_i32(TW))))
    ep_ch = [
        b.add(b.add(k_base, b.mul(m_glob[m], c16)), b.mul(lhalf, c8)) for m in range(TM)
    ]
    ep_ok = [None] * TM
    if KPAD != p.kpg:
        # padded channel -> real one; a padding run (8 channels) is not stored
        for m in range(TM):
            v = ep_ch[m]
            r_in = b.mod(v, b.const_i32(KPAD))
            ep_ok[m] = b.cmp_lt(r_in, b.const_i32(p.kpg))
            ep_ch[m] = b.add(
                b.mul(b.div(v, b.const_i32(KPAD)), b.const_i32(p.kpg)), r_in
            )

    def _epilogue(task, accs) -> None:
        n = b.div(task, n_bands)
        h_tile = b.mul(b.mod(task, n_bands), b.const_i32(TH))
        nbase = b.mul(n, ys_n)
        for tn in range(TN):
            th, tw = ep_px[tn]
            h = b.add(h_tile, th)
            w = b.add(w_tile, tw)
            ok = b.land(b.cmp_lt(h, c_Ho), b.cmp_lt(w, c_Wo))
            pix = b.add(nbase, b.add(b.mul(h, ys_h), b.mul(w, ys_w)))
            for m in range(TM):
                off = b.mul(b.add(pix, ep_ch[m]), c2)
                okm = ok if ep_ok[m] is None else b.land(ok, ep_ok[m])
                voff = b.select(okm, off, params["D_bytes"])
                val = b.vec_cast_f32_to(accs[m * TN + tn], io_type)
                _buf_store_vN(b, p.dtype, y_rsrc, voff, c0, val, 4)

    # ---- the walk.
    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_it)):
            _issue(b.add(task0, b.const_i32(st)), b.const_i32(st))
    zero_acc = b.zero_vec_f32(8)
    with b.scf_for(c0, n_it, c1, iv_name="it") as it:
        _tdm_ring_wait(b, it, NS, n_it)
        b.sync_lds_only()  # band it is visible; slot (it - 1) % NS is free
        nxt = b.add(it, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_it)):
            _issue(b.add(task0, nxt), b.mod(nxt, b.const_i32(NS)))
        slot_base = b.mul(b.mod(it, b.const_i32(NS)), b.const_i32(SLOT))
        accs = [zero_acc] * (TM * TN)
        b.s_setprio(1)
        NKS = len(KSTEPS)
        PD = spec.pf_depth if spec.prefetch else 0
        queue = [_bfrags(slot_base, j) for j in range(min(PD, NKS))]
        for ks in range(NKS):
            if PD:
                bf = queue.pop(0)
                if ks + PD < NKS:
                    queue.append(_bfrags(slot_base, ks + PD))
            else:
                bf = _bfrags(slot_base, ks)
            for m in range(TM):
                for tn in range(TN):
                    i = m * TN + tn
                    bi = tn if SHARE_B else i
                    accs[i] = b.mma(wmma_op, afrag[m][ks], bf[bi], accs[i])
        b.s_setprio(0)
        _epilogue(b.add(task0, it), accs)
    return b.kernel
