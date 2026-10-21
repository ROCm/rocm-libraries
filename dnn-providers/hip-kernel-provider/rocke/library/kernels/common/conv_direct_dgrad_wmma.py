# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct stride-1 dgrad on wave32 WMMA (gfx1250).

Why a separate family
---------------------
On gfx950 the direct dgrad is a pipeline: transpose the weights, reorganize
them, then run the MFMA forward kernel (``DirectConvSpec``) on the transposed
problem. Every piece of it is wave64 MFMA, and the forward kernel draws its
parallelism from the group axis, so it only serves small per-group channel
counts. This kernel computes the same thing directly on wave32 WMMA, for dense
and grouped layers alike, with no weight pre-pass:

    dX[n, h, w, c] = sum_{r, s, k} dY[n, h + PAD - r, w + PAD - s, k] * W[k, r, s, c]

Per workgroup, with M = dX channels and N = dX pixels::

    block_c                     dX channels       (waves_c * tiles_c * 16)
    tile_h x tile_w             dX pixels         (waves_p * tiles_p * 16)
    ck                          dY channels reduced per LDS stage

    for k0 in range(0, kpg, ck):                  # runtime scf.for
        TDM dY[h0 : h0 + tile_h + KH - 1, w0 : w0 + tile_w + KW - 1, k0 : k0 + ck]
            -> LDS halo [row][col][k]              (one 3-D transfer, zero-filled)
        TDM W[k0 : k0 + ck, all taps, c0 : c0 + block_c]
            -> LDS [tap][k][c]                     (one 3-D transfer)
        for k-step, tap (r, s):                    # Python-unrolled
            A = W^T fragment  (ds_load_tr16_b128: the transpose happens on the read)
            B = dY fragment   (halo shifted by (KH-1-r, KW-1-s): no im2col)
            acc[c-tile, px-tile] = wmma_16x16x32(A, B, acc)

The halo origin is ``(h_tile - (KH-1-PAD), w_tile - (KW-1-PAD))``. A transfer
cannot start at a negative coordinate, so a block on the top or left image
edge starts its read at row / column 0 and lands it shifted down / right in
the slot; the rows and columns above / left of it are zeroed once, before the
loop, and never written again. Shifting right wraps the last columns of each
row into the next row's leading (zero) columns: the transfer's column extent is
clipped so that it writes zeros there, which is also why each slot has a
little slack at its end.

The slots form a ring of ``stages``: slot ``i % stages`` holds chunk ``i``; one
barrier per chunk publishes it, after which the slot the previous chunk read
is refilled ``stages - 1`` chunks ahead. Wave ``i < tile_n`` issues image
``i``'s dY transfer, wave ``tile_n`` the weights.

With ``split_barrier`` the per-chunk barrier is split: a wave signals once it
has read the chunk's last fragments (and, as the issuer, once its next
transfer has landed), and waits at the top of the next chunk, so its last
tap's WMMAs run while the slower waves catch up.

Every lane of the accumulator holds one pixel and 8 consecutive dX channels.
The epilogue stages the dX tile in LDS as [image][row][col][channel] and
stores it with one TDM transfer per image, which clips it to the image and to
the group's channels: measured faster than per-lane 16-byte stores, which
touch one 32-byte run per pixel.

Register blocking (``tiles_c x tiles_p`` 16x16 tiles per wave) sets the LDS
reads per WMMA, the lever of this loop: a tap reads ``tiles_c + tiles_p``
fragments for ``tiles_c * tiles_p`` WMMAs. Past 256 VGPRs the backend
addresses the upper register banks.

Stride, dilation and small groups
---------------------------------
A stride-``S`` dgrad splits into ``S * S`` output phases: row ``S * i + a``
collects filter row ``r`` from dY row ``i + delta(r)`` for the one phase ``a``
with ``(a + PAD - r * dil) % S == 0`` (:func:`_axis_plan`). The block tiles the
phase grid, keeps one dY halo for all phases, and holds one accumulator set
per phase; taps that share a halo shift share their B fragments. The
epilogue interleaves the phases in LDS and stores the dense ``S * tile_h x
S * tile_w`` dX tile. Dilation only changes the halo shift of each tap, so it
is the stride-1 case of the same plan.

``pack_groups = G`` treats ``G`` consecutive groups as one super-group of
``G * cpg`` dX and ``G * kpg`` dY channels (contiguous in NHWC). Its weights
are staged once, before the loop, as a block-diagonal matrix (zeros off the
diagonal): the layers this serves are memory-bound, so the zero products cost
less than the per-group blocks they replace.

AOT
---
Batch, spatial extents, the group count and both channel counts are kernargs
(the direct-conv dgrad block of :func:`kernels.common.conv_abi.conv_direct_arg_names`).
The filter and padding shape the halo and the unrolled tap loop, so they are
build-time capabilities, like the tile geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    CodegenPolicy,
    apply_codegen_policy,
    normalize_scheduler_strategy,
)
from rocke.core.ir import I32, IRBuilder, KernelDef, Value

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)

WMMA_K = 32  # K of the 16x16x32 atom
CH_PAD = 8  # channels of row skew past each LDS row (16 bytes)
DGRAD_WMMA_VGPR_BUDGET = 448
_I32_MAX = (1 << 31) - 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _axis_plan(K: int, PAD: int, dil: int, S: int):
    """Per-axis tap plan of a strided / dilated dgrad.

    Output row ``h = S * i + a`` (phase ``a``) collects filter row ``r`` from
    dY row ``i + delta(r)``, ``delta(r) = (a + PAD - r * dil) / S``, for the
    one phase ``a`` that makes it exact. Returns ``([(a, delta - dmin)] per r,
    dmax - dmin, -dmin)``: the halo offset of every tap, the halo rows past
    the tile, and the halo origin shift (the forward pass's padding). Stride 1
    and dilation 1 give ``K - 1 - r``, ``K - 1`` and ``K - 1 - PAD``.
    """
    taps = []
    for r in range(K):
        for a in range(S):
            if (a + PAD - r * dil) % S == 0:
                taps.append((a, (a + PAD - r * dil) // S))
                break
    dmin = min(d for _, d in taps)
    dmax = max(d for _, d in taps)
    return [(a, d - dmin) for a, d in taps], dmax - dmin, -dmin


@dataclass(frozen=True)
class DirectDgradWmmaSpec:
    """One configuration of :func:`build_direct_dgrad_wmma`.

    ``problem`` supplies the build-time capabilities (filter, padding, dtype)
    and the shape used for launch geometry and validation; the channel counts,
    group count and extents are read from the kernargs.
    """

    problem: DirectConvProblem
    name: str = "direct_dgrad_wmma"
    waves_c: int = 2  # waves along dX channels
    waves_p: int = 4  # waves along dX pixels
    tiles_c: int = 4  # 16-channel tiles per wave
    tiles_p: int = 4  # 16-pixel tiles per wave
    tile_w: int = 16  # dX columns per block; tile_h follows from the pixel count
    tile_n: int = 1  # images per block (small maps)
    ck: int = 32  # dY channels per LDS stage (multiple of 32)
    stages: int = 2  # LDS ring depth
    # Read the next tap's fragments before issuing this tap's WMMAs, so the
    # LDS latency hides behind them (costs a second fragment set of VGPRs).
    prefetch: bool = False
    # AMDGPU machine-scheduler strategy (rocke.core.codegen_policy), or None.
    sched: str | None = None
    wave_size: int = 32
    # Groups packed into one block-diagonal super-group (small cpg): the
    # block multiplies pack_groups * cpg dX channels against pack_groups * kpg
    # dY channels, with the off-diagonal weights zero. Bakes cpg and kpg.
    pack_groups: int = 1

    @property
    def stride(self) -> int:
        return self.problem.stride

    @property
    def plan_h(self):
        p = self.problem
        return _axis_plan(p.KH, p.PAD, p.dil_h, p.stride)

    @property
    def plan_w(self):
        p = self.problem
        return _axis_plan(p.KW, p.PAD, p.dil_w, p.stride)

    @property
    def phases(self) -> int:
        return self.problem.stride * self.problem.stride

    @property
    def cpg_eff(self) -> int:
        return self.pack_groups * self.problem.cpg

    @property
    def kpg_eff(self) -> int:
        return self.pack_groups * self.problem.kpg

    @property
    def groups_eff(self) -> int:
        return self.problem.groups // self.pack_groups

    # Split the per-chunk workgroup barrier (s_barrier_signal / _wait): a wave
    # signals once it has read the chunk's last fragments, and waits only at
    # the top of the next chunk, so the last tap's WMMAs cover the skew.
    split_barrier: bool = False
    # Epilogue per wave: each wave stages its own tiles in a private LDS
    # region and stores them with its own TDM transfer, with no block barrier
    # between the staging and the store (needs whole rows of one image per
    # wave: tiles_p * 16 a multiple of tile_w that divides tile_h * tile_w).
    wave_epilogue: bool = False

    @property
    def block_c(self) -> int:
        return self.waves_c * self.tiles_c * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def tile_h(self) -> int:
        d = self.tile_w * self.tile_n
        return self.block_p // d if d > 0 else 0  # 0: rejected by the validator

    @property
    def threads_per_block(self) -> int:
        return self.waves_c * self.waves_p * self.wave_size

    @property
    def halo_h(self) -> int:
        return self.tile_h + self.plan_h[1]

    @property
    def halo_w(self) -> int:
        return self.tile_w + self.plan_w[1]

    @property
    def x_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def w_pitch(self) -> int:
        return self.block_c + CH_PAD

    @property
    def image_elems(self) -> int:
        return self.halo_h * self.halo_w * self.x_pitch

    @property
    def halo_elems(self) -> int:
        return self.tile_n * self.image_elems

    @property
    def wei_elems(self) -> int:
        p = self.problem
        return p.KH * p.KW * self.ck * self.w_pitch

    @property
    def halo_slack(self) -> int:
        # A left-edge block's last row wraps up to KW-1-PAD columns past the ring.
        return self.plan_w[2] * self.x_pitch

    def lds_bytes(self) -> int:
        # Each halo slot carries its own wrap slack (see the module docstring).
        ring = self.stages * (self.halo_elems + self.halo_slack + self.wei_elems)
        epi = self.phases * self.block_p * self.w_pitch  # the staged dX tile
        if self.wave_epilogue:
            n_waves = self.waves_c * self.waves_p
            epi = n_waves * self.tiles_p * 16 * (self.tiles_c * 16 + CH_PAD)
        return 2 * max(ring, epi)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"wc{self.waves_c}",
            f"wp{self.waves_p}",
            f"tc{self.tiles_c}",
            f"tp{self.tiles_p}",
            f"tw{self.tile_w}",
            f"tn{self.tile_n}" if self.tile_n > 1 else "",
            f"ck{self.ck}",
            f"st{self.stages}",
            "pf" if self.prefetch else "",
            f"s{self.sched}" if self.sched else "",
            f"str{p.stride}" if p.stride > 1 else "",
            f"dil{p.dil_h}x{p.dil_w}" if p.is_dilated else "",
            (f"pg{self.pack_groups}c{p.cpg}k{p.kpg}" if self.pack_groups > 1 else ""),
            "sb" if self.split_barrier else "",
            "we" if self.wave_epilogue else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        S = p.stride
        n_ct = _ceil_div(self.cpg_eff, self.block_c)
        n_ht = _ceil_div(_ceil_div(p.H, S), self.tile_h)
        n_wt = _ceil_div(_ceil_div(p.W, S), self.tile_w)
        return (
            _ceil_div(p.N, self.tile_n) * self.groups_eff * n_ht * n_wt * n_ct,
            1,
            1,
        )


def dgrad_wmma_tile_efficiency(spec: DirectDgradWmmaSpec) -> float:
    """Share of ``spec``'s padded tile grid that is real work on its problem."""
    p = spec.problem
    S = p.stride
    hq, wq = _ceil_div(p.H, S), _ceil_div(p.W, S)
    tiles = (
        _ceil_div(p.N, spec.tile_n) * spec.tile_n,
        _ceil_div(hq, spec.tile_h) * spec.tile_h,
        _ceil_div(wq, spec.tile_w) * spec.tile_w,
        _ceil_div(spec.cpg_eff, spec.block_c) * spec.block_c,
    )
    return (p.N * hq * wq * spec.cpg_eff) / (tiles[0] * tiles[1] * tiles[2] * tiles[3])


def dgrad_wmma_reason(spec: DirectDgradWmmaSpec, arch: str = "gfx1250") -> str | None:
    """Why ``spec`` cannot run on ``arch`` (None when it can)."""
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return str(e)
    p = spec.problem
    if target.wave_size != 32 or spec.wave_size != 32:
        return "direct_dgrad_wmma is a wave32 kernel"
    if not target.memory.has_tdm:
        return f"no TDM on {arch}"
    ab = "bf16" if p.dtype == "bf16" else "f16"
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return f"missing wmma_f32_16x16x32_{ab} on {arch}"
    if p.stride not in (1, 2):
        return f"stride {p.stride} not in (1, 2)"
    if p.stride > 1 and p.is_dilated:
        return "a strided dilated dgrad is not supported"
    if p.dil_h < 1 or p.dil_w < 1:
        return f"dilation {p.dil_h}x{p.dil_w} < 1"
    if p.PAD < 0 or spec.plan_h[2] < 0 or spec.plan_w[2] < 0:
        return f"PAD {p.PAD} outside 0..(K-1)*dilation"
    G = spec.pack_groups
    if G < 1:
        return f"pack_groups {G} < 1"
    if G > 1:
        if p.groups % G:
            return f"groups {p.groups} not a multiple of pack_groups {G}"
        if p.cpg not in (2, 4, 8, 16) or p.kpg not in (2, 4, 8, 16):
            return f"packed cpg {p.cpg} / kpg {p.kpg} not in (2, 4, 8, 16)"
        if spec.kpg_eff > spec.ck:
            return f"packed kpg {spec.kpg_eff} > ck {spec.ck} (one dY chunk)"
        if spec.cpg_eff > spec.block_c:
            return f"packed cpg {spec.cpg_eff} > block_c {spec.block_c}"
    if spec.cpg_eff % 8 or spec.kpg_eff % 8:
        return (
            f"cpg {spec.cpg_eff} and kpg {spec.kpg_eff} must be multiples of 8"
            " (16-byte rows)"
        )
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_c, spec.waves_p, spec.tiles_c, spec.tiles_p) < 1:
        return "waves and tiles must be >= 1"
    if spec.stages < 2:
        return "stages must be >= 2"
    tw = spec.tile_w
    if tw < 1 or spec.tile_n < 1:
        return f"tile_w {tw} and tile_n {spec.tile_n} must be >= 1"
    if spec.block_p % (tw * spec.tile_n):
        return f"tile_w {tw} x tile_n {spec.tile_n} must divide {spec.block_p}"
    if spec.tile_n > 1 and (spec.tile_h * tw) % 16:
        return "an image tile must hold whole 16-pixel tiles"
    if spec.tile_n > spec.waves_c * spec.waves_p - 1:
        return "tile_n needs one more wave than images (one dY transfer per image)"
    if (spec.split_barrier or spec.wave_epilogue) and (p.stride > 1 or G > 1):
        return "split_barrier / wave_epilogue serve stride-1 unpacked tiles only"
    if spec.wave_epilogue:
        wpx = spec.tiles_p * 16
        if wpx % tw or (spec.tile_h * tw) % wpx:
            return "wave_epilogue needs whole rows of one image per wave"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if spec.waves_c * spec.waves_p < 2:
        return "needs at least two waves (one TDM issuer per operand)"
    if max(spec.x_pitch, spec.w_pitch, spec.halo_w, spec.halo_h) > 0xFFFF:
        return "tile too wide for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    # Accumulators + one A and one B fragment set per wave. Past 256 the
    # kernel addresses the upper VGPR banks (s_set_vgpr_msb).
    vgprs = 8 * spec.phases * spec.tiles_c * spec.tiles_p + 8 * (
        spec.tiles_c + spec.tiles_p
    ) * (2 if spec.prefetch else 1)
    if vgprs > DGRAD_WMMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    # Global offsets are i32 bytes and the tile index is a flat block id.
    tensor_bytes = (
        2 * p.N * p.Ho * p.Wo * p.total_k,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.H * p.W * p.total_c,
    )
    if max(tensor_bytes) > _I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
    if spec.grid()[0] > _I32_MAX:
        return f"{spec.grid()[0]} workgroups exceed the i32 block id"
    return None


def is_valid_dgrad_wmma_spec(
    spec: DirectDgradWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = dgrad_wmma_reason(spec, arch)
    return (why is None, why or "ok")


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
    ``tile2`` may be a runtime value. Same encoding as
    :meth:`IRBuilder.tdm_descriptor_2d`, plus the dimension-2 fields.
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


def build_direct_dgrad_wmma(
    spec: DirectDgradWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    why = dgrad_wmma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DirectDgradWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW = p.KH, p.KW
    S = p.stride
    G = spec.pack_groups
    NPH = spec.phases
    taps_h, _, PADD_H = spec.plan_h  # PADD: padding of the equivalent forward pass
    taps_w, _, PADD_W = spec.plan_w
    io_type = _io_type(p.dtype)
    ab = "bf16" if p.dtype == "bf16" else "f16"
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=WMMA_K
    )

    WAVE = spec.wave_size
    WC, WP = spec.waves_c, spec.waves_p
    TM, TN = spec.tiles_c, spec.tiles_p
    THREADS = spec.threads_per_block
    BC = spec.block_c
    TH, TW = spec.tile_h, spec.tile_w
    HALO_H, HALO_W = spec.halo_h, spec.halo_w
    CK = spec.ck
    NS = spec.stages
    XP, WPITCH = spec.x_pitch, spec.w_pitch
    HALO, WEI = spec.halo_elems, spec.wei_elems
    NB, IMG = spec.tile_n, spec.image_elems
    HALO_STAGE = HALO + spec.halo_slack
    TAPS = KH * KW

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="dgrad", io_type=io_type)
    A, Bp, D = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    if G > 1:
        # Super-groups of G groups; the packed channel counts are baked.
        c_groups = b.div(params["p_groups"], b.const_i32(G))
        c_cpg = b.const_i32(spec.cpg_eff)
        c_kpg = b.const_i32(spec.kpg_eff)
    else:
        c_groups = params["p_groups"]
        c_cpg = b.div(params["p_total_c"], c_groups)
        c_kpg = b.div(params["p_total_k"], c_groups)
    c_N = params["p_N"]
    c_H, c_W = params["p_Hi"], params["p_Wi"]
    c_Ho, c_Wo = params["p_Ho"], params["p_Wo"]

    tid = b.thread_id_x()
    c_wave = b.const_i32(WAVE)
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    c16 = b.const_i32(16)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_c = b.mod(wave_id, b.const_i32(WC))
    wave_p = b.div(wave_id, b.const_i32(WC))

    # ---- tile of this block: bx = (((nblk * G + g) * n_ht + ht) * n_wt + wt) * n_ct + ct.
    n_ct = b.div(b.add(c_cpg, b.const_i32(BC - 1)), b.const_i32(BC))
    if S > 1:
        # Tiles index the phase grid: output row S * i + a, i < ceil(H / S).
        c_Hq = b.div(b.add(c_H, b.const_i32(S - 1)), b.const_i32(S))
        c_Wq = b.div(b.add(c_W, b.const_i32(S - 1)), b.const_i32(S))
    else:
        c_Hq, c_Wq = c_H, c_W
    n_ht = b.div(b.add(c_Hq, b.const_i32(TH - 1)), b.const_i32(TH))
    n_wt = b.div(b.add(c_Wq, b.const_i32(TW - 1)), b.const_i32(TW))
    t = b.block_id_x()
    ct = b.mod(t, n_ct)
    t = b.div(t, n_ct)
    wt = b.mod(t, n_wt)
    t = b.div(t, n_wt)
    ht = b.mod(t, n_ht)
    t = b.div(t, n_ht)
    g = b.mod(t, c_groups)
    n_base = b.mul(b.div(t, c_groups), b.const_i32(NB))
    c_base = b.mul(ct, b.const_i32(BC))
    h_tile = b.mul(ht, b.const_i32(TH))
    w_tile = b.mul(wt, b.const_i32(TW))
    k_grp = b.mul(g, c_kpg)
    c_grp = b.mul(g, c_cpg)
    w_row = b.mul(c_cpg, b.const_i32(TAPS))  # element stride between k rows
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_kpg, b.const_i32(CK - 1)), c_CK)

    # One LDS block: the halo ring, then the weight ring; the TDM epilogue
    # reuses it from offset 0 once the last chunk is consumed.
    WEI0 = NS * HALO_STAGE
    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="dg_lds")
    halo_lds = wei_lds = lds
    halo_addr = b.trunc(b.smem_addr_of(lds), I32)
    wei_addr = b.add(halo_addr, b.const_i32(WEI0 * 2))

    # Halo origin in dY coordinates and the edge shifts (module docstring).
    h0 = b.sub(h_tile, b.const_i32(PADD_H))
    w0 = b.sub(w_tile, b.const_i32(PADD_W))
    hs = b.smax(b.sub(c0, h0), c0)
    sw = b.smax(b.sub(c0, w0), c0)
    h_start = b.add(h0, hs)
    w_start = b.add(w0, sw)
    z_elems = b.mul(b.add(b.mul(hs, b.const_i32(HALO_W)), sw), b.const_i32(XP))
    dy_w_ext = b.smax(b.smin(b.sub(c_Wo, w_start), b.sub(b.const_i32(HALO_W), sw)), c0)
    dy_h_ext = b.smax(b.sub(c_Ho, h_start), c0)
    dy_tile2 = b.sub(b.const_i32(HALO_H), hs)
    dy_base = b.add(
        b.add(
            b.mul(n_base, params["p_dY_stride_n"]),
            b.mul(h_start, params["p_dY_stride_ho"]),
        ),
        b.add(b.mul(w_start, params["p_dY_stride_wo"]), k_grp),
    )
    c_ext = b.smax(b.smin(b.sub(c_cpg, c_base), b.const_i32(BC)), c0)
    w_base = b.add(b.mul(k_grp, w_row), c_base)

    # ---- zero the halo cells no transfer writes (top / left image edge).
    max_z = (PADD_H * HALO_W + PADD_W) * XP
    if max_z:
        zero_v = b.zero_vec(io_type, 8)
        z_vecs = b.div(z_elems, b.const_i32(8))
        for j in range(0, _ceil_div(max_z, 8), THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, z_vecs)):
                flat = b.mul(v, b.const_i32(8))
                for st in range(NS):
                    for im in range(NB):
                        off = st * HALO_STAGE + im * IMG
                        b.smem_store_vN(
                            halo_lds, [c0, b.add(flat, b.const_i32(off))], zero_v, n=8
                        )

    def _issue(chunk: Value) -> None:
        """TDM chunk ``chunk`` into ring slot chunk % NS."""
        stage = b.mod(chunk, b.const_i32(NS))
        k0 = b.mul(chunk, c_CK)
        k_ext = b.smax(b.smin(b.sub(c_kpg, k0), c_CK), c0)
        stage_off = b.mul(stage, b.const_i32(HALO_STAGE))
        for im in range(NB):
            # Wave im moves image im's halo; an image past N transfers zeros.
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(im))):
                img_off = b.add(stage_off, b.const_i32(im * IMG))
                img = b.add(
                    b.add(dy_base, k0), b.mul(b.const_i32(im), params["p_dY_stride_n"])
                )
                src = b.global_ptr_add(A, b.mul(img, c2))
                dst = b.add(halo_addr, b.mul(b.add(img_off, z_elems), c2))
                h_ext = dy_h_ext
                if NB > 1:
                    n_ok = b.cmp_lt(b.add(n_base, b.const_i32(im)), c_N)
                    h_ext = b.select(n_ok, dy_h_ext, c0)
                    src = b.select(n_ok, src, b.global_ptr_add(A, c0))
                b.tensor_load_to_lds(
                    *_tdm_descriptor_3d(
                        b,
                        src,
                        dst,
                        dims=(k_ext, dy_w_ext, h_ext),
                        strides=(params["p_dY_stride_wo"], params["p_dY_stride_ho"]),
                        tile0=XP,
                        tile1=HALO_W,
                        tile2=dy_tile2,
                    )
                )
        if G > 1:
            return  # the packed weights are staged once, before the loop
        with b.scf_if(b.cmp_eq(wave_id, b.const_i32(NB))):
            src = b.global_ptr_add(Bp, b.mul(b.add(w_base, b.mul(k0, w_row)), c2))
            dst = b.add(wei_addr, b.mul(stage, b.const_i32(WEI * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(c_ext, k_ext, b.const_i32(TAPS)),
                    strides=(w_row, c_cpg),
                    tile0=WPITCH,
                    tile1=CK,
                    tile2=b.const_i32(TAPS),
                )
            )

    # ---- per-lane LDS read offsets.
    # B (dY): lane l reads pixel l % 16 of its tile, k (l // 16) * 16 .. +15.
    px_off: list[Value] = []
    for tn in range(TN):
        flat = b.add(
            b.mul(b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16), l16
        )
        if NB > 1:
            nb = b.div(flat, b.const_i32(TH * TW))
            flat = b.mod(flat, b.const_i32(TH * TW))
        else:
            nb = c0
        th = b.div(flat, b.const_i32(TW))
        tw = b.mod(flat, b.const_i32(TW))
        off = b.add(
            b.mul(b.add(b.mul(th, b.const_i32(HALO_W)), tw), b.const_i32(XP)),
            b.mul(lhalf, c16),
        )
        if NB > 1:
            off = b.add(off, b.mul(nb, b.const_i32(IMG)))
        px_off.append(off)
    # A (W^T): transpose read of a 32 (k) x 16 (c) block of [k][c].
    c8 = b.const_i32(8)
    tr_col = b.mul(b.div(l16, c8), c8)
    tr_row = b.add(b.mul(lhalf, c16), b.mod(lane, c8))
    wei_off: list[Value] = []
    for m in range(TM):
        col = b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16)
        wei_off.append(
            b.add(
                b.mul(tr_row, b.const_i32(WPITCH)),
                b.add(col, b.add(tr_col, b.const_i32(WEI0))),
            )
        )

    def _read_b(stage_base: Value, tn: int, tap_off: int) -> Value:
        addr = b.add(stage_base, b.add(px_off[tn], b.const_i32(tap_off)))
        lo = b.smem_load_vN(halo_lds, c0, addr, dtype=io_type, n=8)
        hi = b.smem_load_vN(halo_lds, c0, b.add(addr, c8), dtype=io_type, n=8)
        return b.vec_concat(lo, hi)

    def _read_a(stage_base: Value, m: int, row_off: int) -> Value:
        addr = b.add(stage_base, b.add(wei_off[m], b.const_i32(row_off)))
        lo = b.ds_read_tr16_b128(wei_lds, c0, addr, dtype=io_type)
        hi = b.ds_read_tr16_b128(
            wei_lds, c0, b.add(addr, b.const_i32(8 * WPITCH)), dtype=io_type
        )
        return b.vec_concat(lo, hi)

    # One step per (k-step, halo shift): the taps sharing a shift (strided
    # phases) share its B fragments. Stride 1 has one tap per shift.
    steps = []
    for ks in range(CK // WMMA_K):
        groups_by_shift: dict = {}
        for r in range(KH):
            for s in range(KW):
                (a_h, dh), (a_w, dw) = taps_h[r], taps_w[s]
                groups_by_shift.setdefault((dh, dw), []).append(
                    (r * KW + s, a_h * S + a_w)
                )
        steps.extend((ks, sh, tl) for sh, tl in groups_by_shift.items())

    SB = spec.split_barrier

    def _wait_landed(kc: Value) -> None:
        """Wait until this wave's transfer of chunk ``kc`` has landed."""
        if NS == 2:
            b.s_wait_tensorcnt(0)
        else:
            # This chunk's transfer is the oldest in flight; younger ones
            # exist only while chunks remain to be issued.
            more = b.cmp_lt(b.add(kc, b.const_i32(NS - 2)), n_chunks)
            with b.scf_if_else(more) as (then_ctx, else_ctx):
                with then_ctx:
                    b.s_wait_tensorcnt(NS - 2)
                with else_ctx:
                    b.s_wait_tensorcnt(0)

    def _signal_next(kc: Value) -> None:
        """Split barrier, first half: this wave's reads of chunk ``kc`` are
        done and its transfer of chunk ``kc + 1`` has landed."""
        b.s_waitcnt(lgkmcnt=0)
        _wait_landed(b.add(kc, c1))
        b.s_barrier_signal(0xFFFFFFFF)

    def _chunk(kc: Value, accs: list[Value]) -> list[Value]:
        """Publish chunk ``kc``, refill the ring, multiply the chunk."""
        if SB:
            b.s_barrier_wait(0xFFFF)
        else:
            _wait_landed(kc)
            b.sync_lds_only()
        nxt = b.add(kc, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_chunks)):
            _issue(nxt)
        stage = b.mod(kc, b.const_i32(NS))
        halo_base = b.mul(stage, b.const_i32(HALO_STAGE))
        wei_base = b.mul(stage, b.const_i32(WEI))

        def _frags(st):
            ks, (dh, dw), tl = st
            tap_off = (dh * HALO_W + dw) * XP + ks * WMMA_K
            bf = [_read_b(halo_base, tn, tap_off) for tn in range(TN)]
            af = [
                [
                    _read_a(wei_base, m, (tap * CK + ks * WMMA_K) * WPITCH)
                    for m in range(TM)
                ]
                for tap, _ in tl
            ]
            return af, bf

        accs = list(accs)
        b.s_setprio(1)
        nxt_frags = _frags(steps[0]) if spec.prefetch else None
        for j, st in enumerate(steps):
            if spec.prefetch:
                af, bf = nxt_frags
                if j + 1 < len(steps):
                    nxt_frags = _frags(steps[j + 1])
            else:
                af, bf = _frags(st)
            if SB and j == len(steps) - 1:
                _signal_next(kc)
            for (_, ph), aft in zip(st[2], af):
                for m in range(TM):
                    for tn in range(TN):
                        i = (ph * TM + m) * TN + tn
                        accs[i] = b.mma(wmma_op, aft[m], bf[tn], accs[i])
        b.s_setprio(0)
        return accs

    def _epilogue(accs: list[Value]) -> None:
        """Stage the dX tile in LDS as [image][row][col][channel] and store it
        with one TDM transfer per image; the transfer clips the tile to the
        image and to the group's channels (the pad columns lie past them)."""
        b.sync_lds_only()
        for ph in range(NPH):
            a_h, a_w = divmod(ph, S)
            for m in range(TM):
                col = b.add(
                    b.mul(b.add(b.mul(wave_c, b.const_i32(TM)), b.const_i32(m)), c16),
                    b.mul(lhalf, c8),
                )
                for tn in range(TN):
                    px = b.add(
                        b.mul(
                            b.add(b.mul(wave_p, b.const_i32(TN)), b.const_i32(tn)), c16
                        ),
                        l16,
                    )
                    if S > 1:
                        # Phase-grid pixel (ti, tj) -> dX tile pixel
                        # (S * ti + a_h, S * tj + a_w).
                        f = px
                        if NB > 1:
                            nb = b.div(f, b.const_i32(TH * TW))
                            f = b.mod(f, b.const_i32(TH * TW))
                        ti = b.div(f, b.const_i32(TW))
                        tj = b.mod(f, b.const_i32(TW))
                        px = b.add(
                            b.mul(
                                b.add(b.mul(ti, b.const_i32(S)), b.const_i32(a_h)),
                                b.const_i32(S * TW),
                            ),
                            b.add(b.mul(tj, b.const_i32(S)), b.const_i32(a_w)),
                        )
                        if NB > 1:
                            px = b.add(px, b.mul(nb, b.const_i32(S * S * TH * TW)))
                    val = b.vec_cast_f32_to(accs[(ph * TM + m) * TN + tn], io_type)
                    b.smem_store_vN(
                        lds, [c0, b.add(b.mul(px, b.const_i32(WPITCH)), col)], val, n=8
                    )
        b.sync_lds_only()
        hx_tile = h_tile if S == 1 else b.mul(h_tile, b.const_i32(S))
        wx_tile = w_tile if S == 1 else b.mul(w_tile, b.const_i32(S))
        for im in range(NB):
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(im))):
                n_img = b.add(n_base, b.const_i32(im))
                off = b.add(
                    b.add(
                        b.mul(n_img, params["p_dX_stride_n"]),
                        b.mul(hx_tile, params["p_dX_stride_hi"]),
                    ),
                    b.add(
                        b.mul(wx_tile, params["p_dX_stride_wi"]), b.add(c_grp, c_base)
                    ),
                )
                h_ext = b.smax(b.sub(c_H, hx_tile), c0)
                dst = b.global_ptr_add(D, b.mul(off, c2))
                if NB > 1:
                    n_ok = b.cmp_lt(n_img, c_N)
                    h_ext = b.select(n_ok, h_ext, c0)
                    dst = b.select(n_ok, dst, b.global_ptr_add(D, c0))
                b.tensor_store_from_lds(
                    *_tdm_descriptor_3d(
                        b,
                        dst,
                        b.add(
                            halo_addr,
                            b.const_i32(im * NPH * TH * TW * WPITCH * 2),
                        ),
                        dims=(c_ext, b.smax(b.sub(c_W, wx_tile), c0), h_ext),
                        strides=(params["p_dX_stride_wi"], params["p_dX_stride_hi"]),
                        tile0=WPITCH,
                        tile1=S * TW,
                        tile2=b.const_i32(S * TH),
                    )
                )
                b.s_wait_tensorcnt(0)

    def _epilogue_wave(accs: list[Value]) -> None:
        """Per-wave staging and TDM store (see ``wave_epilogue``)."""
        if not SB:
            b.sync_lds_only()  # every wave is done with the ring
        WPX = TN * 16
        EP = TM * 16 + CH_PAD
        ROWS = WPX // TW
        base = b.mul(wave_id, b.const_i32(WPX * EP))
        for m in range(TM):
            col = b.add(b.const_i32(m * 16), b.mul(lhalf, c8))
            for tn in range(TN):
                px = b.add(b.const_i32(tn * 16), l16)
                val = b.vec_cast_f32_to(accs[m * TN + tn], io_type)
                b.smem_store_vN(
                    lds,
                    [c0, b.add(base, b.add(b.mul(px, b.const_i32(EP)), col))],
                    val,
                    n=8,
                )
        b.s_waitcnt(lgkmcnt=0)
        px0 = b.mul(wave_p, b.const_i32(WPX))
        img = b.div(px0, b.const_i32(TH * TW))
        row0 = b.div(b.mod(px0, b.const_i32(TH * TW)), b.const_i32(TW))
        n_img = b.add(n_base, img)
        h_w = b.add(h_tile, row0)
        c_w = b.add(c_base, b.mul(wave_c, b.const_i32(TM * 16)))
        off = b.add(
            b.add(
                b.mul(n_img, params["p_dX_stride_n"]),
                b.mul(h_w, params["p_dX_stride_hi"]),
            ),
            b.add(b.mul(w_tile, params["p_dX_stride_wi"]), b.add(c_grp, c_w)),
        )
        c_ext_w = b.smax(b.smin(b.sub(c_cpg, c_w), b.const_i32(TM * 16)), c0)
        h_ext = b.smax(b.sub(c_H, h_w), c0)
        dst = b.global_ptr_add(D, b.mul(off, c2))
        if NB > 1:
            n_ok = b.cmp_lt(n_img, c_N)
            h_ext = b.select(n_ok, h_ext, c0)
            dst = b.select(n_ok, dst, b.global_ptr_add(D, c0))
        b.tensor_store_from_lds(
            *_tdm_descriptor_3d(
                b,
                dst,
                b.add(halo_addr, b.mul(base, c2)),
                dims=(c_ext_w, b.smax(b.sub(c_W, w_tile), c0), h_ext),
                strides=(params["p_dX_stride_wi"], params["p_dX_stride_hi"]),
                tile0=EP,
                tile1=TW,
                tile2=b.const_i32(ROWS),
            )
        )
        b.s_wait_tensorcnt(0)

    if G > 1:
        # Stage the super-group's block-diagonal weights into slot 0 once:
        # zero the slot, then each lane copies one CPG-wide row of one group.
        CPG0, KPG0 = p.cpg, p.kpg
        zero_v = b.zero_vec(io_type, 8)
        nvec = WEI // 8
        for j in range(0, nvec, THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, b.const_i32(nvec))):
                b.smem_store_vN(
                    wei_lds,
                    [c0, b.add(b.mul(v, b.const_i32(8)), b.const_i32(WEI0))],
                    zero_v,
                    n=8,
                )
        b.sync_lds_only()
        VEC = min(CPG0, 8)
        PARTS = CPG0 // VEC
        KQ = G * KPG0
        total = TAPS * KQ * PARTS
        for j in range(0, total, THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, b.const_i32(total))):
                part = b.mod(v, b.const_i32(PARTS))
                rw = b.div(v, b.const_i32(PARTS))
                tap = b.div(rw, b.const_i32(KQ))
                kq = b.mod(rw, b.const_i32(KQ))
                gi = b.div(kq, b.const_i32(KPG0))
                pv = b.mul(part, b.const_i32(VEC))
                gidx = b.add(
                    b.mul(b.add(k_grp, kq), b.const_i32(TAPS * CPG0)),
                    b.add(b.mul(tap, b.const_i32(CPG0)), pv),
                )
                val = b.global_load_vN(Bp, gidx, io_type, VEC)
                loff = b.add(
                    b.mul(b.add(b.mul(tap, b.const_i32(CK)), kq), b.const_i32(WPITCH)),
                    b.add(b.mul(gi, b.const_i32(CPG0)), pv),
                )
                b.smem_store_vN(
                    wei_lds, [c0, b.add(loff, b.const_i32(WEI0))], val, n=VEC
                )

    # ---- main loop over dY channel chunks.
    for st in range(NS - 1):
        with b.scf_if(b.cmp_lt(b.const_i32(st), n_chunks)):
            _issue(b.const_i32(st))
    if SB:
        b.s_waitcnt(lgkmcnt=0)  # the edge zeroing
        _wait_landed(c0)
        b.s_barrier_signal(0xFFFFFFFF)
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0,
        n_chunks,
        c1,
        [(f"acc{i}", zero_acc) for i in range(NPH * TM * TN)],
        iv_name="kc",
    )
    with loop as (kc, carried):
        b.scf_yield(*_chunk(kc, list(carried)))
    if SB:
        b.s_barrier_wait(0xFFFF)
    if spec.wave_epilogue:
        _epilogue_wave(list(loop.results))
    else:
        _epilogue(list(loop.results))
    return b.kernel
