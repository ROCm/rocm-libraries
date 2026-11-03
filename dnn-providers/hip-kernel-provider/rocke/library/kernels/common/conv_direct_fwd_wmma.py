# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct forward convolution on wave32 WMMA (gfx1250).

Why a separate family
---------------------
The direct forward kernels of the other targets (``conv_direct_grouped``,
``conv_direct_nongrouped``) are wave64 MFMA kernels. On wave32 WMMA targets
the forward pass was left to implicit GEMM, which pays an im2col address
computation per K step and has no halo reuse. This kernel computes

    Y[n, p, q, k] = sum_{r, s, c} X[n, S*p + r*dh - PAD, S*q + s*dw - PAD, c] * W[k, r, s, c]

directly, for dense and grouped layers alike. Per workgroup, with M = output
channels and N = output pixels::

    block_k                     output channels   (waves_k * tiles_k * 16)
    tile_h x tile_w             output pixels     (waves_p * tiles_p * 16)
    ck                          input channels reduced per LDS stage

    for c0 in range(0, cpg, ck):                  # runtime scf.for
        TDM X[S*h0 - PAD : ..., S*w0 - PAD : ..., c0 : c0 + ck]
            -> LDS halo [row][col][c]              (one 3-D transfer, zero-filled)
        TDM W[k0 : k0 + block_k, all taps, c0 : c0 + ck]
            -> LDS [tap][k][c]                     (one 3-D transfer)
        for k-step, tap (r, s):                    # Python-unrolled
            A = W fragment    (row k, 16 contiguous c: two plain 16-byte reads)
            B = X fragment    (halo at (S*p + r*dh, S*q + s*dw): no im2col)
            acc[k-tile, px-tile] = wmma_16x16x32(A, B, acc)

The halo holds the input rows and columns the tile reads: ``S * (tile - 1) +
(K - 1) * dil + 1`` of each, densely, so stride and dilation only change the
B read offsets of each tap. A transfer cannot start at a negative coordinate,
so a block on the top or left image edge starts its read at row / column 0
and lands it shifted down / right in the slot; the rows and columns above /
left of it are zeroed once, before the loop, and never written again. Shifting
right wraps the last columns of each row into the next row's leading (zero)
columns: the transfer's column extent is clipped so that it writes zeros
there, which is also why each slot has a little slack at its end.

The slots form a ring of ``stages``: slot ``i % stages`` holds chunk ``i``; one
barrier per chunk publishes it, after which the slot the previous chunk read
is refilled ``stages - 1`` chunks ahead. Wave ``i < tile_n`` issues image
``i``'s X transfer, wave ``tile_n`` the weights.

With ``split_barrier`` the per-chunk barrier is split: a wave signals once it
has read the chunk's last fragments (and, as the issuer, once its next
transfer has landed), and waits at the top of the next chunk, so its last
tap's WMMAs run while the slower waves catch up.

Every lane of the accumulator holds one pixel and 8 consecutive output
channels. The epilogue stages the Y tile in LDS as [image][row][col][channel]
and stores it with one TDM transfer per image (or one per wave with
``wave_epilogue``), which clips it to the image and to the group's channels.

Register blocking (``tiles_k x tiles_p`` 16x16 tiles per wave) sets the LDS
reads per WMMA, the lever of this loop: a tap reads ``tiles_k + tiles_p``
fragments for ``tiles_k * tiles_p`` WMMAs. Past 256 VGPRs the backend
addresses the upper register banks.

``pack_groups = G`` treats ``G`` consecutive groups as one super-group of
``G * cpg`` input and ``G * kpg`` output channels (contiguous in NHWC). Its
weights are staged once, before the loop, as a block-diagonal matrix (zeros
off the diagonal): the layers this serves are memory-bound, so the zero
products cost less than the per-group blocks they replace.

AOT
---
Batch, spatial extents, the group count and both channel counts are kernargs
(the direct-conv forward block of :func:`kernels.common.conv_abi.conv_direct_arg_names`).
The filter, stride, dilation and padding shape the halo and the unrolled tap
loop, so they are build-time capabilities, like the tile geometry.
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
from rocke.core.ir import I32, IRBuilder, KernelDef, Value, VectorType

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    emit_direct_params,
)
from kernels.common.conv_fwd_wmma_gate import (
    WMMA_K,
    WMMA_M,
    WMMA_N,
    wmma_ab_dtype,
    wmma_fwd_gate,
)

CH_PAD = 8  # channels of row skew past each LDS row (16 bytes)
FWD_WMMA_VGPR_BUDGET = 512
_I32_MAX = (1 << 31) - 1


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class DirectFwdWmmaSpec:
    """One configuration of :func:`build_direct_fwd_wmma`.

    ``problem`` supplies the build-time capabilities (filter, padding, stride,
    dilation, dtype) and the shape used for launch geometry and validation;
    the channel counts, group count and extents are read from the kernargs.
    """

    problem: DirectConvProblem
    name: str = "direct_fwd_wmma"
    waves_k: int = 2  # waves along output channels
    waves_p: int = 4  # waves along output pixels
    tiles_k: int = 4  # 16-channel tiles per wave
    tiles_p: int = 4  # 16-pixel tiles per wave
    tile_w: int = 16  # output columns per block; tile_h follows from the pixels
    tile_n: int = 1  # images per block (small maps)
    ck: int = 32  # input channels per LDS stage (multiple of 32)
    stages: int = 2  # LDS ring depth
    # Read the next tap's fragments before issuing this tap's WMMAs, so the
    # LDS latency hides behind them (costs a second fragment set of VGPRs).
    prefetch: bool = False
    # AMDGPU machine-scheduler strategy (rocke.core.codegen_policy), or None.
    sched: str | None = None
    wave_size: int = 32
    # Groups packed into one block-diagonal super-group (small cpg): the
    # block multiplies pack_groups * kpg output channels against
    # pack_groups * cpg input channels, the off-diagonal weights zero. Bakes
    # cpg and kpg.
    pack_groups: int = 1
    # Split the per-chunk workgroup barrier (s_barrier_signal / _wait): a wave
    # signals once it has read the chunk's last fragments, and waits only at
    # the top of the next chunk, so the last tap's WMMAs cover the skew.
    split_barrier: bool = False
    # Epilogue per wave: each wave stages its own tiles in a private LDS
    # region and stores them with its own TDM transfer, with no block barrier
    # between the staging and the store (needs whole rows of one image per
    # wave: tiles_p * 16 a multiple of tile_w that divides tile_h * tile_w).
    wave_epilogue: bool = False
    # Stride 2: keep the halo's even and odd input columns in two separate
    # sub-halos (one TDM transfer each, element stride 2), so a fragment's 16
    # pixels read consecutive LDS rows instead of every other one (a 2-way
    # bank conflict on every B read).
    phase_split: bool = False
    # Stage the weights one filter row (KW taps) at a time: the weight ring
    # advances once per (chunk, filter row) while the halo of a chunk stays,
    # in two slots, for all KH rows. A third of the weight LDS per stage, so
    # wider tiles (more register blocking) and deeper chunks fit.
    row_ring: bool = False
    # As row_ring, one filter tap per weight slot (KH * KW steps per chunk):
    # the smallest weight slot, for the widest tiles and deepest chunks.
    tap_ring: bool = False
    # Ping-pong schedule: the waves form two ranks that run one barrier
    # apart, and every step is "read the fragments | barrier | WMMAs |
    # barrier", so one rank reads LDS while the other multiplies. Replaces
    # the per-chunk barrier and the in-wave fragment prefetch.
    pingpong: bool = False
    # Steps (k-step, tap) per ping-pong phase: one barrier pair per group.
    pp_group: int = 1
    # Reuse B (X) fragments across filter rows: with whole output rows per
    # wave, the fragment of row ``t`` at filter row ``r + 1`` is the one of
    # row ``t + 1`` at filter row ``r``. The taps of one filter column then
    # read ``tiles_p + (KH - 1) * dil * tile_w / 16`` B fragments instead of
    # ``KH * tiles_p`` (register reuse, no extra instructions).
    tap_reuse: bool = False
    # With tap_reuse: build the fragments of filter column ``s + 1`` from those
    # of column ``s`` with an in-wave lane rotation (v_permlane16, lane l <-
    # lane l + dil of the same 16-lane row) plus a 2-lane LDS read for the
    # column that enters at the right edge, instead of a full LDS read.
    tap_shift: bool = False

    @property
    def ring_taps(self) -> int:
        """Taps per weight slot."""
        p = self.problem
        if self.tap_ring:
            return 1
        return p.KW if self.row_ring else p.KH * p.KW

    @property
    def phases_w(self) -> int:
        return self.problem.stride if self.phase_split else 1

    @property
    def halo_cols(self) -> int:
        """Columns of one (phase) sub-halo."""
        p = self.problem
        if self.phase_split:
            return self.tile_w + ((p.KW - 1) * p.dil_w) // p.stride
        return self.halo_w

    @property
    def lead_cols(self) -> int:
        """Most leading zero columns a (phase) sub-halo gets at the left edge."""
        p = self.problem
        return _ceil_div(p.PAD, p.stride) if self.phase_split else p.PAD

    @property
    def phase_elems(self) -> int:
        """One (phase) sub-halo and its wrap slack."""
        return (
            self.halo_h * self.halo_cols * self.x_pitch + self.lead_cols * self.x_pitch
        )

    @property
    def stride(self) -> int:
        return self.problem.stride

    @property
    def cpg_eff(self) -> int:
        return self.pack_groups * self.problem.cpg

    @property
    def kpg_eff(self) -> int:
        return self.pack_groups * self.problem.kpg

    @property
    def groups_eff(self) -> int:
        return self.problem.groups // self.pack_groups

    @property
    def block_k(self) -> int:
        return self.waves_k * self.tiles_k * 16

    @property
    def block_p(self) -> int:
        return self.waves_p * self.tiles_p * 16

    @property
    def tile_h(self) -> int:
        d = self.tile_w * self.tile_n
        return self.block_p // d if d > 0 else 0  # 0: rejected by the validator

    @property
    def threads_per_block(self) -> int:
        return self.waves_k * self.waves_p * self.wave_size

    @property
    def halo_h(self) -> int:
        p = self.problem
        return p.stride * (self.tile_h - 1) + (p.KH - 1) * p.dil_h + 1

    @property
    def halo_w(self) -> int:
        p = self.problem
        return p.stride * (self.tile_w - 1) + (p.KW - 1) * p.dil_w + 1

    @property
    def x_pitch(self) -> int:
        return self.ck + CH_PAD

    @property
    def epi_pitch(self) -> int:
        return self.block_k + CH_PAD

    @property
    def image_elems(self) -> int:
        if self.phase_split:
            return self.phases_w * self.phase_elems
        return self.halo_h * self.halo_w * self.x_pitch

    @property
    def halo_elems(self) -> int:
        return self.tile_n * self.image_elems

    @property
    def wei_elems(self) -> int:
        p = self.problem
        return self.ring_taps * self.block_k * self.x_pitch

    @property
    def halo_slack(self) -> int:
        # A left-edge block's last row wraps up to PAD columns past the ring
        # (a split halo carries the slack in each phase).
        return 0 if self.phase_split else self.problem.PAD * self.x_pitch

    def lds_bytes(self) -> int:
        # Each halo slot carries its own wrap slack (see the module docstring).
        if self.row_ring or self.tap_ring:
            ring = (
                2 * (self.halo_elems + self.halo_slack) + self.stages * self.wei_elems
            )
        else:
            ring = self.stages * (self.halo_elems + self.halo_slack + self.wei_elems)
        epi = self.block_p * self.epi_pitch  # the staged Y tile
        if self.wave_epilogue:
            n_waves = self.waves_k * self.waves_p
            epi = n_waves * self.tiles_p * 16 * (self.tiles_k * 16 + CH_PAD)
        return 2 * max(ring, epi)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}",
            f"p{p.PAD}",
            f"wk{self.waves_k}",
            f"wp{self.waves_p}",
            f"tk{self.tiles_k}",
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
            "ps" if self.phase_split else "",
            "rr" if self.row_ring else "",
            "tr" if self.tap_ring else "",
            "pp" if self.pingpong else "",
            f"pg{self.pp_group}" if self.pp_group > 1 else "",
            "tu" if self.tap_reuse else "",
            "tsh" if self.tap_shift else "",
            flags={"bf16": p.dtype == "bf16"},
        )

    def grid(self) -> tuple[int, int, int]:
        p = self.problem
        n_kt = _ceil_div(self.kpg_eff, self.block_k)
        n_ht = _ceil_div(p.Ho, self.tile_h)
        n_wt = _ceil_div(p.Wo, self.tile_w)
        return (
            _ceil_div(p.N, self.tile_n) * self.groups_eff * n_ht * n_wt * n_kt,
            1,
            1,
        )

    def launch_geometry(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)


def fwd_wmma_tile_efficiency(spec: DirectFwdWmmaSpec) -> float:
    """Share of ``spec``'s padded tile grid that is real work on its problem."""
    p = spec.problem
    tiles = (
        _ceil_div(p.N, spec.tile_n) * spec.tile_n,
        _ceil_div(p.Ho, spec.tile_h) * spec.tile_h,
        _ceil_div(p.Wo, spec.tile_w) * spec.tile_w,
        _ceil_div(spec.kpg_eff, spec.block_k) * spec.block_k,
    )
    return (p.N * p.Ho * p.Wo * spec.kpg_eff) / (
        tiles[0] * tiles[1] * tiles[2] * tiles[3]
    )


def fwd_wmma_reason(spec: DirectFwdWmmaSpec, arch: str = "gfx1250") -> str | None:
    """Why ``spec`` cannot run on ``arch`` (None when it can)."""
    p = spec.problem
    why, target = wmma_fwd_gate(p, arch, "direct_fwd_wmma", spec.wave_size)
    if why:
        return why
    if p.stride not in (1, 2):
        return f"stride {p.stride} not in (1, 2)"
    if p.dil_h < 1 or p.dil_w < 1:
        return f"dilation {p.dil_h}x{p.dil_w} < 1"
    if p.KH != p.KW:
        return f"filter {p.KH}x{p.KW} is not square"
    if p.PAD < 0 or p.PAD > (p.KH - 1) * max(p.dil_h, p.dil_w):
        return f"PAD {p.PAD} outside 0..(K-1)*dilation"
    if p.Ho < 1 or p.Wo < 1:
        return f"empty output {p.Ho}x{p.Wo}"
    G = spec.pack_groups
    if G < 1:
        return f"pack_groups {G} < 1"
    if G > 1:
        if p.groups % G:
            return f"groups {p.groups} not a multiple of pack_groups {G}"
        if p.cpg not in (2, 4, 8, 16) or p.kpg not in (2, 4, 8, 16):
            return f"packed cpg {p.cpg} / kpg {p.kpg} not in (2, 4, 8, 16)"
        if spec.cpg_eff > spec.ck:
            return f"packed cpg {spec.cpg_eff} > ck {spec.ck} (one X chunk)"
        if spec.kpg_eff > spec.block_k:
            return f"packed kpg {spec.kpg_eff} > block_k {spec.block_k}"
    if spec.cpg_eff % 8 or spec.kpg_eff % 8:
        return (
            f"cpg {spec.cpg_eff} and kpg {spec.kpg_eff} must be multiples of 8"
            " (16-byte rows)"
        )
    if spec.ck % WMMA_K or spec.ck <= 0:
        return f"ck {spec.ck} must be a positive multiple of {WMMA_K}"
    if min(spec.waves_k, spec.waves_p, spec.tiles_k, spec.tiles_p) < 1:
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
    if spec.tile_n > spec.waves_k * spec.waves_p - 1:
        return "tile_n needs one more wave than images (one X transfer per image)"
    if spec.pingpong:
        nw = spec.waves_k * spec.waves_p
        issuers = 2 * spec.tile_n + 1 if spec.phase_split else spec.tile_n + 1
        if spec.row_ring or spec.tap_ring or spec.split_barrier or spec.prefetch:
            return "pingpong replaces the ring variants, split_barrier and prefetch"
        if nw % 2 or issuers > nw // 2:
            return "pingpong needs an even wave count with every issuer in rank 0"
        if spec.wave_epilogue:
            return "pingpong uses the block epilogue"
    if spec.pp_group < 1 or (spec.pp_group > 1 and not spec.pingpong):
        return f"pp_group {spec.pp_group}: >= 1, and > 1 only with pingpong"
    if (spec.row_ring or spec.tap_ring) and (G > 1 or spec.split_barrier):
        return "row_ring / tap_ring serve unpacked tiles without split_barrier"
    if spec.row_ring and spec.tap_ring:
        return "row_ring and tap_ring are exclusive"
    if spec.tap_reuse:
        if spec.row_ring or spec.tap_ring:
            return "tap_reuse needs all filter rows of a column in one weight slot"
        if spec.pingpong:
            return "tap_reuse is not combined with pingpong"
        if p.stride != 1 or spec.tile_n != 1 or spec.phase_split:
            return "tap_reuse serves stride 1, one image per block"
        if tw % 16 or (spec.tiles_p * 16) % tw:
            return "tap_reuse needs whole 16-pixel row pieces and whole rows per wave"
    if spec.tap_shift:
        if not spec.tap_reuse:
            return "tap_shift builds on tap_reuse"
        if p.dil_w > 15:
            return "tap_shift rotates within a 16-lane row"
    if spec.phase_split:
        if p.stride != 2:
            return "phase_split serves stride 2 only"
        if spec.tile_n * 2 + 1 > spec.waves_k * spec.waves_p:
            return "phase_split needs 2 * tile_n + 1 waves (one transfer each)"
    if spec.wave_epilogue:
        wpx = spec.tiles_p * 16
        if wpx % tw or (spec.tile_h * tw) % wpx:
            return "wave_epilogue needs whole rows of one image per wave"
    if spec.threads_per_block > target.limits.max_threads_per_block:
        return f"{spec.threads_per_block} threads per block"
    if spec.waves_k * spec.waves_p < 2:
        return "needs at least two waves (one TDM issuer per operand)"
    if max(spec.x_pitch, spec.epi_pitch, spec.halo_w, spec.halo_h) > 0xFFFF:
        return "tile too wide for a TDM descriptor"
    if not target.fits_lds(spec.lds_bytes()):
        return f"LDS {spec.lds_bytes()} B > {target.lds_capacity_bytes}"
    # Accumulators + one A and one B fragment set per wave. Past 256 the
    # kernel addresses the upper VGPR banks (s_set_vgpr_msb).
    vgprs = 8 * spec.tiles_k * spec.tiles_p + 8 * (spec.tiles_k + spec.tiles_p) * (
        2 if spec.prefetch else spec.pp_group
    )
    if spec.tap_reuse:
        vgprs += 8 * ((p.KH - 1) * p.dil_h * (tw // 16))
    if vgprs > FWD_WMMA_VGPR_BUDGET:
        return f"~{vgprs} VGPRs of accumulators and fragments"
    try:
        normalize_scheduler_strategy(spec.sched)
    except ValueError as e:
        return str(e)
    # Global offsets are i32 bytes and the tile index is a flat block id.
    tensor_bytes = (
        2 * p.N * p.H * p.W * p.total_c,
        2 * p.total_k * p.KH * p.KW * p.cpg,
        2 * p.N * p.Ho * p.Wo * p.total_k,
    )
    if max(tensor_bytes) > _I32_MAX:
        return f"tensors of {max(tensor_bytes)} B exceed the i32 byte offsets"
    if spec.grid()[0] > _I32_MAX:
        return f"{spec.grid()[0]} workgroups exceed the i32 block id"
    return None


def is_valid_fwd_wmma_spec(
    spec: DirectFwdWmmaSpec, arch: str = "gfx1250"
) -> tuple[bool, str]:
    why = fwd_wmma_reason(spec, arch)
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


def build_direct_fwd_wmma(spec: DirectFwdWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    why = fwd_wmma_reason(spec, arch)
    if why is not None:
        raise ValueError(f"invalid DirectFwdWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW = p.KH, p.KW
    S = p.stride
    DH, DW = p.dil_h, p.dil_w
    PAD = p.PAD
    G = spec.pack_groups
    io_type = _io_type(p.dtype)
    ab = wmma_ab_dtype(p.dtype)
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma",
        a_dtype=ab,
        b_dtype=ab,
        c_dtype="fp32",
        m=WMMA_M,
        n=WMMA_N,
        k=WMMA_K,
    )

    WAVE = spec.wave_size
    WK, WP = spec.waves_k, spec.waves_p
    TM, TN = spec.tiles_k, spec.tiles_p
    THREADS = spec.threads_per_block
    BK = spec.block_k
    TH, TW = spec.tile_h, spec.tile_w
    HALO_H, HALO_W = spec.halo_h, spec.halo_w
    CK = spec.ck
    NS = spec.stages
    XP, EPITCH = spec.x_pitch, spec.epi_pitch
    HALO, WEI = spec.halo_elems, spec.wei_elems
    NB, IMG = spec.tile_n, spec.image_elems
    HALO_STAGE = HALO + spec.halo_slack
    TAPS = KH * KW
    PS = spec.phase_split
    NPW = spec.phases_w
    HC = spec.halo_cols  # columns of one (phase) sub-halo
    PSTR = spec.phase_elems if PS else 0  # element stride between phases

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.sched:
        apply_codegen_policy(b.kernel, CodegenPolicy(scheduler_strategy=spec.sched))
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    Xp, Wp, Yp = params["A"], params["B"], params["D"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    if G > 1:
        # Super-groups of G groups; the packed channel counts are baked.
        c_groups = b.div(params["p_groups"], b.const_i32(G))
        c_cpg = b.const_i32(spec.cpg_eff)
        c_kpg = b.const_i32(spec.kpg_eff)
        c_cpg0 = b.const_i32(p.cpg)  # weight row length of one group
    else:
        c_groups = params["p_groups"]
        c_cpg = b.div(params["p_total_c"], c_groups)
        c_kpg = b.div(params["p_total_k"], c_groups)
        c_cpg0 = c_cpg
    c_N = params["p_N"]
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
    c16 = b.const_i32(16)
    c8 = b.const_i32(8)
    l16 = b.mod(lane, c16)
    lhalf = b.div(lane, c16)
    wave_k = b.mod(wave_id, b.const_i32(WK))
    wave_p = b.div(wave_id, b.const_i32(WK))

    # ---- tile of this block: bx = (((nblk * G + g) * n_ht + ht) * n_wt + wt) * n_kt + kt.
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
    n_base = b.mul(b.div(t, c_groups), b.const_i32(NB))
    k_base = b.mul(kt, b.const_i32(BK))
    h_tile = b.mul(ht, b.const_i32(TH))  # output rows
    w_tile = b.mul(wt, b.const_i32(TW))
    k_grp = b.mul(g, c_kpg)  # first output channel of the (super-)group
    c_grp = b.mul(g, c_cpg)  # first input channel of the (super-)group
    w_row = b.mul(c_cpg0, b.const_i32(TAPS))  # element stride between k rows
    c_CK = b.const_i32(CK)
    n_chunks = b.div(b.add(c_cpg, b.const_i32(CK - 1)), c_CK)

    # One LDS block: the halo ring, then the weight ring; the TDM epilogue
    # reuses it from offset 0 once the last chunk is consumed.
    RR = spec.row_ring or spec.tap_ring
    NHS = 2 if RR else NS  # halo slots
    WTAPS = spec.ring_taps  # taps per weight slot
    NG = TAPS // WTAPS  # weight slots (steps) per chunk
    WEI0 = NHS * HALO_STAGE
    lds = b.smem_alloc(io_type, [1, spec.lds_bytes() // 2], name_hint="fw_lds")
    halo_addr = b.trunc(b.smem_addr_of(lds), I32)
    wei_addr = b.add(halo_addr, b.const_i32(WEI0 * 2))

    # Halo origin in X coordinates and the edge shifts (module docstring).
    h0 = b.sub(h_tile if S == 1 else b.mul(h_tile, b.const_i32(S)), b.const_i32(PAD))
    w0 = b.sub(w_tile if S == 1 else b.mul(w_tile, b.const_i32(S)), b.const_i32(PAD))
    # Leading zero rows / columns; a padding wider than the filter's reach
    # (dilation along one axis only) can zero a whole (sub-)halo, never more.
    hs = b.smin(b.smax(b.sub(c0, h0), c0), b.const_i32(HALO_H))
    h_start = b.smax(b.add(h0, hs), c0)
    x_h_ext = b.smax(b.sub(c_H, h_start), c0)
    x_tile2 = b.sub(b.const_i32(HALO_H), hs)
    # Per (phase) sub-halo: (first input column, leading zero columns, zeroed
    # elements, column extent). Phase a holds the input columns w0 + a + S*j.
    c_S = b.const_i32(S)
    sub_halos = []
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
        z_elems = b.mul(b.add(b.mul(hs, b.const_i32(HC)), sw), b.const_i32(XP))
        base = b.add(
            b.add(b.mul(n_base, xs_n), b.mul(h_start, xs_h)),
            b.add(b.mul(w_start, xs_w), c_grp),
        )
        sub_halos.append((base, z_elems, w_ext))
    xs_col = b.mul(xs_w, c_S) if PS else xs_w
    k_ext = b.smax(b.smin(b.sub(c_kpg, k_base), b.const_i32(BK)), c0)
    w_base = b.mul(b.add(k_grp, k_base), w_row)

    # ---- zero the halo cells no transfer writes (top / left image edge).
    max_z = (min(PAD, HALO_H) * HC + min(spec.lead_cols, HC)) * XP
    if max_z:
        zero_v = b.zero_vec(io_type, 8)
        for a, (_, z_elems, _) in enumerate(sub_halos):
            z_vecs = b.div(z_elems, b.const_i32(8))
            for j in range(0, _ceil_div(max_z, 8), THREADS):
                v = b.add(tid, b.const_i32(j))
                with b.scf_if(b.cmp_lt(v, z_vecs)):
                    flat = b.mul(v, b.const_i32(8))
                    for st in range(NHS):
                        for im in range(NB):
                            off = st * HALO_STAGE + im * IMG + a * PSTR
                            b.smem_store_vN(
                                lds, [c0, b.add(flat, b.const_i32(off))], zero_v, n=8
                            )

    def _chunk_ext(chunk: Value) -> tuple[Value, Value]:
        k0 = b.mul(chunk, c_CK)  # first input channel of the chunk
        return k0, b.smax(b.smin(b.sub(c_cpg, k0), c_CK), c0)

    def _issue(chunk: Value) -> None:
        """TDM chunk ``chunk`` into ring slot chunk % NS."""
        stage = b.mod(chunk, b.const_i32(NS))
        _issue_halo(chunk, stage)
        if G == 1:  # the packed weights are staged once, before the loop
            _issue_wei(chunk, stage)

    def _issue_halo(chunk: Value, stage: Value) -> None:
        k0, c_ext = _chunk_ext(chunk)
        stage_off = b.mul(stage, b.const_i32(HALO_STAGE))
        for a, (x_base, z_elems, x_w_ext) in enumerate(sub_halos):
            for im in range(NB):
                # Wave im (phase 1: wave NB + 1 + im) moves image im's
                # (sub-)halo; an image past N transfers zeros.
                issuer = im if a == 0 else NB + 1 + im
                # A (sub-)halo that is all padding (its zeroing covers it)
                # issues no transfer: a zero-row tile is not a no-op.
                issue = b.land(
                    b.cmp_eq(wave_id, b.const_i32(issuer)), b.cmp_lt(c0, x_tile2)
                )
                with b.scf_if(issue):
                    img_off = b.add(stage_off, b.const_i32(im * IMG + a * PSTR))
                    img = b.add(b.add(x_base, k0), b.mul(b.const_i32(im), xs_n))
                    src = b.global_ptr_add(Xp, b.mul(img, c2))
                    dst = b.add(halo_addr, b.mul(b.add(img_off, z_elems), c2))
                    h_ext = x_h_ext
                    if NB > 1:
                        n_ok = b.cmp_lt(b.add(n_base, b.const_i32(im)), c_N)
                        h_ext = b.select(n_ok, x_h_ext, c0)
                        src = b.select(n_ok, src, b.global_ptr_add(Xp, c0))
                    b.tensor_load_to_lds(
                        *_tdm_descriptor_3d(
                            b,
                            src,
                            dst,
                            dims=(c_ext, x_w_ext, h_ext),
                            strides=(xs_col, xs_h),
                            tile0=XP,
                            tile1=HC,
                            tile2=x_tile2,
                        )
                    )

    def _issue_wei(chunk: Value, stage: Value, grp: Value | None = None) -> None:
        """Weights of ``chunk`` (all taps, or tap group ``grp`` of WTAPS
        consecutive taps) into weight slot ``stage``."""
        k0, c_ext = _chunk_ext(chunk)
        with b.scf_if(b.cmp_eq(wave_id, b.const_i32(NB))):
            off = b.add(w_base, k0)
            if grp is not None:
                off = b.add(off, b.mul(grp, b.mul(c_cpg0, b.const_i32(WTAPS))))
            src = b.global_ptr_add(Wp, b.mul(off, c2))
            dst = b.add(wei_addr, b.mul(stage, b.const_i32(WEI * 2)))
            b.tensor_load_to_lds(
                *_tdm_descriptor_3d(
                    b,
                    src,
                    dst,
                    dims=(c_ext, k_ext, b.const_i32(WTAPS)),
                    strides=(w_row, c_cpg0),
                    tile0=XP,
                    tile1=BK,
                    tile2=b.const_i32(WTAPS),
                )
            )

    # ---- per-lane LDS read offsets.
    # B (X): lane l reads pixel l % 16 of its tile, c (l // 16) * 16 .. +15.
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
        if S > 1:
            th = b.mul(th, b.const_i32(S))
            if not PS:  # a split halo indexes columns per phase
                tw = b.mul(tw, b.const_i32(S))
        off = b.add(
            b.mul(b.add(b.mul(th, b.const_i32(HC)), tw), b.const_i32(XP)),
            b.mul(lhalf, c16),
        )
        if NB > 1:
            off = b.add(off, b.mul(nb, b.const_i32(IMG)))
        px_off.append(off)
    # A (W): lane l reads output channel l % 16 of its tile, c (l // 16) * 16 .. +15.
    wei_off: list[Value] = []
    for m in range(TM):
        row = b.add(
            b.mul(b.add(b.mul(wave_k, b.const_i32(TM)), b.const_i32(m)), c16), l16
        )
        wei_off.append(
            b.add(
                b.mul(row, b.const_i32(XP)),
                b.add(b.mul(lhalf, c16), b.const_i32(WEI0)),
            )
        )

    def _read16(stage_base: Value, off: Value, const_off: int) -> Value:
        addr = b.add(stage_base, b.add(off, b.const_i32(const_off)))
        lo = b.smem_load_vN(lds, c0, addr, dtype=io_type, n=8)
        hi = b.smem_load_vN(lds, c0, b.add(addr, c8), dtype=io_type, n=8)
        return b.vec_concat(lo, hi)

    TU = spec.tap_reuse
    TSH = spec.tap_shift
    TWF = TW // 16  # 16-pixel fragments per output row
    if TU:
        # lane base of the wave's first fragment (px_off[tn] minus its static part)
        px_base = px_off[0]
        if TSH:
            c_edge = b.const_i32(16 - DW)
            is_edge = b.cmp_ge(l16, c_edge)
            edge_off = b.mul(b.sub(b.smax(l16, c_edge), l16), b.const_i32(XP))

    def tap_off0(ks: int, s: int) -> int:
        return s * DW * XP + ks * WMMA_K

    # v_permlane16 selectors: lane l <- lane l + DW inside each 16-lane row
    # (lanes past the row keep their own value; the edge read replaces them).
    _src = [min(i + DW, 15) for i in range(16)]
    SEL_LO = sum(_src[i] << (4 * i) for i in range(8))
    SEL_HI = sum(_src[8 + i] << (4 * i) for i in range(8))
    SEL_LO = SEL_LO - (1 << 32) if SEL_LO & 0x80000000 else SEL_LO
    SEL_HI = SEL_HI - (1 << 32) if SEL_HI & 0x80000000 else SEL_HI

    def _shift(frag: Value, stage_base: Value, off: int) -> Value:
        """The fragment one filter column (DW pixels) to the right: rotate each
        dword DW lanes down its 16-lane row (v_permlane16); the last DW lanes
        of each row read their new column from the halo. That read is issued
        by the whole wave, but every other lane reads the address of lane
        16 - DW (an LDS broadcast), so it moves 2 * DW distinct 32-byte runs."""
        words = b.vec_bitcast(frag, VectorType(I32, 8))
        lo_sel, hi_sel = b.const_i32(SEL_LO), b.const_i32(SEL_HI)
        moved = [
            b.permlane16(w, w, lo_sel, hi_sel, fi=False, bound_ctrl=False)
            for w in (b.vec_extract(words, i) for i in range(8))
        ]
        out = b.vec_bitcast(b.vec_pack(moved, I32), VectorType(io_type, 16))
        edge = _read16(stage_base, b.add(px_base, edge_off), off)
        return b.select(is_edge, edge, out)

    # One step per (k-step, tap).
    steps = [
        (ks, r, s) for ks in range(CK // WMMA_K) for r in range(KH) for s in range(KW)
    ]

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
        wei_base = b.mul(stage, b.const_i32(WEI)) if G == 1 else c0
        sig = (lambda: _signal_next(kc)) if SB else None
        return _multiply(steps, halo_base, wei_base, accs, sig)

    def _multiply(st_list, halo_base, wei_base, accs, signal=None):
        """The WMMAs of ``st_list`` (k-step, filter row, column) from one halo
        slot and one weight slot; ``signal`` runs before the last step's."""

        cache: dict = {}

        def _frags(st):
            ks, r, s = st
            if PS:
                a, j = (s * DW) % S, (s * DW) // S
                tap_off = a * PSTR + (r * DH * HC + j) * XP + ks * WMMA_K
            else:
                tap_off = (r * DH * HALO_W + s * DW) * XP + ks * WMMA_K
            if TU:
                bf = []
                for tn in range(TN):
                    # static halo offset of this fragment relative to the
                    # wave's first fragment (same lane base for every tn)
                    key = (
                        ks,
                        s,
                        (tn // TWF + r * DH) * HALO_W * XP + (tn % TWF) * 16 * XP,
                    )
                    if key not in cache:
                        prev = (ks, s - 1, key[2]) if TSH and s > 0 else None
                        if prev is not None and prev in cache:
                            cache[key] = _shift(
                                cache[prev], halo_base, key[2] + tap_off0(ks, s)
                            )
                        else:
                            cache[key] = _read16(
                                halo_base, px_base, key[2] + tap_off0(ks, s)
                            )
                    bf.append(cache[key])
            else:
                bf = [_read16(halo_base, px_off[tn], tap_off) for tn in range(TN)]
            tap = (r * KW + s) % WTAPS
            w_off = tap * BK * XP + ks * WMMA_K
            af = [_read16(wei_base, wei_off[m], w_off) for m in range(TM)]
            return af, bf

        if TU:
            # one filter column at a time: its KH rows reuse the B fragments
            st_list = sorted(st_list, key=lambda t: (t[0], t[2], t[1]))
        accs = list(accs)
        b.s_setprio(1)
        nxt_frags = _frags(st_list[0]) if spec.prefetch else None
        for j, st in enumerate(st_list):
            if spec.prefetch:
                af, bf = nxt_frags
                if j + 1 < len(st_list):
                    nxt_frags = _frags(st_list[j + 1])
            else:
                af, bf = _frags(st)
            if signal is not None and j == len(st_list) - 1:
                signal()
            for m in range(TM):
                for tn in range(TN):
                    i = m * TN + tn
                    accs[i] = b.mma(wmma_op, af[m], bf[tn], accs[i])
        b.s_setprio(0)
        return accs

    PP = spec.pingpong
    if PP:
        rank = b.div(wave_id, b.const_i32((WK * WP) // 2))

    def _pp_bar() -> None:
        b.s_barrier_bare(fence="compiler")
        b.sched_barrier(0)

    def _chunk_pp(kc: Value, accs: list[Value]) -> list[Value]:
        """``pingpong``: multiply chunk ``kc``, which has landed (the issuers
        waited for it before the barrier that starts it). Rank 1 runs this a
        barrier behind rank 0."""
        # Ring slot (kc + NS - 1) % NS held chunk kc - 1, whose last reads
        # (rank 1's) drained before the barrier rank 0 just passed.
        nxt = b.add(kc, b.const_i32(NS - 1))
        with b.scf_if(b.cmp_lt(nxt, n_chunks)):
            _issue(nxt)
        stage = b.mod(kc, b.const_i32(NS))
        halo_base = b.mul(stage, b.const_i32(HALO_STAGE))
        wei_base = b.mul(stage, b.const_i32(WEI)) if G == 1 else c0
        accs = list(accs)
        PG = spec.pp_group
        groups = [steps[i : i + PG] for i in range(0, len(steps), PG)]
        for j, grp in enumerate(groups):
            frags = []
            for ks, r, s in grp:
                if PS:
                    a, jj = (s * DW) % S, (s * DW) // S
                    tap_off = a * PSTR + (r * DH * HC + jj) * XP + ks * WMMA_K
                else:
                    tap_off = (r * DH * HALO_W + s * DW) * XP + ks * WMMA_K
                bf = [_read16(halo_base, px_off[tn], tap_off) for tn in range(TN)]
                tap = (r * KW + s) % WTAPS
                w_off = tap * BK * XP + ks * WMMA_K
                af = [_read16(wei_base, wei_off[m], w_off) for m in range(TM)]
                frags.append((af, bf))
            last = j == len(groups) - 1
            if last:
                # The other rank reads this slot one barrier later, and the
                # ring refill after the next barrier must not overtake ours.
                b.s_waitcnt(lgkmcnt=0)
            _pp_bar()
            b.s_setprio(1)
            for af, bf in frags:
                for m in range(TM):
                    for tn in range(TN):
                        i = m * TN + tn
                        accs[i] = b.mma(wmma_op, af[m], bf[tn], accs[i])
            b.s_setprio(0)
            if last:
                # The issuers wait for chunk kc + 1 before the barrier that
                # starts it.
                _wait_landed(b.add(kc, c1))
            _pp_bar()
        return accs

    n_steps = b.mul(n_chunks, b.const_i32(NG))

    def _chunk_rr(kc: Value, accs: list[Value]) -> list[Value]:
        """``row_ring`` / ``tap_ring``: chunk ``kc`` as NG (tap group) steps,
        each on its own weight slot; the chunk's halo slot stays for all."""
        hslot = b.mod(kc, c2)
        halo_base = b.mul(hslot, b.const_i32(HALO_STAGE))
        for r in range(NG):
            u = b.add(b.mul(kc, b.const_i32(NG)), b.const_i32(r))
            # The weight issuer waits for step u's weights; at a chunk's first
            # row the halo issuers wait for its halo (their only transfer in
            # flight: the next chunk's goes out below, after the barrier).
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(NB))):
                if NS == 2:
                    b.s_wait_tensorcnt(0)
                else:
                    more = b.cmp_lt(b.add(u, b.const_i32(NS - 2)), n_steps)
                    with b.scf_if_else(more) as (then_ctx, else_ctx):
                        with then_ctx:
                            b.s_wait_tensorcnt(NS - 2)
                        with else_ctx:
                            b.s_wait_tensorcnt(0)
            if r == 0:
                with b.scf_if(b.cmp_ne(wave_id, b.const_i32(NB))):
                    b.s_wait_tensorcnt(0)
            b.sync_lds_only()
            nxt = b.add(u, b.const_i32(NS - 1))
            with b.scf_if(b.cmp_lt(nxt, n_steps)):
                nc = b.div(nxt, b.const_i32(NG))
                _issue_wei(nc, b.mod(nxt, b.const_i32(NS)), b.mod(nxt, b.const_i32(NG)))
            if r == 0:
                nc = b.add(kc, c1)
                with b.scf_if(b.cmp_lt(nc, n_chunks)):
                    _issue_halo(nc, b.mod(nc, c2))
            wslot = b.mod(u, b.const_i32(NS))
            row_steps = [
                (ks, t // KW, t % KW)
                for ks in range(CK // WMMA_K)
                for t in range(r * WTAPS, (r + 1) * WTAPS)
            ]
            accs = _multiply(row_steps, halo_base, b.mul(wslot, b.const_i32(WEI)), accs)
        return accs

    k_out = b.add(k_grp, k_base)  # first output channel of the block

    def _epilogue(accs: list[Value]) -> None:
        """Stage the Y tile in LDS as [image][row][col][channel] and store it
        with one TDM transfer per image; the transfer clips the tile to the
        image and to the group's channels (the pad columns lie past them)."""
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
        for im in range(NB):
            with b.scf_if(b.cmp_eq(wave_id, b.const_i32(im))):
                n_img = b.add(n_base, b.const_i32(im))
                off = b.add(
                    b.add(b.mul(n_img, ys_n), b.mul(h_tile, ys_h)),
                    b.add(b.mul(w_tile, ys_w), k_out),
                )
                h_ext = b.smax(b.sub(c_Ho, h_tile), c0)
                dst = b.global_ptr_add(Yp, b.mul(off, c2))
                if NB > 1:
                    n_ok = b.cmp_lt(n_img, c_N)
                    h_ext = b.select(n_ok, h_ext, c0)
                    dst = b.select(n_ok, dst, b.global_ptr_add(Yp, c0))
                b.tensor_store_from_lds(
                    *_tdm_descriptor_3d(
                        b,
                        dst,
                        b.add(halo_addr, b.const_i32(im * TH * TW * EPITCH * 2)),
                        dims=(k_ext, b.smax(b.sub(c_Wo, w_tile), c0), h_ext),
                        strides=(ys_w, ys_h),
                        tile0=EPITCH,
                        tile1=TW,
                        tile2=b.const_i32(TH),
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
        k_w = b.add(k_base, b.mul(wave_k, b.const_i32(TM * 16)))
        off = b.add(
            b.add(b.mul(n_img, ys_n), b.mul(h_w, ys_h)),
            b.add(b.mul(w_tile, ys_w), b.add(k_grp, k_w)),
        )
        k_ext_w = b.smax(b.smin(b.sub(c_kpg, k_w), b.const_i32(TM * 16)), c0)
        h_ext = b.smax(b.sub(c_Ho, h_w), c0)
        dst = b.global_ptr_add(Yp, b.mul(off, c2))
        if NB > 1:
            n_ok = b.cmp_lt(n_img, c_N)
            h_ext = b.select(n_ok, h_ext, c0)
            dst = b.select(n_ok, dst, b.global_ptr_add(Yp, c0))
        b.tensor_store_from_lds(
            *_tdm_descriptor_3d(
                b,
                dst,
                b.add(halo_addr, b.mul(base, c2)),
                dims=(k_ext_w, b.smax(b.sub(c_Wo, w_tile), c0), h_ext),
                strides=(ys_w, ys_h),
                tile0=EP,
                tile1=TW,
                tile2=b.const_i32(ROWS),
            )
        )
        b.s_wait_tensorcnt(0)

    if G > 1:
        # Stage the super-group's block-diagonal weights into slot 0 once:
        # zero the slot, then each lane copies one CPG-wide run of one row.
        CPG0, KPG0 = p.cpg, p.kpg
        zero_v = b.zero_vec(io_type, 8)
        nvec = WEI // 8
        for j in range(0, nvec, THREADS):
            v = b.add(tid, b.const_i32(j))
            with b.scf_if(b.cmp_lt(v, b.const_i32(nvec))):
                b.smem_store_vN(
                    lds,
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
                val = b.global_load_vN(Wp, gidx, io_type, VEC)
                loff = b.add(
                    b.mul(b.add(b.mul(tap, b.const_i32(BK)), kq), b.const_i32(XP)),
                    b.add(b.mul(gi, b.const_i32(CPG0)), pv),
                )
                b.smem_store_vN(lds, [c0, b.add(loff, b.const_i32(WEI0))], val, n=VEC)

    # ---- main loop over input channel chunks.
    if RR:
        _issue_halo(c0, c0)
        for st in range(NS - 1):
            with b.scf_if(b.cmp_lt(b.const_i32(st), n_steps)):
                _issue_wei(b.const_i32(st // NG), b.const_i32(st), b.const_i32(st % NG))
    else:
        for st in range(NS - 1):
            with b.scf_if(b.cmp_lt(b.const_i32(st), n_chunks)):
                _issue(b.const_i32(st))
    if SB:
        b.s_waitcnt(lgkmcnt=0)  # the edge zeroing
        _wait_landed(c0)
        b.s_barrier_signal(0xFFFFFFFF)
    if PP:
        _wait_landed(c0)
        b.sync_lds_only()  # chunk 0 and the edge zeroing are visible
        with b.scf_if(b.cmp_eq(rank, c1)):
            _pp_bar()  # rank 1 runs one barrier behind
    zero_acc = b.zero_vec_f32(8)
    loop = b.scf_for_iter(
        c0,
        n_chunks,
        c1,
        [(f"acc{i}", zero_acc) for i in range(TM * TN)],
        iv_name="kc",
    )
    with loop as (kc, carried):
        b.scf_yield(
            *(_chunk_rr if RR else _chunk_pp if PP else _chunk)(kc, list(carried))
        )
    if SB:
        b.s_barrier_wait(0xFFFF)
    if PP:
        with b.scf_if(b.cmp_eq(rank, c0)):
            _pp_bar()  # rank 0 catches up with rank 1's lead barrier
    if spec.wave_epilogue:
        _epilogue_wave(list(loop.results))
    else:
        _epilogue(list(loop.results))
    return b.kernel
