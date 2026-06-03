# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 experimental deep-fused conv + maxpool prototype.

This is intentionally narrow. It proves the single-kernel dataflow we need for
deeper fusion before generalizing the scheduler:

    implicit-GEMM conv0 -> accumulator epilogue -> LDS C-shuffle
    -> 1x1 conv1 -> LDS C-shuffle -> maxpool -> Y

The v1 validator requires each CTA to own a rectangular tile of final pooled
outputs. That keeps pool windows local to the CTA while proving that a
downstream non-elementwise stage can consume upstream MFMA tiles without
materializing intermediates in global memory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

from ...core.ir import F16, I32, IRBuilder, PtrType, Value
from ...helpers.distribution import (
    LoadStoreTraits,
    make_static_distributed_tensor,
    store_tile_cshuffle,
)
from ...helpers.epilogues import _cshuffle_acc_distribution
from ...helpers.geometry import WarpGrid
from ...helpers.layouts import LdsLayout
from ...helpers.loads import CoalescedTileLoader
from ...helpers.spec import SignatureBuilder, kernel_name_join
from ...helpers.tensor_view import make_buffer_resource, make_lds_view
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_smem_frag_contiguous_f16,
)
from ..common.conv_implicit_gemm import (
    ConvAccumulatorEpilogue,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    is_valid_spec as is_valid_conv_spec,
    _apply_accumulator_epilogue,
)


@dataclass(frozen=True)
class FusedConvPoolProblem:
    """Shape contract for the gfx950 conv0 -> conv1 -> maxpool prototype."""

    conv: ConvProblem
    conv1_k: int = 0
    pool_y: int = 2
    pool_x: int = 2
    pool_stride_h: int = 2
    pool_stride_w: int = 2

    @property
    def conv1_channels(self) -> int:
        return self.conv.K if self.conv1_k <= 0 else self.conv1_k

    @property
    def pool_ho(self) -> int:
        return (self.conv.Ho - self.pool_y) // self.pool_stride_h + 1

    @property
    def pool_wo(self) -> int:
        return (self.conv.Wo - self.pool_x) // self.pool_stride_w + 1

    @property
    def total_out(self) -> int:
        return self.conv.N * self.pool_ho * self.pool_wo * self.conv1_channels

    def short(self) -> str:
        p = self.conv
        return (
            f"{p.short()}"
            f"_K1{self.conv1_channels}"
            f"_pool{self.pool_y}x{self.pool_x}"
            f"_s{self.pool_stride_h}x{self.pool_stride_w}"
        )


@dataclass(frozen=True)
class Gfx950DeepFusedConvPoolSpec:
    """One concrete gfx950 deep-fusion prototype configuration."""

    problem: FusedConvPoolProblem
    name: str = "ck_dsl_gfx950_deep_fused_conv_pool"
    tile_m: int = 128
    tile_n: int = 32
    tile_k: int = 16
    pool_tile_h: int = 4
    pool_tile_w: int = 8
    warp_m: int = 2
    warp_n: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16
    pipeline: str = "mem"
    async_dma: bool = False
    unroll_k: bool = False
    acc_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)
    conv1_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)
    cache_input_footprint: bool = False
    direct_conv0_from_input_cache: bool = False

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * 64

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.problem.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"pt{self.pool_tile_h}x{self.pool_tile_w}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"{self.pipeline}_{'async' if self.async_dma else 'sync'}",
            self.acc_epilogue.tag(),
            self.conv1_epilogue.tag(),
            "cshuffle_conv1_pool",
            flags={
                "icache": self.cache_input_footprint,
                "directa": self.direct_conv0_from_input_cache,
                "unrollk": self.unroll_k,
            },
        )

    def conv_spec(self) -> ImplicitGemmConvSpec:
        return ImplicitGemmConvSpec(
            problem=self.problem.conv,
            name=self.name,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            pipeline=self.pipeline,
            epilogue="cshuffle",
            async_dma=self.async_dma,
            unroll_k=self.unroll_k,
            acc_epilogue=self.acc_epilogue,
        )


def make_deep_fused_conv_pool_spec(
    *,
    n: int = 1,
    h: int,
    w: int,
    c: int,
    k0: int,
    k1: int,
    r: int = 3,
    s: int = 3,
    pool_tile_h: int = 4,
    pool_tile_w: int = 8,
    tile_n: int = 32,
    tile_k: int = 16,
    warp_m: int = 2,
    warp_n: int = 1,
    warp_tile_k: int = 16,
    pipeline: str = "mem",
    unroll_k: bool = False,
    async_dma: bool = False,
    cache_input_footprint: bool = False,
    direct_conv0_from_input_cache: bool = False,
) -> Gfx950DeepFusedConvPoolSpec:
    """Build a deep-fusion spec, auto-deriving the constrained ``tile_m``.

    ``tile_m`` must equal the rectangular conv tile that backs one pooled-output
    tile, i.e. ``(pool_tile_h * pool_stride_h) * (pool_tile_w * pool_stride_w)``.
    Deriving it here keeps callers (verify harness, benchmarks, sweeps) from
    setting it inconsistently with ``pool_tile_*`` and tripping the validator.
    """

    conv = ConvProblem(
        N=n,
        Hi=h,
        Wi=w,
        C=c,
        K=k0,
        R=r,
        S=s,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    problem = FusedConvPoolProblem(conv=conv, conv1_k=k1)
    conv_tile_h = pool_tile_h * problem.pool_stride_h
    conv_tile_w = pool_tile_w * problem.pool_stride_w
    tile_m = conv_tile_h * conv_tile_w
    return Gfx950DeepFusedConvPoolSpec(
        problem=problem,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pool_tile_h=pool_tile_h,
        pool_tile_w=pool_tile_w,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=warp_tile_k,
        pipeline=pipeline,
        unroll_k=unroll_k,
        async_dma=async_dma,
        cache_input_footprint=cache_input_footprint,
        direct_conv0_from_input_cache=direct_conv0_from_input_cache,
        acc_epilogue=ConvAccumulatorEpilogue(relu=True),
    )


def is_valid_spec(
    spec: Gfx950DeepFusedConvPoolSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the experimental fused kernel."""

    if arch != "gfx950":
        return False, f"deep fused conv/pool v1 is gfx950-only (got {arch!r})"
    conv_spec = spec.conv_spec()
    ok, why = is_valid_conv_spec(conv_spec, arch=arch)
    if not ok:
        return False, why
    p = spec.problem
    c = p.conv
    if (p.pool_y, p.pool_x, p.pool_stride_h, p.pool_stride_w) != (2, 2, 2, 2):
        return False, "v1 supports only 2x2 stride-2 maxpool"
    if p.pool_ho <= 0 or p.pool_wo <= 0:
        return False, "pool output dimensions must be positive"
    if spec.pipeline not in ("mem", "compv3", "compv4"):
        return False, f"unsupported pipeline {spec.pipeline!r}"
    if spec.async_dma and (
        spec.cache_input_footprint or spec.direct_conv0_from_input_cache
    ):
        return False, "async_dma is only supported with the default conv0 A-load path"
    if spec.unroll_k and spec.async_dma:
        return False, "unroll_k and async_dma are mutually exclusive K-loop schedules"
    if spec.unroll_k and (
        spec.cache_input_footprint or spec.direct_conv0_from_input_cache
    ):
        return False, "unroll_k is only supported with the default conv0 A-load path"
    if c.N != 1:
        return False, f"v1 tiled schedule supports only N=1 (got N={c.N})"
    if spec.pool_tile_h <= 0 or spec.pool_tile_w <= 0:
        return False, "pool_tile_h and pool_tile_w must be positive"
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    if spec.tile_m != conv_tile_h * conv_tile_w:
        return False, (
            f"tile_m={spec.tile_m} must equal rectangular conv tile "
            f"{conv_tile_h}x{conv_tile_w}={conv_tile_h * conv_tile_w}"
        )
    if p.pool_ho % spec.pool_tile_h or p.pool_wo % spec.pool_tile_w:
        return False, (
            f"v1 requires pool dims ({p.pool_ho}, {p.pool_wo}) divisible by "
            f"pool tile ({spec.pool_tile_h}, {spec.pool_tile_w})"
        )
    if c.K > spec.tile_n:
        return False, (
            f"v1 requires one CTA to own all conv channels: K={c.K} > tile_n={spec.tile_n}"
        )
    if p.conv1_channels > spec.tile_n:
        return False, (
            f"v1 requires one CTA to own all conv1 channels: "
            f"K1={p.conv1_channels} > tile_n={spec.tile_n}"
        )
    if c.K % 8:
        return False, "v1 W1 loader requires conv0 channels divisible by 8"
    if spec.tile_m % (spec.warp_m * spec.warp_tile_m):
        return False, "tile_m must divide warp_m * warp_tile_m"
    if spec.tile_n % (spec.warp_n * spec.warp_tile_n):
        return False, "tile_n must divide warp_n * warp_tile_n"
    return True, "ok"


def deep_fused_conv_pool_signature(spec: Gfx950DeepFusedConvPoolSpec):
    """Manifest/launcher signature.

    The first three params match conv's pointer convention, but the third
    pointer is the final pooled output. ``W1`` is declared before the byte-size
    scalars so the HIP packed-args ABI keeps all 64-bit pointer args aligned.
    """

    return (
        SignatureBuilder()
        .ptr("A", "f16")
        .ptr("B", "f16")
        .ptr("Y", "f16")
        .ptr("W1", "f16")
        .scalar("W1_bytes", "i32")
        .scalar("A_bytes", "i32")
        .scalar("B_bytes", "i32")
        .scalar("Y_bytes", "i32")
        .build()
    )


def deep_fused_conv_pool_grid(
    spec: Gfx950DeepFusedConvPoolSpec,
) -> Tuple[int, int, int]:
    p = spec.problem
    return (1, p.pool_ho // spec.pool_tile_h, p.pool_wo // spec.pool_tile_w)


def _stage_accumulators_to_cshuffle_lds(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    *,
    sync: bool = True,
) -> Value:
    """Publish MFMA accumulators to a row-major ``[tile_m, tile_n]`` LDS tile.

    ``sync=False`` skips the trailing ``b.sync()`` so the caller can batch this
    producer barrier with another disjoint-LDS producer (e.g. the W1 load) into
    a single block-wide barrier before the shared consumer reads both tiles.
    """

    atom = spec.atom
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    if len(accs) != mfmas_m * mfmas_n:
        raise ValueError(f"expected {mfmas_m * mfmas_n} accs, got {len(accs)}")

    lds_layout = LdsLayout.cshuffle(tile_m=grid.tile_m, tile_n=grid.tile_n)
    lds_layout.validate()
    c_view = make_lds_view(
        b,
        dtype=F16,
        shape=lds_layout.storage_shape(grid.tile_m),
        name_hint="DeepFusionC_smem",
    )
    c_smem = c_view.base
    c_window = c_view.tile(
        list(lds_layout.storage_shape(grid.tile_m)),
        [b.const_i32(0), b.const_i32(0)],
    )

    dist = _cshuffle_acc_distribution(atom.c_per_lane)
    traits = LoadStoreTraits(distribution=dist, vector_dim_y=1, scalar_per_vector=1)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)

    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[mi * mfmas_n + ni]
            acc_h = b.vec_trunc_f32_to_f16(acc)
            dt = make_static_distributed_tensor(dist, dtype=F16)
            for i in range(atom.c_per_lane):
                dt.set([i, 0], b.vec_extract(acc_h, i))

            tile_m_base = b.add(warp_m_off, b.const_i32(mi * atom.m))
            tile_n_base = b.add(warp_n_off, b.const_i32(ni * atom.n))

            def coord_fn(b_, y_base, _k, *, _mb=tile_m_base, _nb=tile_n_base):
                i = int(y_base[0])
                row_in_atom, col_in_atom = atom.lane_to_output(b_, grid.lane, i)
                return [b_.add(_mb, row_in_atom), b_.add(_nb, col_in_atom)]

            store_tile_cshuffle(b, c_window, dt, traits=traits, coord_fn=coord_fn)

    if sync:
        b.sync()
    return c_smem


def _load_conv1_weights_to_lds(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    w1_rsrc: Value,
    grid: WarpGrid,
    *,
    sync: bool = True,
) -> Value:
    """Load W1[K1, K0] into a padded row-major LDS tile.

    ``sync=False`` skips the trailing ``b.sync()`` so the caller can fold this
    barrier into a single block-wide barrier shared with the conv0 cshuffle
    stage (the two write disjoint LDS tiles, so one barrier suffices before the
    conv1 MFMA reads both).
    """

    w1_smem = b.smem_alloc(F16, [spec.tile_n, spec.problem.conv.K], name_hint="W1_smem")
    loader = CoalescedTileLoader(
        tile_rows=spec.tile_n,
        tile_cols=spec.problem.conv.K,
        block_size=spec.block_size,
        load_vec=8,
    )
    c_k0 = b.const_i32(spec.problem.conv.K)
    c_k1 = b.const_i32(spec.problem.conv1_channels)

    def descriptor(b_: IRBuilder, row: Value, col: Value):
        row_ok = b_.cmp_lt(row, c_k1)
        col_ok = b_.cmp_lt(col, c_k0)
        valid = b_.land(row_ok, col_ok)
        off = b_.add(b_.mul(row, c_k0), col)
        return off, valid

    loader.load(b, tid=grid.tid, smem_dst=w1_smem, descriptor=descriptor, rsrc=w1_rsrc)
    if sync:
        b.sync()
    return w1_smem


def _setup_input_footprint_cache(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    a_rsrc: Value,
    grid: WarpGrid,
) -> Value:
    """Load the unique conv0 input footprint for this pooled-output tile."""

    p = spec.problem
    c = p.conv
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    foot_h = conv_tile_h + (c.R - 1) * c.dH
    foot_w = conv_tile_w + (c.S - 1) * c.dW
    input_smem = b.smem_alloc(F16, [foot_h * foot_w, c.C], name_hint="InputFoot_smem")
    total = foot_h * foot_w * c.C
    elems_per_thread = (total + spec.block_size - 1) // spec.block_size
    c_total = b.const_i32(total)
    c_c = b.const_i32(c.C)
    c_foot_w = b.const_i32(foot_w)
    c_half_bytes = b.const_i32(2)
    oob = b.const_i32((1 << 31) - 1)
    h_base = b.sub(
        b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h * p.pool_stride_h)),
        b.const_i32(c.pH),
    )
    w_base = b.sub(
        b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w * p.pool_stride_w)),
        b.const_i32(c.pW),
    )

    for e in range(elems_per_thread):
        idx = b.add(b.mul(b.const_i32(e), b.const_i32(spec.block_size)), grid.tid)
        idx_ok = b.cmp_lt(idx, c_total)
        safe_idx = b.select(idx_ok, idx, b.const_i32(0))
        ci = b.mod(safe_idx, c_c)
        t0 = b.div(safe_idx, c_c)
        local_w = b.mod(t0, c_foot_w)
        local_h = b.div(t0, c_foot_w)
        global_h = b.add(h_base, local_h)
        global_w = b.add(w_base, local_w)
        h_ok = b.land(
            b.cmp_ge(global_h, b.const_i32(0)), b.cmp_lt(global_h, b.const_i32(c.Hi))
        )
        w_ok = b.land(
            b.cmp_ge(global_w, b.const_i32(0)), b.cmp_lt(global_w, b.const_i32(c.Wi))
        )
        valid = b.land(idx_ok, b.land(h_ok, w_ok))
        off_elems = b.add(
            b.mul(b.add(b.mul(global_h, b.const_i32(c.Wi)), global_w), c_c),
            ci,
        )
        off_bytes = b.mul(off_elems, c_half_bytes)
        safe_off = b.select(valid, off_bytes, oob)
        v = b.buffer_load_f16(a_rsrc, safe_off, b.const_i32(0))
        b.smem_store_f16(input_smem, [t0, ci], v)

    b.sync()
    return input_smem


def _load_conv0_a_tile_from_input_cache(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    conv_spec: ImplicitGemmConvSpec,
    k_off: Value,
    a_dst: Value,
    grid: WarpGrid,
    input_smem: Value,
) -> None:
    """Materialize the conv0 implicit-GEMM A tile from cached input footprint."""

    p = spec.problem
    c = p.conv
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    foot_w = conv_tile_w + (c.S - 1) * c.dW
    total = spec.tile_m * spec.tile_k
    elems_per_thread = (total + spec.block_size - 1) // spec.block_size
    c_total = b.const_i32(total)
    c_tile_k = b.const_i32(spec.tile_k)
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_sc = b.const_i32(c.S * c.C)
    c_c = b.const_i32(c.C)
    c_foot_w = b.const_i32(foot_w)
    c_k_gemm = b.const_i32(c.K_gemm)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    for e in range(elems_per_thread):
        idx = b.add(b.mul(b.const_i32(e), b.const_i32(spec.block_size)), grid.tid)
        idx_ok = b.cmp_lt(idx, c_total)
        safe_idx = b.select(idx_ok, idx, b.const_i32(0))
        row = b.div(safe_idx, c_tile_k)
        col = b.mod(safe_idx, c_tile_k)
        kg = b.add(k_off, col)
        kg_ok = b.cmp_lt(kg, c_k_gemm)

        local_oh = b.div(row, c_conv_tile_w)
        local_ow = b.mod(row, c_conv_tile_w)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)
        # VALU opt: strength-reduce div/mod by C=8 to shift/mask.
        if c.C == 8:
            s_col = b.lshr(rem, b.const_i32(3))
            ci = b.land(rem, b.const_i32(7))
        else:
            s_col = b.div(rem, c_c)
            ci = b.mod(rem, c_c)
        ih = b.add(b.mul(local_oh, b.const_i32(c.sH)), b.mul(r, b.const_i32(c.dH)))
        iw = b.add(b.mul(local_ow, b.const_i32(c.sW)), b.mul(s_col, b.const_i32(c.dW)))
        foot_row = b.add(b.mul(ih, c_foot_w), iw)
        v = b.vec_extract(b.smem_load_vN_f16(input_smem, foot_row, ci, n=1), 0)
        v = b.select(b.land(idx_ok, kg_ok), v, zero_h)
        b.smem_store_f16(a_dst, [row, col], v)


def _load_conv0_a_operand_from_input_cache(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    row: Value,
    k_off: Value,
    col_base: Value,
    frag_len: int,
    input_smem: Value,
) -> Value:
    """Read one MFMA A operand fragment directly from the cached input footprint.

    Optimized for VALU address-math reduction:
    - Hoists row-dependent ``local_oh/local_ow`` out of the per-element loop
      (computed once per fragment, not once per element).
    - Strength-reduces div/mod by ``C=8`` (power-of-2) to shift/mask:
      ``s_col = rem >> 3`` and ``ci = rem & 7`` instead of div/mod.
    """

    p = spec.problem
    c = p.conv
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    foot_w = conv_tile_w + (c.S - 1) * c.dW
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_sc = b.const_i32(c.S * c.C)
    c_c = b.const_i32(c.C)
    c_foot_w = b.const_i32(foot_w)
    c_k_gemm = b.const_i32(c.K_gemm)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    # VALU opt 1: hoist row-dependent coordinates out of the per-element loop.
    # ``row`` is fixed for the entire fragment (one MFMA A operand), so
    # ``local_oh`` and ``local_ow`` are loop-invariant. Computing them once
    # saves ``frag_len - 1`` div/mod pairs (7 pairs for ``frag_len=8``).
    local_oh = b.div(row, c_conv_tile_w)
    local_ow = b.mod(row, c_conv_tile_w)

    # Precompute base offset for spatial indexing (also loop-invariant).
    oh_base = b.mul(local_oh, b.const_i32(c.sH))
    ow_base = b.mul(local_ow, b.const_i32(c.sW))

    elems = []
    for i in range(frag_len):
        kg = b.add(k_off, b.add(col_base, b.const_i32(i)))
        kg_ok = b.cmp_lt(kg, c_k_gemm)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)

        # VALU opt 2: strength-reduce div/mod by C=8 (power-of-2) to shift/mask.
        # Original: ``s_col = div(rem, C)``, ``ci = mod(rem, C)``.
        # For C=8: ``s_col = rem >> 3``, ``ci = rem & 7``.
        # This replaces two fp div-by-recip (``v_rcp`` + ``v_div_fixup``) with
        # one ``v_lshrrev_b32`` and one ``v_and_b32`` — cheaper on VALU.
        if c.C == 8:
            s_col = b.lshr(rem, b.const_i32(3))
            ci = b.land(rem, b.const_i32(7))
        else:
            s_col = b.div(rem, c_c)
            ci = b.mod(rem, c_c)

        ih = b.add(oh_base, b.mul(r, b.const_i32(c.dH)))
        iw = b.add(ow_base, b.mul(s_col, b.const_i32(c.dW)))
        foot_row = b.add(b.mul(ih, c_foot_w), iw)
        raw = b.vec_extract(b.smem_load_vN_f16(input_smem, foot_row, ci, n=1), 0)
        elems.append(b.select(kg_ok, raw, zero_h))
    return b.vec_pack(elems, elems[0].type)


def _epilogue_is_pool_deferrable(epi: ConvAccumulatorEpilogue) -> bool:
    """Whether ``epi`` commutes with maxpool so it can be applied after the pool.

    ReLU, bias add, clamp, and non-negative scale are all monotonic
    non-decreasing, so ``epi(max(xs)) == max(epi(x) for x in xs)``. Applying the
    epilogue to the pooled result (one value per pooled pixel) instead of to every
    conv1 accumulator element (4x more for 2x2 pool) cuts the per-element fmax/etc.
    VALU. A negative scale would turn the outer max into a min, so it is not
    deferrable.
    """
    return epi.scale >= 0.0


def _apply_epilogue_scalar(
    b: IRBuilder, epi: ConvAccumulatorEpilogue, v: Value
) -> Value:
    """Apply a static fp32 epilogue to a single scalar value.

    Mirrors the per-lane transform in ``_apply_accumulator_epilogue`` so the
    deferred-past-pool path is numerically identical to applying it on the accs.
    """
    if epi.is_identity():
        return v
    if epi.bias != 0.0:
        v = b.fadd(v, b.const_f32(epi.bias))
    if epi.scale != 1.0:
        v = b.fmul(v, b.const_f32(epi.scale))
    if epi.relu:
        v = b.fmax(v, b.const_f32(0.0))
    if epi.clamp_min is not None:
        v = b.fmax(v, b.const_f32(epi.clamp_min))
    if epi.clamp_max is not None:
        v = b.fmin(v, b.const_f32(epi.clamp_max))
    return v


def _emit_conv1_1x1_mfma(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    conv_spec: ImplicitGemmConvSpec,
    c0_smem: Value,
    w1_smem: Value,
    grid: WarpGrid,
    defer_epilogue: bool = False,
) -> Sequence[Value]:
    """Compute conv1 as a 1x1 GEMM over staged conv0 activations.

    When ``defer_epilogue`` is set, the raw fp32 accumulators are returned and the
    caller is responsible for applying ``spec.conv1_epilogue`` after the maxpool
    reduction (valid only when the epilogue is pool-deferrable).
    """

    atom = conv_spec.atom
    decoded = decode_mfma_lanes(b, atom, grid.lane)
    m_in_atom = decoded.m_in_atom
    n_in_atom = decoded.n_in_atom
    k_blk = decoded.k_blk
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = conv_spec.k_atoms_per_tile_k
    k_chunks = (spec.problem.conv.K + spec.tile_k - 1) // spec.tile_k
    # The valid_k mask only guards a K tail. When the tiling covers K exactly
    # it is statically dead, so we can skip it and issue wide vector ds_reads.
    needs_mask = k_chunks * spec.tile_k != spec.problem.conv.K
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    accs = [b.zero_vec_f32(atom.c_per_lane) for _ in range(mfmas_m * mfmas_n)]

    for k_chunk in range(k_chunks):
        chunk_base = k_chunk * spec.tile_k
        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(atom.a_per_lane)),
                b.const_i32(chunk_base + kk * conv_spec.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off,
                    b.add(b.const_i32(mi * atom.m), m_in_atom),
                )
                a_rows.append(
                    load_smem_frag_contiguous_f16(
                        b,
                        c0_smem,
                        a_row,
                        col_base,
                        atom.a_per_lane,
                        needs_mask=needs_mask,
                        valid_k=spec.problem.conv.K,
                    )
                )

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off,
                    b.add(b.const_i32(ni * atom.n), n_in_atom),
                )
                b_cols.append(
                    load_smem_frag_contiguous_f16(
                        b,
                        w1_smem,
                        b_row,
                        col_base,
                        atom.b_per_lane,
                        needs_mask=needs_mask,
                        valid_k=spec.problem.conv.K,
                    )
                )

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    accs[flat] = atom.emit(b, a_rows[mi], b_cols[ni], accs[flat])
                    flat += 1

    if defer_epilogue:
        return list(accs)
    return _apply_accumulator_epilogue(b, spec.conv1_epilogue, accs)


def _emit_inline_maxpool_from_cshuffle(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    c_smem: Value,
    y_rsrc: Value,
    grid: WarpGrid,
    epilogue: Optional[ConvAccumulatorEpilogue] = None,
) -> None:
    """Reduce the staged conv tile into final pooled NHWK output.

    When ``epilogue`` is given, it is applied to each pooled fp32 result before
    the fp16 store (the deferred conv1 epilogue, see
    ``_epilogue_is_pool_deferrable``).
    """

    p = spec.problem
    out_k = p.conv1_channels
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w

    # Lever B: tile the gather by (window, k-block). The 2x2 maxpool corner rows
    # depend only on the pooled window, not on the channel k, so processing a
    # contiguous run of ``kvec`` channels per thread amortizes the window decode
    # and the 4 corner-address computations across kvec channels and folds the
    # per-channel scalar ``ds_read_u16`` into a single wide ``ds_read_b{32,64}``.
    # k is the contiguous (column) dim of the row-major [tile_m, tile_n] cshuffle
    # LDS tile, so a kvec-wide read stays within one row. Pick the largest valid
    # width that divides out_k while keeping >= half the block's threads active.
    kvec = 1
    for cand in (8, 4, 2):
        if (
            out_k % cand == 0
            and (spec.pool_tile_h * spec.pool_tile_w * (out_k // cand))
            >= spec.block_size // 2
        ):
            kvec = cand
            break
    kblocks = out_k // kvec
    total_vec = spec.pool_tile_h * spec.pool_tile_w * kblocks
    elems_per_thread = (total_vec + spec.block_size - 1) // spec.block_size
    c_total_vec = b.const_i32(total_vec)
    c_kblocks = b.const_i32(kblocks)
    c_kvec = b.const_i32(kvec)
    c_pool_tile_w = b.const_i32(spec.pool_tile_w)
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_out_k = b.const_i32(out_k)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    neg_inf = b.const_f32(-3.4028234663852886e38)
    block_pool_h = b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h))
    block_pool_w = b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w))

    for e in range(elems_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), b.const_i32(spec.block_size)), grid.tid)
        in_range = b.cmp_lt(vec_idx, c_total_vec)
        safe_vec_idx = b.select(in_range, vec_idx, b.const_i32(0))

        kb = b.mod(safe_vec_idx, c_kblocks)
        k0 = b.mul(kb, c_kvec)
        t0 = b.div(safe_vec_idx, c_kblocks)
        local_pwo = b.mod(t0, c_pool_tile_w)
        local_pho = b.div(t0, c_pool_tile_w)
        global_pho = b.add(block_pool_h, local_pho)
        global_pwo = b.add(block_pool_w, local_pwo)

        accs = [neg_inf] * kvec
        for yy in range(2):
            local_conv_h = b.add(b.mul(local_pho, b.const_i32(2)), b.const_i32(yy))
            for xx in range(2):
                local_conv_w = b.add(b.mul(local_pwo, b.const_i32(2)), b.const_i32(xx))
                conv_m_local = b.add(b.mul(local_conv_h, c_conv_tile_w), local_conv_w)
                v_vec = b.smem_load_vN_f16(c_smem, conv_m_local, k0, n=kvec)
                for j in range(kvec):
                    accs[j] = b.fmax(accs[j], b.cast_to_f32(b.vec_extract(v_vec, j)))

        y_base_elems = b.add(
            b.mul(
                b.add(b.mul(global_pho, b.const_i32(p.pool_wo)), global_pwo), c_out_k
            ),
            k0,
        )
        for j in range(kvec):
            acc = accs[j]
            if epilogue is not None:
                acc = _apply_epilogue_scalar(b, epilogue, acc)
            y_h = b.trunc_f32_to_f16(acc)
            y_off_bytes = b.mul(b.add(y_base_elems, b.const_i32(j)), c_half_bytes)
            safe_off = b.select(in_range, y_off_bytes, oob_sentinel)
            b.buffer_store_f16(y_rsrc, safe_off, b.const_i32(0), y_h)


def _maxpool_is_intra_lane(spec: Gfx950DeepFusedConvPoolSpec, grid: WarpGrid) -> bool:
    """Whether the conv1->maxpool handoff can stay register-resident (no LDS).

    With a single 32x32 MFMA atom per warp and ``warp_n==1``, each lane owns a
    vec<16> accumulator whose 16 slots tile a 4x4 conv-spatial block for one
    channel (= ``lane % 32``). For a 2x2 stride-2 pool that block is exactly
    2x2=4 pool windows, and all four corners of every window live in the *same
    lane's* accumulator -- so the maxpool reduces purely intra-lane with no
    cross-lane shuffle and no cshuffle LDS staging. The exact slot decomposition
    (see ``_emit_inline_maxpool_from_registers``) requires:

      local_conv_h = warp_m_idx*4 + i//4   (warp_m==2 -> 8 conv rows)
      local_conv_w = (lane//32)*4 + i%4    (conv_tile_w==8 -> 8 conv cols)

    so conv_tile must be 8x8 and tile_m==64. Any other geometry falls back to
    the LDS gather path (``_emit_inline_maxpool_from_cshuffle``).
    """
    p = spec.problem
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    return (
        grid.warp_tile_m == 32
        and grid.warp_tile_n == 32
        and grid.mfmas_per_warp_m == 1
        and grid.mfmas_per_warp_n == 1
        and grid.warp_n == 1
        and grid.warp_m == 2
        and p.pool_stride_h == 2
        and p.pool_stride_w == 2
        and conv_tile_h == 8
        and conv_tile_w == 8
        and spec.tile_m == 64
        and p.conv1_channels <= 32
    )


# Pool window (pho_l, pwo_l) -> the four accumulator slots holding its corners.
# Derived from the 32x32 C-fragment layout: slot = (i//4)*4 + (i%4) with
# i//4 = pho_l*2 + yy and i%4 = pwo_l*2 + xx over the 2x2 window (yy,xx in 0..1).
_INTRA_LANE_WINDOW_SLOTS = {
    (0, 0): (0, 1, 4, 5),
    (0, 1): (2, 3, 6, 7),
    (1, 0): (8, 9, 12, 13),
    (1, 1): (10, 11, 14, 15),
}


def _emit_inline_maxpool_from_registers(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    conv1_accs: Sequence[Value],
    y_rsrc: Value,
    grid: WarpGrid,
    epilogue: Optional[ConvAccumulatorEpilogue] = None,
) -> None:
    """Reduce the conv1 accumulators directly into pooled NHWK output.

    Eliminates the conv1->maxpool cshuffle handoff: instead of staging the
    conv1 accs to LDS and re-gathering, each lane reduces its own vec<16>
    accumulator. Gated by :func:`_maxpool_is_intra_lane`. When ``epilogue`` is
    given it is applied once per pooled fp32 result (the deferred conv1
    epilogue, see :func:`_epilogue_is_pool_deferrable`).
    """

    p = spec.problem
    out_k = p.conv1_channels
    acc_vec = conv1_accs[0]

    channel = b.mod(grid.lane, b.const_i32(32))
    m_blk = b.div(grid.lane, b.const_i32(32))
    block_pool_h = b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h))
    block_pool_w = b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w))
    pho_base = b.add(block_pool_h, b.mul(grid.warp_m_idx, b.const_i32(2)))
    pwo_base = b.add(block_pool_w, b.mul(m_blk, b.const_i32(2)))

    in_range = b.cmp_lt(channel, b.const_i32(out_k))
    oob_sentinel = b.const_i32((1 << 31) - 1)
    c_pool_wo = b.const_i32(p.pool_wo)
    c_out_k = b.const_i32(out_k)
    c_half_bytes = b.const_i32(2)

    for pho_l in range(2):
        gpho = b.add(pho_base, b.const_i32(pho_l))
        for pwo_l in range(2):
            gpwo = b.add(pwo_base, b.const_i32(pwo_l))
            s0, s1, s2, s3 = _INTRA_LANE_WINDOW_SLOTS[(pho_l, pwo_l)]
            acc = b.fmax(
                b.fmax(b.vec_extract(acc_vec, s0), b.vec_extract(acc_vec, s1)),
                b.fmax(b.vec_extract(acc_vec, s2), b.vec_extract(acc_vec, s3)),
            )
            if epilogue is not None:
                acc = _apply_epilogue_scalar(b, epilogue, acc)
            y_h = b.trunc_f32_to_f16(acc)
            y_off_elems = b.add(
                b.mul(b.add(b.mul(gpho, c_pool_wo), gpwo), c_out_k), channel
            )
            y_off_bytes = b.mul(y_off_elems, c_half_bytes)
            safe_off = b.select(in_range, y_off_bytes, oob_sentinel)
            b.buffer_store_f16(y_rsrc, safe_off, b.const_i32(0), y_h)


def build_deep_fused_conv_pool(spec: Gfx950DeepFusedConvPoolSpec, arch: str = "gfx950"):
    """Build the gfx950 one-CTA conv0 -> conv1 -> maxpool fused kernel."""

    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid deep fused conv/pool spec for {arch}: {why}")

    conv_spec = replace(spec.conv_spec(), name=spec.name)

    def extra_params(b: IRBuilder) -> Value:
        W1 = b.param(
            "W1", PtrType(F16, "global"), noalias=True, readonly=True, align=16
        )
        W1_bytes = b.param("W1_bytes", I32)
        return make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc

    def m_index_fn(b: IRBuilder, row: Value, _grid: WarpGrid) -> Value:
        """Conv0 M-index callback: maps tile-local row to global (ho, wo) offset.

        VALU opt: strength-reduce div/mod by ``conv_tile_w`` when it's power-of-2.
        For the target shape (pool_tile_w=8, pool_stride_w=2), conv_tile_w=16,
        so ``local_h = row >> 4`` and ``local_w = row & 15`` replace hardware
        div/mod. This is called **per MFMA A fragment** in the main K-loop, so
        it's a hot path for conv0 addressing.
        """
        p = spec.problem
        c = p.conv
        conv_tile_w = spec.pool_tile_w * p.pool_stride_w
        c_conv_tile_w = b.const_i32(conv_tile_w)

        # Check if conv_tile_w is power-of-2 at compile time and strength-reduce.
        if conv_tile_w > 0 and (conv_tile_w & (conv_tile_w - 1)) == 0:
            # Power-of-2: use shift/mask instead of div/mod.
            shift = (conv_tile_w - 1).bit_length()
            local_h = b.lshr(row, b.const_i32(shift))
            local_w = b.land(row, b.const_i32(conv_tile_w - 1))
        else:
            # Not power-of-2: fall back to hardware div/mod.
            local_h = b.div(row, c_conv_tile_w)
            local_w = b.mod(row, c_conv_tile_w)

        global_h = b.add(
            b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h * p.pool_stride_h)),
            local_h,
        )
        global_w = b.add(
            b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w * p.pool_stride_w)),
            local_w,
        )
        return b.add(b.mul(global_h, b.const_i32(c.Wo)), global_w)

    def a_mhw_index_fn(b: IRBuilder, row: Value, grid: WarpGrid):
        """Conv0 A-coord callback: tile-local row -> (n, ho, wo) directly.

        Returns the same (global_h, global_w) that ``m_index_fn`` computes via
        shift/mask, but as separate coords so the decomposed A descriptor can
        consume them without re-deriving them. This bypasses the m-flatten here
        (mul+add) and the descriptor's m -> (n, ho, wo) magic unmerge (~10 VALU
        per A coord, two magic divisions by Wo and Ho). N==1 for this problem,
        so n is the constant 0. Bit-identical to the flattened m path.
        """
        p = spec.problem
        conv_tile_w = spec.pool_tile_w * p.pool_stride_w
        if conv_tile_w > 0 and (conv_tile_w & (conv_tile_w - 1)) == 0:
            shift = (conv_tile_w - 1).bit_length()
            local_h = b.lshr(row, b.const_i32(shift))
            local_w = b.land(row, b.const_i32(conv_tile_w - 1))
        else:
            c_conv_tile_w = b.const_i32(conv_tile_w)
            local_h = b.div(row, c_conv_tile_w)
            local_w = b.mod(row, c_conv_tile_w)
        global_h = b.add(
            b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h * p.pool_stride_h)),
            local_h,
        )
        global_w = b.add(
            b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w * p.pool_stride_w)),
            local_w,
        )
        return b.const_i32(0), global_h, global_w

    def setup_input_cache(
        b: IRBuilder, conv_spec_: ImplicitGemmConvSpec, grid: WarpGrid, a_rsrc
    ):
        return _setup_input_footprint_cache(b, spec, a_rsrc, grid)

    def load_a_tile_from_cache(
        b: IRBuilder,
        conv_spec_: ImplicitGemmConvSpec,
        k_off: Value,
        a_dst: Value,
        grid: WarpGrid,
        cache,
    ) -> None:
        if spec.direct_conv0_from_input_cache:
            return
        _load_conv0_a_tile_from_input_cache(
            b, spec, conv_spec_, k_off, a_dst, grid, cache
        )

    def load_a_operand_from_cache(
        b: IRBuilder,
        conv_spec_: ImplicitGemmConvSpec,
        row: Value,
        k_off: Value,
        col_base: Value,
        frag_len: int,
        grid: WarpGrid,
        cache,
    ) -> Value:
        return _load_conv0_a_operand_from_input_cache(
            b, spec, row, k_off, col_base, frag_len, cache
        )

    def epilogue_override(
        b: IRBuilder,
        conv_spec_: ImplicitGemmConvSpec,
        accs: Sequence[Value],
        grid: WarpGrid,
        y_rsrc: Value,
        w1_rsrc,
    ) -> None:
        # Barrier-merge: the conv0 cshuffle stage (writes DeepFusionC_smem) and
        # the W1 load (writes W1_smem) target disjoint LDS tiles, and the conv1
        # MFMA below reads both. Emit each producer without its own barrier and
        # gate the consumer on a single block-wide barrier. This also lets the
        # W1 global loads overlap the conv0 cshuffle LDS stores (a free partial
        # W1-hoist) instead of being serialized behind a redundant barrier.
        c_smem = _stage_accumulators_to_cshuffle_lds(
            b, conv_spec_, accs, grid, sync=False
        )
        w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
        b.sync()
        # VALU opt: ReLU/bias/clamp/(scale>=0) are monotonic, so the conv1
        # epilogue commutes with maxpool. Defer it past the pool to apply it once
        # per pooled pixel instead of per conv1 accumulator element (~4x fewer
        # fmax for 2x2 pool). conv1 output is consumed only by the pool, so this
        # is exact.
        defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
        conv1_accs = _emit_conv1_1x1_mfma(
            b, spec, conv_spec_, c_smem, w1_smem, grid, defer_epilogue=defer
        )
        deferred_epi = spec.conv1_epilogue if defer else None
        if _maxpool_is_intra_lane(spec, grid):
            # Handoff #2 eliminated: each lane's vec<16> conv1 accumulator already
            # holds the 4 pool windows it owns (intra-lane, no shuffle), so reduce
            # straight to global output -- skips the second cshuffle staging LDS
            # store, its barrier, and the gather-side reads/converts.
            _emit_inline_maxpool_from_registers(
                b, spec, conv1_accs, y_rsrc, grid, epilogue=deferred_epi
            )
        else:
            conv1_smem = _stage_accumulators_to_cshuffle_lds(
                b, conv_spec_, conv1_accs, grid
            )
            _emit_inline_maxpool_from_cshuffle(
                b, spec, conv1_smem, y_rsrc, grid, epilogue=deferred_epi
            )

    return build_implicit_gemm_conv(
        conv_spec,
        arch=arch,
        extra_params=extra_params,
        m_index_fn=m_index_fn,
        a_mhw_index_fn=a_mhw_index_fn,
        input_cache_setup=(
            setup_input_cache
            if (spec.cache_input_footprint or spec.direct_conv0_from_input_cache)
            else None
        ),
        a_load_override=(
            load_a_tile_from_cache
            if (spec.cache_input_footprint or spec.direct_conv0_from_input_cache)
            else None
        ),
        a_operand_override=load_a_operand_from_cache
        if spec.direct_conv0_from_input_cache
        else None,
        epilogue_override=epilogue_override,
    )
