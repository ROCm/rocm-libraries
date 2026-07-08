# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Family-agnostic deep-fused conv + maxpool prototype (MFMA + WMMA).

This is the arch-parametric home of the deep-fusion prototype that proves the
single-kernel dataflow for deeper fusion:

    implicit-GEMM conv0 -> accumulator epilogue -> LDS C-shuffle
    -> 1x1 conv1 -> LDS C-shuffle -> maxpool -> Y

The conv0/conv1/pool glue is authored **once** and driven by the resolved
:class:`~rocke.core.arch.MmaOp`: every place that used to hardcode the MFMA
atom (accumulator scatter, conv1 operand loads, MMA emit, per-lane fragment
widths) now goes through ``op`` -- ``op.{a,b,c}_layout().coord(...)``,
``op.{a,b,c}_frag_len``, ``op.m/n``, and the target-neutral
:meth:`IRBuilder.mma`. The same body therefore emits the wave64 MFMA 32x32x16
path (gfx950) and the wave32 WMMA 16x16x16 path (gfx1201) with no per-family
branching in the numeric core. Block size derives from ``wave_size`` so the
cooperative loaders and the maxpool gather tile correctly on both.

Thin per-arch shims (``instances/gfx950`` / ``instances/gfx1201``) select the
warp-tile geometry + wave size and re-export the builder under their public
names. The MFMA-32x32 intra-lane maxpool fast path is kept here, geometry-gated,
so it stays a pure perf optimization that naturally disables itself for WMMA.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ...core.ir import F16, I32, IRBuilder, PtrType, Value
from ...runtime.hip_module import Runtime
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
from ...helpers.mfma_gemm_inner import load_smem_frag_contiguous_f16
from .conv_implicit_gemm import (
    ConvAccumulatorEpilogue,
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
    is_valid_spec as is_valid_conv_spec,
    _apply_accumulator_epilogue,
    _resolve_conv_op,
)
from .manifest_runner.utils import as_u8_buffer, nbytes, require_numpy

__all__ = [
    "FusedConvPoolProblem",
    "DeepFusedConvPoolSpec",
    "make_deep_fused_conv_pool_spec",
    "is_valid_spec",
    "deep_fused_conv_pool_signature",
    "deep_fused_conv_pool_grid",
    "run_deep_fused_conv_pool_fp16_manifest_problem",
    "build_deep_fused_conv_pool",
]


@dataclass(frozen=True)
class FusedConvPoolProblem:
    """Shape contract for the conv0 -> conv1 -> maxpool prototype."""

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
class DeepFusedConvPoolSpec:
    """One concrete deep-fusion configuration (arch-parametric).

    ``wave_size`` + ``warp_tile_*`` are the only fields a per-arch shim must set
    differently: gfx950 picks (64, 32x32x16) and gfx1201 picks (32, 16x16x16).
    ``block_size`` derives from ``wave_size`` so the cooperative loaders and the
    maxpool gather tile correctly on both ISAs.
    """

    problem: FusedConvPoolProblem
    name: str = "rocke_deep_fused_conv_pool"
    tile_m: int = 128
    tile_n: int = 32
    tile_k: int = 16
    conv1_tile_k: int = 0
    pool_tile_h: int = 4
    pool_tile_w: int = 8
    warp_m: int = 2
    warp_n: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16
    wave_size: int = 64
    pipeline: str = "mem"
    async_dma: bool = False
    unroll_k: bool = False
    acc_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)
    conv1_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)
    cache_input_footprint: bool = False
    direct_conv0_from_input_cache: bool = False

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

    @property
    def effective_conv1_tile_k(self) -> int:
        return self.tile_k if self.conv1_tile_k <= 0 else self.conv1_tile_k

    def kernel_name(self) -> str:
        conv1_k_part = (
            f"c1k{self.effective_conv1_tile_k}"
            if self.effective_conv1_tile_k != self.tile_k
            else ""
        )
        return kernel_name_join(
            self.name,
            self.problem.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            conv1_k_part,
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
        # ``epilogue_override`` owns the entire write-back for both families, so
        # the built-in epilogue is dead code; ``epilogue`` here only selects the
        # ``is_valid`` gate. The WMMA conv gate accepts only ``default`` while
        # the MFMA path historically used ``cshuffle`` -- keep MFMA byte-exact.
        epilogue = "default" if self.wave_size == 32 else "cshuffle"
        conv_name = kernel_name_join(
            self.name,
            (
                f"c1k{self.effective_conv1_tile_k}"
                if self.effective_conv1_tile_k != self.tile_k
                else ""
            ),
        )
        return ImplicitGemmConvSpec(
            problem=self.problem.conv,
            name=conv_name,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            wave_size=self.wave_size,
            pipeline=self.pipeline,
            epilogue=epilogue,
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
    conv1_tile_k: int = 0,
    warp_m: int = 2,
    warp_n: int = 1,
    warp_tile_m: int = 32,
    warp_tile_n: int = 32,
    warp_tile_k: int = 16,
    wave_size: int = 64,
    name: str = "rocke_deep_fused_conv_pool",
    pipeline: str = "mem",
    unroll_k: bool = False,
    async_dma: bool = False,
    cache_input_footprint: bool = False,
    direct_conv0_from_input_cache: bool = False,
) -> DeepFusedConvPoolSpec:
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
        Y=r,
        X=s,
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
    return DeepFusedConvPoolSpec(
        problem=problem,
        name=name,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        conv1_tile_k=conv1_tile_k,
        pool_tile_h=pool_tile_h,
        pool_tile_w=pool_tile_w,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_m=warp_tile_m,
        warp_tile_n=warp_tile_n,
        warp_tile_k=warp_tile_k,
        wave_size=wave_size,
        pipeline=pipeline,
        unroll_k=unroll_k,
        async_dma=async_dma,
        cache_input_footprint=cache_input_footprint,
        direct_conv0_from_input_cache=direct_conv0_from_input_cache,
        acc_epilogue=ConvAccumulatorEpilogue(relu=True),
    )


def is_valid_spec(
    spec: DeepFusedConvPoolSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the experimental fused kernel.

    Arch-neutral: any arch whose underlying conv spec validates (MFMA on CDNA,
    WMMA on RDNA4) is accepted; the conv gate itself enforces the per-family
    atom/pipeline/epilogue constraints.
    """

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
    conv1_tile_k = spec.effective_conv1_tile_k
    if conv1_tile_k <= 0:
        return False, f"conv1_tile_k must be positive (got {conv1_tile_k})"
    if conv1_tile_k % spec.warp_tile_k:
        return False, "conv1_tile_k must be divisible by warp_tile_k"
    if conv1_tile_k < spec.warp_tile_k:
        return False, "conv1_tile_k must be at least one warp_tile_k"
    return True, "ok"


def deep_fused_conv_pool_signature(spec: DeepFusedConvPoolSpec):
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
    spec: DeepFusedConvPoolSpec,
) -> Tuple[int, int, int]:
    p = spec.problem
    return (1, p.pool_ho // spec.pool_tile_h, p.pool_wo // spec.pool_tile_w)


def _stage_accumulators_to_cshuffle_lds(
    b: IRBuilder,
    op,
    accs: Sequence[Value],
    grid: WarpGrid,
    *,
    sync: bool = True,
) -> Value:
    """Publish MMA accumulators to a row-major ``[tile_m, tile_n]`` LDS tile.

    Fully ``op``-driven: the per-lane fragment width (``op.c_frag_len``) and the
    slot -> (row, col) scatter (``op.c_layout().coord``) come from the resolved
    MMA op, so this stages both the MFMA column-distributed vec<16> accumulator
    (gfx950) and the WMMA vec<8> accumulator (gfx1201) with one body. For MFMA
    ``op.c_layout()`` is byte-identical to ``MfmaAtom.lane_to_output``, so the
    gfx950 store path is unchanged.

    ``sync=False`` skips the trailing ``b.sync()`` so the caller can batch this
    producer barrier with another disjoint-LDS producer (e.g. the W1 load) into
    a single block-wide barrier before the shared consumer reads both tiles.
    """

    c_frag_len = op.c_frag_len
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

    dist = _cshuffle_acc_distribution(c_frag_len)
    traits = LoadStoreTraits(distribution=dist, vector_dim_y=1, scalar_per_vector=1)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    c_map = op.c_layout()

    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[mi * mfmas_n + ni]
            acc_h = b.vec_trunc_f32_to_f16(acc)
            dt = make_static_distributed_tensor(dist, dtype=F16)
            for i in range(c_frag_len):
                dt.set([i, 0], b.vec_extract(acc_h, i))

            tile_m_base = b.add(warp_m_off, b.const_i32(mi * op.m))
            tile_n_base = b.add(warp_n_off, b.const_i32(ni * op.n))

            def coord_fn(b_, y_base, _k, *, _mb=tile_m_base, _nb=tile_n_base):
                i = int(y_base[0])
                row_in_atom, col_in_atom = c_map.coord(b_, grid.lane, i)
                return [b_.add(_mb, row_in_atom), b_.add(_nb, col_in_atom)]

            store_tile_cshuffle(b, c_window, dt, traits=traits, coord_fn=coord_fn)

    if sync:
        b.sync()
    return c_smem


def _load_conv1_weights_to_lds(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
    w1_rsrc: Value,
    grid: WarpGrid,
    *,
    sync: bool = True,
) -> Value:
    """Load W1[K1, K0] into a padded row-major LDS tile.

    ``sync=False`` skips the trailing ``b.sync()`` so the caller can fold this
    barrier into a single block-wide barrier shared with the conv0 cshuffle
    stage (the two write disjoint LDS tiles, so one barrier suffices before the
    conv1 MMA reads both).
    """

    w1_smem = b.smem_alloc(F16, [spec.tile_n, spec.problem.conv.K], name_hint="W1_smem")
    # W1 is tiny (K1 x K0, normally 32x32), so larger WMMA CTAs can have more
    # threads than 8-wide load chunks. Pick the widest vector width that gives
    # every thread a uniform chunk instead of hard-coding b128 loads.
    loader = CoalescedTileLoader.from_tile(
        tile_rows=spec.tile_n,
        tile_cols=spec.problem.conv.K,
        block_size=spec.block_size,
        max_vec=8,
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


def _can_use_specialized_conv0_a_loader(spec: DeepFusedConvPoolSpec) -> bool:
    """Whether the target-shape conv0 A load can bypass TensorDescriptor math."""

    c = spec.problem.conv
    return (
        spec.wave_size == 32
        and not spec.cache_input_footprint
        and not spec.direct_conv0_from_input_cache
        and c.N == 1
        and c.C == 8
        and c.Y == 3
        and c.X == 3
        and c.sH == 1
        and c.sW == 1
        and c.pH == 1
        and c.pW == 1
        and c.dH == 1
        and c.dW == 1
    )


def _load_conv0_a_tile_specialized(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
    conv_spec: ImplicitGemmConvSpec,
    k_off: Value,
    a_dst: Value,
    grid: WarpGrid,
    a_rsrc: Value,
) -> None:
    """Specialized NHWC conv0 A loader for the fixed deep-fusion target.

    This replaces the generic TensorDescriptor path for
    ``N=1,C=8,Y=X=3,stride=1,pad=1``. It keeps the shared coalesced loader's
    thread distribution but computes the A offset directly:

    ``row -> (local_oh, local_ow)``, ``kg -> (y, x, c)``, then
    ``((global_h + y - pH) * Wi + global_w + x - pW) * C + c``.
    """

    p = spec.problem
    c = p.conv
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_wi = b.const_i32(c.Wi)
    c_c = b.const_i32(c.C)
    c_sc = b.const_i32(c.X * c.C)  # 24 for the target shape.
    c_k_gemm = b.const_i32(c.K_gemm)

    h_base = b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h * p.pool_stride_h))
    w_base = b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w * p.pool_stride_w))

    loader = CoalescedTileLoader(
        tile_rows=spec.tile_m,
        tile_cols=spec.tile_k,
        block_size=spec.block_size,
        load_vec=CoalescedTileLoader.choose_vec(
            tile_rows=spec.tile_m,
            tile_cols=spec.tile_k,
            block_size=spec.block_size,
            max_vec=8,
        ),
    )

    def descriptor(b_: IRBuilder, row: Value, col: Value):
        kg = b_.add(k_off, col)
        if conv_tile_w > 0 and (conv_tile_w & (conv_tile_w - 1)) == 0:
            shift = (conv_tile_w - 1).bit_length()
            local_oh = b_.lshr(row, b_.const_i32(shift))
            local_ow = b_.land(row, b_.const_i32(conv_tile_w - 1))
        else:
            local_oh = b_.div(row, c_conv_tile_w)
            local_ow = b_.mod(row, c_conv_tile_w)

        r = b_.div(kg, c_sc)
        rem = b_.mod(kg, c_sc)
        s_col = b_.lshr(rem, b_.const_i32(3))
        ci = b_.land(rem, b_.const_i32(7))

        hi = b_.sub(b_.add(b_.add(h_base, local_oh), r), b_.const_i32(c.pH))
        wi = b_.sub(b_.add(b_.add(w_base, local_ow), s_col), b_.const_i32(c.pW))
        h_ok = b_.land(
            b_.cmp_ge(hi, b_.const_i32(0)), b_.cmp_lt(hi, b_.const_i32(c.Hi))
        )
        w_ok = b_.land(b_.cmp_ge(wi, b_.const_i32(0)), b_.cmp_lt(wi, c_wi))
        kg_ok = b_.cmp_lt(kg, c_k_gemm)
        valid = b_.land(kg_ok, b_.land(h_ok, w_ok))
        off_elems = b_.add(b_.mul(b_.add(b_.mul(hi, c_wi), wi), c_c), ci)
        return off_elems, valid

    loader.load(b, tid=grid.tid, smem_dst=a_dst, descriptor=descriptor, rsrc=a_rsrc)


def _setup_input_footprint_cache(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
    a_rsrc: Value,
    grid: WarpGrid,
) -> Value:
    """Load the unique conv0 input footprint for this pooled-output tile."""

    p = spec.problem
    c = p.conv
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    foot_h = conv_tile_h + (c.Y - 1) * c.dH
    foot_w = conv_tile_w + (c.X - 1) * c.dW
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
    spec: DeepFusedConvPoolSpec,
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
    foot_w = conv_tile_w + (c.X - 1) * c.dW
    total = spec.tile_m * spec.tile_k
    elems_per_thread = (total + spec.block_size - 1) // spec.block_size
    c_total = b.const_i32(total)
    c_tile_k = b.const_i32(spec.tile_k)
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_sc = b.const_i32(c.X * c.C)
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
    spec: DeepFusedConvPoolSpec,
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
    foot_w = conv_tile_w + (c.X - 1) * c.dW
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_sc = b.const_i32(c.X * c.C)
    c_c = b.const_i32(c.C)
    c_foot_w = b.const_i32(foot_w)
    c_k_gemm = b.const_i32(c.K_gemm)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    # VALU opt 1: hoist row-dependent coordinates out of the per-element loop.
    local_oh = b.div(row, c_conv_tile_w)
    local_ow = b.mod(row, c_conv_tile_w)

    oh_base = b.mul(local_oh, b.const_i32(c.sH))
    ow_base = b.mul(local_ow, b.const_i32(c.sW))

    elems = []
    for i in range(frag_len):
        kg = b.add(k_off, b.add(col_base, b.const_i32(i)))
        kg_ok = b.cmp_lt(kg, c_k_gemm)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)

        # VALU opt 2: strength-reduce div/mod by C=8 (power-of-2) to shift/mask.
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


def _epilogue_is_relu_only(epi: ConvAccumulatorEpilogue) -> bool:
    """Whether the deferred epilogue is exactly ``relu(x)``."""

    return (
        epi.bias == 0.0
        and epi.scale == 1.0
        and epi.relu
        and epi.clamp_min is None
        and epi.clamp_max is None
    )


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


def _emit_conv1_1x1(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
    conv_spec: ImplicitGemmConvSpec,
    op,
    c0_smem: Value,
    w1_smem: Value,
    grid: WarpGrid,
    defer_epilogue: bool = False,
) -> Sequence[Value]:
    """Compute conv1 as a 1x1 GEMM over staged conv0 activations (op-driven).

    Single body for both families: the operand lane->(mn, k) decomposition comes
    from ``op.a_layout()/b_layout()`` (the MFMA 32x32x16 and WMMA 16x16x16 maps
    are both K-contiguous, so a lane's fragment is ``smem[row, k_base:k_base+L]``),
    the fragment width is ``op.{a,b}_frag_len``, and the matmul is emitted via the
    target-neutral :meth:`IRBuilder.mma`. The K-tail mask (``valid_k=K0``) is
    handled by :func:`load_smem_frag_contiguous_f16`, which ``_emit_frag_smem_load``
    cannot do -- needed for shapes where ``tile_k`` overhangs ``K0`` (toy K0=8).

    When ``defer_epilogue`` is set the raw fp32 accumulators are returned and the
    caller applies ``spec.conv1_epilogue`` after the maxpool reduction (valid only
    when the epilogue is pool-deferrable).
    """

    a_map = op.a_layout()
    b_map = op.b_layout()
    a_mn_in_atom, a_k_base = a_map.coord(b, grid.lane, 0)
    b_k_base, b_mn_in_atom = b_map.coord(b, grid.lane, 0)
    a_frag = op.a_frag_len
    b_frag = op.b_frag_len
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    K0 = spec.problem.conv.K
    conv1_tile_k = spec.effective_conv1_tile_k
    k_atoms = conv1_tile_k // conv_spec.warp_tile_k
    k_chunks = (K0 + conv1_tile_k - 1) // conv1_tile_k
    # The valid_k mask only guards a K tail. When the tiling covers K exactly it
    # is statically dead, so we skip it and issue wide vector ds_reads.
    needs_mask = k_chunks * conv1_tile_k != K0
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    accs = [b.zero_vec_f32(op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]

    for k_chunk in range(k_chunks):
        chunk_base = k_chunk * conv1_tile_k
        for kk in range(k_atoms):
            tile_off = b.const_i32(chunk_base + kk * conv_spec.warp_tile_k)
            a_col_base = b.add(a_k_base, tile_off)
            b_col_base = b.add(b_k_base, tile_off)
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off,
                    b.add(b.const_i32(mi * op.m), a_mn_in_atom),
                )
                a_rows.append(
                    load_smem_frag_contiguous_f16(
                        b,
                        c0_smem,
                        a_row,
                        a_col_base,
                        a_frag,
                        needs_mask=needs_mask,
                        valid_k=K0,
                    )
                )

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off,
                    b.add(b.const_i32(ni * op.n), b_mn_in_atom),
                )
                b_cols.append(
                    load_smem_frag_contiguous_f16(
                        b,
                        w1_smem,
                        b_row,
                        b_col_base,
                        b_frag,
                        needs_mask=needs_mask,
                        valid_k=K0,
                    )
                )

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                    flat += 1

    if defer_epilogue:
        return list(accs)
    return _apply_accumulator_epilogue(b, spec.conv1_epilogue, accs)


def _emit_inline_maxpool_from_cshuffle(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
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

    # Tile the gather by (window, k-block). The 2x2 maxpool corner rows depend
    # only on the pooled window, not on channel k, so processing a contiguous run
    # of ``kvec`` channels per thread amortizes the window decode + 4 corner
    # addresses across kvec channels and folds the per-channel scalar
    # ``ds_read_u16`` into a single wide ``ds_read_b{32,64}``. k is the contiguous
    # (column) dim of the row-major [tile_m, tile_n] cshuffle LDS tile, so a
    # kvec-wide read stays within one row. Pick the largest valid width that
    # divides out_k while keeping >= half the block's threads active.
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
        halves = []
        for j in range(kvec):
            acc = accs[j]
            if epilogue is not None:
                acc = _apply_epilogue_scalar(b, epilogue, acc)
            halves.append(b.trunc_f32_to_f16(acc))
        # The ``kvec`` channels are contiguous in NHWK Y, and ``y_base_elems`` is
        # a multiple of ``kvec`` (k0 = kb*kvec, out_k % kvec == 0), so the byte
        # offset is kvec*2-aligned -- pack and emit one coalesced wide store
        # (b32/b64/b128) instead of kvec scalar b16 stores (runbook 6.2). OOB
        # vec_idx steers the base offset to the sentinel; the buffer rsrc drops it.
        base_off_bytes = b.mul(y_base_elems, c_half_bytes)
        safe_base = b.select(in_range, base_off_bytes, oob_sentinel)
        if kvec >= 2:
            y_vec = b.vec_pack(halves, F16)
            b.buffer_store_vN_f16(y_rsrc, safe_base, b.const_i32(0), y_vec, kvec // 2)
        else:
            b.buffer_store_f16(y_rsrc, safe_base, b.const_i32(0), halves[0])


def _maxpool_is_intra_lane(spec: DeepFusedConvPoolSpec, grid: WarpGrid) -> bool:
    """Whether the conv1->maxpool handoff can stay register-resident (no LDS).

    MFMA-32x32 fast path: with a single 32x32 MFMA atom per warp and
    ``warp_n==1``, each lane owns a vec<16> accumulator whose 16 slots tile a 4x4
    conv-spatial block for one channel (= ``lane % 32``). For a 2x2 stride-2 pool
    that block is exactly 2x2=4 pool windows, all four corners of every window
    living in the *same lane's* accumulator -- so the maxpool reduces purely
    intra-lane with no cross-lane shuffle and no cshuffle LDS staging.

    The gate checks the exact 32x32 geometry, so it naturally returns ``False``
    for WMMA (warp_tile 16x16), which falls through to the layout-agnostic
    cshuffle-LDS gather path.
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


def _maxpool_is_intra_lane_wmma(
    spec: DeepFusedConvPoolSpec, grid: WarpGrid, op
) -> bool:
    """WMMA analogue of :func:`_maxpool_is_intra_lane`: can the conv1->maxpool
    handoff stay register-resident on RDNA4 (wave32, 16x16x16)?

    The gfx12 acc layout gives lane ``l`` the fragment ``row=(l//16)*8+slot,
    col=l%16`` for ``slot`` in 0..7 -- i.e. a lane owns 8 consecutive conv-M rows
    at one channel. With ``conv_tile_w == 16`` (one atom wide) the conv-M index is
    ``conv_h*16 + conv_w``, so atom ``mi`` is conv row ``warp_m_idx*2 + mi`` and a
    lane's 8 slots are conv columns ``(l//16)*8 .. +7`` (one half of the row).

    For a 2x2 stride-2 pool with ``warp_m == pool_tile_h`` and
    ``mfmas_per_warp_m == 2``, each warp owns exactly one pool row
    (``pho == warp_m_idx``, conv rows ``2*pho`` and ``2*pho+1`` = atoms mi=0,1).
    A window's two conv columns ``2*pwo, 2*pwo+1`` (2*pwo even) never cross the
    8-column lane-half boundary, so all four corners live in the *same lane*
    across the two adjacent m-tile accs -- intra-lane, no cshuffle LDS staging.
    """
    p = spec.problem
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    return (
        grid.wave_size == 32
        and op.m == 16
        and op.n == 16
        and grid.warp_tile_m == 16
        and grid.warp_tile_n == 16
        and grid.mfmas_per_warp_m == 2
        and grid.warp_n == 1
        and p.pool_stride_h == 2
        and p.pool_stride_w == 2
        and conv_tile_w == 16
        and grid.warp_m == spec.pool_tile_h
        and p.conv1_channels <= 32
    )


def _emit_wmma_maxpool_from_registers(
    b: IRBuilder,
    spec: DeepFusedConvPoolSpec,
    conv1_accs: Sequence[Value],
    y_rsrc: Value,
    grid: WarpGrid,
    op,
    epilogue: Optional[ConvAccumulatorEpilogue] = None,
) -> None:
    """RDNA4 register-resident maxpool (no conv1->maxpool LDS handoff).

    Gated by :func:`_maxpool_is_intra_lane_wmma`. Each lane reduces the four
    corners of every pool window it owns straight from its conv1 WMMA
    accumulators: ``conv1_accs[mi*mfmas_n + ni]`` slot ``s`` holds conv pixel
    (conv_h = warp_m_idx*2 + mi, conv_w = (lane//16)*8 + s) at channel
    ``ni*16 + lane%16``. A lane owns conv columns of one row-half (8 cols = 4 pool
    windows); ``pho == warp_m_idx`` is fixed per warp. When ``epilogue`` is given
    it is applied once per pooled fp32 result (the deferred conv1 epilogue).
    """

    p = spec.problem
    out_k = p.conv1_channels
    mfmas_n = grid.mfmas_per_warp_n

    col = b.mod(grid.lane, b.const_i32(16))  # channel within an n-atom
    half = b.div(grid.lane, b.const_i32(16))  # which 8-col half of the conv row
    block_pool_h = b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h))
    block_pool_w = b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w))
    gpho = b.add(block_pool_h, grid.warp_m_idx)  # pho == warp_m_idx
    # pwo = half*4 + w_local; conv cols 2*pwo,2*pwo+1 -> slots 2*w_local,2*w_local+1.
    pwo_base = b.add(block_pool_w, b.mul(half, b.const_i32(4)))

    oob_sentinel = b.const_i32((1 << 31) - 1)
    c_pool_wo = b.const_i32(p.pool_wo)
    c_out_k = b.const_i32(out_k)
    c_half_bytes = b.const_i32(2)
    row_off = b.mul(gpho, c_pool_wo)  # gpho*pool_wo, shared across windows

    for w_local in range(4):
        gpwo = b.add(pwo_base, b.const_i32(w_local))
        s0 = 2 * w_local
        s1 = 2 * w_local + 1
        pix_off = b.mul(b.add(row_off, gpwo), c_out_k)  # (gpho*pool_wo+gpwo)*out_k
        for ni in range(mfmas_n):
            acc_top = conv1_accs[0 * mfmas_n + ni]  # mi=0 -> conv row 2*pho
            acc_bot = conv1_accs[1 * mfmas_n + ni]  # mi=1 -> conv row 2*pho+1
            top0 = b.vec_extract(acc_top, s0)
            top1 = b.vec_extract(acc_top, s1)
            bot0 = b.vec_extract(acc_bot, s0)
            bot1 = b.vec_extract(acc_bot, s1)
            pool3 = b.fmax3(top0, top1, bot0)
            if epilogue is not None and _epilogue_is_relu_only(epilogue):
                acc = b.fmax3(pool3, bot1, b.const_f32(0.0))
            else:
                acc = b.fmax(pool3, bot1)
            if epilogue is not None and not _epilogue_is_relu_only(epilogue):
                acc = _apply_epilogue_scalar(b, epilogue, acc)
            y_h = b.trunc_f32_to_f16(acc)
            channel = b.add(b.const_i32(ni * 16), col)
            in_range = b.cmp_lt(channel, c_out_k)
            y_off_bytes = b.mul(b.add(pix_off, channel), c_half_bytes)
            safe_off = b.select(in_range, y_off_bytes, oob_sentinel)
            b.buffer_store_f16(y_rsrc, safe_off, b.const_i32(0), y_h)


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
    spec: DeepFusedConvPoolSpec,
    conv1_accs: Sequence[Value],
    y_rsrc: Value,
    grid: WarpGrid,
    epilogue: Optional[ConvAccumulatorEpilogue] = None,
) -> None:
    """Reduce the conv1 accumulators directly into pooled NHWK output (MFMA-only).

    Eliminates the conv1->maxpool cshuffle handoff: instead of staging the conv1
    accs to LDS and re-gathering, each lane reduces its own vec<16> accumulator.
    Gated by :func:`_maxpool_is_intra_lane`. When ``epilogue`` is given it is
    applied once per pooled fp32 result (the deferred conv1 epilogue).
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


def build_deep_fused_conv_pool(spec: DeepFusedConvPoolSpec, arch: str = "gfx950"):
    """Build the one-CTA conv0 -> conv1 -> maxpool fused kernel for ``arch``."""

    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid deep fused conv/pool spec for {arch}: {why}")

    conv_spec = spec.conv_spec()
    op = _resolve_conv_op(conv_spec, arch)

    def extra_params(b: IRBuilder) -> Value:
        W1 = b.param(
            "W1", PtrType(F16, "global"), noalias=True, readonly=True, align=16
        )
        W1_bytes = b.param("W1_bytes", I32)
        return make_buffer_resource(b, W1, num_bytes=W1_bytes).rsrc

    def m_index_fn(b: IRBuilder, row: Value, _grid: WarpGrid) -> Value:
        """Conv0 M-index callback: tile-local row -> global (ho, wo) offset.

        VALU opt: strength-reduce div/mod by ``conv_tile_w`` when power-of-2.
        Called per MFMA A fragment in the main K-loop, so a hot path.
        """
        p = spec.problem
        c = p.conv
        conv_tile_w = spec.pool_tile_w * p.pool_stride_w
        c_conv_tile_w = b.const_i32(conv_tile_w)

        if conv_tile_w > 0 and (conv_tile_w & (conv_tile_w - 1)) == 0:
            shift = (conv_tile_w - 1).bit_length()
            local_h = b.lshr(row, b.const_i32(shift))
            local_w = b.land(row, b.const_i32(conv_tile_w - 1))
        else:
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
        consume them without re-deriving them (bypasses the m-flatten and the
        descriptor's m -> (n, ho, wo) magic unmerge). N==1, so n is constant 0.
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

    def setup_specialized_a_loader(
        b: IRBuilder, conv_spec_: ImplicitGemmConvSpec, grid: WarpGrid, a_rsrc
    ):
        return a_rsrc

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

    def load_a_tile_specialized(
        b: IRBuilder,
        conv_spec_: ImplicitGemmConvSpec,
        k_off: Value,
        a_dst: Value,
        grid: WarpGrid,
        a_rsrc,
    ) -> None:
        _load_conv0_a_tile_specialized(b, spec, conv_spec_, k_off, a_dst, grid, a_rsrc)

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
        # MMA below reads both. Emit each producer without its own barrier and
        # gate the consumer on a single block-wide barrier. This also lets the
        # W1 global loads overlap the conv0 cshuffle LDS stores.
        c_smem = _stage_accumulators_to_cshuffle_lds(b, op, accs, grid, sync=False)
        w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid, sync=False)
        b.sync()
        # VALU opt: ReLU/bias/clamp/(scale>=0) are monotonic, so the conv1
        # epilogue commutes with maxpool. Defer it past the pool to apply once
        # per pooled pixel instead of per conv1 acc element (~4x fewer fmax).
        defer = _epilogue_is_pool_deferrable(spec.conv1_epilogue)
        conv1_accs = _emit_conv1_1x1(
            b, spec, conv_spec_, op, c_smem, w1_smem, grid, defer_epilogue=defer
        )
        deferred_epi = spec.conv1_epilogue if defer else None
        if _maxpool_is_intra_lane(spec, grid):
            # Handoff eliminated: each lane's vec<16> conv1 accumulator already
            # holds the 4 pool windows it owns (intra-lane, no shuffle), so reduce
            # straight to global output.
            _emit_inline_maxpool_from_registers(
                b, spec, conv1_accs, y_rsrc, grid, epilogue=deferred_epi
            )
        elif _maxpool_is_intra_lane_wmma(spec, grid, op):
            # RDNA4 analogue: the 2x2 corners live in the same lane across the two
            # adjacent m-tile accs, so skip the cshuffle LDS handoff entirely.
            _emit_wmma_maxpool_from_registers(
                b, spec, conv1_accs, y_rsrc, grid, op, epilogue=deferred_epi
            )
        else:
            conv1_smem = _stage_accumulators_to_cshuffle_lds(b, op, conv1_accs, grid)
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
            else (
                setup_specialized_a_loader
                if _can_use_specialized_conv0_a_loader(spec)
                else None
            )
        ),
        a_load_override=(
            load_a_tile_from_cache
            if (spec.cache_input_footprint or spec.direct_conv0_from_input_cache)
            else (
                load_a_tile_specialized
                if _can_use_specialized_conv0_a_loader(spec)
                else None
            )
        ),
        a_operand_override=(
            load_a_operand_from_cache if spec.direct_conv0_from_input_cache else None
        ),
        epilogue_override=epilogue_override,
    )


def run_deep_fused_conv_pool_fp16_manifest_problem(
    manifest: dict, _shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Manifest-runner problem for the fp16 fused conv0->conv1->pool kernel."""
    np = require_numpy()
    cv = [int(x) for x in manifest["conv"]]
    if len(cv) < 13:
        raise ValueError("conv manifest needs [N,H,W,C,K,R,S,sH,sW,pH,pW,dH,dW]")
    N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW = cv[:13]
    pool = [int(x) for x in manifest["pool"]]
    pool_y, pool_x, pool_sh, pool_sw = pool[:4]
    K1 = int(manifest["conv1"]["K1"])
    _, pool_ho, pool_wo, _ = [int(x) for x in manifest["pool_output_shape"]]

    Ho = (Hi + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (Wi + 2 * pW - dW * (S - 1) - 1) // sW + 1

    seed = int(manifest.get("seed", 123))
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((N, Hi, Wi, C)).astype(np.float32) * 0.25).astype(
        np.float16
    )
    B0 = (rng.standard_normal((K, R, S, C)).astype(np.float32) * 0.25).astype(
        np.float16
    )
    W1 = (rng.standard_normal((K1, K)).astype(np.float32) * 0.25).astype(np.float16)
    Y = np.zeros((N, pool_ho, pool_wo, K1), dtype=np.float16)

    gx, gy, gz = [int(x) for x in manifest["grid_explicit"]]
    grid = (gx, gy, gz)
    block = (int(manifest["threads_per_block"]), 1, 1)
    conv0_flop = N * Ho * Wo * K * R * S * C
    conv1_flop = N * Ho * Wo * K1 * K
    flop = 2.0 * (conv0_flop + conv1_flop)
    bytes_xfer = 2.0 * (A.size + B0.size + W1.size + Y.size)

    def make_args(rt: Runtime):
        A_dev = rt.alloc(nbytes(A))
        B_dev = rt.alloc(nbytes(B0))
        Y_dev = rt.alloc(nbytes(Y))
        W1_dev = rt.alloc(nbytes(W1))
        rt.memcpy_h2d(A_dev, as_u8_buffer(A), nbytes(A))
        rt.memcpy_h2d(B_dev, as_u8_buffer(B0), nbytes(B0))
        rt.memcpy_h2d(W1_dev, as_u8_buffer(W1), nbytes(W1))
        rt.memset(Y_dev, 0, nbytes(Y))
        args = struct.pack(
            "<QQQQiiii",
            A_dev,
            B_dev,
            Y_dev,
            W1_dev,
            nbytes(W1),
            nbytes(A),
            nbytes(B0),
            nbytes(Y),
        )
        return args, (A_dev, B_dev, Y_dev, W1_dev)

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, Y.size
        rt.memcpy_d2h(as_u8_buffer(Y), ptrs[2], nbytes(Y))
        Ap = np.pad(A, ((0, 0), (pH, pH), (pW, pW), (0, 0)))
        C0 = np.zeros((N, Ho, Wo, K), dtype=np.float32)
        for r in range(R):
            for s in range(S):
                row_start = r * dH
                col_start = s * dW
                x = Ap[
                    :,
                    row_start : row_start + Ho * sH : sH,
                    col_start : col_start + Wo * sW : sW,
                    :,
                ].astype(np.float32)
                w = B0[:, r, s, :].astype(np.float32)
                C0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)
        C0 = np.maximum(C0, 0.0).astype(np.float16).astype(np.float32)
        C1 = np.einsum("nhwk,ok->nhwo", C0, W1.astype(np.float32), optimize=True)
        C1 = np.maximum(C1, 0.0).astype(np.float16).astype(np.float32)
        ref = np.empty((N, pool_ho, pool_wo, K1), dtype=np.float32)
        for ho in range(pool_ho):
            for wo in range(pool_wo):
                h0 = ho * pool_sh
                w0 = wo * pool_sw
                patch = C1[:, h0 : h0 + pool_y, w0 : w0 + pool_x, :]
                ref[:, ho, wo, :] = patch.max(axis=(1, 2))
        ref_h = ref.astype(np.float16)
        ref_f32 = ref_h.astype(np.float32)
        tol = 1e-2
        err = np.abs(Y.astype(np.float32) - ref_f32)
        bad = err > tol + tol * np.abs(ref_f32)
        return float(err.max()), int(np.count_nonzero(bad)), Y.size

    return make_args, grid, block, flop, bytes_xfer, check
