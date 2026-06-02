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
from typing import Sequence, Tuple

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
from ...helpers.mfma_gemm_inner import decode_mfma_lanes
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
    tile_k: int = 32
    pool_tile_h: int = 4
    pool_tile_w: int = 8
    warp_m: int = 2
    warp_n: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16
    acc_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)
    conv1_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue(relu=True)

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
            self.acc_epilogue.tag(),
            self.conv1_epilogue.tag(),
            "cshuffle_conv1_pool",
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
            pipeline="mem",
            epilogue="cshuffle",
            acc_epilogue=self.acc_epilogue,
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
    if c.K > spec.tile_k:
        return False, (
            f"v1 1x1 conv requires conv0 channels K0={c.K} <= tile_k={spec.tile_k}"
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
) -> Value:
    """Publish MFMA accumulators to a row-major ``[tile_m, tile_n]`` LDS tile."""

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

    b.sync()
    return c_smem


def _load_conv1_weights_to_lds(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    w1_rsrc: Value,
    grid: WarpGrid,
) -> Value:
    """Load W1[K1, K0] into a padded row-major LDS tile."""

    w1_smem = b.smem_alloc(F16, [spec.tile_n, spec.tile_k], name_hint="W1_smem")
    loader = CoalescedTileLoader(
        tile_rows=spec.tile_n,
        tile_cols=spec.tile_k,
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
    b.sync()
    return w1_smem


def _masked_smem_frag_f16(
    b: IRBuilder,
    smem: Value,
    row: Value,
    col_base: Value,
    frag_len: int,
    valid_k: int,
) -> Value:
    """Load a f16 fragment, zeroing lanes beyond the logical K extent."""

    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))
    elems = []
    c_valid_k = b.const_i32(valid_k)
    for i in range(frag_len):
        col = b.add(col_base, b.const_i32(i))
        raw = b.vec_extract(b.smem_load_vN_f16(smem, row, col, n=1), 0)
        ok = b.cmp_lt(col, c_valid_k)
        elems.append(b.select(ok, raw, zero_h))
    return b.vec_pack(elems, elems[0].type)


def _emit_conv1_1x1_mfma(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    conv_spec: ImplicitGemmConvSpec,
    c0_smem: Value,
    w1_smem: Value,
    grid: WarpGrid,
) -> Sequence[Value]:
    """Compute conv1 as a 1x1 GEMM over staged conv0 activations."""

    atom = conv_spec.atom
    decoded = decode_mfma_lanes(b, atom, grid.lane)
    m_in_atom = decoded.m_in_atom
    n_in_atom = decoded.n_in_atom
    k_blk = decoded.k_blk
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = conv_spec.k_atoms_per_tile_k
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    accs = [b.zero_vec_f32(atom.c_per_lane) for _ in range(mfmas_m * mfmas_n)]

    for kk in range(k_atoms):
        col_base = b.add(
            b.mul(k_blk, b.const_i32(atom.a_per_lane)),
            b.const_i32(kk * conv_spec.warp_tile_k),
        )
        a_rows = []
        for mi in range(mfmas_m):
            a_row = b.add(
                warp_m_off,
                b.add(b.const_i32(mi * atom.m), m_in_atom),
            )
            a_rows.append(
                _masked_smem_frag_f16(
                    b,
                    c0_smem,
                    a_row,
                    col_base,
                    atom.a_per_lane,
                    spec.problem.conv.K,
                )
            )

        b_cols = []
        for ni in range(mfmas_n):
            b_row = b.add(
                warp_n_off,
                b.add(b.const_i32(ni * atom.n), n_in_atom),
            )
            b_cols.append(
                _masked_smem_frag_f16(
                    b,
                    w1_smem,
                    b_row,
                    col_base,
                    atom.b_per_lane,
                    spec.problem.conv.K,
                )
            )

        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = atom.emit(b, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1

    return _apply_accumulator_epilogue(b, spec.conv1_epilogue, accs)


def _emit_inline_maxpool_from_cshuffle(
    b: IRBuilder,
    spec: Gfx950DeepFusedConvPoolSpec,
    c_smem: Value,
    y_rsrc: Value,
    grid: WarpGrid,
) -> None:
    """Reduce the staged conv tile into final pooled NHWK output."""

    p = spec.problem
    out_k = p.conv1_channels
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    total = spec.pool_tile_h * spec.pool_tile_w * out_k
    elems_per_thread = (total + spec.block_size - 1) // spec.block_size
    c_total = b.const_i32(total)
    c_k = b.const_i32(out_k)
    c_pool_tile_w = b.const_i32(spec.pool_tile_w)
    c_conv_tile_w = b.const_i32(conv_tile_w)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    neg_inf = b.const_f32(-3.4028234663852886e38)
    block_pool_h = b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h))
    block_pool_w = b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w))

    for e in range(elems_per_thread):
        local_idx = b.add(b.mul(b.const_i32(e), b.const_i32(spec.block_size)), grid.tid)
        local_in_range = b.cmp_lt(local_idx, c_total)
        safe_local_idx = b.select(local_in_range, local_idx, b.const_i32(0))

        k = b.mod(safe_local_idx, c_k)
        t0 = b.div(safe_local_idx, c_k)
        local_pwo = b.mod(t0, c_pool_tile_w)
        t1 = b.div(t0, c_pool_tile_w)
        local_pho = t1
        global_pho = b.add(block_pool_h, local_pho)
        global_pwo = b.add(block_pool_w, local_pwo)
        in_range = local_in_range

        acc = neg_inf
        for yy in range(2):
            local_conv_h = b.add(b.mul(local_pho, b.const_i32(2)), b.const_i32(yy))
            for xx in range(2):
                local_conv_w = b.add(b.mul(local_pwo, b.const_i32(2)), b.const_i32(xx))
                conv_m_local = b.add(b.mul(local_conv_h, c_conv_tile_w), local_conv_w)
                v_h = b.vec_extract(b.smem_load_vN_f16(c_smem, conv_m_local, k, n=1), 0)
                acc = b.fmax(acc, b.cast_to_f32(v_h))

        y_h = b.trunc_f32_to_f16(acc)
        y_off_elems = b.add(
            b.mul(b.add(b.mul(global_pho, b.const_i32(p.pool_wo)), global_pwo), c_k),
            k,
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
        p = spec.problem
        c = p.conv
        conv_tile_w = spec.pool_tile_w * p.pool_stride_w
        local_h = b.div(row, b.const_i32(conv_tile_w))
        local_w = b.mod(row, b.const_i32(conv_tile_w))
        global_h = b.add(
            b.mul(b.block_id_y(), b.const_i32(spec.pool_tile_h * p.pool_stride_h)),
            local_h,
        )
        global_w = b.add(
            b.mul(b.block_id_z(), b.const_i32(spec.pool_tile_w * p.pool_stride_w)),
            local_w,
        )
        return b.add(b.mul(global_h, b.const_i32(c.Wo)), global_w)

    def epilogue_override(
        b: IRBuilder,
        conv_spec_: ImplicitGemmConvSpec,
        accs: Sequence[Value],
        grid: WarpGrid,
        y_rsrc: Value,
        w1_rsrc,
    ) -> None:
        c_smem = _stage_accumulators_to_cshuffle_lds(b, conv_spec_, accs, grid)
        w1_smem = _load_conv1_weights_to_lds(b, spec, w1_rsrc, grid)
        conv1_accs = _emit_conv1_1x1_mfma(b, spec, conv_spec_, c_smem, w1_smem, grid)
        conv1_smem = _stage_accumulators_to_cshuffle_lds(
            b, conv_spec_, conv1_accs, grid
        )
        _emit_inline_maxpool_from_cshuffle(b, spec, conv1_smem, y_rsrc, grid)

    return build_implicit_gemm_conv(
        conv_spec,
        arch=arch,
        extra_params=extra_params,
        m_index_fn=m_index_fn,
        epilogue_override=epilogue_override,
    )
