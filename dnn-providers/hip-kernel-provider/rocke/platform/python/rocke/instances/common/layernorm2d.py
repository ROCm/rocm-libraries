# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LayerNorm2D forward kernel instance builder (CK Tile ``02_layernorm2d`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/02_layernorm2d``. For
each row of an ``(M, N)`` activation tensor, computes:

    mean[m]    = sum_n(X[m,n]) / N
    var[m]     = sum_n((X[m,n] - mean[m])^2) / N    (stable Welford)
    inv_std[m] = 1 / sqrt(var[m] + eps)
    Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * gamma[n] + beta[n]

The kernel is expressed entirely against the CK Tile-inspired
:class:`rocke.helpers.TensorView` / :class:`rocke.helpers.TileWindow`
abstractions for I/O, :func:`rocke.helpers.io.load_vec_as_f32` /
:func:`pack_f32_to` for dtype-promoted ingest/egress, and
:func:`rocke.helpers.reduction.block_lds_reduce` for the cross-thread
sum. The bare-IR ``smem_alloc`` / ``smem_load_vN_f32`` /
``global_load_vN`` calls that used to dominate this file are gone.

What we cover today:
  - Dtypes ``f16`` / ``bf16`` for X/gamma/beta/Y (compute in f32)
  - Optional save of ``mean`` / ``inv_std`` per row (CK Tile's
    ``save_mean_var`` traits)
  - Single-pass row reduction with a numerically-stable Welford
    ``(mean, M2, count)`` block merge (no ``E[X^2] - E[X]^2``
    catastrophic cancellation for the post-residual |mean| >> sigma
    activations LayerNorm sees in transformer blocks)

Performance shape:
  - One CTA per row, ``block_size`` threads
  - Each thread loads ``elems_per_thread`` f16 / bf16 elements in
    ``vec``-wide chunks; the values are kept in f32 registers so the
    second pass doesn't re-load from HBM
  - One LDS f32 buffer of ``block_size`` words is reused for the
    ``s1`` and ``s2`` LDS-tree reductions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type, store_scalar_from_f32
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    row_norm_needs_two_pass,
    tree_reduce,
    welford_block_reduce_stable,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import pass2_row_chunks, sweep_row_chunks
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class LayerNorm2DSpec:
    """One LayerNorm2D forward instance."""

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_mean_invstd: bool = False
    wave_size: int = 64
    name: str = "rocke_layernorm2d_fwd"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"smv": self.save_mean_invstd},
        )


def is_valid_spec(spec: LayerNorm2DSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one LayerNorm2D config on ``arch``.

    Pure elementwise + twin LDS-tree reductions (no MFMA): the only
    architecture facts that matter are the per-WG LDS capacity and max
    threads/block, both sourced from :class:`rocke.core.arch.ArchTarget`
    so an unknown arch / over-budget ``block_size`` is rejected with a
    structured reason. The three f32 Welford reduction buffers
    (``3 * block_size`` words: mean / M2 / count) fit both gfx942
    (64 KiB) and gfx950 (160 KiB), so gfx950 behavior is unchanged.

    The per-thread register cap
    (:data:`rocke.helpers.reduction.REGISTER_TILE_MAX_ELEMS_PER_THREAD`)
    only bounds the cached *single-pass* path: a row whose
    ``elems_per_thread`` overflows it builds the streaming *two-pass*
    kernel (re-reads X from HBM for pass 2) instead of being rejected, so
    large-N shapes that don't fit in VGPRs are still supported.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    # Cap ``elems_per_thread`` only when the cached single-pass path would
    # be used; the two-pass path streams X twice and carries no per-row
    # register cache, so it is not bounded by the VGPR budget.
    cap = (
        None
        if row_norm_needs_two_pass(spec.elems_per_thread)
        else REGISTER_TILE_MAX_ELEMS_PER_THREAD
    )
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=cap,
        )
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    # Three f32 Welford reduction buffers (mean + M2 + count),
    # ``block_size`` words each.
    bytes_lds = 3 * spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_layernorm2d(spec: LayerNorm2DSpec) -> KernelDef:
    """Build the IR for one LayerNorm2D forward instance.

    Kernel signature:
      ``(X: ptr, Gamma: ptr, Beta: ptr, Y: ptr,
         [Mean: ptr, InvStd: ptr,]
         M: i32, N: i32, eps: f32)``

    Grid layout: ``grid_x = M``, ``block = (block_size, 1, 1)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid layernorm2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Beta = b.param(
        "Beta", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_mean_invstd:
        Mean = b.param("Mean", PtrType(io_ty, "global"), noalias=True, writeonly=True)
        InvStd = b.param(
            "InvStd", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    M = b.param("M", I32)  # noqa: F841 - ABI symmetry with CK Tile
    _ = b.param("N", I32)  # noqa: F841 - validated by caller; equals n_per_block
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    # CK Tile-style data abstractions. X / Y are 2D packed views over
    # the full activation tensor; the per-row tile pins its origin to
    # ``row``. Gamma / Beta are 1D vectors over N -- handled as plain
    # views since they're accessed with a single coordinate.
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    y_view = make_naive_tensor_view_packed(Y, shape=(1, N), dtype=io_ty)
    g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty)
    b_view = make_global_view(Beta, shape=(N,), dtype=io_ty)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    y_tile = make_tile_window(y_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    # LDS scratch for the numerically-stable Welford block merge. The
    # count-weighted ``(mean, M2, count)`` triple combiner is not a plain
    # associative add, so it cannot ride the generic ``block_lds_reduce``
    # / ``block_lds_reduce_pair`` ``combine=`` path; it needs a dedicated
    # three-channel LDS tree (one ``block_size``-wide f32 buffer per
    # channel). The extra channel vs the old two-buffer sum/sumsq form
    # costs ``block_size * 4 == 1 KB`` for the default ``BS=256``, well
    # within the CU LDS budget.
    lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
    lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
    lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    # Pass 1: ``sweep_row_chunks`` plays the role of CK Tile's
    # ``sweep_tile``: it streams the row through ``vec``-wide chunks,
    # invokes ``pass1_body(n_off, x_scalars)`` per chunk, and (with
    # ``cache=True``) records the f32 scalars so pass 2 doesn't
    # re-load from HBM.
    #
    # Per-chunk we materialise ``chunk_sum`` (sum of x) and ``chunk_sumsq``
    # (sum of x^2) via :func:`rocke.helpers.reduction.tree_reduce` so each fold has
    # critical-path depth ``log2(VEC)`` instead of ``VEC``. The
    # per-chunk partial is then merged once into the running per-thread
    # ``sum_p`` / ``sumsq_p`` scalars; this matches the latency structure
    # of CK Tile's ``BlockNormReduce`` per-Y sweep (where
    # ``MeanDistributedTensor`` gets folded one Y-position at a time, but
    # each Y position is a tree-reduce internally). The per-thread
    # ``(mean, M2, count)`` Welford triple is derived from these partials
    # after the sweep and merged across threads with the count-weighted
    # stable combiner.
    # Two-pass (large-N) selection mirrors legacy CK's ``isSweepOnce``:
    # a row whose ``elems_per_thread`` overflows the per-thread register
    # tile cannot be cached in VGPRs, so pass 1 only streams + reduces
    # (``cache=False``) and pass 2 re-reads X from HBM. For every in-budget
    # config ``two_pass`` is False and the cached single-pass path below is
    # byte-identical to the pre-two-pass kernel.
    two_pass = row_norm_needs_two_pass(spec.elems_per_thread)

    sum_p = b.const_f32(0.0)
    sumsq_p = b.const_f32(0.0)

    def pass1_body(_n_off, x_scalars):
        nonlocal sum_p, sumsq_p
        sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
        sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
        sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))

    sweep_res = sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=not two_pass,
    )

    # Numerically-stable Welford block merge. Each thread reads exactly
    # ``elems_per_thread`` elements (``N == block_size * elems_per_thread``)
    # so the per-thread count is the compile-time ``elems_per_thread``.
    # Derive this thread's partial Welford triple ``(mean_p, m2_p, count)``
    # from the per-thread ``sum_p`` / ``sumsq_p`` accumulators:
    #
    #     mean_p = sum_p / count
    #     m2_p   = sumsq_p - mean_p * sum_p     ( = Sum (x - mean_p)^2 )
    #
    # then merge across the workgroup with the count-weighted
    # ``(mean, M2, count)`` combiner (CK ``BlockwiseWelford::Merge``),
    # which has no catastrophic cancellation when ``|mean| >> sigma`` --
    # unlike the previous ``var = E[X^2] - E[X]^2`` block form. The helper
    # returns the biased (population) variance ``M2_total / count_total``,
    # exactly the statistic the old form computed but accurately.
    count_p = float(spec.elems_per_thread)
    inv_count_p = b.const_f32(1.0 / count_p)
    mean_p = b.fmul(sum_p, inv_count_p)
    m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

    mean, var = welford_block_reduce_stable(
        b,
        mean_p,
        m2_p,
        b.const_f32(count_p),
        lds_mean,
        lds_m2,
        lds_count,
        tid,
        block_size=BS,
    )
    inv_std = b.rsqrt(b.fadd(var, eps))

    if spec.save_mean_invstd:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            store_scalar_from_f32(b, Mean, row, mean, dtype=spec.dtype)
            store_scalar_from_f32(b, InvStd, row, inv_std, dtype=spec.dtype)

    # Pass 2: normalise, scale by gamma, shift by beta, write Y. The
    # pass2 helper pulls cached f32 scalars out of the pass1 sweep
    # result and stores the truncated f16/bf16 vector back to the
    # tile window per chunk.
    #
    # Reordering ``(x - mean) * inv_std * gamma + beta`` to
    # ``(x - mean) * (inv_std * gamma) + beta`` drops the critical
    # path from four serial f32 ops to three: ``fsub(x, mean)`` and
    # ``fmul(inv_std, gv[i])`` run in parallel (no shared operand
    # between them), then one ``fmul`` followed by one ``fadd``.
    # The op count is unchanged; this is purely a scheduling win
    # (and FMA-fusion-friendly should the lowering ever set
    # ``fp-contract=fast`` on ``arith.fadd``/``arith.fmul``).
    # In the single-pass path ``x_scalars`` are the cached f32 values from
    # pass 1; in the two-pass path the cache is empty, so re-stream X from
    # HBM (the same ``x_tile`` window) before normalising.
    def pass2_body(n_off, _k, x_scalars):
        if two_pass:
            x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        bv = b_view.load_vec_as_f32(b, [n_off], n=VEC)
        return [
            b.fadd(
                b.fmul(b.fsub(x_scalars[i], mean), b.fmul(inv_std, gv[i])),
                bv[i],
            )
            for i in range(VEC)
        ]

    pass2_row_chunks(
        b,
        y_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass2_body,
        cached_f32=sweep_res.cached,
    )

    return b.kernel


def layernorm2d_grid(m: int, spec: LayerNorm2DSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def layernorm2d_signature(spec: LayerNorm2DSpec):
    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Gamma", spec.dtype)
        .ptr("Beta", spec.dtype)
        .ptr("Y", spec.dtype)
    )
    if spec.save_mean_invstd:
        sb.ptr("Mean", spec.dtype).ptr("InvStd", spec.dtype)
    return sb.scalar("M", "i32").scalar("N", "i32").scalar("eps", "f32").build()
