# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LayerNorm2D forward kernel instance builder (CK Tile ``02_layernorm2d`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/02_layernorm2d``. For
each row of an ``(M, N)`` activation tensor, computes:

    mean[m]    = sum_n(X[m,n]) / N
    var[m]     = sum_n(X[m,n]^2) / N - mean[m]^2
    inv_std[m] = 1 / sqrt(var[m] + eps)
    Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * gamma[n] + beta[n]

The kernel is expressed entirely against the CK Tile-inspired
:class:`ck_dsl.helpers.TensorView` / :class:`ck_dsl.helpers.TileWindow`
abstractions for I/O, :func:`ck_dsl.helpers.io.load_vec_as_f32` /
:func:`pack_f32_to` for dtype-promoted ingest/egress, and
:func:`ck_dsl.helpers.reduction.block_lds_reduce` for the cross-thread
sum. The bare-IR ``smem_alloc`` / ``smem_load_vN_f32`` /
``global_load_vN`` calls that used to dominate this file are gone.

What we cover today:
  - Dtypes ``f16`` / ``bf16`` for X/gamma/beta/Y (compute in f32)
  - Optional save of ``mean`` / ``inv_std`` per row (CK Tile's
    ``save_mean_var`` traits)
  - Single-pass row reduction using ``E[X^2] - E[X]^2``

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
from typing import Callable, List, Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type, store_scalar_from_f32
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


def _balanced_combine(
    values: List[Value], combine: Callable[[Value, Value], Value]
) -> Value:
    """Balanced-tree fold of ``values`` under ``combine``.

    Used for the per-chunk fold of f32 partials in pass 1. Without an
    explicit tree, the serial ``fadd(fadd(fadd(s, x0), x1), ...)``
    pattern has critical-path depth ``len(values)``; the LLVM ``reassoc``
    fastmath flag is not set on ``arith.fadd`` (see
    ``core/lower_llvm.py::_op_arith_fadd``) so the optimiser cannot
    re-shape the chain on our behalf. Emitting the tree explicitly
    drops the depth to ``ceil(log2(len(values)))``, mirroring the
    "leaves first, then merges" structure CK Tile gets from its
    ``sweep_tile_span`` over a per-Y register tile.
    """
    if not values:
        raise ValueError("_balanced_combine requires at least one value")
    cur = list(values)
    while len(cur) > 1:
        nxt: List[Value] = []
        i = 0
        while i + 1 < len(cur):
            nxt.append(combine(cur[i], cur[i + 1]))
            i += 2
        if i < len(cur):
            nxt.append(cur[i])
        cur = nxt
    return cur[0]


def _paired_block_lds_reduce_sum(
    b: IRBuilder,
    val_a: Value,
    val_c: Value,
    lds_a: Value,
    lds_c: Value,
    tid: Value,
    *,
    block_size: int,
) -> Tuple[Value, Value]:
    """Twin-channel block sum reduction sharing one barrier schedule.

    Functionally equivalent to two back-to-back
    :func:`ck_dsl.helpers.reduction.block_lds_reduce` calls (one for
    ``val_a``, one for ``val_c``), but interleaves the two channels'
    LDS writes and reads inside a *single* halving loop so the
    ``s_barrier`` between halving steps is amortised across both
    reductions. For ``block_size == 256`` this cuts the sync count
    from ``2 * (log2(256) + 1) == 18`` down to ``log2(256) + 1 == 9``
    and the LDS round-trip count in half -- a real perf win for the
    row-norm sum + sumsq fold (LayerNorm) since the two reductions
    used to dominate the cross-thread wallclock on the LDS-bound path.

    The implementation mirrors :func:`block_lds_reduce` line-for-line
    except every step touches both LDS buffers. The caller owns both
    ``lds_a`` / ``lds_c`` allocations; both must be at least
    ``block_size`` f32 slots wide.
    """
    if val_a.type.name != "f32" or val_c.type.name != "f32":
        raise ValueError("_paired_block_lds_reduce_sum expects f32 inputs")

    b.smem_store_vN_f32(lds_a, [tid], val_a, 1)
    b.smem_store_vN_f32(lds_c, [tid], val_c, 1)
    b.sync()

    n = block_size
    while n > 1:
        half = n // 2
        c_half = b.const_i32(half)
        in_first = b.cmp_lt(tid, c_half)
        with b.scf_if(in_first):
            j = b.add(tid, c_half)
            a_a = b.vec_extract(b.smem_load_vN_f32(lds_a, tid, n=1), 0)
            c_a = b.vec_extract(b.smem_load_vN_f32(lds_a, j, n=1), 0)
            a_c = b.vec_extract(b.smem_load_vN_f32(lds_c, tid, n=1), 0)
            c_c = b.vec_extract(b.smem_load_vN_f32(lds_c, j, n=1), 0)
            b.smem_store_vN_f32(lds_a, [tid], b.fadd(a_a, c_a), 1)
            b.smem_store_vN_f32(lds_c, [tid], b.fadd(a_c, c_c), 1)
        b.sync()
        n = half

    out_a = b.vec_extract(b.smem_load_vN_f32(lds_a, b.const_i32(0), n=1), 0)
    out_c = b.vec_extract(b.smem_load_vN_f32(lds_c, b.const_i32(0), n=1), 0)
    return out_a, out_c


@dataclass(frozen=True)
class LayerNorm2DSpec:
    """One LayerNorm2D forward instance."""

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_mean_invstd: bool = False
    wave_size: int = 64
    name: str = "ck_dsl_layernorm2d_fwd"

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
    threads/block, both sourced from :class:`ck_dsl.core.arch.ArchTarget`
    so an unknown arch / over-budget ``block_size`` is rejected with a
    structured reason. The two f32 reduction buffers (``2 * block_size``
    words) fit both gfx942 (64 KiB) and gfx950 (160 KiB), so gfx950
    behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    # Two f32 LDS reduction buffers (sum + sumsq), ``block_size`` words each.
    bytes_lds = 2 * spec.block_size * 4
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

    # LDS scratch for the two block-wide reductions. We allocate two
    # ``block_size``-sized f32 buffers so the sum (``s1``) and the
    # sum-of-squares (``s2``) folds can share a *single* halving
    # schedule via :func:`_paired_block_lds_reduce_sum` instead of
    # paying for two back-to-back ``block_lds_reduce`` round-trips
    # (which would double the ``s_barrier`` count). The extra LDS
    # cost is ``block_size * 4 == 1 KB`` for the default ``BS=256``,
    # well within the 64 KB CU budget.
    lds_s1 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_s1").base
    lds_s2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_s2").base

    # Pass 1: ``sweep_row_chunks`` plays the role of CK Tile's
    # ``sweep_tile``: it streams the row through ``vec``-wide chunks,
    # invokes ``pass1_body(n_off, x_scalars)`` per chunk, and (with
    # ``cache=True``) records the f32 scalars so pass 2 doesn't
    # re-load from HBM.
    #
    # Per-chunk we materialise ``chunk_s1`` (sum of x) and ``chunk_s2``
    # (sum of x^2) via :func:`_balanced_combine` so each fold has
    # critical-path depth ``log2(VEC)`` instead of ``VEC``. The
    # per-chunk partial is then merged once into the running ``s1`` /
    # ``s2`` scalars; this matches the latency structure of CK Tile's
    # ``BlockNormReduce`` per-Y sweep (where ``MeanDistributedTensor``
    # gets folded one Y-position at a time, but each Y position is a
    # tree-reduce internally).
    s1 = b.const_f32(0.0)
    s2 = b.const_f32(0.0)

    def pass1_body(_n_off, x_scalars):
        nonlocal s1, s2
        sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
        s1 = b.fadd(s1, _balanced_combine(list(x_scalars), b.fadd))
        s2 = b.fadd(s2, _balanced_combine(sq_scalars, b.fadd))

    sweep_res = sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=True,
    )

    # Twin-channel cross-thread reduction: one halving schedule
    # produces both ``total_s1`` and ``total_s2`` with half the
    # ``s_barrier`` count of the original double ``block_lds_reduce``
    # call.
    total_s1, total_s2 = _paired_block_lds_reduce_sum(
        b, s1, s2, lds_s1, lds_s2, tid, block_size=BS
    )

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean = b.fmul(total_s1, rcp_n)
    second_moment = b.fmul(total_s2, rcp_n)
    var = b.fsub(second_moment, b.fmul(mean, mean))
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
    def pass2_body(n_off, _k, x_scalars):
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
