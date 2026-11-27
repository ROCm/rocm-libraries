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
from ...helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    block_lds_reduce_pair,
    block_lds_reduce_pair_with_wave_prologue,
    row_norm_needs_two_pass,
    tree_reduce,
    welford_block_reduce,
    welford_block_reduce_stable,
    welford_block_reduce_stable_wave_prologue,
    welford_block_reduce_wave_prologue,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import (
    pass2_row_chunks,
    pass2_row_chunks_dynamic,
    sweep_row_chunks,
    sweep_row_chunks_dynamic,
)
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class LayerNorm2DSpec:
    """One LayerNorm2D forward instance.

    ``dynamic=False`` (the default) is the original, compile-time-shaped
    kernel: ``n_per_block`` is the exact row width this build is
    specialized to, byte-identical to the pre-AOT-generic emission.

    ``dynamic=True`` builds an AOT-generic kernel instead: the row width
    becomes the runtime ``N`` kernarg (``cols`` in the docstrings below),
    read and used in the IR rather than left dead. ``n_per_block`` is then
    reinterpreted as a *capacity* -- the largest row width this build's
    register tiling supports -- and the caller/dispatcher must only launch
    it against requests with ``cols <= n_per_block``; the kernel itself has
    no way to check that at runtime. See :func:`build_layernorm2d`'s
    dynamic branch.

    ``cheap_reduce=True`` (default ``False``) swaps the numerically-stable
    Welford ``(mean, M2, count)`` block merge for the naive
    ``var = E[X^2] - E[X]^2`` two-pass form (:func:`rocke.helpers.reduction.welford_block_reduce`),
    matching MIOpen's own ``MIOpenLayerNorm.cpp`` reduction shape (plain
    ``sum``/``sum_sq`` LDS-tree fold, 2 fp32 adds/step instead of a full
    Welford pairwise combine). This trades away the catastrophic-cancellation
    safety the module docstring calls out (``|mean| >> sigma`` activations)
    for a much cheaper per-barrier-step payload -- an opt-in lever,
    never the default. Validate against a reference before trusting it for
    any specific workload's value range.

    ``cheap_reduce=True`` combined with ``fast_reduce=True`` additionally
    swaps the ``(sum, sum_sq)`` block merge's barriered LDS tree for
    :func:`rocke.helpers.reduction.block_lds_reduce_pair_with_wave_prologue`
    (wave-XOR-first, one cross-warp barrier instead of ``log2(block_size)``):
    the smallest row shapes are launch/sync-bound (the per-sample staged
    timing floor is dominated by the kernel's barrier count, not its
    arithmetic), so this trims the critical path between the launch's start
    and stop events rather than any FLOP count.
    """

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_mean_invstd: bool = False
    wave_size: int = 64
    dynamic: bool = False
    fast_reduce: bool = False
    cheap_reduce: bool = False
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
            flags={
                "smv": self.save_mean_invstd,
                "dyn": self.dynamic,
                "fr": self.fast_reduce,
                "cr": self.cheap_reduce,
            },
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

    if spec.dynamic:
        # The dynamic (AOT-generic) body always streams X twice (no
        # per-row register cache -- a runtime trip count can't back one
        # sized at build time, see build_layernorm2d's dynamic branch), so
        # it carries no VGPR cap, and ``n_per_block`` is a capacity, not
        # the real row width, so its exact-divisibility check doesn't
        # apply: the actual runtime ``cols`` is masked-tail-safe for any
        # value, not just one that divides ``block_size*vec``.
        ok, why = validate_io(
            IOSpecRule(dtype=spec.dtype, block_size=spec.block_size, vec=spec.vec)
        )
        if not ok:
            return False, why
    else:
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

    if spec.fast_reduce and spec.block_size % spec.wave_size != 0:
        return False, (
            f"fast_reduce requires block_size {spec.block_size} divisible by "
            f"wave_size {spec.wave_size}"
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

    ``spec.dynamic=True`` builds the AOT-generic variant instead (see
    :func:`_build_layernorm2d_dynamic`): one compiled binary serves any
    runtime row width up to ``spec.n_per_block`` (now a capacity), rather
    than being specialized to an exact ``N``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid layernorm2d spec: {why}")

    if spec.dynamic:
        return _build_layernorm2d_dynamic(spec)

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
    if spec.cheap_reduce:
        # MIOpen-style cheap reduction: plain (sum, sum_sq) fused-barrier
        # fold (see welford_block_reduce_stable's docstring, cheap_reduce
        # flag note), ``var = E[X^2] - E[X]^2`` computed once after the
        # tree instead of a per-step Welford merge -- opt-in, less
        # numerically stable, see LayerNorm2DSpec's docstring.
        reduce_fn = (
            welford_block_reduce_wave_prologue if spec.fast_reduce else welford_block_reduce
        )
        kwargs = {"wave_size": spec.wave_size} if spec.fast_reduce else {}
        mean, var = reduce_fn(
            b,
            sum_p,
            sumsq_p,
            spec.elems_per_thread,
            lds_mean,
            lds_m2,
            tid,
            block_size=BS,
            **kwargs,
        )
    else:
        count_p = float(spec.elems_per_thread)
        inv_count_p = b.const_f32(1.0 / count_p)
        mean_p = b.fmul(sum_p, inv_count_p)
        m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

        if spec.fast_reduce:
            mean, var = welford_block_reduce_stable_wave_prologue(
                b,
                mean_p,
                m2_p,
                b.const_f32(count_p),
                lds_mean,
                lds_m2,
                lds_count,
                tid,
                block_size=BS,
                wave_size=spec.wave_size,
            )
        else:
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


def _build_layernorm2d_dynamic(spec: LayerNorm2DSpec) -> KernelDef:
    """AOT-generic LayerNorm2D: one compiled binary serves any runtime row
    width ``cols`` (the real ``N`` kernarg), up to ``spec.n_per_block``'s
    capacity.

    The main vectorized sweep is a genuine runtime-trip-count loop
    (:func:`rocke.helpers.sweep.sweep_row_chunks_dynamic` /
    ``pass2_row_chunks_dynamic``, built on ``scf.for``); the remainder
    (``cols`` mod ``block_size*vec``, which the main loop's runtime trip
    count leaves uncovered) is a ``vec``-wide, Python-unrolled *masked*
    scalar tail -- exactly :func:`build_layernorm2d`'s own compile-time-HW
    tail idiom (``batchnorm2d.py`` uses the identical shape), just with
    the bound (``cols``) a runtime Value instead of a Python int. Always
    streams X twice: a runtime trip count can't back the static path's
    per-thread register cache, so pass 2 re-reads X from HBM rather than
    reusing a build-time-sized cache array.
    """
    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec
    cap = spec.n_per_block  # capacity; only used to size the dummy view shape

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
    _ = b.param("M", I32)  # noqa: F841 - row count; already dynamic via grid size
    cols = b.param("N", I32)
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    # Row stride is the runtime ``cols`` (not a compile-time shape): the
    # engine's TensorDescriptor already accepts an SSA Value stride (see
    # helpers/tensor_view.py), so addressing just works once the stride is
    # a Value instead of an int. ``shape=`` only sizes the (unused for
    # addressing) descriptor rank/capacity bookkeeping.
    x_view = make_global_view(X, shape=(1, cap), dtype=io_ty, strides=(cols, 1))
    y_view = make_global_view(Y, shape=(1, cap), dtype=io_ty, strides=(cols, 1))
    g_view = make_global_view(Gamma, shape=(cap,), dtype=io_ty)
    b_view = make_global_view(Beta, shape=(cap,), dtype=io_ty)
    x_tile = make_tile_window(x_view, lengths=(1, 1), origin=(row, b.const_i32(0)))
    y_tile = make_tile_window(y_view, lengths=(1, 1), origin=(row, b.const_i32(0)))

    lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
    lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
    lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    # --- Pass 1: runtime-trip-count sweep + masked scalar tail ----------
    def pass1_body(_n_off, x_scalars, iter_vars):
        sum_p, sumsq_p = iter_vars
        sq_scalars = [b.fmul(xi, xi) for xi in x_scalars]
        sum_p = b.fadd(sum_p, tree_reduce(b, b.fadd, list(x_scalars)))
        sumsq_p = b.fadd(sumsq_p, tree_reduce(b, b.fadd, sq_scalars))
        return [sum_p, sumsq_p]

    (sum_p, sumsq_p), main_elems, _tail = sweep_row_chunks_dynamic(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        length=cols,
        iter_args=[("sum_p", b.const_f32(0.0)), ("sumsq_p", b.const_f32(0.0))],
        body=pass1_body,
    )

    tail_count_p = b.const_f32(0.0)
    for tc in range(VEC):
        idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
        valid = b.cmp_lt(idx_col, cols)
        idx_safe = b.smin(idx_col, b.sub(cols, b.const_i32(1)))
        flat = b.add(b.mul(row, cols), idx_safe)
        x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
        x_masked = b.select(valid, x_raw, b.const_f32(0.0))
        valid_f = b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
        sum_p = b.fadd(sum_p, x_masked)
        sumsq_p = b.fadd(sumsq_p, b.fmul(x_masked, x_masked))
        tail_count_p = b.fadd(tail_count_p, valid_f)

    # ``main_elems`` is the TOTAL row elements the main loop covers (all BS
    # threads' share, matching the ``main_elems_per_row`` naming in
    # build_batchnorm2d's compile-time tail); dividing by BS gives back
    # THIS thread's own main-loop count (the Welford count is per-thread).
    if spec.cheap_reduce:
        # MIOpen-style cheap reduction (see LayerNorm2DSpec's docstring):
        # masked tail elements already contribute exactly 0 to sum_p/sumsq_p,
        # so a plain commutative sum across all BS threads totals exactly
        # ``cols`` real elements -- dividing by the runtime ``cols`` kernarg
        # directly needs no per-thread count reduction (no lds_count, no
        # is_empty guard) at all, unlike the Welford paths below.
        pair_reduce_fn = (
            block_lds_reduce_pair_with_wave_prologue if spec.fast_reduce else block_lds_reduce_pair
        )
        pair_kwargs = {"wave_size": spec.wave_size} if spec.fast_reduce else {}
        total_sum, total_sumsq = pair_reduce_fn(
            b, sum_p, sumsq_p, lds_mean, lds_m2, tid, block_size=BS,
            combine_a="sum", combine_c="sum", **pair_kwargs,
        )
        one_f32 = b.const_f32(1.0)
        cols_f32 = b.sitofp_f32(cols)
        inv_cols = b.fdiv(one_f32, cols_f32)
        mean = b.fmul(total_sum, inv_cols)
        sq_mean = b.fmul(total_sumsq, inv_cols)
        var = b.fsub(sq_mean, b.fmul(mean, mean))
    else:
        main_per_thread = b.div(main_elems, b.const_i32(BS))
        count_p = b.fadd(b.sitofp_f32(main_per_thread), tail_count_p)
        # A thread can own zero elements when ``cols < BS`` (row shorter than
        # the block); guard the reciprocal so it reports a finite empty
        # partial (mean=0, M2=0) instead of 1/0 -> inf -> 0*inf -> NaN, which
        # would otherwise poison the whole block's welford merge.
        is_empty_p = b.fcmp("oeq", count_p, b.const_f32(0.0))
        zero_recip = b.const_f32(0.0)
        recip_p = b.fdiv(b.const_f32(1.0), count_p)
        inv_count_p = b.select(is_empty_p, zero_recip, recip_p)
        mean_p = b.fmul(sum_p, inv_count_p)
        m2_p = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))

        if spec.fast_reduce:
            mean, var = welford_block_reduce_stable_wave_prologue(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid,
                block_size=BS, wave_size=spec.wave_size,
            )
        else:
            mean, var = welford_block_reduce_stable(
                b, mean_p, m2_p, count_p, lds_mean, lds_m2, lds_count, tid, block_size=BS
            )
    inv_std = b.rsqrt(b.fadd(var, eps))

    if spec.save_mean_invstd:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            store_scalar_from_f32(b, Mean, row, mean, dtype=spec.dtype)
            store_scalar_from_f32(b, InvStd, row, inv_std, dtype=spec.dtype)

    # --- Pass 2: runtime-trip-count write-back + masked (scf_if) tail ----
    def pass2_body(n_off):
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

    trip_count = b.div(cols, b.const_i32(BS * VEC))
    pass2_row_chunks_dynamic(
        b,
        y_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        main_elems_per_row=main_elems,
        trip_count=trip_count,
        body=pass2_body,
    )

    for tc in range(VEC):
        idx_col = b.add(b.add(main_elems, b.const_i32(tc * BS)), tid)
        valid = b.cmp_lt(idx_col, cols)
        with b.scf_if(valid):
            flat = b.add(b.mul(row, cols), idx_col)
            x_raw = load_scalar_as_f32(b, X, flat, dtype=spec.dtype)
            gv = g_view.load_vec_as_f32(b, [idx_col], n=1)
            bv = b_view.load_vec_as_f32(b, [idx_col], n=1)
            y_val = b.fadd(b.fmul(b.fsub(x_raw, mean), b.fmul(inv_std, gv[0])), bv[0])
            store_scalar_from_f32(b, Y, flat, y_val, dtype=spec.dtype)

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
