# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime-N LayerNorm2D forward kernel instance builder.

This is the dynamic-N sibling of :mod:`rocke.instances.common.layernorm2d`,
mirroring the way :mod:`rocke.instances.common.rmsnorm2d_dynamic` relates to
the static :mod:`rocke.instances.common.rmsnorm2d`. The static instance bakes
``N`` at build time (its ``sweep_row_chunks`` is Python-unrolled over a
compile-time ``elems_per_thread`` and the per-thread count is the constant
``elems_per_thread``), so one binary serves exactly one ``N``. This instance
reads ``N`` from the kernel's runtime i32 argument instead, so a **single
binary serves every ``N`` that is a multiple of ``vec``**:

    mean[m]    = sum_n(X[m,n]) / N
    var[m]     = sum_n((X[m,n] - mean[m])^2) / N     (population variance)
    inv_std[m] = 1 / sqrt(var[m] + eps)
    Y[m,n]     = (X[m,n] - mean[m]) * inv_std[m] * gamma[n] + beta[n]

It is a deliberate, minimal delta from the static kernel:

  * both reductions (sum and sum-of-squares) are folded in a single
    ``scf.for`` grid-stride loop over the runtime column count
    (``upper = N / vec``) carrying ``(sum, sumsq, count)`` in f32 iter-args
    -- the same runtime-extent-loop pattern the GEMMs use to drive their
    K-loop off a runtime ``K`` param
    (:mod:`rocke.instances.gfx1151.wmma_gemm`), and the same shape as
    ``rmsnorm2d_dynamic`` (which carries one partial instead of three);
  * because ``N`` is runtime the per-thread element count is runtime too,
    so the count is carried in the loop (``count += vec`` per group) and
    passed to the **stable Welford block merge** verbatim -- the same
    count-weighted ``(mean, M2, count)`` combiner the static kernel uses,
    which avoids the catastrophic cancellation of ``var = E[X^2] - E[X]^2``
    for the ``|mean| >> sigma`` post-residual activations LayerNorm sees;
  * a lane that folds **zero** groups (possible when ``N / vec < block_size``,
    e.g. N=1024, vec=8 -> only 128 of 256 lanes are busy) supplies a
    **finite** ``(mean_p=0, m2_p=0, count_p=0)`` triple via a ``count > 0``
    select, so the Welford merge's ``count_b_over_count = 0`` guard folds it
    away without a NaN (``0/0`` mean would otherwise poison the tree);
  * this kernel is inherently **two-pass** (X is re-streamed from HBM for the
    normalise pass) because the runtime element count cannot be cached in a
    per-thread register tile -- exactly like ``rmsnorm2d_dynamic``.

**Precondition: ``N % vec == 0``.** With flat row-major addressing row ``m``
begins at element ``m*N``, so a vector-``vec`` load of a row is only correctly
aligned when ``N`` is a multiple of ``vec``. This is enforced at selection time
by a ``multiple_of`` constraint in ``family.json`` rather than inside the
kernel, and holds for every transformer hidden size we target (2048, 2432,
3072, 4096, ...). Because ``N % vec == 0`` there is no ragged remainder, so the
kernel has no tail path.

The ABI is byte-identical to the static kernel
``(X, Gamma, Beta, Y [, Mean, InvStd], M, N, eps)`` -- the static builder
already declares ``N`` as a runtime param, it just discards it. So the same
:class:`rocke.helpers.spec.SignatureBuilder` signature and grid apply, and the
AOT catalog engine drives it with no adapter change.

**gfx1151 gotcha inherited:** ``wave_size=32`` MUST be set. The Welford block
merge's wave prologue / lane-mask shuffles miscompile at the default wave64 on
RDNA3, exactly as in ``rmsnorm2d_dynamic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import (
    io_ir_type,
    load_vec_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from ...helpers.reduction import tree_reduce, welford_block_reduce_stable
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ...helpers.tensor_view import make_lds_view


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class LayerNorm2DDynamicSpec:
    """One runtime-N LayerNorm2D forward instance.

    Unlike :class:`LayerNorm2DSpec` there is no ``n_per_block`` -- ``N`` is a
    runtime kernel argument, so ``block_size`` / ``vec`` are the only
    (perf-only) knobs and the emitted binary is valid for every ``N`` that is a
    multiple of ``vec`` (see the module docstring on alignment).
    """

    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_mean_invstd: bool = False
    wave_size: int = 64
    name: str = "rocke_layernorm2d_fwd_dyn"

    def kernel_name(self) -> str:
        # No ``N{n}`` token: one binary serves every N (multiple of vec).
        return kernel_name_join(
            self.name,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"smv": self.save_mean_invstd},
        )


def is_valid_spec(
    spec: LayerNorm2DDynamicSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one runtime-N LayerNorm2D config on ``arch``.

    N-divisibility and the per-thread VGPR (``elems_per_thread``) cap that
    bound the static kernel do not apply here: N is runtime and the reduction
    streams via an ``scf.for`` carrying three f32 partials, so the only
    architecture facts that matter are the max threads/block and the three f32
    Welford LDS buffers (``3 * block_size`` words: mean / M2 / count).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r} (expected f16/bf16)"
    if spec.vec not in (2, 4, 8):
        return False, f"vec must be one of {{2,4,8}} (got {spec.vec})"
    if spec.block_size <= 0:
        return False, f"block_size must be positive (got {spec.block_size})"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    # Three f32 Welford reduction buffers (mean + M2 + count), block_size words.
    bytes_lds = 3 * spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_layernorm2d_dynamic(spec: LayerNorm2DDynamicSpec) -> KernelDef:
    """Build the IR for one runtime-N LayerNorm2D forward instance.

    Kernel signature (identical to the static kernel):
      ``(X: ptr, Gamma: ptr, Beta: ptr, Y: ptr,
         [Mean: ptr, InvStd: ptr,]
         M: i32, N: i32, eps: f32)``

    Precondition (enforced by selection, not here): ``N % vec == 0``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid layernorm2d_dynamic spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec

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
    M = b.param("M", I32)  # noqa: F841 (row count implied by the grid; ABI parity)
    N = b.param("N", I32)  # runtime column count -- the whole point of this kernel
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    c_vec = b.const_i32(VEC)
    c_bs = b.const_i32(BS)
    c_vec_f = b.const_f32(float(VEC))
    zero_f = b.const_f32(0.0)

    # Flat row-major addressing (GEMM style): X[row, col] lives at ``row*N +
    # col``. N % vec == 0 (see module docstring) keeps every row start
    # vec-aligned, so the vectorised loads below are always aligned.
    row_base = b.mul(row, N)

    # Number of full VEC-wide groups in a row (exact -- N % vec == 0).
    n_vec = b.div(N, c_vec)

    lds_mean = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_mean").base
    lds_m2 = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_m2").base
    lds_count = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_count").base

    # ---- Pass 1: sum and sum-of-squares (streaming grid-stride) ----
    #
    # Lane ``tid`` folds VEC-groups g = tid, tid+BS, tid+2BS, ... < n_vec into
    # carried f32 partials (sum, sumsq) and a carried element count. This is the
    # runtime-extent accumulating loop the GEMMs use for their K-loop, folding
    # both x and x*x (and bumping the count) instead of a WMMA fragment.
    p1 = b.scf_for_iter(
        tid,
        n_vec,
        c_bs,
        [("s", zero_f), ("s2", zero_f), ("cnt", zero_f)],
        iv_name="g1",
    )
    with p1 as (g1, (s, s2, cnt)):
        off = b.add(row_base, b.mul(g1, c_vec))
        xs = load_vec_as_f32(b, X, off, dtype=spec.dtype, n=VEC)
        chunk_sq = [b.fmul(xi, xi) for xi in xs]
        new_s = b.fadd(s, tree_reduce(b, b.fadd, list(xs)))
        new_s2 = b.fadd(s2, tree_reduce(b, b.fadd, chunk_sq))
        new_cnt = b.fadd(cnt, c_vec_f)
        b.scf_yield(new_s, new_s2, new_cnt)
    sum_p, sumsq_p, count_p = p1.results[0], p1.results[1], p1.results[2]

    # Per-lane Welford triple from the (sum, sumsq, count) partials:
    #     mean_p = sum_p / count_p
    #     m2_p   = sumsq_p - mean_p * sum_p      ( = Sum (x - mean_p)^2 )
    # A zero-count lane (N/vec < block_size) must supply a FINITE triple, else
    # ``0/0 = NaN`` in mean_p poisons the Welford merge (its count-weighted
    # combiner multiplies delta by count_b_over_count, and NaN*0 = NaN). Guard
    # with ``count_p > 0`` selects so empty lanes contribute (0, 0, 0).
    has = b.fcmp("ogt", count_p, zero_f)
    inv_cnt = b.rcp(count_p)  # +inf when count_p == 0
    mean_raw = b.fmul(sum_p, inv_cnt)  # 0*inf = NaN when empty -> select away
    mean_p = b.select(has, mean_raw, zero_f)
    m2_raw = b.fsub(sumsq_p, b.fmul(mean_p, sum_p))
    m2_p = b.select(has, m2_raw, zero_f)

    # ---- Stable Welford block merge (reused verbatim from the static kernel).
    # gfx1151 wave32 gotcha lives inside this helper's wave prologue, hence
    # wave_size=32 on the spec.
    mean, var = welford_block_reduce_stable(
        b,
        mean_p,
        m2_p,
        count_p,
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

    # ---- Pass 2: normalise, scale by gamma, shift by beta, write Y (re-stream
    # X). Same grid-stride column walk. Gamma / Beta are indexed by column only
    # (shape [N]); Y shares X's row-major layout so it reuses ``off``.
    # ``(x - mean) * (inv_std * gamma) + beta`` mirrors the static kernel's
    # reordered multiply (fsub and the inv_std*gamma fmul run in parallel).
    p2 = b.scf_for(tid, n_vec, c_bs, iv_name="g2")
    with p2 as g2:
        col = b.mul(g2, c_vec)
        off = b.add(row_base, col)
        xs = load_vec_as_f32(b, X, off, dtype=spec.dtype, n=VEC)
        gs = load_vec_as_f32(b, Gamma, col, dtype=spec.dtype, n=VEC)
        betas = load_vec_as_f32(b, Beta, col, dtype=spec.dtype, n=VEC)
        out = [
            b.fadd(b.fmul(b.fsub(xs[i], mean), b.fmul(inv_std, gs[i])), betas[i])
            for i in range(VEC)
        ]
        store_vec(b, Y, off, pack_f32_to(b, out, dtype=spec.dtype), n=VEC)

    return b.kernel


def layernorm2d_dynamic_grid(
    m: int, spec: LayerNorm2DDynamicSpec
) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def layernorm2d_dynamic_signature(spec: LayerNorm2DDynamicSpec):
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
