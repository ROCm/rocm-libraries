# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime-N RMSNorm2D forward kernel instance builder.

This is the dynamic-N sibling of :mod:`rocke.instances.common.rmsnorm2d`.
The static instance bakes ``N`` at build time (the reduction is a
Python-unrolled ``sweep_row_chunks`` over a compile-time
``elems_per_thread`` and the divisor is ``const_f32(N)``), so one binary
serves exactly one ``N``. This instance reads ``N`` from the kernel's
runtime i32 argument instead, so a **single binary serves every ``N``
that is a multiple of ``vec``**:

    rms[m]     = sqrt(sum_n(X[m,n]^2) / N + eps)
    inv_rms[m] = 1 / rms[m]
    Y[m,n]     = X[m,n] * inv_rms[m] * gamma[n]

It is a deliberate, minimal delta from the static kernel:

  * the sum-of-squares reduction is an ``scf.for`` grid-stride loop over
    the runtime column count (``upper = N / vec``) carrying the partial
    in an iter-arg -- the same runtime-extent-loop pattern the GEMMs use
    to drive their K-loop off a runtime ``K`` param
    (:mod:`rocke.instances.gfx1201.wmma_gemm`);
  * the divisor is ``rcp(sitofp_f32(N))`` -- runtime int->float;
  * the cross-lane block reduction is reused **verbatim** -- it reduces
    one f32 partial per lane and is independent of ``N``, so the
    gfx1151 wave32 gotcha (``wave_size=32`` MUST be set, else the
    lane-mask-32 shuffle miscompiles) is inherited unchanged.

**Precondition: ``N % vec == 0``.** With flat row-major addressing row
``m`` begins at element ``m*N``, so a vector-``vec`` load of a row is only
correctly aligned when ``N`` is a multiple of ``vec`` (otherwise
successive rows start off the ``vec``-element boundary and the vectorised
``global_load`` is misaligned). This is not a real limit for the target
workloads -- every ComfyUI RMSNorm hidden size we care about (Flux 3072,
SD3.5 2432, 2048, 4096, ...) is a multiple of 8 -- and it is enforced at
selection time by a ``multiple_of`` constraint in ``family.json`` rather
than inside the kernel. Because ``N % vec == 0`` there is no ragged
remainder, so the kernel has no tail path: it is a clean vectorised
grid-stride loop.

The ABI is byte-identical to the static kernel
``(X, Gamma, Y [, InvRms], M, N, eps)`` -- the static builder already
declares ``N`` as a runtime param, it just discards it. So the same
:class:`rocke.helpers.spec.SignatureBuilder` signature and grid apply,
and the AOT catalog engine drives it with no adapter change.
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
from ...helpers.reduction import (
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ...helpers.tensor_view import make_lds_view


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class RMSNorm2DDynamicSpec:
    """One runtime-N RMSNorm2D forward instance.

    Unlike :class:`RMSNorm2DSpec` there is no ``n_per_block`` -- ``N`` is
    a runtime kernel argument, so ``block_size`` / ``vec`` are the only
    (perf-only) knobs and the emitted binary is valid for every ``N`` that
    is a multiple of ``vec`` (see the module docstring on alignment).
    """

    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_inv_rms: bool = False
    wave_size: int = 64
    name: str = "rocke_rmsnorm2d_fwd_dyn"

    def kernel_name(self) -> str:
        # No ``N{n}`` token: one binary serves every N (multiple of vec).
        return kernel_name_join(
            self.name,
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"sr": self.save_inv_rms},
        )


def is_valid_spec(spec: RMSNorm2DDynamicSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one runtime-N RMSNorm2D config on ``arch``.

    N-divisibility and the per-thread VGPR (``elems_per_thread``) cap that
    bound the static kernel do not apply here: N is runtime and the
    reduction streams via an ``scf.for`` carrying a single f32 partial, so
    the only architecture facts that matter are the max threads/block and
    the one f32 LDS reduction buffer of ``block_size`` words.
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

    # One f32 LDS reduction buffer of ``block_size`` words.
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_rmsnorm2d_dynamic(spec: RMSNorm2DDynamicSpec) -> KernelDef:
    """Build the IR for one runtime-N RMSNorm2D forward instance.

    Kernel signature (identical to the static kernel):
      ``(X: ptr, Gamma: ptr, Y: ptr,
         [InvRms: ptr,]
         M: i32, N: i32, eps: f32)``

    Precondition (enforced by selection, not here): ``N % vec == 0``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid rmsnorm2d_dynamic spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC = spec.block_size, spec.vec

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_inv_rms:
        InvRms = b.param(
            "InvRms", PtrType(io_ty, "global"), noalias=True, writeonly=True
        )
    M = b.param("M", I32)  # noqa: F841 (row count implied by the grid; ABI parity)
    N = b.param("N", I32)  # runtime column count -- the whole point of this kernel
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    c_vec = b.const_i32(VEC)
    c_bs = b.const_i32(BS)

    # Flat row-major addressing (GEMM style): X[row, col] lives at
    # ``row*N + col``. N % vec == 0 (see module docstring) keeps every row
    # start vec-aligned, so the vectorised loads below are always aligned.
    row_base = b.mul(row, N)

    # Number of full VEC-wide groups in a row (exact -- N % vec == 0).
    n_vec = b.div(N, c_vec)

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # ---- Pass 1: sum of squares (streaming grid-stride over columns) ----
    #
    # Lane ``tid`` folds VEC-groups g = tid, tid+BS, tid+2BS, ... < n_vec
    # into a carried f32 partial. This is the runtime-extent accumulating
    # loop the GEMMs use for their K-loop (wmma_gemm.py:128), just folding
    # x*x instead of a WMMA fragment.
    p1 = b.scf_for_iter(tid, n_vec, c_bs, [("s2", b.const_f32(0.0))], iv_name="g1")
    with p1 as (g1, (s2,)):
        off = b.add(row_base, b.mul(g1, c_vec))
        xs = load_vec_as_f32(b, X, off, dtype=spec.dtype, n=VEC)
        chunk_sq = [b.fmul(xi, xi) for xi in xs]
        b.scf_yield(b.fadd(s2, tree_reduce(b, b.fadd, chunk_sq)))
    s2 = p1.results[0]

    # ---- Cross-lane block reduction (reused verbatim; N-independent) ----
    if spec.block_size % spec.wave_size == 0:
        total_s2 = block_lds_reduce_with_wave_prologue(
            b,
            s2,
            lds,
            tid,
            block_size=spec.block_size,
            combine="sum",
            wave_size=spec.wave_size,
        )
    else:
        total_s2 = block_lds_reduce(b, s2, lds, tid, block_size=BS, combine="sum")

    # Runtime divisor: mean of squares over N, then rsqrt.
    rcp_n = b.rcp(b.sitofp_f32(N))
    mean_sq = b.fmul(total_s2, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps))

    if spec.save_inv_rms:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            store_scalar_from_f32(b, InvRms, row, inv_rms, dtype=spec.dtype)

    # ---- Pass 2: scale by inv_rms * gamma and write Y (re-stream X) ----
    #
    # Same grid-stride column walk. Gamma is indexed by column only
    # (shape [N]); Y shares X's row-major layout so it uses the same
    # ``off``. ``x * (inv_rms * gv)`` mirrors the static kernel's reordered
    # multiply.
    p2 = b.scf_for(tid, n_vec, c_bs, iv_name="g2")
    with p2 as g2:
        col = b.mul(g2, c_vec)
        off = b.add(row_base, col)
        xs = load_vec_as_f32(b, X, off, dtype=spec.dtype, n=VEC)
        gs = load_vec_as_f32(b, Gamma, col, dtype=spec.dtype, n=VEC)
        out = [b.fmul(xs[i], b.fmul(inv_rms, gs[i])) for i in range(VEC)]
        store_vec(b, Y, off, pack_f32_to(b, out, dtype=spec.dtype), n=VEC)

    return b.kernel


def rmsnorm2d_dynamic_grid(m: int, spec: RMSNorm2DDynamicSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def rmsnorm2d_dynamic_signature(spec: RMSNorm2DDynamicSpec):
    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Gamma", spec.dtype)
        .ptr("Y", spec.dtype)
    )
    if spec.save_inv_rms:
        sb.ptr("InvRms", spec.dtype)
    return sb.scalar("M", "i32").scalar("N", "i32").scalar("eps", "f32").build()
