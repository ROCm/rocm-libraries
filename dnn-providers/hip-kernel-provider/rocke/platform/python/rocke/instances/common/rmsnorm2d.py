# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""RMSNorm2D forward kernel instance builder (CK Tile ``10_rmsnorm2d`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/10_rmsnorm2d``. For each
row of an ``(M, N)`` activation tensor:

    rms[m]     = sqrt(sum_n(X[m,n]^2) / N + eps)
    inv_rms[m] = 1 / rms[m]
    Y[m,n]     = X[m,n] * inv_rms[m] * gamma[n]

This is the layer-norm-without-mean variant used by Llama / Mistral /
Gemma-style language models. Architecturally identical to
:mod:`rocke.instances.common.layernorm2d` (one CTA per row, one LDS tree
reduction, two-pass body); the only differences are the absent mean
subtraction, the dropped beta term, and a single reduction over ``s2``
instead of two.

The kernel uses the same CK Tile-inspired :class:`TensorView` /
:class:`TileWindow` / :func:`block_lds_reduce` helpers as layernorm; the
visible delta vs the C++ reference is essentially three lines of code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type, store_scalar_from_f32
from ...helpers.reduction import (
    REGISTER_TILE_MAX_ELEMS_PER_THREAD,
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    row_norm_needs_two_pass,
    tree_reduce,
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
class RMSNorm2DSpec:
    """One RMSNorm2D forward instance."""

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_inv_rms: bool = False
    wave_size: int = 64
    name: str = "rocke_rmsnorm2d_fwd"

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
            flags={"sr": self.save_inv_rms},
        )


def is_valid_spec(spec: RMSNorm2DSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one RMSNorm2D config on ``arch``.

    This kernel is a pure elementwise + LDS-tree-reduction norm: it uses
    no MFMA atoms, so the only architecture facts that matter are the
    per-WG LDS capacity and the max threads/block. Both are sourced from
    :class:`rocke.core.arch.ArchTarget` so the predicate rejects an
    unknown arch (or an over-budget ``block_size`` / LDS request) with a
    structured reason rather than failing later at lower/launch time. The
    one f32 reduction buffer (``block_size`` words) fits both gfx942
    (64 KiB) and gfx950 (160 KiB) for every valid ``block_size``, so
    gfx950 behavior is unchanged.

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

    # One f32 LDS reduction buffer of ``block_size`` words.
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_rmsnorm2d(spec: RMSNorm2DSpec) -> KernelDef:
    """Build the IR for one RMSNorm2D forward instance.

    Kernel signature:
      ``(X: ptr, Gamma: ptr, Y: ptr,
         [InvRms: ptr,]
         M: i32, N: i32, eps: f32)``
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid rmsnorm2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

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
    M = b.param("M", I32)  # noqa: F841
    _ = b.param("N", I32)  # noqa: F841
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    y_view = make_naive_tensor_view_packed(Y, shape=(1, N), dtype=io_ty)
    g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    y_tile = make_tile_window(y_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # Pass 1: ``sweep_row_chunks`` streams X once, the lambda accumulates
    # the sum-of-squares, and the f32 scalars are cached for pass 2.
    #
    # Per-chunk reduction is a balanced tree (depth log2(VEC)) folded
    # once into the running ``s2`` partial; vs the prior straight-line
    # ``s2 = fadd(s2, x_i*x_i)`` chain (depth VEC per chunk), this
    # halves the critical-path latency the AMDGPU pipeline sees for
    # the per-thread accumulator. CK Tile's ``BlockNormReduce`` does
    # the same fold via ``sweep_tile_span`` but for the canonical
    # 2D distribution; here the per-thread cardinality maps to the
    # VEC-wide chunk so the tree pays off uniformly.
    # Two-pass (large-N) selection mirrors legacy CK's ``isSweepOnce``:
    # a row whose ``elems_per_thread`` overflows the per-thread register
    # tile cannot be cached in VGPRs, so pass 1 only streams + reduces
    # (``cache=False``) and pass 2 re-reads X from HBM. For every in-budget
    # config ``two_pass`` is False and the cached single-pass path below is
    # byte-identical to the pre-two-pass kernel.
    two_pass = row_norm_needs_two_pass(spec.elems_per_thread)

    s2 = b.const_f32(0.0)

    def pass1_body(_n_off, x_scalars):
        nonlocal s2
        chunk_sq = [b.fmul(xi, xi) for xi in x_scalars]
        s2 = b.fadd(s2, tree_reduce(b, b.fadd, chunk_sq))

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

    # Cross-thread reduction. The wave-aligned path is the CK Tile
    # ``BlockReduce2dSync`` (warp XOR butterfly) + ``CrossWarpSync``
    # (one ``num_warps``-slot LDS round + tree-combine) shape, mirroring
    # ``instances/common/reduce.py``: for BS=256 / wave64 it replaces the
    # 8-round LDS tree (8 syncs) with six cross-lane shuffles + one
    # ``sync``. The fallback keeps the canonical full LDS tree for any
    # block size that isn't a clean multiple of ``wave_size`` (e.g. a
    # wave32 target), preserving the wave-agnostic correctness path.
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

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean_sq = b.fmul(total_s2, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps))

    if spec.save_inv_rms:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            store_scalar_from_f32(b, InvRms, row, inv_rms, dtype=spec.dtype)

    # Pass 2: scale by ``inv_rms * gamma``, write output via the same
    # sweep helper. Reordering the multiplies as ``x * (inv_rms * gv)``
    # exposes the ``inv_rms * gv`` term as a parallel-issuable op vs the
    # prior ``(x * inv_rms) * gv`` chain (both have depth 2, but the
    # reordered form has the two operands of the inner ``fmul`` rotate
    # together so the AMDGPU scheduler can interleave more freely vs
    # the prior form where ``inv_rms`` was a "hot" shared operand for
    # every inner ``fmul``). Matches the form used by CK Tile's
    # ``Rmsnorm2dFwdPipelineOnePass`` sweep_tile body.
    # In the single-pass path ``x_scalars`` are the cached f32 values from
    # pass 1; in the two-pass path the cache is empty, so re-stream X from
    # HBM (the same ``x_tile`` window) before scaling.
    def pass2_body(n_off, _k, x_scalars):
        if two_pass:
            x_scalars = x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        return [b.fmul(x_scalars[i], b.fmul(inv_rms, gv[i])) for i in range(VEC)]

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


def rmsnorm2d_grid(m: int, spec: RMSNorm2DSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def rmsnorm2d_signature(spec: RMSNorm2DSpec):
    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Gamma", spec.dtype)
        .ptr("Y", spec.dtype)
    )
    if spec.save_inv_rms:
        sb.ptr("InvRms", spec.dtype)
    return sb.scalar("M", "i32").scalar("N", "i32").scalar("eps", "f32").build()
