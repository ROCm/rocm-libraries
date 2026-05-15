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
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type, store_scalar_from_f32
from ..helpers.reduction import block_lds_reduce
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ..helpers.sweep import pass2_row_chunks, sweep_row_chunks
from ..helpers.tensor_view import (
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


def is_valid_spec(spec: LayerNorm2DSpec) -> Tuple[bool, str]:
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )


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

    # LDS scratch for the two block-wide reductions (s1 then s2 share
    # the same buffer because their lifetimes don't overlap).
    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # Pass 1: ``sweep_row_chunks`` plays the role of CK Tile's
    # ``sweep_tile``: it streams the row through ``vec``-wide chunks,
    # invokes ``pass1_body(n_off, x_scalars)`` per chunk, and (with
    # ``cache=True``) records the f32 scalars so pass 2 doesn't
    # re-load from HBM. The lambda mutates ``s1`` / ``s2`` via
    # ``nonlocal`` to accumulate the moments.
    s1 = b.const_f32(0.0)
    s2 = b.const_f32(0.0)

    def pass1_body(_n_off, x_scalars):
        nonlocal s1, s2
        for xi in x_scalars:
            s1 = b.fadd(s1, xi)
            s2 = b.fadd(s2, b.fmul(xi, xi))

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

    total_s1 = block_lds_reduce(b, s1, lds, tid, block_size=BS, combine="sum")
    total_s2 = block_lds_reduce(b, s2, lds, tid, block_size=BS, combine="sum")

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
    def pass2_body(n_off, _k, x_scalars):
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        bv = b_view.load_vec_as_f32(b, [n_off], n=VEC)
        return [
            b.fadd(
                b.fmul(b.fmul(b.fsub(x_scalars[i], mean), inv_std), gv[i]),
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
