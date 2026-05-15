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
:mod:`ck_dsl.instances.layernorm2d` (one CTA per row, one LDS tree
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
class RMSNorm2DSpec:
    """One RMSNorm2D forward instance."""

    n_per_block: int
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    save_inv_rms: bool = False
    wave_size: int = 64
    name: str = "ck_dsl_rmsnorm2d_fwd"

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


def is_valid_spec(spec: RMSNorm2DSpec) -> Tuple[bool, str]:
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )


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
    s2 = b.const_f32(0.0)

    def pass1_body(_n_off, x_scalars):
        nonlocal s2
        for xi in x_scalars:
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

    total_s2 = block_lds_reduce(b, s2, lds, tid, block_size=BS, combine="sum")

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean_sq = b.fmul(total_s2, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps))

    if spec.save_inv_rms:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            store_scalar_from_f32(b, InvRms, row, inv_rms, dtype=spec.dtype)

    # Pass 2: scale by gamma, write output via the same sweep helper.
    def pass2_body(n_off, _k, x_scalars):
        gv = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        return [b.fmul(b.fmul(x_scalars[i], inv_rms), gv[i]) for i in range(VEC)]

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
