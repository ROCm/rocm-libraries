# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Row-wise reduction kernel instance builder (CK Tile ``05_reduce`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/05_reduce``. For each row
of an ``(M, N)`` tensor produces one scalar per row by applying:

    sum :  Y[m] = sum_n(X[m,n])
    max :  Y[m] = max_n(X[m,n])
    mean:  Y[m] = sum_n(X[m,n]) / N

Compute is in f32 internally; the output dtype matches the input dtype
(f16 in / f16 out, etc.).

The kernel uses the same CK Tile-inspired :class:`TensorView` and
:func:`block_lds_reduce` helpers as the norm kernels; ``op`` simply
picks the reduction combiner (``"sum"`` vs ``"max"``) and an optional
``rcp(N)`` post-multiply for ``"mean"``.
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
from ..helpers.sweep import sweep_row_chunks
from ..helpers.tensor_view import (
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]
ReduceOp = Literal["sum", "max", "mean"]


@dataclass(frozen=True)
class Reduce2DSpec:
    """One row-reduction instance."""

    n_per_block: int
    op: ReduceOp = "sum"
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    name: str = "ck_dsl_reduce2d"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.op,
            self.dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def is_valid_spec(spec: Reduce2DSpec) -> Tuple[bool, str]:
    if spec.op not in ("sum", "max", "mean"):
        return False, f"unsupported op {spec.op!r}"
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
        )
    )


_NEG_INF_F32 = -3.4028234663852886e38


def build_reduce2d(spec: Reduce2DSpec) -> KernelDef:
    """Build the IR for one row-reduction instance.

    Kernel signature: ``(X: ptr, Y: ptr, M: i32, N: i32)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid reduce2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841
    _ = b.param("N", I32)  # noqa: F841

    tid = b.thread_id_x()
    row = b.block_id_x()

    # CK Tile-style: make_naive_tensor_view_packed(X, (1, N)) gives us a
    # packed row-major view; make_tile_window pins the origin to ``row``
    # so the sweep below indexes within a single row.
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    if spec.op in ("sum", "mean"):
        acc = b.const_f32(0.0)
        combine: Literal["sum", "max"] = "sum"
    else:
        acc = b.const_f32(_NEG_INF_F32)
        combine = "max"

    # sweep_row_chunks plays the role of CK Tile's ``sweep_tile``: it
    # invokes ``body(n_off, x_scalars)`` once per per-thread chunk and
    # threads the f32-promoted lane scalars in. The accumulator is
    # rebound across iterations via the closed-over ``acc`` -- the
    # ``nonlocal`` keyword makes the lambda mutate the outer binding.
    def body(_n_off, x_scalars):
        nonlocal acc
        for xi in x_scalars:
            acc = b.fadd(acc, xi) if combine == "sum" else b.fmax(acc, xi)

    sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=body,
    )

    total = block_lds_reduce(b, acc, lds, tid, block_size=BS, combine=combine)

    if spec.op == "mean":
        total = b.fmul(total, b.rcp(b.const_f32(float(N))))

    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        store_scalar_from_f32(b, Y, row, total, dtype=spec.dtype)

    return b.kernel


def reduce2d_grid(m: int, spec: Reduce2DSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def reduce2d_signature(spec: Reduce2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .build()
    )
