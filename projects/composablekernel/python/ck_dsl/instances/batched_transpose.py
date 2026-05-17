# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched 2D transpose kernel instance.

DSL counterpart of CK Tile's ``example/ck_tile/35_batched_transpose``.
Same per-batch algorithm as :mod:`ck_dsl.instances.transpose`:

  1. Coalesced HBM read into LDS staging (one ``vec``-wide load per
     thread).
  2. ``__syncthreads``.
  3. Column-strided LDS read packs into one output vector + one
     coalesced HBM store per thread.

The batched extension just adds a batch axis on top:

* Grid becomes ``(W/tile_n, H/tile_m, batch_count)``.
* ``block_id_z`` picks the per-batch pointer offset for X and Y.
* The per-batch element strides default to ``H*W`` (contiguous packed
  batches) and can be overridden at launch time via the
  ``batch_stride_x`` / ``batch_stride_y`` runtime args.

What we cover today:

* Dtypes ``f16`` / ``bf16`` (same as the underlying 2D transpose).
* Square tiles ``tile_m == tile_n in {16, 32, 64}`` (same as the
  underlying 2D transpose; non-square is a v2 follow-up shared with
  :class:`Transpose2DSpec`).
* Vec widths ``{2, 4, 8}`` for both DMA phases.
* H/W must be divisible by the tile size, batch_stride_x defaults
  to ``H*W`` (caller can override for non-contiguous batches).

Implementation note: the batch dim sits on ``block_id_z`` rather than
folding into one of the planar tile counters. This keeps the per-batch
launch trivially parallel and lets the kernel reuse the
single-batch indexing logic byte-for-byte (modulo the per-batch
pointer offset added at the top of the kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    kernel_name_join,
    validate_io,
)
from ..helpers.tensor_view import make_global_view, make_lds_view


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class BatchedTranspose2DSpec:
    """One batched transpose kernel configuration.

    The tile-level geometry mirrors :class:`Transpose2DSpec`; only the
    batch axis is added here. ``batch_count`` is informational
    (informs ``batched_transpose2d_grid``); the kernel itself reads
    ``block_id_z`` so the launch grid's z extent is what selects the
    batch index at runtime.
    """

    tile_m: int = 64
    tile_n: int = 64
    vec: int = 8
    dtype: DType = "f16"
    lds_pad: int = 8
    name: str = "ck_dsl_batched_transpose2d"

    @property
    def block_size(self) -> int:
        return (self.tile_m * self.tile_n) // self.vec

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"{self.tile_m}x{self.tile_n}",
            f"v{self.vec}",
            f"p{self.lds_pad}",
        )


def is_valid_spec(spec: BatchedTranspose2DSpec) -> Tuple[bool, str]:
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
        )
    )
    if not ok:
        return ok, why
    if spec.tile_m not in (16, 32, 64) or spec.tile_n not in (16, 32, 64):
        return False, "tile_m/tile_n must be in {16, 32, 64}"
    if spec.tile_m != spec.tile_n:
        return False, "non-square tiles not yet supported"
    if (spec.tile_m * spec.tile_n) % spec.vec:
        return False, "tile area must be divisible by vec"
    if spec.block_size > 1024:
        return False, (f"block_size {spec.block_size} > 1024 hardware cap")
    return True, "ok"


def build_batched_transpose2d(spec: BatchedTranspose2DSpec) -> KernelDef:
    """Build the IR for one batched transpose instance.

    Kernel signature::

        (X: ptr<dtype, global>,    # input  (batch_count * H * W layout)
         Y: ptr<dtype, global>,    # output (batch_count * W * H layout)
         H: i32, W: i32,
         batch_stride_x: i32,      # per-batch element stride for X
         batch_stride_y: i32)      # per-batch element stride for Y
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid batched_transpose2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    TM, TN, vec, BS = spec.tile_m, spec.tile_n, spec.vec, spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    H = b.param("H", I32)
    W = b.param("W", I32)
    batch_stride_x = b.param("batch_stride_x", I32)
    batch_stride_y = b.param("batch_stride_y", I32)

    tid = b.thread_id_x()

    c_vec = b.const_i32(vec)
    c_TN_chunks = b.const_i32(TN // vec)
    c_TM_chunks = b.const_i32(TM // vec)

    # Per-batch pointer offsets (in elements). The descriptor below
    # rides these into its origin's first axis so the offset folds
    # into the same per-thread address calculation as the planar
    # 2D transpose.
    batch_idx = b.block_id_z()
    batch_off_x = b.mul(batch_idx, batch_stride_x)
    batch_off_y = b.mul(batch_idx, batch_stride_y)

    tile_x = b.block_id_x()
    tile_y = b.block_id_y()

    h0 = b.mul(tile_y, b.const_i32(TM))
    w0 = b.mul(tile_x, b.const_i32(TN))

    # 3D views with a leading batch axis whose stride is the per-batch
    # element stride (passed in as a runtime arg). The middle and inner
    # axes are exactly what the single-batch transpose uses.
    x_view = make_global_view(
        X, shape=(1, TM, TN), dtype=io_ty, strides=(batch_stride_x, W, 1)
    )
    y_view = make_global_view(
        Y, shape=(1, TN, TM), dtype=io_ty, strides=(batch_stride_y, H, 1)
    )
    x_tile = x_view.tile(lengths=(1, TM, TN), origin=(batch_idx, h0, w0))
    y_tile = y_view.tile(lengths=(1, TN, TM), origin=(batch_idx, w0, h0))

    lds_view = make_lds_view(
        b, dtype=io_ty, shape=(TM, TN + spec.lds_pad), name_hint="lds_xpose"
    )
    lds_tile = lds_view.tile(lengths=(TM, TN), origin=(b.const_i32(0), b.const_i32(0)))

    # Phase 1: coalesced global -> LDS.
    row1 = b.div(tid, c_TN_chunks)
    col1_chunk = b.mod(tid, c_TN_chunks)
    col1 = b.mul(col1_chunk, c_vec)

    x_vec = x_tile.load_vec(b, b.const_i32(0), row1, col1, n=vec)
    lds_tile.store_vec(b, row1, col1, value=x_vec, n=vec)
    b.sync()

    # Phase 2: column-strided LDS reads, coalesced global stores.
    col2 = b.div(tid, c_TM_chunks)
    row2_chunk = b.mod(tid, c_TM_chunks)
    row2_base = b.mul(row2_chunk, c_vec)

    elems = [
        lds_tile.load_scalar(b, b.add(row2_base, b.const_i32(i)), col2)
        for i in range(vec)
    ]
    out_vec = b.vec_pack(elems, io_ty)
    y_tile.store_vec(b, b.const_i32(0), col2, row2_base, value=out_vec, n=vec)

    # Suppress "unused parameter" warnings for batch_off_x / batch_off_y
    # by referencing them in the IR (the offset arithmetic above already
    # folds them via the descriptor's leading-axis stride).
    _ = batch_off_x  # noqa: F841
    _ = batch_off_y  # noqa: F841

    return b.kernel


def batched_transpose2d_grid(
    h: int, w: int, batch_count: int, spec: BatchedTranspose2DSpec
) -> Tuple[int, int, int]:
    """Return the launch grid: ``(W/tile_n, H/tile_m, batch_count)``."""
    nx = (w + spec.tile_n - 1) // spec.tile_n
    ny = (h + spec.tile_m - 1) // spec.tile_m
    return (nx, ny, batch_count)


def batched_transpose2d_signature(spec: BatchedTranspose2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("H", "i32")
        .scalar("W", "i32")
        .scalar("batch_stride_x", "i32")
        .scalar("batch_stride_y", "i32")
        .build()
    )


__all__ = [
    "BatchedTranspose2DSpec",
    "batched_transpose2d_grid",
    "batched_transpose2d_signature",
    "build_batched_transpose2d",
    "is_valid_spec",
]
