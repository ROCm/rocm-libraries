# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Not vendored: written for this provider against flydsl 0.3.4 (pip wheel). Its
# structure follows AITER's FlyDSL MoE gather-reduce epilogue:
#   upstream: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/moe_gather_reduce.py
# What is taken from it: one workgroup per gathered row, the row's index read
# once per workgroup, the row folded into a 64-bit base address so each buffer
# descriptor addresses one row (its 32-bit offsets never span the whole table),
# a zero-length descriptor that switches off the reads of a row that may not be
# read, and 16-byte raw-dword copies. What differs: a plain copy (no route
# weights, no top-k reduction, no LDS), the index read with a scalar load (so
# the row's address stays in SGPRs), INT32 or INT64 indices, an outer axis
# (gather along any axis of a packed tensor), the row width a runtime argument,
# and a one-element-per-lane variant for rows whose bytes are not a multiple of
# 16.

"""Gather along one axis: ``y[a, i, b] = x[a, indices[i], b]``.

With ``x`` packed as ``[outer, axis_extent, inner]`` and ``y`` packed as
``[outer, num_indices, inner]`` (``indices`` read flat, any rank, ``num_indices``
elements), an embedding lookup is ``outer = 1``: ``x`` is the ``[V, H]`` table,
``inner = H``.

Grid ``(num_indices, ceil(inner / (BLOCK_THREADS * vec_elems)), outer)``, block
``(BLOCK_THREADS, 1, 1)``. Every lane moves ``vec_elems`` contiguous elements:
one 16-byte dwordx4 copy in a ``vector`` object (which needs ``inner`` bytes to be
a multiple of 16 and 16-byte aligned bases, so every lane's vector is whole), or
one element in a scalar object, which serves every row width.

Row addresses are 64-bit: ``x``'s row starts at
``x + ((a * axis_extent + index) * inner) * elem_bytes`` computed in i64, so a
table past 4 GiB (Qwen's 248320 x 5120 bf16 is 2.4 GiB, and grows with the
vocabulary) is addressed exactly. Only one row's bytes must fit the 32-bit
buffer offset.

An index outside ``[0, axis_extent)`` is undefined behaviour in the op's contract
(hipDNN's CPU reference throws). Here it reads nothing: its row's descriptor has
zero records, so every load returns 0 and the output row is written as zeros.
No memory outside ``x`` is ever read.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from kernels.common import buffer_ops

BLOCK_THREADS = 256
VECTOR_BYTES = 16

_ELEM_BYTES = {"bf16": 2, "f16": 2, "f32": 4}
_INDEX_BYTES = {"i32": 4, "i64": 8}


def kernel_name(dtype_str: str, index_dtype_str: str, vector: bool) -> str:
    return f"flydsl_gather_{dtype_str}_{index_dtype_str}_{'v16' if vector else 's'}"


def vec_elems(dtype_str: str, vector: bool) -> int:
    """Elements one lane moves: a 16-byte vector, or one element."""
    return VECTOR_BYTES // _ELEM_BYTES[dtype_str] if vector else 1


def build_gather_module(dtype_str: str, index_dtype_str: str, vector: bool):
    """Build the gather launcher for one (dtype, index dtype, access width).

    The data dtype only sets the element size: values are copied as raw bits.
    """
    if dtype_str not in _ELEM_BYTES:
        raise ValueError(f"unsupported dtype {dtype_str!r}")
    if index_dtype_str not in _INDEX_BYTES:
        raise ValueError(f"unsupported index dtype {index_dtype_str!r}")

    elem_bytes = _ELEM_BYTES[dtype_str]
    index_is_64 = index_dtype_str == "i64"
    lane_elems = vec_elems(dtype_str, vector)
    block_elems = BLOCK_THREADS * lane_elems
    # Raw bits: a 16-bit element moves as an i16, a 32-bit one as an i32, a
    # vector as four i32 dwords.
    scalar_ty = fx.Int16 if elem_bytes == 2 else fx.Int32

    @flyc.kernel(name=kernel_name(dtype_str, index_dtype_str, vector))
    def gather_kernel(
        X: fx.Pointer,
        Indices: fx.Pointer,
        Y: fx.Pointer,
        num_indices: fx.Int32,
        axis_extent: fx.Int32,
        inner: fx.Int32,
    ):
        index_pos = fx.Int32(fx.block_idx.x)
        chunk = fx.Int32(fx.block_idx.y)
        outer_pos = fx.Int32(fx.block_idx.z)
        tid = fx.Int32(fx.thread_idx.x)

        # The workgroup's index, read with a scalar load so it -- and every row
        # address derived from it -- stays wave-uniform in SGPRs, as a buffer
        # descriptor's base must be.
        index_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(Indices)),
            num_records_bytes=fx.Int64(num_indices)
            * fx.Int64(_INDEX_BYTES[index_dtype_str]),
        )
        if const_expr(index_is_64):
            lo = buffer_ops.buffer_load(
                index_rsrc, index_pos * fx.Int32(2), vec_width=1, is_scalar=True
            )
            hi = buffer_ops.buffer_load(
                index_rsrc,
                index_pos * fx.Int32(2) + fx.Int32(1),
                vec_width=1,
                is_scalar=True,
            )
            index = (fx.Int64(fx.Int32(hi)) << fx.Int64(32)) | (
                fx.Int64(fx.Int32(lo)) & fx.Int64(0xFFFFFFFF)
            )
        else:
            index = fx.Int64(
                fx.Int32(
                    buffer_ops.buffer_load(
                        index_rsrc, index_pos, vec_width=1, is_scalar=True
                    )
                )
            )

        # One unsigned compare covers both bounds: a negative index is a huge
        # unsigned one.
        in_range = fx.Uint64(index) < fx.Uint64(fx.Int64(axis_extent))
        row = in_range.select(index, fx.Int64(0))

        inner_i64 = fx.Int64(inner)
        row_bytes = inner * fx.Int32(elem_bytes)
        x_row = (fx.Int64(outer_pos) * fx.Int64(axis_extent) + row) * inner_i64
        y_row = (
            fx.Int64(outer_pos) * fx.Int64(num_indices) + fx.Int64(index_pos)
        ) * inner_i64
        x_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(X)) + x_row * fx.Int64(elem_bytes),
            num_records_bytes=fx.Int64(in_range.select(row_bytes, fx.Int32(0))),
        )
        y_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(Y)) + y_row * fx.Int64(elem_bytes),
            num_records_bytes=fx.Int64(row_bytes),
        )

        lane = chunk * fx.Int32(BLOCK_THREADS) + tid
        first = lane * fx.Int32(lane_elems)
        if first < inner:
            if const_expr(vector):
                # 16 bytes per lane: offsets in dwords.
                data = buffer_ops.buffer_load(
                    x_rsrc, lane * fx.Int32(4), vec_width=4, dtype=fx.Int32
                )
                buffer_ops.buffer_store(data, y_rsrc, lane * fx.Int32(4))
            else:
                data = buffer_ops.buffer_load(
                    x_rsrc, first, vec_width=1, dtype=scalar_ty
                )
                buffer_ops.buffer_store(data, y_rsrc, first)

    @flyc.jit
    def launch_gather(
        X: fx.Pointer,
        Indices: fx.Pointer,
        Y: fx.Pointer,
        num_indices: fx.Int32,
        axis_extent: fx.Int32,
        inner: fx.Int32,
        outer: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008  framework idiom
    ):
        chunks = (fx.Int64(inner) + fx.Int64(block_elems - 1)) // fx.Int64(block_elems)
        launcher = gather_kernel(X, Indices, Y, num_indices, axis_extent, inner)
        launcher.launch(
            grid=(fx.Int64(num_indices), chunks, fx.Int64(outer)),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gather
