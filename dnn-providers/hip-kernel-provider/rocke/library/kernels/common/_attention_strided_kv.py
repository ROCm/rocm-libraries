# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime byte-stride addressing for the tiled split-KV segment kernels.

K/V have logical [batch, head, token, dim] coordinates and unit dim stride.
The binding checks alignment and bounds before packing these arguments.
Mirrored by instance_attention_strided_kv_internal.h in the C++ engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, IRBuilder, Value


@dataclass(frozen=True)
class StridedKvParams:
    batch: Value
    head: Value
    token: Value
    span: Value


def declare_strided_kv_params(b: IRBuilder, name: str) -> StridedKvParams:
    return StridedKvParams(
        b.param(f"{name}_stride_batch_bytes", I64),
        b.param(f"{name}_stride_head_bytes", I64),
        b.param(f"{name}_stride_token_bytes", I32),
        b.param(f"{name}_span_bytes", I32),
    )


def strided_kv_resource(
    b: IRBuilder, ptr: Value, p: StridedKvParams, seq: Value, head: Value
) -> Value:
    batch_offset = b.mul(b.zext(seq, I64), p.batch)
    head_offset = b.mul(b.zext(head, I64), p.head)
    base = b.global_ptr_add(ptr, b.add(batch_offset, head_offset))
    return b.buffer_rsrc(base, p.span)


def strided_kv_offset(
    b: IRBuilder,
    p: StridedKvParams,
    tile: Value,
    linear: Value,
    valid_length: Value,
    head_size: int,
    tile_size: int,
) -> Value:
    token = b.add(
        b.mul(tile, b.const_i32(tile_size)), b.div(linear, b.const_i32(head_size))
    )
    dim = b.mod(linear, b.const_i32(head_size))
    valid = b.cmp_lt(token, valid_length)
    # Clamp before multiplying: a padded tile coordinate must not overflow even
    # if the last valid row just fits the binding's buffer-offset bound.
    safe_token = b.select(valid, token, b.const_i32(0))
    offset = b.add(b.mul(safe_token, p.token), b.mul(dim, b.const_i32(2)))
    # The binding limits span to 0x7fff0000 bytes. An out-of-range buffer load
    # supplies zero to LDS without reading adjacent heads/batches or stale KV.
    return b.select(valid, offset, b.const_i32(0x7FFFFFFF))
