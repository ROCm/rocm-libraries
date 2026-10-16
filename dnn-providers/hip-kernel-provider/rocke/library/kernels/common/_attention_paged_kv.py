# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Logical-token gather into the existing tiled decode LDS layout.

The caller bounds both contiguous KV pools to 0x7fff0000 bytes. Page-table
indices are clamped independently of the buffer-load mask, including prefetch
past the final tile. Page IDs and sequence lengths are caller-owned metadata.
Mirrored by instance_attention_paged_kv_internal.h.
"""

from rocke.core.ir import I32


def paged_kv_offset(
    b, table, seq_base, head, tile, linear, length, hd, page, tile_size, heads
):
    token = b.add(b.mul(tile, b.const_i32(tile_size)), b.div(linear, b.const_i32(hd)))
    dim = b.mod(linear, b.const_i32(hd))
    valid = b.cmp_lt(token, length)
    safe_token = b.select(valid, token, b.const_i32(0))
    page_index = b.div(safe_token, b.const_i32(page))
    physical = b.global_load(table, b.add(seq_base, page_index), I32, align=4)
    page_offset = b.mul(physical, b.const_i32(page * heads * hd * 2))
    row = b.mod(safe_token, b.const_i32(page))
    row_offset = b.mul(row, b.const_i32(heads * hd * 2))
    head_offset = b.mul(head, b.const_i32(hd * 2))
    dim_offset = b.mul(dim, b.const_i32(2))
    offset = b.add(b.add(page_offset, row_offset), b.add(head_offset, dim_offset))
    return b.select(valid, offset, b.const_i32(0x7FFFFFFF))
