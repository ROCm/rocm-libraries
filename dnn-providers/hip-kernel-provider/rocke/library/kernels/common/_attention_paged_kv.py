# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Logical-token gather into the existing tiled decode LDS layout.

The caller bounds both contiguous KV pools to 0x7fff0000 bytes. Page-table
indices are clamped independently of the buffer-load mask, including prefetch
past the final tile. Page IDs and sequence lengths are caller-owned metadata.
Mirrored by instance_attention_paged_kv_internal.h.
"""

from rocke.core.ir import I32
from rocke.helpers.transforms import TensorDescriptor, unmerge


def paged_kv_token(b, tile, linear, hd, tile_size):
    """Decompose a compute-tile coordinate into logical token and dimension."""
    coords = ("row", "dim")
    desc = TensorDescriptor.naive(
        "paged_tile", lengths=(tile_size, hd), coord_names=coords
    ).transform(unmerge("linear", into=coords, dims=(tile_size, hd)))
    lower = desc.unmerge_lower(b, linear=linear)
    token_desc = TensorDescriptor.naive(
        "paged_token",
        lengths=(1, tile_size),
        strides=(tile_size, 1),
        coord_names=("tile", "row"),
    )
    token, _ = token_desc.offset(b, tile=tile, row=lower["row"])
    return token, lower["dim"]


def paged_kv_page(b, safe_token, page):
    """Split an already sanitized token before the caller's page lookup."""
    coords = ("logical_page", "page_token")
    desc = TensorDescriptor.naive(
        "paged_page", lengths=(1, page), coord_names=coords
    ).transform(unmerge("safe_token", into=coords, dims=(1, page)))
    lower = desc.unmerge_lower(b, safe_token=safe_token)
    return lower["logical_page"], lower["page_token"]


def paged_kv_byte_offset(b, physical, row, head, dim, hd, page, heads):
    """Flatten checked physical coordinates with explicit two-byte KV strides."""
    desc = TensorDescriptor.naive(
        "paged_kv_bytes",
        lengths=(1, page, heads, hd),
        strides=(page * heads * hd * 2, heads * hd * 2, hd * 2, 2),
        coord_names=("physical_page", "page_token", "kv_head", "dim"),
    )
    offset, _ = desc.offset(
        b, physical_page=physical, page_token=row, kv_head=head, dim=dim
    )
    return offset


def paged_kv_offset(
    b, table, seq_base, head, tile, linear, length, hd, page, tile_size, heads
):
    token, dim = paged_kv_token(b, tile, linear, hd, tile_size)
    valid = b.cmp_lt(token, length)
    safe_token = b.select(valid, token, b.const_i32(0))
    page_index, row = paged_kv_page(b, safe_token, page)
    physical = b.global_load(table, b.add(seq_base, page_index), I32, align=4)
    offset = paged_kv_byte_offset(b, physical, row, head, dim, hd, page, heads)
    return b.select(valid, offset, b.const_i32(0x7FFFFFFF))
