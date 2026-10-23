# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Internal addressing choices for the tiled 3D attention loaders.

Inputs have already passed the caller's raw type and geometry validation.
Transfer width, storage conversion and address width are independent choices.
"""

from enum import Enum


class AttentionKVLoaderKind(Enum):
    STRIDED = "strided"
    PAGED_DESCRIPTOR = "paged_descriptor"
    PAGED_TOKEN_GATHER = "paged_token_gather"
    PAGED_MULTI_PAGE = "paged_multi_page"


def resolve_attention_kv_loader(
    arch: str,
    kv_layout: str,
    page_size: int,
    tile_size: int,
    *,
    use_i64_kv_addr: bool = False
) -> AttentionKVLoaderKind:
    """Resolve the current implementation without changing its eligibility.

    gfx942 retains its descriptor for subpage tiles. gfx950 retains the T32
    token gather and uses whole-page DMA for larger multi-page tiles or
    caches that need 64-bit page addresses.
    """
    if arch not in ("gfx942", "gfx950"):
        raise ValueError("tiled KV loader requires gfx942/gfx950")
    if kv_layout == "strided":
        return AttentionKVLoaderKind.STRIDED
    if kv_layout != "paged":
        raise ValueError("kv_layout must be 'paged' or 'strided'")
    if (
        arch == "gfx950"
        and page_size >= 16
        and tile_size > page_size
        and (tile_size in (64, 128) or use_i64_kv_addr)
    ):
        return AttentionKVLoaderKind.PAGED_MULTI_PAGE
    gather = tile_size > page_size if arch == "gfx942" else tile_size != page_size
    return (
        AttentionKVLoaderKind.PAGED_TOKEN_GATHER
        if gather
        else AttentionKVLoaderKind.PAGED_DESCRIPTOR
    )
