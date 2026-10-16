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


def resolve_attention_kv_loader(
    arch: str, kv_layout: str, page_size: int, tile_size: int
) -> AttentionKVLoaderKind:
    """Resolve the current implementation without changing its eligibility.

    gfx942 retains its descriptor for subpage tiles; gfx950 uses token gather.
    A future whole-page DMA loader can own separate transfer geometry checks.
    """
    if arch not in ("gfx942", "gfx950"):
        raise ValueError("tiled KV loader requires gfx942/gfx950")
    if kv_layout == "strided":
        return AttentionKVLoaderKind.STRIDED
    if kv_layout != "paged":
        raise ValueError("kv_layout must be 'paged' or 'strided'")
    gather = tile_size > page_size if arch == "gfx942" else tile_size != page_size
    return (
        AttentionKVLoaderKind.PAGED_TOKEN_GATHER
        if gather
        else AttentionKVLoaderKind.PAGED_DESCRIPTOR
    )
