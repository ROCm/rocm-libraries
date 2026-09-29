# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit, sweep-only gfx942 unified-attention candidates."""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry

from .tuning_common import AttentionGeometryVariant, make_tuning_candidate


# Absolute T=64/128 points are already represented by one of these multipliers
# for every supported block_size {16,32,64}; omit duplicate candidates.
_TILES = ("1x", "2x", "4x", "8x")


def _variants():
    # The generic gfx942 body supports one or two 16-row atoms per warp.
    for tile in _TILES:
        for nw in (1, 2, 4, 8):
            for mw in (16, 32):
                yield AttentionGeometryVariant(
                    arch="gfx942",
                    path="2d",
                    codepath="narrow",
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=mw,
                )
    # CDNA3 wide paths use the selectable 32x32x8 atom.
    for codepath in ("wide32x8", "transposed_x8"):
        for tile in _TILES:
            for nw in (1, 2, 4, 8):
                yield AttentionGeometryVariant(
                    arch="gfx942",
                    path="2d",
                    codepath=codepath,
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=32,
                )
    # This builder owns real BLOCK_M=128 / 256-thread geometry internally; its
    # spec is a discriminator and not another free tiling axis.
    yield AttentionGeometryVariant(
        arch="gfx942",
        path="2d",
        codepath="gfx942_4warp",
        builder_kind="gfx942_4warp_gqa",
        tile_policy="64",
        num_warps=1,
        block_m_per_warp=32,
    )
    for tile in ("1x", "half"):
        # ``half`` is translated below by the explicit builder through a
        # concrete token; use per-block variants because it is only legal for
        # block sizes >=32.
        for segments in (8, 16, 32, 64, 128):
            yield AttentionGeometryVariant(
                arch="gfx942",
                path="3d",
                codepath="splitkv",
                builder_kind="tiled_3d",
                tile_policy=tile,
                num_segments=segments,
            )


GFX942_TUNING_VARIANTS = tuple(_variants())


def register(registry: CandidateRegistry) -> None:
    for variant in GFX942_TUNING_VARIANTS:
        registry.register(make_tuning_candidate(variant))
