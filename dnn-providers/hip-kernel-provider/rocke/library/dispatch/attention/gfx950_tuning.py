# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Explicit, sweep-only gfx950 unified-attention candidates."""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry

from .tuning_common import AttentionGeometryVariant, make_tuning_candidate


# Absolute T=64/128 points are already represented by one of these multipliers
# for every supported block_size {16,32,64}; omit duplicate candidates.
_TILES = ("1x", "2x", "4x", "8x")


def _variants():
    # Narrow 16x16 path supports all four CTA widths.
    for backend in ("llvm", "hipcc"):
        for tile in _TILES:
            for nw in (1, 2, 4, 8):
                yield AttentionGeometryVariant(
                    arch="gfx950",
                    path="2d",
                    codepath="narrow",
                    builder_kind="tiled",
                    tile_policy=tile,
                    num_warps=nw,
                    block_m_per_warp=16,
                    compile_backend=backend,
                )
    # gfx950 32x32 paths require M-per-warp=32 and reject nw=8.
    for codepath in ("wide32", "transposed32"):
        for backend in ("llvm", "hipcc"):
            for tile in _TILES:
                for nw in (1, 2, 4):
                    yield AttentionGeometryVariant(
                        arch="gfx950",
                        path="2d",
                        codepath=codepath,
                        builder_kind="tiled",
                        tile_policy=tile,
                        num_warps=nw,
                        block_m_per_warp=32,
                        compile_backend=backend,
                    )
    # gfx950 3D has fixed BLOCK_M=16 and T=block_size; segment count is the
    # load-balancing geometry exposed by the kernel.
    for segments in (8, 16, 32, 64, 128):
        yield AttentionGeometryVariant(
            arch="gfx950",
            path="3d",
            codepath="splitkv",
            builder_kind="tiled_3d",
            tile_policy="1x",
            num_segments=segments,
        )


GFX950_TUNING_VARIANTS = tuple(_variants())


def register(registry: CandidateRegistry) -> None:
    for variant in GFX950_TUNING_VARIANTS:
        registry.register(make_tuning_candidate(variant))
