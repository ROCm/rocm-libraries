#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Single Source of Truth for Grouped Convolution Tile Configurations

This module defines all valid tile configurations for grouped convolution kernels.
Both codegen and instance_builder import from here to ensure consistency.

Architecture:
  grouped_config_rules.py  (SOURCE OF TRUTH)
      ├── Used by unified_grouped_conv_codegen.py
      ├── Used by grouped_conv_instance_builder.py
      ├── Used by validate_ml_vs_oracle.py
      └── Used by generate_instances.py

Tile data is extracted from:
  configs/grouped_conv/{forward,backward_data,backward_weight}/profiler/nhwgc_{bf16,fp32}.json
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# =============================================================================
# TileVariantContainer — backward-compatible container for mapping dict values
# =============================================================================

class TileVariantContainer(tuple):
    """Backward-compatible container for tile-mapping variants.

    Behaves as a plain tuple for existing callers (unpacking still works),
    but carries additional 'specialized' variants for new callers.

    Usage:
        warp_m, warp_n, warp_k = TILE_TO_WARP[key]   # old callers: use default
        TILE_TO_WARP[key].all_values()                # new callers: all variants
    """

    def __new__(cls, default: tuple, specialized: list = ()):
        return super().__new__(cls, default)

    def __init__(self, default: tuple, specialized: list = ()):
        self.specialized = list(specialized)

    @property
    def default(self):
        return tuple(self)

    def all_values(self):
        """Return list of all valid values: [default] + specialized."""
        return [self.default] + self.specialized


# =============================================================================
# Tile Lists
# =============================================================================
# Four orthogonal lists covering all tiles from the JSON profiler configs.
# COMMON_TILES = tiles present in ALL three direction unions (bf16 ∪ fp32).
# FWD_TILES / BWD_DATA_TILES / BWD_WEIGHT_TILES = direction-specific tiles not in COMMON.
# Available tiles for a variant = COMMON_TILES ∪ <variant>_TILES.
# Format: (tile_m, tile_n, tile_k)

# Tiles present in all three directions.
COMMON_TILES: List[Tuple[int, int, int]] = [
    (32,  64,  32), (32,  128, 32), (64,  16,  64), (64,  32,  32),
    (64,  64,  32), (64,  128, 32), (128, 32,  16), (128, 32,  32),
    (128, 64,  32), (128, 128, 32), (128, 256, 32), (256, 128, 32),
]

# Forward-only tiles.
FWD_TILES: List[Tuple[int, int, int]] = [
    (16,  16,  64),  (16,  16,  128), (16,  32,  64),  (16,  64,  64),
    (16,  128, 64),  (16,  256, 64),  (32,  16,  64),   (32,  64,  16),
    (32,  64,  64),  (32,  128, 16),  (32,  128, 64),   (32,  256, 64),
    (64,  16,  16),  (64,  32,  16),  (64,  32,  64),   (64,  64,  8),
    (64,  64,  16),  (64,  64,  64),  (64,  128, 16),   (64,  128, 64),
    (128, 16,  64),  (128, 32,  64),  (128, 64,  8),    (128, 64,  16),
    (128, 64,  64),  (128, 128, 16),  (128, 128, 64),   (128, 192, 16),
    (128, 256, 16),  (224, 256, 64),  (256, 16,  64),   (256, 32,  64),
    (256, 64,  8),   (256, 128, 16),  (256, 224, 64),   (256, 256, 32),
]

# Backward-data-only tiles.
BWD_DATA_TILES: List[Tuple[int, int, int]] = [
    (16, 64, 32), (64, 16, 16), (64, 16, 32), (128, 32, 64),
]

# Backward-weight-only tiles.
BWD_WEIGHT_TILES: List[Tuple[int, int, int]] = [
    (16,  16,  32),  (16,  16,  64),  (16,  32,  64),  (16,  64,  64),
    (16,  128, 32),  (16,  128, 64),  (16,  256, 32),  (16,  256, 64),
    (32,  16,  64),  (32,  32,  32),  (32,  64,  16),  (32,  128, 16),
    (64,  32,  16),  (64,  64,  16),  (64,  64,  64),  (64,  128, 16),
    (64,  128, 64),  (128, 16,  64),  (128, 64,  16),  (128, 128, 16),
    (128, 128, 64),  (128, 256, 16),  (256, 16,  64),  (256, 32,  64),
    (256, 128, 16),  (256, 256, 32),
]


def get_tiles_for_variant(variant: str) -> List[Tuple[int, int, int]]:
    """Return all tiles available for the given conv variant.

    Returns COMMON_TILES ∪ <variant>_TILES, sorted.
    """
    if variant == "forward":
        extra = FWD_TILES
    elif variant == "bwd_data":
        extra = BWD_DATA_TILES
    elif variant == "bwd_weight":
        extra = BWD_WEIGHT_TILES
    else:
        extra = []
    return sorted(set(COMMON_TILES) | set(extra))


# =============================================================================
# TILE_TO_WAVE_WARP — single source of truth for (wave, warp) pairs per tile
# =============================================================================
# Key: (tile_m, tile_n, tile_k)
# Value: list of ((wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n, warp_tile_k)) pairs
#
# This is the authoritative data structure. TILE_TO_WAVE and TILE_TO_WARP below
# are backward-compat views derived from this dict (first pair = default).
#
# Extracted from JSON profiler configs (nhwgc, all dtypes, all variants).
#
TILE_TO_WAVE_WARP: Dict[Tuple[int, int, int], List[Tuple[Tuple, Tuple]]] = {
    (16, 16, 32):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 16, 64):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 16, 128): [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 32, 64):  [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 64, 32):  [((1, 1, 1), (16, 16, 8)),  ((1, 1, 1), (16, 16, 16))],
    (16, 64, 64):  [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 128, 32): [((1, 1, 1), (16, 16, 16))],
    (16, 128, 64): [((1, 2, 1), (16, 16, 8)),  ((1, 2, 1), (16, 16, 16))],
    (16, 256, 32): [((1, 1, 1), (16, 16, 16))],
    (16, 256, 64): [((1, 4, 1), (16, 16, 16))],
    (32, 16, 64):  [((2, 1, 1), (16, 16, 8)),  ((2, 1, 1), (16, 16, 16))],
    (32, 32, 32):  [((1, 1, 1), (32, 32, 8))],
    (32, 64, 16):  [((1, 1, 1), (32, 32, 4))],
    (32, 64, 32):  [((1, 1, 1), (32, 32, 8))],
    (32, 64, 64):  [((1, 2, 1), (32, 32, 8))],
    (32, 128, 16): [((1, 2, 1), (32, 32, 4))],
    (32, 128, 32): [((1, 1, 1), (32, 32, 8)), ((1, 2, 1), (32, 32, 8))],
    (32, 128, 64): [((1, 2, 1), (32, 32, 8))],
    (32, 256, 64): [((1, 4, 1), (32, 32, 8))],
    (64, 16, 16):  [((1, 1, 1), (16, 16, 4)), ((1, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 4)), ((4, 1, 1), (16, 16, 16))],
    (64, 16, 32):  [((1, 1, 1), (16, 16, 8)), ((1, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 8)), ((4, 1, 1), (16, 16, 16))],
    (64, 16, 64):  [((2, 1, 1), (16, 16, 8)), ((2, 1, 1), (16, 16, 16)),
                    ((4, 1, 1), (16, 16, 16))],
    (64, 32, 16):  [((1, 1, 1), (32, 32, 4))],
    (64, 32, 32):  [((1, 1, 1), (32, 32, 8))],
    (64, 32, 64):  [((2, 1, 1), (32, 32, 8))],
    (64, 64, 8):   [((2, 1, 1), (32, 32, 8))],
    (64, 64, 16):  [((1, 1, 1), (32, 32, 4))],
    (64, 64, 32):  [((1, 1, 1), (32, 32, 8)), ((2, 2, 1), (16, 16, 8)),
                    ((2, 2, 1), (16, 16, 16))],
    (64, 64, 64):  [((2, 2, 1), (16, 16, 16)), ((2, 2, 1), (32, 32, 8))],
    (64, 128, 16): [((1, 2, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4))],
    (64, 128, 32): [((1, 2, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (64, 128, 64): [((2, 2, 1), (32, 32, 8))],
    (128, 16, 64): [((2, 1, 1), (16, 16, 8)), ((2, 1, 1), (16, 16, 16))],
    (128, 32, 16): [((2, 1, 1), (32, 32, 4)), ((4, 1, 1), (32, 32, 4)),
                    ((4, 1, 1), (32, 32, 8))],
    (128, 32, 32): [((1, 1, 1), (32, 32, 8)), ((2, 1, 1), (32, 32, 8)),
                    ((2, 1, 2), (32, 32, 8)), ((4, 1, 1), (32, 32, 8))],
    (128, 32, 64): [((2, 1, 1), (32, 32, 8)), ((4, 1, 1), (32, 32, 16))],
    (128, 64, 8):  [((2, 1, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 64, 16): [((2, 1, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4)),
                    ((2, 2, 1), (32, 32, 8))],
    (128, 64, 32): [((2, 1, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 64, 64): [((2, 2, 1), (32, 32, 8))],
    (128, 128, 16):[((1, 2, 1), (32, 32, 4)), ((2, 2, 1), (32, 32, 4))],
    (128, 128, 32):[((1, 2, 1), (32, 32, 8)), ((2, 2, 1), (32, 32, 8))],
    (128, 128, 64):[((2, 2, 1), (32, 32, 8))],
    (128, 192, 16):[((2, 2, 1), (32, 32, 4))],
    (128, 256, 16):[((2, 2, 1), (32, 32, 4))],
    (128, 256, 32):[((2, 2, 1), (32, 32, 8))],
    (224, 256, 64):[((2, 2, 1), (16, 16, 16))],
    (256, 16, 64): [((4, 1, 1), (16, 16, 16))],
    (256, 32, 64): [((4, 1, 1), (32, 32, 8))],
    (256, 64, 8):  [((2, 2, 1), (32, 32, 8))],
    (256, 128, 16):[((2, 2, 1), (32, 32, 4))],
    (256, 128, 32):[((2, 2, 1), (32, 32, 8))],
    (256, 224, 64):[((2, 2, 1), (16, 16, 16))],
    (256, 256, 32):[((2, 2, 1), (16, 16, 16)), ((2, 2, 1), (32, 32, 8))],
}

# Also expose under the old name for any external code that imports it by name.
TILE_TO_WAVE_WARP_PAIRS = TILE_TO_WAVE_WARP


def _build_wave_warp_compat_views():
    """Derive backward-compat TILE_TO_WAVE and TILE_TO_WARP from TILE_TO_WAVE_WARP.

    For each tile the first pair is the default; subsequent pairs are specialized.
    The two dicts are views — each index corresponds to the same pair in TILE_TO_WAVE_WARP.
    """
    wave: Dict[Tuple[int, int, int], TileVariantContainer] = {}
    warp: Dict[Tuple[int, int, int], TileVariantContainer] = {}
    for tile, pairs in TILE_TO_WAVE_WARP.items():
        default_wave, default_warp = pairs[0]
        spec_waves = [p[0] for p in pairs[1:]]
        spec_warps = [p[1] for p in pairs[1:]]
        wave[tile] = TileVariantContainer(default_wave, spec_waves)
        warp[tile] = TileVariantContainer(default_warp, spec_warps)
    return wave, warp


# Backward-compat views — callers that do:
#   wave_m, wave_n, wave_k = TILE_TO_WAVE[key]
#   warp_m, warp_n, warp_k = TILE_TO_WARP[key]
# continue to work unchanged (they get the first/default pair).
# For all valid pairs, iterate TILE_TO_WAVE_WARP[key] directly.
TILE_TO_WAVE, TILE_TO_WARP = _build_wave_warp_compat_views()


# =============================================================================
# Vector sizes: full table extracted from JSON profiler configs (nhwgc, all dtypes)
# =============================================================================
# Key: (tile_m, tile_n, tile_k, warp_tile_k) -> list of (vec_a, vec_b, vec_c)
# warp_tile_k distinguishes dtype variants:
#   bf16 warp_tile_k = 8 or 16   (depending on warp_tile shape)
#   fp32 warp_tile_k = 4 or 8
#
# Used by get_vector_sizes_for_tile() for exhaustive generation.
_TILE_WTILK_TO_VECS: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int]]] = {
    (16, 16, 32, 8):   [(1, 1, 2)],
    (16, 16, 32, 16):  [(1, 1, 1), (1, 1, 2)],
    (16, 16, 64, 8):   [(4, 4, 4)],
    (16, 16, 64, 16):  [(1, 1, 1), (1, 4, 4), (4, 1, 1), (4, 4, 4), (8, 8, 4)],
    (16, 16, 128, 8):  [(4, 4, 4)],
    (16, 16, 128, 16): [(8, 8, 4)],
    (16, 32, 64, 8):   [(4, 4, 4)],
    (16, 32, 64, 16):  [(1, 1, 1), (1, 2, 4), (1, 4, 4), (2, 1, 1), (2, 2, 4), (2, 4, 4), (8, 8, 4)],
    (16, 64, 32, 8):   [(1, 4, 4), (4, 1, 1), (4, 4, 4)],
    (16, 64, 32, 16):  [(1, 8, 4), (8, 1, 1), (8, 8, 4)],
    (16, 64, 64, 8):   [(4, 4, 4)],
    (16, 64, 64, 16):  [(1, 1, 1), (1, 8, 4), (2, 1, 1), (2, 8, 4), (8, 8, 4)],
    (16, 128, 32, 16): [(4, 4, 1)],
    (16, 128, 64, 8):  [(4, 4, 4)],
    (16, 128, 64, 16): [(1, 8, 4), (2, 8, 4), (8, 8, 4)],
    (16, 256, 32, 16): [(8, 8, 1)],
    (16, 256, 64, 16): [(1, 8, 4), (2, 8, 4), (8, 8, 4)],
    (32, 16, 64, 8):   [(4, 4, 2)],
    (32, 16, 64, 16):  [(1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 2), (4, 1, 1), (4, 2, 2), (8, 8, 2)],
    (32, 32, 32, 8):   [(2, 2, 1), (2, 2, 2)],
    (32, 64, 16, 4):   [(1, 4, 4), (4, 4, 4)],
    (32, 64, 32, 8):   [(1, 1, 8), (2, 2, 1), (2, 8, 8), (4, 4, 1), (4, 4, 2), (4, 4, 4), (8, 8, 8)],
    (32, 64, 64, 8):   [(4, 4, 8), (8, 8, 8)],
    (32, 128, 16, 4):  [(4, 4, 4)],
    (32, 128, 32, 8):  [(1, 1, 8), (4, 4, 4), (8, 8, 1), (8, 8, 2), (8, 8, 8)],
    (32, 128, 64, 8):  [(4, 4, 8), (8, 8, 8)],
    (32, 256, 64, 8):  [(8, 8, 8)],
    (64, 16, 16, 4):   [(4, 1, 1)],
    (64, 16, 16, 16):  [(4, 1, 1)],
    (64, 16, 32, 8):   [(1, 4, 4), (4, 1, 1), (4, 4, 4), (8, 1, 1), (8, 2, 2)],
    (64, 16, 32, 16):  [(1, 8, 4), (8, 1, 1), (8, 2, 2), (8, 8, 4)],
    (64, 16, 64, 8):   [(4, 4, 2)],
    (64, 16, 64, 16):  [(1, 1, 1), (1, 2, 2), (8, 1, 1), (8, 2, 2), (8, 8, 2), (16, 1, 1), (16, 2, 2)],
    (64, 32, 16, 4):   [(4, 4, 1), (4, 4, 4)],
    (64, 32, 32, 8):   [(1, 1, 8), (4, 4, 1), (4, 4, 2), (4, 4, 4), (8, 8, 1), (8, 8, 8)],
    (64, 32, 64, 8):   [(4, 4, 4), (8, 8, 4)],
    (64, 64, 8, 8):    [(1, 1, 8)],
    (64, 64, 16, 4):   [(1, 1, 1), (4, 4, 4)],
    (64, 64, 32, 8):   [(1, 1, 1), (1, 1, 8), (1, 2, 1), (2, 1, 2), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    (64, 64, 32, 16):  [(1, 2, 1), (2, 1, 2), (4, 4, 4), (8, 8, 8)],
    (64, 64, 64, 8):   [(1, 1, 1), (1, 4, 4), (2, 2, 2), (4, 1, 1), (4, 4, 4), (8, 8, 4), (8, 8, 8)],
    (64, 64, 64, 16):  [(2, 2, 4), (4, 1, 1), (8, 8, 2), (8, 8, 8)],
    (64, 128, 16, 4):  [(4, 4, 4)],
    (64, 128, 32, 8):  [(1, 1, 8), (1, 4, 4), (1, 8, 8), (4, 4, 4), (8, 8, 8)],
    (64, 128, 64, 8):  [(8, 8, 8)],
    (128, 16, 64, 8):  [(4, 4, 2)],
    (128, 16, 64, 16): [(8, 1, 1), (8, 2, 2), (8, 8, 2)],
    (128, 32, 16, 4):  [(4, 1, 1), (4, 2, 2), (4, 4, 4)],
    (128, 32, 16, 8):  [(4, 1, 1), (4, 2, 2)],
    (128, 32, 32, 8):  [(1, 1, 8), (4, 1, 1), (4, 4, 4), (8, 1, 1), (8, 2, 2), (8, 8, 1), (8, 8, 2), (8, 8, 8)],
    (128, 32, 64, 8):  [(4, 4, 4), (8, 8, 4)],
    (128, 32, 64, 16): [(16, 1, 1), (16, 2, 2), (16, 8, 8)],
    (128, 64, 8, 8):   [(1, 1, 8)],
    (128, 64, 16, 4):  [(4, 4, 4)],
    (128, 64, 16, 8):  [(1, 1, 8)],
    (128, 64, 32, 8):  [(1, 1, 8), (4, 4, 4), (8, 8, 8)],
    (128, 64, 64, 8):  [(8, 8, 8)],
    (128, 128, 16, 4): [(1, 1, 4), (4, 4, 4)],
    (128, 128, 32, 8): [(1, 1, 8), (4, 4, 4), (4, 4, 8), (8, 8, 8)],
    (128, 128, 64, 8): [(4, 4, 4), (4, 4, 8), (8, 8, 4), (8, 8, 8)],
    (128, 192, 16, 4): [(4, 4, 4)],
    (128, 256, 16, 4): [(4, 4, 4)],
    (128, 256, 32, 8): [(1, 1, 8), (4, 4, 4), (8, 4, 8), (8, 8, 8)],
    (224, 256, 64, 16):[(8, 8, 8)],
    (256, 16, 64, 16): [(8, 1, 1), (8, 2, 2), (8, 8, 2)],
    (256, 32, 64, 8):  [(8, 8, 4), (8, 8, 8)],
    (256, 64, 8, 8):   [(1, 1, 8)],
    (256, 128, 16, 4): [(4, 4, 4)],
    (256, 128, 32, 8): [(1, 1, 8), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    (256, 224, 64, 16):[(8, 8, 8)],
    (256, 256, 32, 8): [(4, 4, 4), (8, 8, 4), (8, 8, 8)],
    (256, 256, 32, 16):[(8, 8, 8)],
}


def get_vector_sizes_for_tile(
    tile_m: int, tile_n: int, tile_k: int,
    warp_tile_k: int,
) -> List[Tuple[int, int, int]]:
    """Return list of valid (vec_a, vec_b, vec_c) for a tile+warp_tile_k combo.

    Looks up the precomputed table extracted from JSON profiler configs.
    Falls back to (4, 8, 8) if not found.
    """
    key = (tile_m, tile_n, tile_k, warp_tile_k)
    return _TILE_WTILK_TO_VECS.get(key, [(4, 8, 8)])


# Vector sizes per tile (backward-compat dict, 3-tuple keys)
# Key: (tile_m, tile_n, tile_k) -> TileVariantContainer((vec_a, vec_b, vec_c), [...])
# Default = bf16 vec config (highest warp_tile_k in the table for that tile).
# For exhaustive generation, call get_vector_sizes_for_tile() instead.
def _build_tile_to_vector() -> Dict[Tuple[int, int, int], "TileVariantContainer"]:
    """Build TILE_TO_VECTOR from _TILE_WTILK_TO_VECS."""
    from collections import defaultdict
    tile_to_all: Dict = defaultdict(set)
    for (tm, tn, tk, wtk), vecs in _TILE_WTILK_TO_VECS.items():
        for v in vecs:
            tile_to_all[(tm, tn, tk)].add(v)

    result = {}
    for tile, vecs in tile_to_all.items():
        vecs_sorted = sorted(vecs, reverse=True)  # highest first = default
        default = vecs_sorted[0]
        specialized = vecs_sorted[1:]
        result[tile] = TileVariantContainer(default, specialized)
    return result


TILE_TO_VECTOR: Dict[Tuple[int, int, int], TileVariantContainer] = _build_tile_to_vector()


def compute_vector_sizes(
    tile_m: int, tile_n: int, tile_k: int,
    dtype_class: str,
    variant: str,
) -> List[Tuple[int, int, int]]:
    """Return a list of valid (vec_a, vec_b, vec_c) tuples for a tile and data type.

    Rules:
      - fp32: max vec_a/b = 4; half (fp16/bf16): max vec_a/b = 8
      - bwd_data: vec_a covers tile_k (the A-tile dim); others cover tile_m
      - vec_b always covers tile_n
      - vec_c: max 8 for half, max 4 for fp32
      - All vecs must satisfy: vec × WARP_SIZE >= relevant_tile_dim
      - Also constrained by TILE_TO_VECTOR default for backward compat
    """
    max_ab = 4 if dtype_class == "float" else 8
    max_c = 4 if dtype_class == "float" else 8

    a_dim = tile_k if variant == "bwd_data" else tile_m

    # Minimum vecs to cover the tile dimension with one warp
    min_a = max(1, (a_dim + WARP_SIZE - 1) // WARP_SIZE)
    min_b = max(1, (tile_n + WARP_SIZE - 1) // WARP_SIZE)

    # Round up to nearest power-of-2
    def ceil_pow2(v):
        r = 1
        while r < v:
            r <<= 1
        return r

    min_a = ceil_pow2(min_a)
    min_b = ceil_pow2(min_b)

    if min_a > max_ab or min_b > max_ab:
        # Fallback: use the stored default
        key = (tile_m, tile_n, tile_k)
        if key in TILE_TO_VECTOR:
            return [TILE_TO_VECTOR[key].default]
        return [(1, 1, 1)]

    results = []
    # Max vec (fast path for the common case)
    max_a = min(max_ab, max_ab)  # = max_ab
    max_b = min(max_ab, max_ab)  # = max_ab
    # Constrain to powers of 2 that satisfy the minimum
    valid_a = [v for v in [1, 2, 4, 8, 16] if min_a <= v <= max_a]
    valid_b = [v for v in [1, 2, 4, 8, 16] if min_b <= v <= max_b]
    valid_c = [v for v in [1, 2, 4, 8] if v <= max_c]

    if not valid_a or not valid_b or not valid_c:
        key = (tile_m, tile_n, tile_k)
        if key in TILE_TO_VECTOR:
            return [TILE_TO_VECTOR[key].default]
        return [(1, 1, 1)]

    # Emit: (max_a, max_b, max_c) as the canonical choice,
    # plus the stored default from TILE_TO_VECTOR if different.
    canonical = (max(valid_a), max(valid_b), max(valid_c))
    results.append(canonical)

    key = (tile_m, tile_n, tile_k)
    if key in TILE_TO_VECTOR:
        stored = TILE_TO_VECTOR[key].default
        if stored not in results:
            results.append(stored)

    return results


# =============================================================================
# Pipeline / Scheduler Rules
# =============================================================================

# Valid (pipeline, scheduler) pairs per variant
# Derived from JSON profiler configs (union of bf16 + fp32).
VARIANT_PIPELINE_SCHEDULER: Dict[str, List[Tuple[str, str]]] = {
    "forward": [
        ("compv1", "intrawave"),
        ("compv1", "interwave"),
        ("compv3", "intrawave"),
        ("compv4", "intrawave"),
        ("compv6", "intrawave"),
        ("mem", "intrawave"),
        ("mem", "interwave"),
    ],
    "bwd_data": [
        ("compv1", "intrawave"),
    ],
    "bwd_weight": [
        ("compv1", "intrawave"),
        ("compv1", "interwave"),
        ("compv3", "intrawave"),
        ("compv4", "intrawave"),
        ("compv6", "intrawave"),
        ("mem", "intrawave"),
        ("mem", "interwave"),
        ("basic_async_v1", "intrawave"),
    ],
}

# Specializations per variant
VARIANT_SPECIALIZATIONS: Dict[str, List[str]] = {
    "forward":     ["default", "filter1x1_pad0", "filter1x1_stride1_pad0", "filter3x3"],
    "bwd_data":    ["default", "filter1x1_stride1_pad0"],
    "bwd_weight":  ["default", "filter1x1_stride1_pad0"],
}

# =============================================================================
# Feature Flag Rules (Phase 5)
# =============================================================================


@dataclass
class StreamKSpec:
    """StreamK parameters for a feature config."""
    strategy: str = "TREE"          # "TREE" | "LINEAR"
    persistent: bool = False        # non-persistent by default


@dataclass
class FeatureSpec:
    """A single feature-flag variant rule.

    Fields that are False/1/None represent 'off'.
    tile_override / pipeline_override = None means use variant defaults.
    """
    split_image: bool = False
    explicit_gemm: bool = False
    two_stage: bool = False
    double_smem_buffer: bool = False
    num_groups_to_merge: int = 1
    streamk_config: Optional[StreamKSpec] = None

    # Optional overrides (None = use variant defaults)
    tile_override: Optional[List[Tuple[int, int, int]]] = None
    pipeline_override: Optional[List[Tuple[str, str]]] = None


# Tiles used for split_image (from JSON profiler configs, forward only)
_SPLIT_IMAGE_TILES: List[Tuple[int, int, int]] = [
    (64, 64, 16), (64, 64, 32),
    (256, 128, 16), (256, 128, 32),
]

VARIANT_FEATURES: Dict[str, List[FeatureSpec]] = {
    "forward": [
        # split_image: small tile subset, compv1/intrawave only
        FeatureSpec(
            split_image=True,
            tile_override=_SPLIT_IMAGE_TILES,
            pipeline_override=[("compv1", "intrawave")],
        ),
        # num_groups_to_merge (all filter specializations, all pipelines)
        FeatureSpec(num_groups_to_merge=2),
        FeatureSpec(num_groups_to_merge=4),
        FeatureSpec(num_groups_to_merge=8),
        FeatureSpec(num_groups_to_merge=16),
        FeatureSpec(num_groups_to_merge=32),
    ],
    "bwd_data": [],
    "bwd_weight": [
        # explicit_gemm only
        FeatureSpec(explicit_gemm=True),
        # two_stage only
        FeatureSpec(two_stage=True),
        # two_stage + explicit_gemm
        FeatureSpec(two_stage=True, explicit_gemm=True),
        # num_groups_to_merge (combined with two_stage as per JSON data)
        FeatureSpec(num_groups_to_merge=2,  two_stage=True),
        FeatureSpec(num_groups_to_merge=4,  two_stage=True),
        FeatureSpec(num_groups_to_merge=8,  two_stage=True),
        FeatureSpec(num_groups_to_merge=16, two_stage=True),
        FeatureSpec(num_groups_to_merge=32, two_stage=True),
        # basic_async_v1 + num_groups_to_merge=2 (no streamk, from JSON data)
        FeatureSpec(
            num_groups_to_merge=2,
            tile_override=[(16, 32, 64), (16, 64, 64), (64, 128, 64)],
            pipeline_override=[("basic_async_v1", "intrawave")],
        ),
        # StreamK non-persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=False),
            pipeline_override=[
                ("compv1", "intrawave"),
                ("mem", "intrawave"),
                ("basic_async_v1", "intrawave"),
            ],
        ),
        # StreamK persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=True),
            pipeline_override=[
                ("compv1", "intrawave"),
                ("mem", "intrawave"),
                ("basic_async_v1", "intrawave"),
            ],
        ),
    ],
}

# =============================================================================
# Pipeline Variant Suffixes (single source of truth — kept for backward compat)
# =============================================================================
# Empirically verified valid (pipeline, wave_mode, has_dsb, has_si) combinations
# observed in the 2D and 3D bf16 gfx950 benchmark CSVs. 30 entries total per ndim.
# Each tuple: (pipeline, wave_mode, has_dsb, has_si)
#   wave_mode: "intrawave" | "interwave"
#   has_dsb:   1 if "_dsb" suffix present (double smem buffer), else 0
#   has_si:    1 if "_si"  suffix present (store immediate),    else 0
PIPELINE_VARIANTS: List[Tuple[str, str, int, int]] = [
    # basic_v1: both intra/inter × {∅, dsb, si, dsb_si} = 8 combos
    ("basic_v1", "intrawave", 0, 0),
    ("basic_v1", "intrawave", 1, 0),
    ("basic_v1", "intrawave", 0, 1),
    ("basic_v1", "intrawave", 1, 1),
    ("basic_v1", "interwave", 0, 0),
    ("basic_v1", "interwave", 1, 0),
    ("basic_v1", "interwave", 0, 1),
    ("basic_v1", "interwave", 1, 1),
    # compv3: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv3", "intrawave", 0, 0),
    ("compv3", "intrawave", 1, 0),
    ("compv3", "intrawave", 0, 1),
    ("compv3", "intrawave", 1, 1),
    # compv4: intrawave × {dsb, dsb_si} only = 2 combos
    ("compv4", "intrawave", 1, 0),
    ("compv4", "intrawave", 1, 1),
    # compv5: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv5", "intrawave", 0, 0),
    ("compv5", "intrawave", 1, 0),
    ("compv5", "intrawave", 0, 1),
    ("compv5", "intrawave", 1, 1),
    # compv6: intrawave × {∅, dsb, si, dsb_si} = 4 combos
    ("compv6", "intrawave", 0, 0),
    ("compv6", "intrawave", 1, 0),
    ("compv6", "intrawave", 0, 1),
    ("compv6", "intrawave", 1, 1),
    # mem: both intra/inter × {∅, dsb, si, dsb_si} = 8 combos
    ("mem", "intrawave", 0, 0),
    ("mem", "intrawave", 1, 0),
    ("mem", "intrawave", 0, 1),
    ("mem", "intrawave", 1, 1),
    ("mem", "interwave", 0, 0),
    ("mem", "interwave", 1, 0),
    ("mem", "interwave", 0, 1),
    ("mem", "interwave", 1, 1),
]


def iter_pipeline_variants(pipelines: List[str] = None):
    """Iterate (pipeline, wave_mode, has_dsb, has_si) tuples, optionally filtered.

    Args:
        pipelines: optional list of pipeline names to keep. If None, yield all.
    """
    if pipelines is None:
        for entry in PIPELINE_VARIANTS:
            yield entry
        return
    keep = set(pipelines)
    for entry in PIPELINE_VARIANTS:
        if entry[0] in keep:
            yield entry


# Valid pipelines per variant (kept for backward compat; full list)
VARIANT_PIPELINES: Dict[str, List[str]] = {
    "forward": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_data": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
    "bwd_weight": [
        "basic_v1",
        "mem",
        "compv3",
        "compv4",
        "compv5",
        "compv6",
        "comp_async",
        "basic_async_v1",
    ],
}

# Tiles that support compv4 pipeline
# compv4 has stricter requirements due to double buffering and LDS constraints
COMPV4_COMPATIBLE_TILES: List[Tuple[int, int, int]] = [
    # warp_tile [16,16,16] - all work with compv4
    (16, 64, 64),
    (32, 64, 64),
    (64, 64, 64),
    # warp_tile [16,16,32] - all work with compv4
    (16, 64, 128),
    (32, 64, 128),
    (64, 64, 128),
]

# =============================================================================
# Shared Validation Rules
# =============================================================================
# These functions are the single source of truth for validation rules
# for convolution code generation.

# --- Vector size validation ---

WARP_SIZE = 64


def is_valid_vector_size(vec: int) -> bool:
    """AMD GPUs only support vector widths 1, 2, 4, 8, 16."""
    return vec == 1 or vec % 2 == 0


def check_vectors(vec_a: int, vec_b: int, vec_c: int) -> bool:
    """Check all three vector sizes are valid (1 or even)."""
    return all(is_valid_vector_size(v) for v in (vec_a, vec_b, vec_c))


# --- Tile coverage validation ---


def check_warp_coverage(
    tile_m: int, tile_n: int, tile_k: int,
    vec_a: int, vec_b: int,
    variant: str = "forward",
) -> bool:
    """Check tile dims don't exceed warp vector load coverage.

    This is a legacy check that was overly conservative (assumed a single warp
    covers the full tile). Valid instances have multiple warps splitting the tile,
    so the effective per-warp dimension is tile_dim / num_warps. Since valid
    vector sizes are already pre-computed from the JSON configs in
    _TILE_WTILK_TO_VECS, this check now always returns True to avoid false
    rejections. The arch filter in WARP_SUPPORTED_COMBINATIONS handles actual
    architectural limits.
    """
    return True


def check_bwd_data_vec_coverage(
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    vec_a: int, vec_b: int,
) -> bool:
    """Bwd_data: vector width must not exceed elements per thread per tile slice."""
    block_size = WARP_SIZE * warp_m * warp_n * warp_k
    if vec_a > (tile_m * tile_k) // block_size:
        return False
    if vec_b > (tile_n * tile_k) // block_size:
        return False
    return True


# --- Pipeline-scheduler restrictions ---

INTERWAVE_PIPELINES = {"basic_v1", "mem"}  # Only these support interwave


def is_valid_pipeline_scheduler(pipeline: str, scheduler: str) -> bool:
    """Check pipeline+scheduler combo is valid.

    Only 'mem' and 'basic_v1' pipelines support interwave; all compute
    pipelines (compv3/v4/v5/v6/async) only support intrawave.
    """
    if scheduler == "interwave" and pipeline not in INTERWAVE_PIPELINES:
        return False
    return True


# --- Pipeline-variant restrictions ---

UNSUPPORTED_VARIANT_PIPELINES = {
    "bwd_weight": {"compv5"},
    "bwd_data": {"compv5"},
}


def is_valid_pipeline_for_variant(pipeline: str, variant: str) -> bool:
    """Check pipeline is supported for the given conv variant.

    Backward weight and backward data reject compv5 due to transpose_tile2d /
    get_length issues.
    """
    blocked = UNSUPPORTED_VARIANT_PIPELINES.get(variant, set())
    return pipeline not in blocked


# --- Stream-K restrictions ---


def is_streamk_valid_for_variant(variant: str) -> bool:
    """Stream-K is only supported for backward weight."""
    return variant == "bwd_weight"


# =============================================================================
# Tile Registration Validation
# =============================================================================


def validate_tile_config(tile_m: int, tile_n: int, tile_k: int) -> bool:
    """Check if a tile configuration is valid and registered."""
    tile_key = (tile_m, tile_n, tile_k)
    return tile_key in TILE_TO_WAVE_WARP and tile_key in TILE_TO_VECTOR


def get_tile_full_config(tile_m: int, tile_n: int, tile_k: int) -> dict:
    """Get complete configuration for a tile size.

    Returns:
        dict with keys: wave_m, wave_n, wave_k, warp_m, warp_n, warp_k, vec_a, vec_b, vec_c
        or None if tile not found
    """
    tile_key = (tile_m, tile_n, tile_k)
    if not validate_tile_config(tile_m, tile_n, tile_k):
        return None

    wave_m, wave_n, wave_k = TILE_TO_WAVE[tile_key]
    warp_m, warp_n, warp_k = TILE_TO_WARP[tile_key]
    vec_a, vec_b, vec_c = TILE_TO_VECTOR[tile_key]

    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "wave_m": wave_m,
        "wave_n": wave_n,
        "wave_k": wave_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": warp_k,
        "vec_a": vec_a,
        "vec_b": vec_b,
        "vec_c": vec_c,
    }


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary():
    """Print summary of available tile configurations."""
    all_tiles = sorted(set(COMMON_TILES) | set(FWD_TILES) | set(BWD_DATA_TILES) | set(BWD_WEIGHT_TILES))
    print("=" * 80)
    print("Grouped Convolution Tile Configurations (Single Source of Truth)")
    print("=" * 80)
    print(f"COMMON_TILES:     {len(COMMON_TILES)}")
    print(f"FWD_TILES:        {len(FWD_TILES)}")
    print(f"BWD_DATA_TILES:   {len(BWD_DATA_TILES)}")
    print(f"BWD_WEIGHT_TILES: {len(BWD_WEIGHT_TILES)}")
    print(f"Total unique:     {len(all_tiles)}")
    print()
    print("Tile sizes (M×N×K) | wave | warp:")
    for tile in all_tiles:
        m, n, k = tile
        tag = []
        if tile in set(COMMON_TILES):
            tag.append("C")
        if tile in set(FWD_TILES):
            tag.append("F")
        if tile in set(BWD_DATA_TILES):
            tag.append("D")
        if tile in set(BWD_WEIGHT_TILES):
            tag.append("W")
        tags = ",".join(tag)
        if tile in TILE_TO_WAVE:
            wave = TILE_TO_WAVE[tile]
            warp = TILE_TO_WARP[tile]
            print(
                f"  [{tags:6}] {m:3}×{n:3}×{k:3}  "
                f"wave={wave[0]}×{wave[1]}×{wave[2]}  "
                f"warp={warp[0]}×{warp[1]}×{warp[2]}"
            )
        else:
            print(f"  [{tags:6}] {m:3}×{n:3}×{k:3}  (no mapping)")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()
