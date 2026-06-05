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

"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow importing tile_math from same directory
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).parent.resolve()
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from arch_specs_generated import WARP_TILE_SUPPORTED_COMBINATIONS

from tile_math import (
    get_valid_vec_sizes as _tm_get_valid_vec_sizes,
    get_valid_wave_warp_pairs as _tm_get_valid_wave_warp_pairs,
)

WARP_SIZE = 64

# Dtype string to dtype_key mapping (for tile_math filter functions).
DTYPE_TO_DTYPE_KEY: Dict[str, str] = {
    "fp16": "fp16_fp16_fp32",
    "bf16": "bf16_bf16_fp32",
    "fp32": "fp32_fp32_fp32",
}

# =============================================================================
# Tile Lists
# =============================================================================
# Four orthogonal lists covering all tiles from configs corresponding old CK 
# grouped conv instances.
# COMMON_TILES = tiles present in ALL three directions.
# FWD_TILES / BWD_DATA_TILES / BWD_WEIGHT_TILES = direction-specific tiles.
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

# Override the tile sizes for split-image feature.
_SPLIT_IMAGE_TILES: List[Tuple[int, int, int]] = [
    (64, 64, 16), (64, 64, 32),
    (256, 128, 16), (256, 128, 32),
]

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
# Mapping from tile to wave configs to restrict the number of configurations.
# =============================================================================
# Per-variant mapping: tile → list of curated wave configs.
# Given a (tile, wave) pair, valid warp_tile shapes are derived mathematically
# from WARP_TILE_SUPPORTED_COMBINATIONS + divisibility rules.
# The selection of the wave configs is based on the wave configurations from
# the old CK grouped convolution instances.

_FWD_TILE_TO_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (16, 16, 64): [(1, 1, 1)],
    (16, 16, 128): [(1, 1, 1)],
    (16, 32, 64): [(1, 2, 1)],
    (16, 64, 64): [(1, 2, 1)],
    (16, 128, 64): [(1, 2, 1)],
    (16, 256, 64): [(1, 4, 1)],
    (32, 16, 64): [(2, 1, 1)],
    (32, 64, 16): [(1, 1, 1)],
    (32, 64, 32): [(1, 1, 1)],
    (32, 64, 64): [(1, 2, 1)],
    (32, 128, 16): [(1, 2, 1)],
    (32, 128, 32): [(1, 2, 1)],
    (32, 128, 64): [(1, 2, 1)],
    (32, 256, 64): [(1, 4, 1)],
    (64, 16, 16): [(1, 1, 1)],
    (64, 16, 64): [(2, 1, 1)],
    (64, 32, 16): [(1, 1, 1)],
    (64, 32, 32): [(1, 1, 1)],
    (64, 32, 64): [(2, 1, 1)],
    (64, 64, 8): [(2, 1, 1)],
    (64, 64, 16): [(1, 1, 1)],
    (64, 64, 32): [(1, 1, 1), (2, 2, 1)],
    (64, 64, 64): [(2, 2, 1)],
    (64, 128, 16): [(1, 2, 1), (2, 2, 1)],
    (64, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (64, 128, 64): [(2, 2, 1)],
    (128, 16, 64): [(2, 1, 1)],
    (128, 32, 16): [(2, 1, 1)],
    (128, 32, 32): [(2, 1, 1), (2, 1, 2)],
    (128, 32, 64): [(2, 1, 1)],
    (128, 64, 8): [(2, 1, 1), (2, 2, 1)],
    (128, 64, 16): [(2, 1, 1), (2, 2, 1)],
    (128, 64, 32): [(2, 1, 1), (2, 2, 1)],
    (128, 64, 64): [(2, 2, 1)],
    (128, 128, 16): [(1, 2, 1), (2, 2, 1)],
    (128, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (128, 128, 64): [(2, 2, 1)],
    (128, 192, 16): [(2, 2, 1)],
    (128, 256, 16): [(2, 2, 1)],
    (128, 256, 32): [(2, 2, 1)],
    (224, 256, 64): [(2, 2, 1)],
    (256, 16, 64): [(4, 1, 1)],
    (256, 32, 64): [(4, 1, 1)],
    (256, 64, 8): [(2, 2, 1)],
    (256, 128, 16): [(2, 2, 1)],
    (256, 128, 32): [(2, 2, 1)],
    (256, 224, 64): [(2, 2, 1)],
    (256, 256, 32): [(2, 2, 1)],
}

_BWD_DATA_TILE_TO_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (16, 64, 32): [(1, 1, 1)],
    (32, 64, 32): [(1, 1, 1)],
    (32, 128, 32): [(1, 2, 1)],
    (64, 16, 16): [(4, 1, 1)],
    (64, 16, 32): [(1, 1, 1), (4, 1, 1)],
    (64, 16, 64): [(4, 1, 1)],
    (64, 32, 32): [(1, 1, 1)],
    (64, 64, 32): [(1, 1, 1)],
    (64, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (128, 32, 16): [(4, 1, 1)],
    (128, 32, 32): [(2, 1, 1), (4, 1, 1)],
    (128, 32, 64): [(4, 1, 1)],
    (128, 64, 32): [(2, 1, 1), (2, 2, 1)],
    (128, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (128, 256, 32): [(2, 2, 1)],
    (256, 128, 32): [(2, 2, 1)],
}

_BWD_WEIGHT_TILE_TO_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (16, 16, 32): [(1, 1, 1)],
    (16, 16, 64): [(1, 1, 1)],
    (16, 32, 64): [(1, 2, 1)],
    (16, 64, 64): [(1, 2, 1)],
    (16, 128, 32): [(1, 1, 1)],
    (16, 128, 64): [(1, 2, 1)],
    (16, 256, 32): [(1, 1, 1)],
    (16, 256, 64): [(1, 4, 1)],
    (32, 16, 64): [(2, 1, 1)],
    (32, 32, 32): [(1, 1, 1)],
    (32, 64, 16): [(1, 1, 1)],
    (32, 64, 32): [(1, 1, 1)],
    (32, 128, 16): [(1, 2, 1)],
    (32, 128, 32): [(1, 1, 1), (1, 2, 1)],
    (64, 16, 64): [(2, 1, 1)],
    (64, 32, 16): [(1, 1, 1)],
    (64, 32, 32): [(1, 1, 1)],
    (64, 64, 16): [(1, 1, 1)],
    (64, 64, 32): [(1, 1, 1)],
    (64, 64, 64): [(2, 2, 1)],
    (64, 128, 16): [(1, 2, 1), (2, 2, 1)],
    (64, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (64, 128, 64): [(2, 2, 1)],
    (128, 16, 64): [(2, 1, 1)],
    (128, 32, 16): [(2, 1, 1)],
    (128, 32, 32): [(1, 1, 1), (2, 1, 1)],
    (128, 64, 16): [(2, 1, 1), (2, 2, 1)],
    (128, 64, 32): [(2, 1, 1), (2, 2, 1)],
    (128, 128, 16): [(1, 2, 1), (2, 2, 1)],
    (128, 128, 32): [(1, 2, 1), (2, 2, 1)],
    (128, 128, 64): [(2, 2, 1)],
    (128, 256, 16): [(2, 2, 1)],
    (128, 256, 32): [(2, 2, 1)],
    (256, 16, 64): [(4, 1, 1)],
    (256, 32, 64): [(4, 1, 1)],
    (256, 128, 16): [(2, 2, 1)],
    (256, 128, 32): [(2, 2, 1)],
    (256, 256, 32): [(2, 2, 1)],
}

# Warp shapes excluded from code generation.
# These are possible shapes, but old CK doesn't 
# use them. However, they can potentially be useful.
_EXCLUDED_WARP_SHAPES: Set[Tuple[int, int, int]] = {
    (16, 16, 32), (4, 64, 16), (64, 4, 16),
}


def get_wave_configs(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[Tuple[int, int, int]]:
    """Return wave configs for a tile+variant.

    Falls back to generic [(1, 1, 1)] for unknown tiles.
    """
    table = {
        "forward": _FWD_TILE_TO_WAVES,
        "bwd_data": _BWD_DATA_TILE_TO_WAVES,
        "bwd_weight": _BWD_WEIGHT_TILE_TO_WAVES,
    }.get(variant, {})
    return table.get((tile_m, tile_n, tile_k), [(1, 1, 1)])


# =============================================================================
# Wave config to warp config mappings
# =============================================================================
# Per-variant mapping: wave → list of warp_tile.
# The curated list is still
# filtered by divisibility and intersected with the arch/dtype-supported combos,
# so per-dtype warp_tile_k correctness is preserved. 
# Waves not in the map fall back to [] (no warp tiles).

_FWD_WAVE_TO_WARPS: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (1, 1, 1): [(16, 16, 4), (16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (1, 2, 1): [(16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (1, 4, 1): [(16, 16, 16), (32, 32, 8)],
    (2, 1, 1): [(16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (2, 1, 2): [(32, 32, 8)],
    (2, 2, 1): [(16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (4, 1, 1): [(16, 16, 16), (32, 32, 8)],
}

_BWD_DATA_WAVE_TO_WARPS: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (1, 1, 1): [(16, 16, 8), (16, 16, 16), (32, 32, 8)],
    (1, 2, 1): [(32, 32, 8)],
    (2, 1, 1): [(32, 32, 8)],
    (2, 2, 1): [(32, 32, 8)],
    (4, 1, 1): [(16, 16, 4), (16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8), (32, 32, 16)],
}

_BWD_WEIGHT_WAVE_TO_WARPS: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
    (1, 1, 1): [(16, 16, 8), (16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (1, 2, 1): [(16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (1, 4, 1): [(16, 16, 16)],
    (2, 1, 1): [(16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (2, 2, 1): [(16, 16, 16), (32, 32, 4), (32, 32, 8)],
    (4, 1, 1): [(16, 16, 16), (32, 32, 8)],
}


# Finer (tile, wave) → warp_tile map. 
# When a (tile, wave) key is present here, it overrides the coarser wave-only map above.
# The wave-only map crosses a wave's warp set with every
# tile sharing that wave, whereas this map lists only the warp tiles actually
# observed for that exact (tile, wave). Keys absent here fall back to the wave map.

_FWD_TILE_WAVE_TO_WARPS: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], List[Tuple[int, int, int]]] = {
    ((16, 16, 64), (1, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 16, 128), (1, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 32, 64), (1, 2, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 64, 64), (1, 2, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 128, 64), (1, 2, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 256, 64), (1, 4, 1)): [(16, 16, 16)],
    ((32, 16, 64), (2, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((32, 64, 16), (1, 1, 1)): [(32, 32, 4)],
    ((32, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((32, 64, 64), (1, 2, 1)): [(32, 32, 8)],
    ((32, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((32, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((32, 128, 64), (1, 2, 1)): [(32, 32, 8)],
    ((32, 256, 64), (1, 4, 1)): [(32, 32, 8)],
    ((64, 16, 16), (1, 1, 1)): [(16, 16, 4), (16, 16, 16)],
    ((64, 16, 64), (2, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((64, 32, 16), (1, 1, 1)): [(32, 32, 4)],
    ((64, 32, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 32, 64), (2, 1, 1)): [(32, 32, 8)],
    ((64, 64, 8), (2, 1, 1)): [(32, 32, 8)],
    ((64, 64, 16), (1, 1, 1)): [(32, 32, 4)],
    ((64, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 64, 32), (2, 2, 1)): [(16, 16, 8), (16, 16, 16)],
    ((64, 64, 64), (2, 2, 1)): [(32, 32, 8)],
    ((64, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((64, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((64, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((64, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((64, 128, 64), (2, 2, 1)): [(32, 32, 8)],
    ((128, 16, 64), (2, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((128, 32, 16), (2, 1, 1)): [(32, 32, 4)],
    ((128, 32, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 32, 32), (2, 1, 2)): [(32, 32, 8)],
    ((128, 32, 64), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 8), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 8), (2, 2, 1)): [(32, 32, 8)],
    ((128, 64, 16), (2, 1, 1)): [(32, 32, 4)],
    ((128, 64, 16), (2, 2, 1)): [(32, 32, 4), (32, 32, 8)],
    ((128, 64, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 64, 64), (2, 2, 1)): [(32, 32, 8)],
    ((128, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((128, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((128, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 128, 64), (2, 2, 1)): [(32, 32, 8)],
    ((128, 192, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 256, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 256, 32), (2, 2, 1)): [(32, 32, 8)],
    ((224, 256, 64), (2, 2, 1)): [(16, 16, 16)],
    ((256, 16, 64), (4, 1, 1)): [(16, 16, 16)],
    ((256, 32, 64), (4, 1, 1)): [(32, 32, 8)],
    ((256, 64, 8), (2, 2, 1)): [(32, 32, 8)],
    ((256, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((256, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((256, 224, 64), (2, 2, 1)): [(16, 16, 16)],
    ((256, 256, 32), (2, 2, 1)): [(16, 16, 16), (32, 32, 8)],
}

_BWD_DATA_TILE_WAVE_TO_WARPS: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], List[Tuple[int, int, int]]] = {
    ((16, 64, 32), (1, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((32, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((32, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((64, 16, 16), (4, 1, 1)): [(16, 16, 4), (16, 16, 16)],
    ((64, 16, 32), (1, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((64, 16, 32), (4, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((64, 16, 64), (4, 1, 1)): [(16, 16, 16)],
    ((64, 32, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((64, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 32, 16), (4, 1, 1)): [(32, 32, 4), (32, 32, 8)],
    ((128, 32, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 32, 32), (4, 1, 1)): [(32, 32, 8)],
    ((128, 32, 64), (4, 1, 1)): [(32, 32, 16)],
    ((128, 64, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((128, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 256, 32), (2, 2, 1)): [(32, 32, 8)],
    ((256, 128, 32), (2, 2, 1)): [(32, 32, 8)],
}

_BWD_WEIGHT_TILE_WAVE_TO_WARPS: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], List[Tuple[int, int, int]]] = {
    ((16, 16, 32), (1, 1, 1)): [(16, 16, 8), (16, 16, 16)],
    ((16, 16, 64), (1, 1, 1)): [(16, 16, 16)],
    ((16, 32, 64), (1, 2, 1)): [(16, 16, 16)],
    ((16, 64, 64), (1, 2, 1)): [(16, 16, 16)],
    ((16, 128, 32), (1, 1, 1)): [(16, 16, 16)],
    ((16, 128, 64), (1, 2, 1)): [(16, 16, 16)],
    ((16, 256, 32), (1, 1, 1)): [(16, 16, 16)],
    ((16, 256, 64), (1, 4, 1)): [(16, 16, 16)],
    ((32, 16, 64), (2, 1, 1)): [(16, 16, 16)],
    ((32, 32, 32), (1, 1, 1)): [(32, 32, 8)],
    ((32, 64, 16), (1, 1, 1)): [(32, 32, 4)],
    ((32, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((32, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((32, 128, 32), (1, 1, 1)): [(32, 32, 8)],
    ((32, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((64, 16, 64), (2, 1, 1)): [(16, 16, 16)],
    ((64, 32, 16), (1, 1, 1)): [(32, 32, 4)],
    ((64, 32, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 64, 16), (1, 1, 1)): [(32, 32, 4)],
    ((64, 64, 32), (1, 1, 1)): [(32, 32, 8)],
    ((64, 64, 64), (2, 2, 1)): [(16, 16, 16), (32, 32, 8)],
    ((64, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((64, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((64, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((64, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((64, 128, 64), (2, 2, 1)): [(32, 32, 8)],
    ((128, 16, 64), (2, 1, 1)): [(16, 16, 16)],
    ((128, 32, 16), (2, 1, 1)): [(32, 32, 4)],
    ((128, 32, 32), (1, 1, 1)): [(32, 32, 8)],
    ((128, 32, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 16), (2, 1, 1)): [(32, 32, 4)],
    ((128, 64, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 64, 32), (2, 1, 1)): [(32, 32, 8)],
    ((128, 64, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 128, 16), (1, 2, 1)): [(32, 32, 4)],
    ((128, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 128, 32), (1, 2, 1)): [(32, 32, 8)],
    ((128, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((128, 128, 64), (2, 2, 1)): [(32, 32, 8)],
    ((128, 256, 16), (2, 2, 1)): [(32, 32, 4)],
    ((128, 256, 32), (2, 2, 1)): [(32, 32, 8)],
    ((256, 16, 64), (4, 1, 1)): [(16, 16, 16)],
    ((256, 32, 64), (4, 1, 1)): [(32, 32, 8)],
    ((256, 128, 16), (2, 2, 1)): [(32, 32, 4)],
    ((256, 128, 32), (2, 2, 1)): [(32, 32, 8)],
    ((256, 256, 32), (2, 2, 1)): [(32, 32, 8)],
}


def get_all_valid_vector_sizes(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    wt_m: int, wt_n: int, wt_k: int,
    dtype_key: str,
) -> Set[Tuple[int, int, int]]:
    """Return the set of (vec_a, vec_b, vec_c) triples tile_math considers valid.

    Thin wrapper around tile_math.get_valid_vec_sizes for use as a hard gate:
    curated/strategy vec triples are kept only if present in this set.
    """
    return set(_tm_get_valid_vec_sizes(
        tile_m, tile_n, tile_k,
        wave_m, wave_n, wave_k,
        wt_m, wt_n, wt_k,
        dtype_key,
    ))


def get_warp_configs_for_tile_and_wave(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    dtype_key: str, arch: str = "gfx942", variant: str = "forward",
) -> List[Tuple[int, int, int]]:
    """Return curated warp_tile shapes for a (variant, wave), filtered for this tile.

    Prefers the finer (tile, wave) → warp map when that exact key is present;
    otherwise falls back to the coarser wave-only map. The result is then kept
    only for shapes that:
      - are arch/dtype-supported (WARP_TILE_SUPPORTED_COMBINATIONS[arch][dtype_key]),
      - are not in _EXCLUDED_WARP_SHAPES,
      - divide the macro tile: tile_m % (wave_m*warp_tile_m) == 0 and
        tile_n % (wave_n*warp_tile_n) == 0.
    """
    tile = (tile_m, tile_n, tile_k)
    wave = (wave_m, wave_n, wave_k)
    fine_table = {
        "forward": _FWD_TILE_WAVE_TO_WARPS,
        "bwd_data": _BWD_DATA_TILE_WAVE_TO_WARPS,
        "bwd_weight": _BWD_WEIGHT_TILE_WAVE_TO_WARPS,
    }.get(variant, {})
    if (tile, wave) in fine_table:
        curated = fine_table[(tile, wave)]
    else:
        curated = {
            "forward": _FWD_WAVE_TO_WARPS,
            "bwd_data": _BWD_DATA_WAVE_TO_WARPS,
            "bwd_weight": _BWD_WEIGHT_WAVE_TO_WARPS,
        }.get(variant, {}).get(wave, [])
    supported = {
        (wt[0], wt[1], wt[2])
        for wt in WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {}).get(dtype_key, [])
    }
    return [
        wt for wt in curated
        if wt not in _EXCLUDED_WARP_SHAPES
        and wt in supported
        and tile_m % (wave_m * wt[0]) == 0
        and tile_n % (wave_n * wt[1]) == 0
    ]


def get_wave_warp_pairs(
    tile_m: int, tile_n: int, tile_k: int,
    variant: str, dtype_key: str, arch: str = "gfx942",
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return (wave, warp_tile) pairs: curated waves x curated warp_tiles.

    Combines curated wave configs with the curated per-variant wave→warp map
    (both from profiler JSON), filtered by arch/dtype support and divisibility.

    The curated pairs are gated against _tm_get_valid_wave_warp_pairs (imported from tile_math.py) 
    and any curated pair rejected is dropped with a warning.
    """
    tm_pairs = set(_tm_get_valid_wave_warp_pairs(
        tile_m, tile_n, tile_k, dtype_key, arch,
    ))
    result = []
    for wave in get_wave_configs(tile_m, tile_n, tile_k, variant):
        for wt in get_warp_configs_for_tile_and_wave(
            tile_m, tile_n, tile_k, *wave, dtype_key, arch, variant,
        ):
            if (wave, wt) not in tm_pairs:
                logging.warning(
                    "Dropping curated wave/warp pair %s rejected by tile_math: "
                    "tile=(%d,%d,%d) variant=%s dtype_key=%s",
                    (wave, wt), tile_m, tile_n, tile_k, variant, dtype_key,
                )
                continue
            result.append((wave, wt))
    return result


# =============================================================================
# Vector Size Strategies (tile + dtype based, wave/warp independent)
# =============================================================================
#
# Convolution GEMM tensors vectorize along different logical dimensions:
#   Forward:    vec_a,vec_b → C (channels), vec_c → K (output channels)
#   BWD data:   vec_a → K, vec_b,vec_c → C
#   BWD weight: vec_a → K, vec_b,vec_c → C
#
# This creates asymmetric vec patterns in profiler configs (e.g. (8,8,4) for
# forward where C supports vec=8 but K only supports vec=4). Each strategy
# produces one (vec_a, vec_b, vec_c) triple from the dtype-determined max.


class VecStrategy(Enum):
    """Vectorization strategies for grouped convolution GEMM configs.

    Each strategy produces a single (vec_a, vec_b, vec_c) triple derived
    from the dtype-determined maximum vector sizes.
    """
    GENERIC = "generic"                          # (1, 1, 1) — minimum fallback
    UNIFORM_MAX = "uniform_max"                  # (max, max, max) — balanced throughput
    MAX_AB_HALF_C = "max_ab_half_c"              # (max, max, max/2) — fwd (8,8,4) pattern
    MAX_A_MIN_BC = "max_a_min_bc"                # (max, 1, 1) — bwd (4,1,1)/(8,1,1) pattern
    MIN_A_MAX_BC = "min_a_max_bc"                # (1, max, max) — bwd (1,4,4)/(1,8,8) pattern
    HALF_UNIFORM = "half_uniform"                # (max/2, max/2, max/2) — (2,2,2) pattern
    QUARTER_AB_MAX_C = "quarter_ab_max_c"        # (max/2, max/2, max) — fwd (4,4,8)/(1,1,8) pattern
    MAX_AB_QUARTER_C = "max_ab_quarter_c"        # (max, max, max/4) — half (8,8,2), fp32 (4,4,1)
    MAX_A_QUARTER_BC = "max_a_quarter_bc"        # (max, max/4, max/4) — half (8,2,2)
    MIN_AB_MAX_C = "min_ab_max_c"                # (1, 1, max) — half (1,1,8), fp32 (1,1,4)
    MAX_A_HALF_BC = "max_a_half_bc"              # (max, max/2, max/2) — fp32 (4,2,2)
    MAX_AB_MIN_C = "max_ab_min_c"                # (max, max, 1) — half (8,8,1)
    MIN_AB_QUARTER_C = "min_ab_quarter_c"        # (1, 1, max/4) — half (1,1,2)
    MIN_A_MAX_B_HALF_C = "min_a_max_b_half_c"    # (1, max, max/2) — half (1,8,4), fp32 (1,4,2)
    HALF_A_MAX_BC = "half_a_max_bc"              # (max/2, max, max) — fp32 (2,4,4)
    HALF_A_MIN_BC = "half_a_min_bc"              # (max/2, 1, 1) — fp32 (2,1,1)


def _max_vec(dtype_class: str) -> int:
    """Return the maximum vector width for a dtype class."""
    if dtype_class == "float":
        return 4
    elif dtype_class in ("half", "fp16", "bf16"):
        return 8
    else:
        raise ValueError(f"Unknown dtype class: {dtype_class}")


def compute_vector_size(
    strategy: VecStrategy,
    dtype_class: str,
) -> Tuple[int, int, int]:
    """Compute a (vec_a, vec_b, vec_c) triple for a given strategy and dtype.

    Args:
        strategy: Which vectorization pattern to use.
        dtype_class: "float" (fp32) or "half" (fp16/bf16).

    Returns:
        A single (vec_a, vec_b, vec_c) triple.
    """
    m = _max_vec(dtype_class)
    h = max(1, m // 2)
    q = max(1, m // 4)

    if strategy == VecStrategy.GENERIC:
        return (1, 1, 1)
    elif strategy == VecStrategy.UNIFORM_MAX:
        return (m, m, m)
    elif strategy == VecStrategy.MAX_AB_HALF_C:
        return (m, m, h)
    elif strategy == VecStrategy.MAX_A_MIN_BC:
        return (m, 1, 1)
    elif strategy == VecStrategy.MIN_A_MAX_BC:
        return (1, m, m)
    elif strategy == VecStrategy.HALF_UNIFORM:
        return (h, h, h)
    elif strategy == VecStrategy.QUARTER_AB_MAX_C:
        return (h, h, m)
    elif strategy == VecStrategy.MAX_AB_QUARTER_C:
        return (m, m, q)
    elif strategy == VecStrategy.MAX_A_QUARTER_BC:
        return (m, q, q)
    elif strategy == VecStrategy.MIN_AB_MAX_C:
        return (1, 1, m)
    elif strategy == VecStrategy.MAX_A_HALF_BC:
        return (m, h, h)
    elif strategy == VecStrategy.MAX_AB_MIN_C:
        return (m, m, 1)
    elif strategy == VecStrategy.MIN_AB_QUARTER_C:
        return (1, 1, q)
    elif strategy == VecStrategy.MIN_A_MAX_B_HALF_C:
        return (1, m, h)
    elif strategy == VecStrategy.HALF_A_MAX_BC:
        return (h, m, m)
    elif strategy == VecStrategy.HALF_A_MIN_BC:
        return (h, 1, 1)
    else:
        return (1, 1, 1)


# =============================================================================
# Per-Tile VecStrategy Tables (extracted from JSON profiler configs)
# =============================================================================
# Per-variant mapping: tile → {dtype_class → list[VecStrategy]}.
# Tables are dtype-class-keyed (Step 7): each strategy's triple is only emitted
# for the dtype class it was observed with in JSON, so no cross-dtype "bleed"
# (e.g. emitting half (8,8,8) for a tile that only used fp32 (4,4,4)).
# Derived per (variant, tile, dtype_class) via greedy set-cover over VecStrategy.
# Tiles not in the table fall back to [GENERIC] (dtype-independent).

_FWD_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 16, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 16, 128): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 64, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 128, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (16, 256, 64): {"half": [VecStrategy.MAX_AB_HALF_C]},
    (32, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (32, 64, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (32, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (32, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 256, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 16, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC]},
    (64, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (64, 32, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C, VecStrategy.MAX_AB_MIN_C]},
    (64, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (64, 64, 16): {"float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.HALF_UNIFORM, VecStrategy.MIN_AB_MAX_C], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (64, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C], "float": [VecStrategy.MAX_AB_HALF_C]},
    (128, 32, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.HALF_UNIFORM, VecStrategy.MIN_AB_MAX_C]},
    (128, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (128, 64, 16): {"half": [VecStrategy.MIN_AB_MAX_C], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 64, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (128, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 192, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (224, 256, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 16, 64): {"half": [VecStrategy.MAX_AB_QUARTER_C]},
    (256, 32, 64): {"half": [VecStrategy.MAX_AB_HALF_C]},
    (256, 64, 8): {"half": [VecStrategy.MIN_AB_MAX_C]},
    (256, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_AB_MAX_C]},
    (256, 224, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 256, 32): {"half": [VecStrategy.UNIFORM_MAX]},
}


_BWD_DATA_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 64, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_B_HALF_C], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 16, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC]},
    (64, 16, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC, VecStrategy.MIN_A_MAX_B_HALF_C], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX], "float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC]},
    (128, 32, 16): {"half": [VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_HALF_BC]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC], "float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX], "float": [VecStrategy.UNIFORM_MAX]},
}


_BWD_WEIGHT_TILE_STRATEGIES: Dict[Tuple[int, int, int], Dict[str, List[VecStrategy]]] = {
    (16, 16, 32): {"half": [VecStrategy.GENERIC, VecStrategy.MIN_AB_QUARTER_C]},
    (16, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.HALF_UNIFORM, VecStrategy.HALF_A_MIN_BC]},
    (16, 32, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_AB_HALF_C], "float": [VecStrategy.HALF_A_MAX_BC]},
    (16, 64, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_AB_HALF_C, VecStrategy.MIN_A_MAX_B_HALF_C]},
    (16, 128, 64): {"half": [VecStrategy.MIN_A_MAX_B_HALF_C]},
    (16, 256, 32): {"half": [VecStrategy.MAX_AB_MIN_C]},
    (16, 256, 64): {"half": [VecStrategy.MIN_A_MAX_B_HALF_C]},
    (32, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.MAX_A_HALF_BC]},
    (32, 64, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MIN_A_MAX_BC]},
    (32, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (32, 128, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.MAX_AB_MIN_C]},
    (64, 16, 64): {"half": [VecStrategy.GENERIC, VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (64, 32, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 32, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 16): {"float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX]},
    (64, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 64, 64): {"half": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_HALF_C, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.HALF_A_MIN_BC], "float": [VecStrategy.GENERIC, VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC, VecStrategy.MIN_A_MAX_BC]},
    (64, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (64, 128, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 16, 64): {"half": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (128, 32, 16): {"float": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_A_MIN_BC]},
    (128, 32, 32): {"half": [VecStrategy.UNIFORM_MAX, VecStrategy.MAX_AB_QUARTER_C, VecStrategy.MAX_A_QUARTER_BC, VecStrategy.MAX_AB_MIN_C]},
    (128, 64, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 64, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (128, 128, 64): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.HALF_UNIFORM]},
    (128, 256, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (128, 256, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 16, 64): {"half": [VecStrategy.MAX_A_MIN_BC, VecStrategy.MAX_A_QUARTER_BC]},
    (256, 32, 64): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 16): {"float": [VecStrategy.UNIFORM_MAX]},
    (256, 128, 32): {"half": [VecStrategy.UNIFORM_MAX]},
    (256, 256, 32): {"half": [VecStrategy.MAX_AB_HALF_C, VecStrategy.HALF_UNIFORM]},
}


def get_vec_strategies(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
    dtype_class: Optional[str] = None,
) -> List[VecStrategy]:
    """Return curated VecStrategy list for a tile+variant (from profiler JSON).

    Tables are dtype-class-keyed. If ``dtype_class`` ("half"/"float") is given,
    return only that dtype's strategies. If None, return the union across dtype
    classes (preserving order, deduplicated). Falls back to [GENERIC] for tiles
    not in the table.
    """
    table = {
        "forward": _FWD_TILE_STRATEGIES,
        "bwd_data": _BWD_DATA_TILE_STRATEGIES,
        "bwd_weight": _BWD_WEIGHT_TILE_STRATEGIES,
    }.get(variant, {})
    entry = table.get((tile_m, tile_n, tile_k))
    if entry is None:
        return [VecStrategy.GENERIC]
    if dtype_class is not None:
        return entry.get(dtype_class, [])
    # Union across dtype classes, deduplicated, order-preserving.
    out: List[VecStrategy] = []
    seen = set()
    for dc in ("half", "float"):
        for s in entry.get(dc, []):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# =============================================================================
# Explicit Per-Tile Vec Overrides (long tail — don't fit any strategy family)
# =============================================================================
# A small set of (variant, tile) → extra vec triples used by the JSON profiler
# configs that no VecStrategy formula produces. These are added on top of the
# strategy-derived vecs (superset semantics). Includes bwd_data's vec_a=16
# (A vectorizes along K, beyond the dtype-class max) and a handful of
# idiosyncratic per-tile triples.

_EXTRA_VEC_TRIPLES: Dict[str, Dict[Tuple[int, int, int], Dict[str, List[Tuple[int, int, int]]]]] = {
    "forward": {
        (32, 64, 64): {"float": [(4, 4, 8)]},
        (32, 128, 64): {"float": [(4, 4, 8)]},
        (64, 64, 32): {"half": [(1, 2, 1), (2, 1, 2)], "float": [(1, 2, 1), (2, 1, 2)]},
        (128, 128, 32): {"float": [(4, 4, 8)]},
        (128, 128, 64): {"float": [(4, 4, 8)]},
        (256, 128, 32): {"half": [(2, 2, 2)]},
    },
    "bwd_data": {
        (64, 16, 32): {"float": [(8, 1, 1), (8, 2, 2)]},
        (64, 16, 64): {"half": [(16, 1, 1), (16, 2, 2)]},
        (128, 32, 16): {"half": [(4, 2, 2)]},
        (128, 32, 32): {"float": [(8, 1, 1), (8, 2, 2)]},
        (128, 32, 64): {"half": [(16, 1, 1), (16, 2, 2), (16, 8, 8)]},
        (128, 256, 32): {"half": [(8, 4, 8)]},
    },
    "bwd_weight": {
        (16, 16, 32): {"float": [(1, 1, 2)]},
        (16, 16, 64): {"half": [(1, 4, 4)]},
        (16, 32, 64): {"half": [(1, 2, 4), (1, 4, 4), (2, 1, 1), (2, 2, 4), (2, 4, 4)]},
        (16, 64, 64): {"half": [(2, 1, 1), (2, 8, 4)]},
        (16, 128, 32): {"half": [(4, 4, 1)]},
        (16, 128, 64): {"half": [(2, 8, 4)]},
        (16, 256, 64): {"half": [(2, 8, 4)]},
        (32, 16, 64): {"half": [(1, 2, 2), (2, 1, 1), (2, 2, 2), (4, 2, 2)]},
        (32, 32, 32): {"half": [(2, 2, 1), (2, 2, 2)]},
        (32, 64, 32): {"half": [(2, 2, 1), (2, 8, 8), (4, 4, 1), (4, 4, 2)]},
        (64, 16, 64): {"half": [(1, 2, 2)], "float": [(8, 2, 2)]},
        (64, 32, 32): {"half": [(4, 4, 1), (4, 4, 2)]},
        (64, 64, 32): {"half": [(2, 2, 2)]},
        (64, 64, 64): {"half": [(1, 4, 4), (2, 2, 2), (2, 2, 4)]},
        (128, 32, 32): {"float": [(8, 2, 2)]},
    },
}


def get_extra_vec_triples(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
    dtype_class: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return explicit per-tile vec triples not produced by any VecStrategy.

    Dtype-class-keyed (Step 7). If ``dtype_class`` is given, return only that
    dtype's extra triples; if None, return the union across dtype classes.
    Empty for tiles whose vecs are fully covered by strategies.
    """
    per_dc = _EXTRA_VEC_TRIPLES.get(variant, {}).get((tile_m, tile_n, tile_k))
    if not per_dc:
        return []
    if dtype_class is not None:
        return per_dc.get(dtype_class, [])
    out: List[Tuple[int, int, int]] = []
    seen = set()
    for dc in ("half", "float"):
        for tr in per_dc.get(dc, []):
            if tr not in seen:
                seen.add(tr)
                out.append(tr)
    return out


# =============================================================================
# Pipeline / Scheduler Rules (per-tile, replaces cross-product)
# =============================================================================

_COMPV4_SET: Set[Tuple[int, int, int]] = set(COMPV4_COMPATIBLE_TILES)


# =============================================================================
# Curated per-tile (pipeline, scheduler) map (extracted from JSON profiler configs)
# =============================================================================
# Per-variant mapping: tile → list of (pipeline, scheduler) pairs observed in JSON.
# Preferred over the rule-based computation below; tiles absent here fall back to
# the shape-based rules. This bounds pipeline/scheduler over-generation (the
# rules emit every pipeline permissible for a shape, whereas JSON benchmarked
# only a specific subset per tile).

_FWD_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 16, 128): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 32, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 64, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 64, 16): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave')],
    (32, 64, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 128, 16): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave')],
    (32, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 16, 16): [('compv1', 'intrawave')],
    (64, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 32, 16): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave')],
    (64, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 64, 8): [('compv1', 'intrawave')],
    (64, 64, 16): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 64, 64): [('compv3', 'intrawave')],
    (64, 128, 16): [('compv1', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (64, 128, 64): [('compv3', 'intrawave')],
    (128, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (128, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 64, 8): [('compv1', 'intrawave')],
    (128, 64, 16): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 64, 64): [('compv3', 'intrawave')],
    (128, 128, 16): [('compv1', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave'), ('compv4', 'intrawave')],
    (128, 128, 64): [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
    (128, 192, 16): [('compv1', 'intrawave')],
    (128, 256, 16): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (224, 256, 64): [('compv3', 'intrawave')],
    (256, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 32, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 64, 8): [('compv1', 'intrawave')],
    (256, 128, 16): [('compv1', 'intrawave')],
    (256, 128, 32): [('compv1', 'interwave'), ('compv1', 'intrawave')],
    (256, 224, 64): [('compv3', 'intrawave')],
    (256, 256, 32): [('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
}

_BWD_DATA_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 64, 32): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave')],
    (64, 16, 16): [('compv1', 'intrawave')],
    (64, 16, 32): [('compv1', 'intrawave')],
    (64, 16, 64): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'intrawave')],
    (128, 32, 64): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'intrawave')],
    (256, 128, 32): [('compv1', 'intrawave')],
}

_BWD_WEIGHT_TILE_PIPELINES: Dict[Tuple[int, int, int], List[Tuple[str, str]]] = {
    (16, 16, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (16, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 32, 64): [('basic_async_v1', 'intrawave'), ('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 64, 64): [('basic_async_v1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 128, 32): [('compv1', 'intrawave')],
    (16, 128, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (16, 256, 32): [('compv1', 'intrawave')],
    (16, 256, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 16, 64): [('compv1', 'interwave'), ('compv1', 'intrawave'), ('mem', 'interwave'), ('mem', 'intrawave')],
    (32, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (32, 64, 16): [('compv1', 'intrawave')],
    (32, 64, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (32, 128, 16): [('compv1', 'intrawave')],
    (32, 128, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (64, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (64, 32, 16): [('compv1', 'intrawave')],
    (64, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (64, 64, 16): [('compv1', 'intrawave')],
    (64, 64, 32): [('compv1', 'intrawave')],
    (64, 64, 64): [('basic_async_v1', 'intrawave'), ('compv1', 'intrawave')],
    (64, 128, 16): [('compv1', 'intrawave')],
    (64, 128, 32): [('compv1', 'intrawave')],
    (64, 128, 64): [('basic_async_v1', 'intrawave')],
    (128, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (128, 32, 16): [('compv1', 'intrawave')],
    (128, 32, 32): [('compv1', 'intrawave'), ('compv6', 'intrawave'), ('mem', 'intrawave')],
    (128, 64, 16): [('compv1', 'intrawave')],
    (128, 64, 32): [('compv1', 'intrawave')],
    (128, 128, 16): [('compv1', 'intrawave')],
    (128, 128, 32): [('compv1', 'intrawave')],
    (128, 128, 64): [('compv1', 'interwave'), ('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
    (128, 256, 16): [('compv1', 'intrawave')],
    (128, 256, 32): [('compv1', 'intrawave')],
    (256, 16, 64): [('mem', 'interwave'), ('mem', 'intrawave')],
    (256, 32, 64): [('basic_async_v1', 'intrawave')],
    (256, 128, 16): [('compv1', 'intrawave')],
    (256, 128, 32): [('compv1', 'intrawave')],
    (256, 256, 32): [('compv3', 'intrawave'), ('compv4', 'intrawave'), ('compv6', 'intrawave')],
}


def get_pipelines_for_tile(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[Tuple[str, str]]:
    """Return list of (pipeline, scheduler) pairs for a tile shape and variant.

    Prefers the curated per-tile map (from profiler JSON) when the tile is present;
    otherwise falls back to the shape-based rules below. The curated map bounds
    pipeline/scheduler over-generation; the rules cover tiles not seen in JSON.
    """
    tile_key = (tile_m, tile_n, tile_k)
    curated = {
        "forward": _FWD_TILE_PIPELINES,
        "bwd_data": _BWD_DATA_TILE_PIPELINES,
        "bwd_weight": _BWD_WEIGHT_TILE_PIPELINES,
    }.get(variant, {})
    if tile_key in curated:
        return list(curated[tile_key])

    tile_area = tile_m * tile_n
    min_dim = min(tile_m, tile_n)

    if variant == "forward":
        pipes: List[Tuple[str, str]] = [("compv1", "intrawave")]
        if tile_k >= 32:
            pipes.append(("compv1", "interwave"))
        if min_dim <= 32 and tile_k >= 64:
            pipes.append(("mem", "intrawave"))
            pipes.append(("mem", "interwave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv3", "intrawave"))
        if tile_key in _COMPV4_SET or (tile_area >= 16384 and tile_k >= 32):
            pipes.append(("compv4", "intrawave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv6", "intrawave"))
        return pipes

    elif variant == "bwd_data":
        return [("compv1", "intrawave")]

    elif variant == "bwd_weight":
        pipes = [("compv1", "intrawave")]
        if tile_k >= 64:
            pipes.append(("compv1", "interwave"))
        if tile_k >= 32:
            pipes.append(("mem", "intrawave"))
        if tile_k >= 64:
            pipes.append(("mem", "interwave"))
        if tile_area >= 4096 and tile_k >= 32:
            pipes.append(("compv3", "intrawave"))
        if tile_key in _COMPV4_SET or (tile_area >= 16384 and tile_k >= 32):
            pipes.append(("compv4", "intrawave"))
        if tile_k >= 32:
            pipes.append(("compv6", "intrawave"))
        if tile_k == 64:
            pipes.append(("basic_async_v1", "intrawave"))
        return pipes

    return [("compv1", "intrawave")]


# =============================================================================
# Specialization Rules (per-tile, replaces cross-product)
# =============================================================================


def get_specs_for_tile(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[str]:
    """Return list of specialization strings for a tile shape and variant.

    Rule-based specialization assignment — assigns only the specializations
    that make sense for the given tile dimensions.
    """
    if variant == "forward":
        if tile_k == 8:
            return ["default"]
        elif tile_k == 16:
            specs = ["default", "filter1x1_pad0", "filter1x1_stride1_pad0"]
            if tile_m * tile_n <= 4096:
                specs.append("filter3x3")
            return specs
        else:  # tile_k >= 32
            specs = ["default", "filter1x1_pad0", "filter1x1_stride1_pad0"]
            if tile_m * tile_n <= 4096:
                specs.append("filter3x3")
            return specs
    elif variant in ("bwd_data", "bwd_weight"):
        return ["default", "filter1x1_stride1_pad0"]

    return ["default"]


# =============================================================================
# Pipeline / Scheduler Rules (variant-level, kept for backward compat)
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


# Tiles used per feature (derived from JSON profiler analysis).
# Restricting features to specific tiles prevents config explosion.
_FWD_GM_TILES: List[Tuple[int, int, int]] = [
    (64, 16, 16), (128, 32, 32),
]
_BWD_EG_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 64), (16, 32, 64), (16, 64, 64), (16, 128, 64), (16, 256, 64),
    (32, 16, 64), (64, 16, 64), (128, 16, 64), (128, 128, 64),
    (256, 16, 64), (256, 256, 32),
]
_BWD_EG_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 64), (16, 32, 64), (16, 64, 64),
    (32, 16, 64), (64, 16, 64), (128, 16, 64), (256, 16, 64),
]
_BWD_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 32), (64, 64, 64),
]
_BWD_GM2_2S_TILES: List[Tuple[int, int, int]] = [
    (32, 32, 32), (32, 64, 32),
]
_BWD_GM4_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 128, 32), (32, 64, 32), (64, 32, 32),
]
_BWD_GM8_2S_TILES: List[Tuple[int, int, int]] = [
    (16, 256, 32), (32, 128, 32), (128, 32, 32),
]
_BWD_SK_TILES: List[Tuple[int, int, int]] = [
    (16, 16, 32), (16, 32, 64), (32, 16, 64),
    (64, 16, 64), (64, 64, 64), (128, 32, 32),
]

VARIANT_FEATURES: Dict[str, List[FeatureSpec]] = {
    "forward": [
        # split_image: small tile subset, compv1/intrawave only
        FeatureSpec(
            split_image=True,
            tile_override=_SPLIT_IMAGE_TILES,
            pipeline_override=[("compv1", "intrawave")],
        ),
        # num_groups_to_merge (restricted to tiles that benefit from merging)
        FeatureSpec(num_groups_to_merge=8,  tile_override=_FWD_GM_TILES),
        FeatureSpec(num_groups_to_merge=16, tile_override=[(64, 16, 16)]),
        FeatureSpec(num_groups_to_merge=32, tile_override=[(64, 16, 16)]),
    ],
    "bwd_data": [],
    "bwd_weight": [
        # explicit_gemm only: tiles with tile_k=64 (larger internal GEMM)
        FeatureSpec(explicit_gemm=True, tile_override=_BWD_EG_TILES),
        # two_stage only
        FeatureSpec(two_stage=True, tile_override=_BWD_2S_TILES),
        # two_stage + explicit_gemm
        FeatureSpec(two_stage=True, explicit_gemm=True, tile_override=_BWD_EG_2S_TILES),
        # num_groups_to_merge + two_stage combinations
        FeatureSpec(num_groups_to_merge=2, two_stage=True, tile_override=_BWD_GM2_2S_TILES),
        FeatureSpec(num_groups_to_merge=4, two_stage=True, tile_override=_BWD_GM4_2S_TILES),
        FeatureSpec(num_groups_to_merge=8, two_stage=True, tile_override=_BWD_GM8_2S_TILES),
        # basic_async_v1 + num_groups_to_merge=2
        FeatureSpec(
            num_groups_to_merge=2,
            tile_override=[(16, 32, 64), (16, 64, 64), (64, 128, 64)],
            pipeline_override=[("basic_async_v1", "intrawave")],
        ),
        # StreamK non-persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=False),
            tile_override=_BWD_SK_TILES,
            pipeline_override=[
                ("compv1", "intrawave"),
                ("mem", "intrawave"),
                ("basic_async_v1", "intrawave"),
            ],
        ),
        # StreamK persistent
        FeatureSpec(
            streamk_config=StreamKSpec(strategy="TREE", persistent=True),
            tile_override=_BWD_SK_TILES,
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

# =============================================================================
# Shared Validation Rules
# =============================================================================
# These functions are the single source of truth for validation rules
# for convolution code generation.

# --- Vector size validation ---


def is_valid_vector_size(vec: int) -> bool:
    """AMD GPUs only support vector widths 1, 2, 4, 8, 16."""
    return vec == 1 or vec % 2 == 0


def check_vectors(vec_a: int, vec_b: int, vec_c: int) -> bool:
    """Check all three vector sizes are valid (1 or even)."""
    return all(is_valid_vector_size(v) for v in (vec_a, vec_b, vec_c))



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
# Depthwise Convolution Parameter Space
# =============================================================================

DEPTHWISE_TILE_SIZES: List[Tuple[int, int]] = [
    (8, 8), (14, 28), (16, 16), (28, 28), (32, 32),
]

DEPTHWISE_FILTER_SIZES: List[int] = [3, 5]

DEPTHWISE_STRIDES: List[Tuple[int, int]] = [(1, 1), (2, 2)]

# Curated depthwise configs matching the JSON profiler set.
# Each tuple: (tile_h, tile_w, filt, str_h, str_w, sub_h, sub_w, nbatch, in_vec, out_vec)
# Padding is derived: pad = (filt - 1) // 2.
# Validated by is_valid_depthwise_config() at module load time.
DEPTHWISE_PARAMS: List[Tuple[int, ...]] = [
    # Filter 3, Stride (1,1)
    (8,  8,  3, 1, 1, 2, 2, 8, 2, 2),
    (16, 16, 3, 1, 1, 1, 4, 8, 8, 8),
    (16, 16, 3, 1, 1, 2, 2, 1, 2, 2),
    (28, 28, 3, 1, 1, 4, 4, 1, 8, 8),
    (32, 32, 3, 1, 1, 4, 4, 1, 8, 8),
    # Filter 3, Stride (2,2)
    (14, 28, 3, 2, 2, 2, 4, 1, 8, 8),
    (16, 16, 3, 2, 2, 1, 4, 1, 8, 8),
    (16, 16, 3, 2, 2, 1, 4, 2, 8, 8),
    (16, 16, 3, 2, 2, 2, 2, 1, 2, 2),
    (16, 16, 3, 2, 2, 2, 2, 1, 8, 8),
    (32, 32, 3, 2, 2, 2, 8, 1, 8, 8),
    (32, 32, 3, 2, 2, 4, 4, 1, 4, 4),
    (32, 32, 3, 2, 2, 4, 4, 1, 8, 8),
    (32, 32, 3, 2, 2, 4, 4, 2, 8, 8),
    # Filter 5, Stride (1,1)
    (8,  8,  5, 1, 1, 1, 1, 1, 1, 1),
    (8,  8,  5, 1, 1, 2, 2, 8, 2, 2),
    (16, 16, 5, 1, 1, 1, 4, 1, 8, 8),
    (16, 16, 5, 1, 1, 1, 4, 8, 8, 8),
    (28, 28, 5, 1, 1, 4, 4, 8, 8, 8),
    (32, 32, 5, 1, 1, 4, 4, 4, 8, 8),
]


def get_depthwise_configs():
    """Get curated depthwise convolution configurations.

    Returns the profiler config set, with each entry validated by
    tile_math.is_valid_depthwise_config().

    Returns:
        List of tile_math.DepthwiseConfig objects.
    """
    from tile_math import DepthwiseConfig, is_valid_depthwise_config

    configs = []
    for params in DEPTHWISE_PARAMS:
        th, tw, filt, sh, sw, sub_h, sub_w, nb, iv, ov = params
        pad = (filt - 1) // 2
        cfg = DepthwiseConfig(th, tw, filt, sh, sw, pad, pad, nb, sub_h, sub_w, iv, ov)
        assert is_valid_depthwise_config(cfg), f"Invalid depthwise config: {params}"
        configs.append(cfg)
    return configs


if __name__ == "__main__":
    all_tiles = sorted(set(COMMON_TILES) | set(FWD_TILES) | set(BWD_DATA_TILES) | set(BWD_WEIGHT_TILES))
    print(f"Total unique tiles: {len(all_tiles)}")
    print(f"  COMMON: {len(COMMON_TILES)}, FWD: {len(FWD_TILES)}, "
          f"BWD_DATA: {len(BWD_DATA_TILES)}, BWD_WEIGHT: {len(BWD_WEIGHT_TILES)}")
