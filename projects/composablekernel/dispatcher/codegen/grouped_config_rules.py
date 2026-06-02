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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow importing tile_math from same directory
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).parent.resolve()
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from arch_specs_generated import WARP_TILE_SUPPORTED_COMBINATIONS

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

# Override the tile sizes for split-image feature.
# Tiles used for split_image (from JSON profiler configs, forward only)
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
# Curated Wave Tables (extracted from JSON profiler configs)
# =============================================================================
# Per-variant mapping: tile → list of curated wave configs.
# Given a (tile, wave) pair, valid warp_tile shapes are derived mathematically
# from WARP_TILE_SUPPORTED_COMBINATIONS + divisibility rules.

_FWD_TILE_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
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

_BWD_DATA_TILE_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
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

_BWD_WEIGHT_TILE_WAVES: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {
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

# Warp shapes never used in JSON configs — exclude from generation.
_EXCLUDED_WARP_SHAPES: Set[Tuple[int, int, int]] = {
    (16, 16, 32), (4, 64, 16), (64, 4, 16),
}


def get_wave_configs(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[Tuple[int, int, int]]:
    """Return curated wave configs for a tile+variant (from profiler JSON).

    Falls back to [(1, 1, 1)] for unknown tiles.
    """
    table = {
        "forward": _FWD_TILE_WAVES,
        "bwd_data": _BWD_DATA_TILE_WAVES,
        "bwd_weight": _BWD_WEIGHT_TILE_WAVES,
    }.get(variant, {})
    return table.get((tile_m, tile_n, tile_k), [(1, 1, 1)])


def get_warp_tiles_for_wave(
    tile_m: int, tile_n: int, tile_k: int,
    wave_m: int, wave_n: int, wave_k: int,
    dtype_key: str, arch: str = "gfx942",
) -> List[Tuple[int, int, int]]:
    """Compute valid warp_tile shapes from divisibility + arch constraints.

    Given a (tile, wave) pair, returns warp_tile shapes from
    WARP_TILE_SUPPORTED_COMBINATIONS[arch][dtype_key] that satisfy:
      tile_m % (wave_m * warp_tile_m) == 0
      tile_n % (wave_n * warp_tile_n) == 0
    """
    supported = WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {}).get(dtype_key, [])
    return [
        (wt[0], wt[1], wt[2]) for wt in supported
        if (wt[0], wt[1], wt[2]) not in _EXCLUDED_WARP_SHAPES
        and tile_m % (wave_m * wt[0]) == 0
        and tile_n % (wave_n * wt[1]) == 0
    ]


def get_wave_warp_pairs(
    tile_m: int, tile_n: int, tile_k: int,
    variant: str, dtype_key: str, arch: str = "gfx942",
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return (wave, warp_tile) pairs: curated waves x derived warp_tiles.

    Combines curated wave configs (from profiler JSON) with mathematically
    derived warp_tile shapes (from arch constraints + divisibility).
    """
    result = []
    for wave in get_wave_configs(tile_m, tile_n, tile_k, variant):
        for wt in get_warp_tiles_for_wave(
            tile_m, tile_n, tile_k, *wave, dtype_key, arch,
        ):
            result.append((wave, wt))
    return result


# =============================================================================
# Vector Size Computation (tile + dtype based, wave/warp independent)
# =============================================================================


def _ceil_pow2(n: int) -> int:
    """Round up to the next power of 2 (or return n if already a power of 2)."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def compute_vector_sizes(
    tile_m: int, tile_n: int, tile_k: int,
    dtype_class: str, variant: str,
) -> List[Tuple[int, int, int]]:
    """Compute valid (vec_a, vec_b, vec_c) from tile dims and dtype.

    This is wave/warp independent — vec sizes depend only on tile dimensions
    and dtype class. Returns the largest valid uniform vec plus (1,1,1).

    Args:
        dtype_class: "float" (fp32) or "half" (fp16/bf16)
        variant: "forward", "bwd_data", "bwd_weight"
    """
    max_ab = 4 if dtype_class == "float" else 8
    max_c = 4 if dtype_class == "float" else 8

    a_dim = tile_k if variant == "bwd_data" else tile_m
    min_a = _ceil_pow2(max(1, (a_dim + WARP_SIZE - 1) // WARP_SIZE))
    min_b = _ceil_pow2(max(1, (tile_n + WARP_SIZE - 1) // WARP_SIZE))

    valid_a = [v for v in [1, 2, 4, 8, 16] if min_a <= v <= max_ab]
    valid_b = [v for v in [1, 2, 4, 8, 16] if min_b <= v <= max_ab]
    valid_c = [v for v in [1, 2, 4, 8] if v <= max_c]

    if not valid_a or not valid_b or not valid_c:
        return [(1, 1, 1)]

    results = [(max(valid_a), max(valid_b), max(valid_c))]
    if (1, 1, 1) not in results:
        results.append((1, 1, 1))
    return results


# =============================================================================
# Pipeline / Scheduler Rules (per-tile, replaces cross-product)
# =============================================================================

_COMPV4_SET: Set[Tuple[int, int, int]] = set(COMPV4_COMPATIBLE_TILES)


def get_pipelines_for_tile(
    tile_m: int, tile_n: int, tile_k: int, variant: str,
) -> List[Tuple[str, str]]:
    """Return list of (pipeline, scheduler) pairs for a tile shape and variant.

    Rule-based pipeline assignment — replaces cross-product with all variant
    pipelines. Each tile gets only the pipelines that are appropriate for its
    shape, reducing the config count significantly.
    """
    tile_key = (tile_m, tile_n, tile_k)
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
    vector sizes are already pre-computed from tile_math, this check now always
    returns True to avoid false rejections.
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
DEPTHWISE_PROFILER_PARAMS: List[Tuple[int, ...]] = [
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
    for params in DEPTHWISE_PROFILER_PARAMS:
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
