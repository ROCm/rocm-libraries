#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA hardware and kernel specifications.

This file consolidates ALL architecture-dependent facts, tile constraints,
feature rules, tile tables, and profile definitions.

Sections:
    1. Architecture capabilities (dtypes, pipelines, hdims)
    2. Tile hardware parameters (LDS, warp classes, element sizes)
    3. Tile constraints (per-arch rules that reject invalid tiles)
    4. Feature compatibility rules (pipeline × feature flag interactions)
    5. Variant-specific tile tables (BWD, SplitKV combine, etc.)
    6. Receipt filters (named subsets for deployment profiles)
    7. Profiles (named predicates for codegen filtering)
"""

import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Ensure this directory is on sys.path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from symbol_map import (  # noqa: E402
    BWD_DTYPE_MAP,
    FWD_DTYPE_MAP,
    canonical_bias,
    canonical_mask,
    canonical_qscale,
)


# =============================================================================
# 1. Architecture capabilities
# =============================================================================

# Dtypes supported per GPU architecture.
# Source: warp_gemm_dispatcher.hpp template specializations (MFMA for gfx9*, WMMA for gfx11/12)
ARCH_DTYPES: Dict[str, List[str]] = {
    "gfx90a": ["fp16", "bf16", "fp32"],
    "gfx942": ["fp16", "bf16", "fp32", "fp8", "fp8bf16", "fp8fp32", "bf8"],
    "gfx950": [
        "fp16",
        "bf16",
        "fp32",
        "fp8",
        "fp8bf16",
        "fp8fp32",
        "bf8",
        "mxfp8",
        "mxfp4",
    ],
    "gfx1100": ["fp16", "bf16"],
    "gfx1201": ["fp16", "bf16", "fp8", "fp8bf16"],  # WMMA fp8 dispatchers (gfx12)
}

# Supported hdim combinations per dtype family.
# hdim_q must satisfy ceil_to_qualified_tile_length() in tile_fmha_shape.hpp:
#   valid hdim_q: 32, 48, 64, 80, 96, 128, 160, 192, 256
# hdim_v is constrained only by warp tile divisibility (no ceil function).
SUPPORTED_HDIMS: Dict[str, List[Tuple[int, int]]] = {
    "fp16": [
        (32, 32),
        (64, 64),
        (80, 96),
        (96, 128),
        (128, 128),
        (160, 160),
        (192, 128),
        (192, 192),
        (256, 256),
    ],
    "bf16": [
        (32, 32),
        (64, 64),
        (80, 96),
        (96, 128),
        (128, 128),
        (160, 160),
        (192, 128),
        (192, 192),
        (256, 256),
    ],
    "fp32": [
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 128),
        (128, 128),
        (192, 192),
        (256, 256),
    ],
    "fp8": [(64, 64), (128, 128), (192, 128), (256, 256)],
    "fp8bf16": [(64, 64), (128, 128), (192, 128), (256, 256)],
    "fp8fp32": [(128, 128)],
    "bf8": [(64, 64), (128, 128), (192, 128), (256, 256)],
    "mxfp8": [(128, 128), (256, 256)],
    "mxfp4": [(128, 128), (256, 256)],
}

# Architecture metadata for validation and codegen
ARCH_METADATA: Dict[str, dict] = {
    "gfx90a": {
        "family": "cdna2",
        "arch_tag": "ck_tile::gfx9_t",
        "supported_dtypes": ["fp16", "bf16", "fp32"],
        "supported_pipelines": [
            "qr",
            "qr_async",
            "qs",
            "qr_pagedkv",
            "qr_nwarp_sshuffle",
            "appendkv",
            "bwd",
        ],
        "supports_trload": False,
        "supports_v3": False,
    },
    "gfx942": {
        "family": "cdna3",
        "arch_tag": "ck_tile::gfx9_t",
        "supported_dtypes": [
            "fp16",
            "bf16",
            "fp32",
            "fp8",
            "fp8fp16",
            "fp8bf16",
            "fp8fp32",
            "bf8",
        ],
        "supported_pipelines": [
            "qr",
            "qr_async",
            "qs",
            "qr_pagedkv",
            "qr_nwarp_sshuffle",
            "appendkv",
            "bwd",
        ],
        "supports_trload": False,
        "supports_v3": False,
    },
    "gfx950": {
        "family": "cdna4",
        "arch_tag": "ck_tile::gfx9_t",
        "supported_dtypes": [
            "fp16",
            "bf16",
            "fp32",
            "fp8",
            "fp8fp16",
            "fp8bf16",
            "fp8fp32",
            "bf8",
        ],
        "supported_pipelines": [
            "qr",
            "qr_async",
            "qs",
            "qr_async_trload",
            "qr_async_trload_v3",
            "v3",
            "qr_pagedkv",
            "qr_nwarp_sshuffle",
            "appendkv",
            "bwd",
        ],
        "supports_trload": True,
        "supports_v3": True,
    },
    "gfx1100": {
        "family": "rdna3",
        "arch_tag": "ck_tile::gfx11_t",
        "supported_dtypes": ["fp16", "bf16"],
        "supported_pipelines": [
            "qr",
            "qr_pagedkv",
            "qr_nwarp_sshuffle",
            "appendkv",
            "bwd",
        ],
        "supports_trload": False,
        "supports_v3": False,
    },
    "gfx1201": {
        "family": "rdna4",
        "arch_tag": "ck_tile::gfx12_t",
        "supported_dtypes": ["fp16", "bf16", "fp8", "fp8bf16"],
        "supported_pipelines": [
            "qr",
            "qr_pagedkv",
            "qr_nwarp_sshuffle",
            "appendkv",
            "bwd",
        ],
        "supports_trload": False,
        "supports_v3": False,
    },
}


# =============================================================================
# 2. Tile hardware parameters
# =============================================================================

# LDS capacity per pipeline type (bytes) — used for non-async pipelines.
# Source: arch.hpp get_smem_capacity() returns 65536 (64 KiB) for gfx90a/gfx942.
# Async pipelines (qr_async*) compute LDS budgets from the exact CK formula
# in tile_passes_all_constraints() since they use 3× double-buffered KV tiles
# and target gfx950 which has 163840 bytes (160 KiB) LDS.
LDS_LIMITS: Dict[str, int] = {
    "qr": 65536,
    "qs": 65536,
}

# Warp configurations: (wm0, wn0, wk0) — MFMA/WMMA instruction tile sizes.
# Source: warp_gemm_dispatcher.hpp template specializations.
WARP_CLASSES: Dict[str, List[Tuple[int, int, int]]] = {
    "fp16": [(32, 32, 16), (16, 16, 32)],
    "bf16": [(32, 32, 16), (16, 16, 32)],
    "fp32": [(16, 16, 16)],
    "fp8": [(32, 32, 32)],
    "fp8bf16": [(32, 32, 32)],
    "fp8fp32": [(32, 32, 32)],
    "bf8": [(32, 32, 32)],
    "mxfp8": [(32, 32, 64)],
    "mxfp4": [(16, 16, 128)],
}

# Element sizes in bytes — sizeof() of the C++ storage type.
# Source: half_t=2, bf16_t=2, float=4, fp8_t/bf8_t=1.
# mxfp4: pk_fp4_t packs 2 fp4 elements into 1 byte (uint8_t), so elem_size=1.
ELEMENT_SIZES: Dict[str, int] = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp8": 1,
    "fp8bf16": 1,
    "fp8fp32": 1,
    "bf8": 1,
    "mxfp8": 1,
    "mxfp4": 1,
}

# Block tile dimensions to sweep — union of all tile sizes used across CK example factories.
# Must be multiples of the warp tile (16 or 32 depending on dtype).
# 192, 384 included for high-occupancy configs on large sequence lengths.
VALID_BM0 = [16, 32, 64, 128, 192, 256]
VALID_BN0 = [16, 32, 64, 96, 128, 192, 256, 384]
VALID_BK0 = [16, 32, 64, 96, 128, 256]

# K0_MAX_SUBMAX_MAP: maps hdim_q → kSubQKHeaddim (the padded K-tile length).
# Source: tile_fmha_shape.hpp ceil_to_qualified_tile_length<Headdim>().
#   32→32(pow2), 48→48(special), 64→64(pow2), 80→96(special),
#   96→128(special), 128→128(pow2), 160→256(special), 192→192(special), 256→256(pow2).
K0_MAX_SUBMAX_MAP: Dict[int, int] = {
    32: 32,
    48: 48,
    64: 64,
    80: 96,
    96: 128,
    128: 128,
    160: 256,
    192: 192,
    256: 256,
}


# =============================================================================
# 3. Tile constraints
# =============================================================================


def check_gfx9_tile_constraints(
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    pipeline: str,
    bm0: int,
    bn0: int,
    bk0: int,
) -> bool:
    """Gfx9 compatibility rules.

    Source: fmha_fwd.py CompatibilityRuleFactoryGfx9.check_hdim_tile().
    Applies to gfx90a, gfx942, gfx950 for pipelines in {qr, qr_async, qs}.
    Note: CK factory is stricter (bm0==128 only for non-128 hdims); we allow
    {64, 128, 192, 256} to let the tile engine explore more configurations.
    """
    if dtype == "fp32":
        return True
    if pipeline not in ("qr", "qr_async", "qs"):
        return True
    if (hdim_q, hdim_v) == (128, 128) and bn0 != 128:
        return False
    if (hdim_q, hdim_v) == (128, 128) and pipeline == "qr_async" and bm0 != 128:
        return False
    if (hdim_q, hdim_v) != (128, 128) and bm0 not in (64, 128, 192, 256):
        return False
    if (hdim_q, hdim_v) == (128, 128) and pipeline != "qr_async" and bk0 == 64:
        return False
    return True


def check_gfx950_tile_constraints(
    hdim_q: int,
    hdim_v: int,
    pipeline: str,
    bm0: int,
    bn0: int,
) -> bool:
    """Gfx950 trload/v3 constraints.

    Source: fmha_fwd.py CompatibilityRuleFactoryGfx950.check_tile_pipeline().
    Note: CK enforces biconditional (v3_tile ↔ v3_pipeline); we only enforce
    v3_pipeline → v3_tile since non-v3 pipelines may still use bm0=256.
    """
    if pipeline == "qr_async_trload":
        if (hdim_q, hdim_v) == (128, 128) and bn0 == 128:
            return False
        if (hdim_q, hdim_v) not in [(64, 64), (128, 128)]:
            return False
    is_v3_tile = bm0 == 256
    is_v3_pipeline = pipeline == "qr_async_trload_v3"
    # v3 pipeline requires bm0=256; other pipelines also allow bm0=256
    if is_v3_pipeline and not is_v3_tile:
        return False
    return True


def check_qr_mfma_insts(
    arch: str,
    hdim_q: int,
    pipeline: str,
    bn0: int,
    bk0: int,
    wn0: int,
    wk0: int,
) -> bool:
    """NumMfmaInsts % 8 == 0 check.

    Source: block_fmha_pipeline_qr_ks_vs.hpp static_assert at line ~490.
    Full C++ formula: (kM0/WarpM)*(kN0/WarpN)*(kK0/WarpK) / (MWarp*NWarp).
    We simplify to (bn0/wn0)*(bk0/wk0), omitting (bm0/wm0)/(rm0*rn0) which
    equals 1 for all current fp16/bf16/fp32/fp8 tiles, or a power-of-2 factor
    for mxfp8/mxfp4 that doesn't change the mod-8 result.  This is conservative:
    it can only reject tiles the full formula would also reject, never the reverse.
    Only applies to qr pipeline + hdim_q==256 + CDNA (gfx9*).
    """
    if pipeline != "qr" or hdim_q != 256:
        return True
    if not arch.startswith("gfx9"):
        return True
    num_mfma = (bn0 // wn0) * (bk0 // wk0)
    if num_mfma % 8 != 0:
        return False
    return True


def tile_passes_all_constraints(
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    pipeline: str,
    bm0: int,
    bn0: int,
    bk0: int,
    wm0: int,
    wn0: int,
    wk0: int,
) -> bool:
    """Master constraint check — returns True if the tile is valid."""
    elem_size = ELEMENT_SIZES.get(dtype, 2)
    lds_limit = LDS_LIMITS.get(pipeline, 65536)

    # LDS capacity check (pipeline-dependent formula)
    if pipeline in ("qr_async", "qr_async_trload", "qr_async_trload_v3"):
        # Async pipeline: Q is in registers. LDS holds NumKVLdsBuffers (=3) copies of
        # max(SingleKSize, SingleVSize). Derived from GetSmemSizeKV() in
        # block_fmha_pipeline_qx_ks_vs_custom_policy.hpp.
        #
        # SingleVSize formula (MakeVLdsBlockDescriptor):
        #   Banks=32, PixelsPerRow = Banks*4/sizeof(dtype) = 32*4/elem_size
        #   kKPack = 16/elem_size (GetSmemKPackV)
        #   NPerRow = PixelsPerRow/kKPack
        #   SingleVSize = (bk1/kKPack) * (hdim_v/NPerRow) * (PixelsPerRow + kKPack)
        # For bf16: PixelsPerRow=64, kKPack=8, NPerRow=8
        #   SingleVSize = (32/8)*(hdim_v/8)*(64+8) = 4*(hdim_v/8)*72 = 36*hdim_v
        #
        # SingleKSize formula (GetSingleSmemElementSpaceSize, async branch):
        #   KPack = 16/elem_size, KVector = alignment (gfx950: 16/elem_size = 8 for bf16)
        #   LanesPerK = bk0/KVector, LaneGroups = 64/LanesPerK
        #   NumIssues = bn0/(LaneGroups*NumWarps)
        #   SingleKSize = NumIssues*NumWarps*(64*KVector + KPack)
        #
        bk1 = 32  # kK1 in TileFmhaShape — design choice from fmha_fwd.py tile defs
        num_warps = bm0 // wm0
        # Banks: arch.hpp get_n_lds_banks() — 64 for gfx950, 32 for older
        banks = 64 if arch == "gfx950" else 32
        pixels_per_row = banks * 4 // elem_size  # Banks * 4bytes / sizeof(dtype)
        k_pack = 16 // elem_size  # GetSmemKPackV: 16 / sizeof(dtype)
        n_per_row = pixels_per_row // k_pack
        single_v = (bk1 // k_pack) * (hdim_v // n_per_row) * (pixels_per_row + k_pack)

        # KVector: GetAlignmentK in custom_policy.hpp — MaxLoadSizeInBytes / sizeof(dtype)
        # gfx950 uses dwordx4 (16 bytes), older uses dword (4 bytes)
        k_vector = 16 // elem_size if arch == "gfx950" else 4 // elem_size
        lanes_per_k = bk0 // k_vector if k_vector > 0 else 1
        lane_groups = 64 // lanes_per_k if lanes_per_k > 0 else 1  # WarpSize=64
        num_issues = (
            bn0 // (lane_groups * num_warps) if (lane_groups * num_warps) > 0 else 0
        )
        single_k = num_issues * num_warps * (64 * k_vector + k_pack)

        single_buf_bytes = max(single_k, single_v) * elem_size
        # NumPrefetchK = NumPrefetchV = 3 (async_default_policy.hpp)
        num_kv_buffers = 3
        # Q uses registers (QLoadOnce=true), so GetSmemSizeQ() = 0.
        total_lds = single_buf_bytes * num_kv_buffers
        # gfx950 HW LDS limit: arch.hpp get_smem_capacity() = 163840 (160 KiB)
        if total_lds > 163840:
            return False
    else:
        # Non-async (qr/qs): Q and K tiles share LDS simultaneously
        if (bm0 * bk0 + bn0 * bk0) * elem_size > lds_limit:
            return False
    # bk0 range
    if bk0 > hdim_q:
        return False
    # hdim_q divisibility (tile_fmha_shape.hpp:60)
    if hdim_q % bk0 != 0:
        return False
    # Warp alignment
    if bm0 % wm0 != 0 or bk0 % wk0 != 0 or bn0 % wn0 != 0:
        return False
    # MFMA inst count
    if not check_qr_mfma_insts(arch, hdim_q, pipeline, bn0, bk0, wn0, wk0):
        return False
    # Async DMA distribution constraint (MakeKLdsStoreBlockDescriptor, custom_policy.hpp).
    # NumIssues = kNPerBlock / (LaneGroups * NumWarps) must be a positive integer, where
    # LaneGroups = WarpSize / LanesPerK = 64 / (bk0 / KVector).
    # Equivalently: (bn0 * bk0) % (kBlockSize * KVector) == 0.
    # KVector = MaxLoadSizeInBytes / sizeof(dtype): gfx950=16/2=8, older=4/2=2 for bf16.
    if pipeline == "qr_async" and arch.startswith("gfx9"):
        kvector = 16 // elem_size if arch == "gfx950" else 4 // elem_size
        num_warps = bm0 // wm0
        block_size = num_warps * 64  # WarpSize = 64
        if (bn0 * bk0) % (block_size * kvector) != 0:
            return False
    # Arch constraints
    if arch in ("gfx90a", "gfx942", "gfx950"):
        if not check_gfx9_tile_constraints(
            dtype, hdim_q, hdim_v, pipeline, bm0, bn0, bk0
        ):
            return False
    if arch == "gfx950":
        if not check_gfx950_tile_constraints(hdim_q, hdim_v, pipeline, bm0, bn0):
            return False
    return True


# =============================================================================
# 4. Feature compatibility rules
# =============================================================================

# Supported mask, bias, and boolean values for feature products.
# These are the template enum values in CK's FMHA traits structs.
MASKS = ["no", "causal", "generic"]
BIASES = ["no", "bias", "alibi"]
BOOLS = ["t", "f"]

# Dtype groups matching CK's _DT_* classification in fmha_fwd.py factory classes.
DT_FP16_BF16 = {"fp16", "bf16"}
DT_FP8 = {"fp8bf16", "fp8", "bf8"}
DT_FP8FP32 = {"fp8fp32"}
DT_FP32 = {"fp32"}


def check_logits_bias(logits: str, bias: str) -> bool:
    """logits_soft_cap requires no bias.

    Source: fmha_fwd.py CompatibilityRuleFactory.check_feature().
    """
    return not (logits == "t" and bias != "no")


def check_group_mode_padding(mode: str, spad: str, skpad: str) -> bool:
    """Group mode requires spad=t and skpad=t.

    Source: fmha_fwd.py CompatibilityRuleFactory.check_feature() +
    block_fmha_pipeline static_asserts for padding.
    """
    if mode == "group":
        return spad == "t" and skpad == "t"
    return True


# =============================================================================
# 5. Variant-specific tile tables
# =============================================================================

# --- SplitKV combine ---
SPLITKV_COMBINE_HDIMS_FP16 = [32, 64, 96, 128, 256]
SPLITKV_COMBINE_HDIMS_FP8 = [64, 128, 256]

# --- BWD dq_dk_dv: main tile per hdim ---
# Source: fmha_bwd.py KernelComponentFactoryGfx9.get_dq_dk_dv_tiles().
# Format: (bm0, bn0, bk0, bn1, bk1, bk0max, tile6, tile7, tile8)
BWD_DQ_DK_DV_TILES_FP16: Dict[Tuple[int, int], Tuple[int, ...]] = {
    (32, 32): (32, 128, 32, 32, 32, 32, 64, 32, 32),
    (64, 64): (32, 128, 64, 32, 64, 32, 32, 64, 64),
    (96, 128): (32, 128, 96, 32, 96, 32, 32, 96, 96),
    (128, 128): (16, 128, 128, 16, 128, 16, 32, 128, 128),
    (256, 256): (16, 64, 256, 16, 256, 16, 32, 256, 256),
}

# Extra tiles for specific hdims: (tile_tuple, tag, is_batch_only).
# Source: fmha_bwd.py KernelComponentFactoryGfx950 trload/small/bn192 variants.
BWD_DQ_DK_DV_EXTRA_TILES: Dict[
    Tuple[int, int], List[Tuple[Tuple[int, ...], str, bool]]
] = {
    (64, 64): [
        ((32, 128, 64, 32, 64, 32, 32, 64, 64), "trload", False),
        ((32, 16, 64, 32, 64, 32, 16, 64, 64), "small", True),
    ],
    (128, 128): [
        ((16, 16, 128, 16, 128, 16, 16, 128, 128), "small", True),
        ((16, 192, 128, 16, 128, 16, 32, 128, 128), "bn192", False),
        ((32, 128, 128, 32, 128, 32, 32, 128, 128), "trload", False),
    ],
}

BWD_EXTRA_PAD_COMBOS = [("f", "f"), ("8", "8")]
BWD_SMALL_DROPOUTS = ["no"]

BWD_DOT_DO_O_HDIMS = [32, 64, 96, 128, 256]
BWD_CONVERT_DQ_HDIMS = [32, 64, 96, 128, 256]
BWD_CONVERT_DQ_TILE_GROUPS: Dict[int, int] = {32: 1, 64: 1, 96: 1, 128: 1, 256: 1}

BWD_DROPOUTS = ["no", "dropout_wg16", "dropout_wg16_storerandval"]
BWD_PAD_COMBOS = [
    ("f", "f"),
    ("f", "t"),
    ("f", "8"),
    ("t", "f"),
    ("t", "t"),
    ("t", "8"),
    ("8", "8"),
]

# --- BWD wave/warp lookup (non-derivable patterns) ---
# Source: fmha_bwd.py FmhaBwdDQDKDVTileSize constructors (F_rm0..F_rk2, F_wk1).
# Key: (bm0, bn0, bk0, trload_flag).  Value: {wave: 9-tuple, warp_k1: int}.
BWD_DQ_WAVE_WARP: Dict[Tuple[int, int, int, str], dict] = {
    (16, 16, 128, "t"): {"wave": (1, 1, 1, 1, 1, 1, 1, 1, 1), "warp_k1": 16},
    (16, 64, 256, "f"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 16},
    (16, 128, 128, "f"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 16},
    (16, 192, 128, "t"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 16},
    (32, 16, 64, "t"): {"wave": (1, 1, 1, 1, 1, 1, 1, 1, 1), "warp_k1": 16},
    (32, 128, 32, "f"): {"wave": (1, 4, 1, 4, 1, 1, 2, 2, 1), "warp_k1": 16},
    (32, 128, 64, "f"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 16},
    (32, 128, 64, "t"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 32},
    (32, 128, 96, "f"): {"wave": (1, 4, 1, 4, 1, 1, 2, 2, 1), "warp_k1": 16},
    (32, 128, 128, "t"): {"wave": (1, 4, 1, 4, 1, 1, 1, 4, 1), "warp_k1": 32},
    (64, 128, 32, "f"): {"wave": (2, 4, 1, 4, 1, 1, 2, 4, 1), "warp_k1": 16},
    (64, 128, 64, "f"): {"wave": (2, 4, 1, 4, 1, 1, 2, 4, 1), "warp_k1": 16},
    (64, 128, 128, "f"): {"wave": (2, 4, 1, 4, 1, 1, 2, 4, 1), "warp_k1": 16},
}


# =============================================================================
# 6. Receipt filters
# =============================================================================


class Receipt(IntEnum):
    """Named receipt levels for deployment profiles.

    These are deployment-specific filters, not derived from C++ constraints.
    They control which kernel subsets are emitted for different integration
    targets (PyTorch, AITER, Flash-Attention, etc.).
    """

    CK_DEFAULT = 0
    CK_EXTENDED = 1
    FLASH_FWD = 2
    FLASH_BWD = 3
    PYTORCH = 4
    AITER_BATCH = 100
    AITER_GROUP = 200
    AITER_BWD_BATCH = 300
    AITER_BWD_GROUP = 400
    AITER_CPP = 600
    FP32_ALL = 800
    FP32_MIN = 801
    FP8_TEST = 888


RECEIPT_FILTERS: Dict[int, Callable[[str, object], bool]] = {
    0: lambda dtype, spec: dtype != "fp32",
    2: lambda dtype, spec: (
        dtype in ("fp16", "bf16")
        and getattr(spec, "bias", "no") in ("no", "alibi")
        and getattr(spec, "qscale", "no") == "no"
        and getattr(spec, "skip", "f") == "f"
        and getattr(spec, "sink", "f") == "f"
    ),
    4: lambda dtype, spec: (
        dtype in ("fp16", "bf16")
        and getattr(spec, "bias", "no") in ("no", "bias")
        and getattr(spec, "qscale", "no") == "no"
        and getattr(spec, "skip", "f") == "f"
        and getattr(spec, "logits", "f") == "f"
    ),
    100: lambda dtype, spec: dtype in ("fp16", "bf16", "fp8bf16"),
    200: lambda dtype, spec: dtype in ("fp16", "bf16", "fp8bf16"),
    600: lambda dtype, spec: dtype in ("fp16", "bf16", "fp8bf16"),
    888: lambda dtype, spec: dtype in ("fp8bf16", "fp8fp32"),
    800: lambda dtype, spec: (
        dtype == "fp32"
        and getattr(spec, "skip", "f") == "f"
        and getattr(spec, "logits", "f") == "f"
    ),
}


def receipt_filter(receipt: int, dtype: str, spec) -> bool:
    """Apply receipt-level filter. Returns True if the kernel should be kept."""
    fn = RECEIPT_FILTERS.get(receipt)
    if fn is None:
        return dtype != "fp32"
    return fn(dtype, spec)


# =============================================================================
# 7. Profiles
# =============================================================================

PROFILE_ALIASES: Dict[str, str] = {str(r.value): r.name.lower() for r in Receipt}


@dataclass(frozen=True)
class FmhaProfile:
    name: str
    predicate: Callable[[dict], bool]

    def allows(self, config: dict) -> bool:
        return self.predicate(config)


def _dtype_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["data_type"] in set(allowed)


def _mode_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["mode"] in set(allowed)


def _family_is(config: dict, allowed: Iterable[str]) -> bool:
    return config["signature"]["family"] in set(allowed)


def _common_row_major_filter(config: dict) -> bool:
    return config["signature"]["vlayout"] == "r"


def _bias_is(config: dict, allowed: Iterable[str]) -> bool:
    return canonical_bias(config["signature"]["bias"]) in set(allowed)


def _qscale_is(config: dict, allowed: Iterable[str]) -> bool:
    return canonical_qscale(config["signature"]["qscale"]) in set(allowed)


def _no_skip_or_logits(config: dict) -> bool:
    return (not config["signature"]["skip_min_seqlen_q"]) and (
        not config["signature"]["logits"]
    )


PROFILES: Dict[str, FmhaProfile] = {
    "ck_default": FmhaProfile(
        "ck_default", lambda c: c["signature"]["data_type"] != "fp32"
    ),
    "ck_extended": FmhaProfile(
        "ck_extended", lambda c: c["signature"]["data_type"] != "fp32"
    ),
    "flash_fwd": FmhaProfile(
        "flash_fwd",
        lambda c: (
            _family_is(c, {"fwd", "fwd_splitkv", "fwd_appendkv", "fwd_pagedkv"})
            and _dtype_is(c, {"fp16", "bf16"})
            and _common_row_major_filter(c)
            and _bias_is(c, {"no", "alibi"})
            and _qscale_is(c, {"no"})
            and not c["signature"]["skip_min_seqlen_q"]
        ),
    ),
    "flash_bwd": FmhaProfile(
        "flash_bwd",
        lambda c: (
            _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
            and _dtype_is(c, {"fp16", "bf16"})
        ),
    ),
    "pytorch": FmhaProfile(
        "pytorch",
        lambda c: (
            _dtype_is(c, {"fp16", "bf16"})
            and _common_row_major_filter(c)
            and _bias_is(c, {"no", "bias"})
            and _qscale_is(c, {"no"})
            and _no_skip_or_logits(c)
            and not c["signature"].get("sink", False)
        ),
    ),
    "aiter_batch": FmhaProfile(
        "aiter_batch",
        lambda c: (
            _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
            and _mode_is(c, {"batch"})
            and _common_row_major_filter(c)
            and (
                c["signature"]["data_type"] != "fp8bf16"
                or c["signature"]["hdim_q"] in {128, 192}
            )
        ),
    ),
    "aiter_group": FmhaProfile(
        "aiter_group",
        lambda c: (
            _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
            and _mode_is(c, {"group"})
            and _common_row_major_filter(c)
        ),
    ),
    "aiter_bwd_batch": FmhaProfile(
        "aiter_bwd_batch",
        lambda c: (
            _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
            and _dtype_is(c, {"fp16", "bf16"})
            and _mode_is(c, {"batch"})
        ),
    ),
    "aiter_bwd_group": FmhaProfile(
        "aiter_bwd_group",
        lambda c: (
            _family_is(c, {"bwd_dot_do_o", "bwd_dq_dk_dv", "bwd_convert_dq"})
            and _dtype_is(c, {"fp16", "bf16"})
            and _mode_is(c, {"group"})
        ),
    ),
    "aiter_cpp": FmhaProfile(
        "aiter_cpp",
        lambda c: (
            _dtype_is(c, {"fp16", "bf16", "fp8bf16"})
            and _common_row_major_filter(c)
            and (
                c["signature"]["data_type"] != "fp8bf16"
                or c["signature"]["hdim_q"] in {128, 192}
            )
        ),
    ),
    "fp32_all": FmhaProfile(
        "fp32_all", lambda c: _dtype_is(c, {"fp32"}) and _no_skip_or_logits(c)
    ),
    "fp32_min": FmhaProfile(
        "fp32_min",
        lambda c: (
            _dtype_is(c, {"fp32"})
            and _mode_is(c, {"batch"})
            and c["signature"]["hdim_q"] in {48, 128}
            and c["signature"]["hdim_v"] in {48, 128}
            and canonical_bias(c["signature"]["bias"]) == "no"
            and not c["signature"]["lse"]
            and not c["signature"]["dropout"]
            and canonical_qscale(c["signature"]["qscale"]) == "no"
        ),
    ),
    "fp8_test": FmhaProfile(
        "fp8_test",
        lambda c: (
            _dtype_is(c, {"fp8bf16", "fp8fp32"})
            and c["signature"]["hdim_q"] in {128, 192}
            and _common_row_major_filter(c)
        ),
    ),
    "all": FmhaProfile("all", lambda _: True),
}


def normalize_profile(
    profile: Optional[str] = None, receipt: Optional[str] = None
) -> str:
    if profile:
        return PROFILE_ALIASES.get(str(profile), str(profile))
    if receipt is not None:
        return PROFILE_ALIASES.get(str(receipt), str(receipt))
    return "ck_default"


def get_profile(
    profile: Optional[str] = None, receipt: Optional[str] = None
) -> FmhaProfile:
    normalized = normalize_profile(profile=profile, receipt=receipt)
    if normalized not in PROFILES:
        raise KeyError(f"Unknown FMHA profile: {normalized}")
    return PROFILES[normalized]


def profile_allows(
    config: dict, profile: Optional[str] = None, receipt: Optional[str] = None
) -> bool:
    return get_profile(profile=profile, receipt=receipt).allows(config)


# =============================================================================
# 8. Validation helpers (for unified_fmha_codegen)
# =============================================================================

_DEFAULTS = {
    "tile": [128, 64, 32, 128, 32, 128],
    "wave": [2, 2, 1, 2, 2, 1, 1, 1, 1],
    "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
    "padding": [True, True, True, True],
    "block_per_cu": 1,
    "num_wave_groups": 1,
    "selection_rank": 0,
}

_GLOBAL_RULES = {
    "hdim_192_128_no_bias_dropout": True,
    "logits_requires_no_bias": True,
    "group_mode_requires_padding": True,
    "hdim_divisible_by": 8,
}


def load_arch_specs() -> dict:
    """Return synthetic arch_specs dict compatible with unified_fmha_codegen."""
    return {
        "architectures": ARCH_METADATA,
        "defaults": _DEFAULTS,
        "global_rules": _GLOBAL_RULES,
        "splitkv_combine": {"combine_bn1": 32},
    }


# =============================================================================
# 9. Config validation (for unified_fmha_codegen)
# =============================================================================


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_config(
    config: dict, arch_specs: Optional[dict] = None
) -> "ValidationResult":
    """Validate an FMHA kernel config against all rules."""
    arch_specs = arch_specs or load_arch_specs()
    result = ValidationResult()

    sig = config["signature"]
    alg = config["algorithm"]
    arch = config["arch"]

    architectures = arch_specs.get("architectures", ARCH_METADATA)
    if arch not in architectures:
        result.add_error(f"Unsupported FMHA target architecture: {arch}")
        return result

    arch_info = architectures[arch]
    global_rules = arch_specs.get("global_rules", _GLOBAL_RULES)
    dtype = sig["data_type"]
    family = sig["family"]
    pipeline = alg["pipeline"]
    canonical_mask(sig["mask"])
    bias = canonical_bias(sig["bias"])

    # Family validation
    supported_families = {
        "fwd",
        "fwd_pagedkv",
        "fwd_splitkv",
        "fwd_splitkv_combine",
        "fwd_appendkv",
        "batch_prefill",
        "bwd_dot_do_o",
        "bwd_dq_dk_dv",
        "bwd_convert_dq",
    }
    if family not in supported_families:
        result.add_error(f"Unsupported FMHA family: {family}")

    # Dtype validation
    supported_dtypes = set(arch_info["supported_dtypes"])
    if dtype not in supported_dtypes:
        result.add_error(f"dtype {dtype} is not supported on {arch}")

    if family.startswith("bwd") and dtype not in BWD_DTYPE_MAP:
        result.add_error(
            f"Backward family {family} only supports {sorted(BWD_DTYPE_MAP)}"
        )

    if (
        family.startswith("fwd")
        and not family.startswith("fwd_append")
        and dtype not in FWD_DTYPE_MAP
    ):
        result.add_error(f"Forward family {family} does not recognize dtype {dtype}")

    # Pipeline validation
    if (
        family != "fwd_splitkv_combine"
        and pipeline not in arch_info["supported_pipelines"]
    ):
        result.add_error(f"pipeline {pipeline} is not supported on {arch}")

    if pipeline in {"v3", "qr_async_trload_v3"} and not arch_info.get(
        "supports_v3", False
    ):
        result.add_warning(f"v3 pipeline on {arch} requires supports_v3 in arch specs")

    if pipeline == "qr_async_trload" and not arch_info.get("supports_trload", False):
        result.add_error("qr_async_trload requires a trload-capable architecture")

    # Global rules
    hdim_q = sig["hdim_q"]
    hdim_v = sig["hdim_v"]
    divisor = global_rules.get("hdim_divisible_by", 8)
    if hdim_q % divisor != 0 or hdim_v % divisor != 0:
        result.add_error(f"Head dimensions must be multiples of {divisor}")

    if global_rules.get("hdim_192_128_no_bias_dropout"):
        if (
            hdim_q == 192
            and hdim_v == 128
            and (bias != "no" or sig.get("dropout", False))
        ):
            result.add_warning(
                "hdim (192,128) with bias/dropout has limited tile support"
            )

    if global_rules.get("logits_requires_no_bias"):
        if bias != "no" and sig.get("logits", False):
            result.add_error("logits_soft_cap cannot be combined with bias")

    if pipeline in {"qr_async_trload", "v3", "qr_async_trload_v3"} and (
        hdim_q != hdim_v or hdim_q not in {64, 128}
    ):
        result.add_error(f"{pipeline} only supports symmetric head dims 64 or 128")

    # Tile validation
    tile = alg["tile"]
    expected_tile_len = 9 if family == "bwd_dq_dk_dv" else 6
    if len(tile) != expected_tile_len or len(alg["wave"]) != 9 or len(alg["warp"]) != 9:
        result.add_error(
            f"tile/wave/warp must have {expected_tile_len}/9/9 elements for {family}"
        )

    # MFMA instruction count check for qr/h256/CDNA
    _1d_families = {"bwd_dot_do_o", "bwd_convert_dq"}
    if (
        pipeline == "qr"
        and hdim_q == 256
        and family not in _1d_families
        and arch_info.get("family", "").startswith("cdna")
        and len(tile) >= 3
        and len(alg["wave"]) >= 2
        and len(alg["warp"]) >= 3
    ):
        wm, wn, wk = alg["warp"][0], alg["warp"][1], alg["warp"][2]
        gm, gn = alg["wave"][0], alg["wave"][1]
        if wm > 0 and wn > 0 and wk > 0 and gm > 0 and gn > 0:
            num_mfma = (tile[0] // wm) * (tile[1] // wn) * (tile[2] // wk) // (gm * gn)
            if num_mfma % 8 != 0:
                result.add_error(
                    f"NumMfmaInsts={num_mfma} must be divisible by 8 for qr/h256/CDNA"
                )

    if alg["block_per_cu"] <= 0 and alg["block_per_cu"] != -1:
        result.add_error("block_per_cu must be positive or -1 (auto)")
    if alg["num_wave_groups"] <= 0:
        result.add_error("num_wave_groups must be positive")

    # --- Family-specific rules ---
    if family == "batch_prefill":
        if sig.get("vlayout", "r") != "r":
            result.add_error("batch_prefill only supports row-major V layout")
        if not sig.get("paged_kv", False):
            result.add_error("batch_prefill requires paged_kv=true")
        ps = sig.get("page_size", 0)
        if ps <= 0 or (ps & (ps - 1)) != 0:
            result.add_error("batch_prefill page_size must be a positive power of two")
        if sig.get("mode", "batch") != "group":
            result.add_error("batch_prefill requires group mode")
        if pipeline != "qr_async":
            result.add_error("batch_prefill currently uses qr_async pipeline")

    if family == "fwd_appendkv":
        if sig.get("mode", "batch") != "batch":
            result.add_error("fwd_appendkv uses batch-mode public API surface")
        if pipeline != "appendkv":
            result.add_error("fwd_appendkv must use appendkv pipeline")
        if sig.get("vlayout", "r") != "r":
            result.add_error("fwd_appendkv currently only supports row-major V")

    if family == "fwd_splitkv_combine":
        if sig.get("mode", "batch") not in {"batch", "group"}:
            result.add_error("fwd_splitkv_combine requires batch or group mode")
        combine_bn1 = arch_specs.get("splitkv_combine", {}).get("combine_bn1", 32)
        if len(tile) > 3 and tile[3] != combine_bn1:
            result.add_error(f"fwd_splitkv_combine requires bn1={combine_bn1}")
        if len(tile) > 3 and (hdim_v < tile[3] or hdim_v % tile[3] != 0):
            result.add_error("fwd_splitkv_combine requires hdim_v divisible by bn1")

    if family == "fwd_pagedkv":
        if pipeline != "qr_pagedkv":
            result.add_error("fwd_pagedkv must use qr_pagedkv pipeline")
        if not sig.get("paged_kv", False):
            result.add_error("fwd_pagedkv requires paged_kv=true")
        if sig.get("vlayout", "r") != "r":
            result.add_error("fwd_pagedkv currently only supports row-major V")

    if family == "fwd_splitkv":
        if pipeline not in {"qr", "qr_nwarp_sshuffle"}:
            result.add_error("fwd_splitkv must use qr or qr_nwarp_sshuffle pipeline")
        if sig.get("vlayout", "r") != "r":
            result.add_error("fwd_splitkv currently only supports row-major V")

    if family == "fwd" and sig.get("vlayout", "r") != "r":
        result.add_warning("dispatcher forward examples currently assume row-major V")

    return result
