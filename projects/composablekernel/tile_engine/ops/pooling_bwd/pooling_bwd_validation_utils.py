#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validation utilities for pooling-backward tile_engine configurations.

The pooling-backward kernel is a flat 1D scatter (`din[indices[i]] = dout[i]`
or atomic-add when windows overlap), so its tile parameters are far simpler
than the forward pooling kernel: only `block_size` and `vector_size`.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datatype helpers
# ---------------------------------------------------------------------------

ELEMENT_SIZE_MAP = {
    "fp8": 1,
    "bf8": 1,
    "int8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
}

DTYPE_STRING_MAP = {
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
    "fp32": "float",
    "fp64": "double",
}

SUPPORTED_DATATYPES = list(DTYPE_STRING_MAP.keys())
SUPPORTED_POOLING_DIMS = ("2d", "3d")


def element_size(datatype: str) -> int:
    datatype = datatype.lower()
    if datatype not in ELEMENT_SIZE_MAP:
        raise ValueError(
            f"Unsupported data type: '{datatype}'. "
            f"Supported: {list(ELEMENT_SIZE_MAP.keys())}"
        )
    return ELEMENT_SIZE_MAP[datatype]


def get_dtype_string(datatype: str) -> str:
    return DTYPE_STRING_MAP.get(datatype, "float")


# ---------------------------------------------------------------------------
# Tile-config validators
# ---------------------------------------------------------------------------


def validate_block_size(block_size: int) -> Tuple[bool, str]:
    if block_size <= 0:
        return False, f"block_size ({block_size}) must be > 0"
    if block_size > 1024:
        return False, f"block_size ({block_size}) exceeds 1024"
    if block_size & (block_size - 1) != 0:
        return False, f"block_size ({block_size}) must be a power of two"
    return True, ""


def validate_vector_size(vector_size: int) -> Tuple[bool, str]:
    if vector_size not in (1, 2, 4):
        return False, f"vector_size ({vector_size}) must be 1, 2, or 4"
    return True, ""


def is_tile_config_valid(
    block_size: int,
    vector_size: int,
    datatype: str,
    fast_mode: bool = False,
) -> bool:
    ok, err = validate_block_size(block_size)
    if not ok:
        logger.debug(f"Block size check failed: {err}")
        return False

    ok, err = validate_vector_size(vector_size)
    if not ok:
        logger.debug(f"Vector size check failed: {err}")
        return False

    if fast_mode:
        return True

    return True


def is_trait_combination_valid(pooling_dim: str, has_overlap: bool) -> bool:
    if pooling_dim not in SUPPORTED_POOLING_DIMS:
        logger.debug(f"Invalid pooling dim: '{pooling_dim}'")
        return False
    _ = has_overlap
    return True


def is_datatype_supported(datatype: str) -> bool:
    return datatype.lower() in ELEMENT_SIZE_MAP
