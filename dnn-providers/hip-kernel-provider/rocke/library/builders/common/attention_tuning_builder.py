# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compatibility re-export of dispatcher-owned attention tuning helpers.

Spec construction lives next to its executable tuning candidates in
:mod:`dispatch.attention.tuning_specs`. Harnesses that still import this module
keep working without making dispatch depend on builders.
"""

from __future__ import annotations

from dispatch.attention.tuning_specs import (
    BLOCK_M_PER_WARP,
    NUM_SEGMENTS,
    NUM_WARPS,
    TILE_POLICIES,
    TILE_POLICIES_3D,
    WAVES_PER_EU,
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    build_explicit_attention_2d,
    build_explicit_attention_3d,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
    resolve_tile_policy,
)

__all__ = [
    "BLOCK_M_PER_WARP",
    "NUM_SEGMENTS",
    "NUM_WARPS",
    "TILE_POLICIES",
    "TILE_POLICIES_3D",
    "WAVES_PER_EU",
    "ExplicitAttention2DConfig",
    "ExplicitAttention3DConfig",
    "build_explicit_attention_2d",
    "build_explicit_attention_3d",
    "make_explicit_attention_2d_spec",
    "make_explicit_attention_3d_specs",
    "resolve_tile_policy",
]
