# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1201 (RDNA4, wave32, WMMA) deep-fused conv + maxpool — thin arch shim.

The kernel body lives in :mod:`rocke.instances.common.deep_fused_conv_pool`,
authored once and driven by the resolved :class:`~rocke.core.arch.MmaOp` so the
same code emits both the MFMA (gfx950) and WMMA (gfx1201) paths. This module only
pins the gfx1201 geometry (wave32, 16x16x16 WMMA atom) and re-exports the common
spec/builder under gfx1201 public names (kernel name
``rocke_gfx1201_deep_fused_conv_pool``).

The MFMA-32x32 intra-lane maxpool fast path in the common body is geometry-gated
and naturally disables itself here (warp_tile 16x16), so WMMA takes the
layout-agnostic cshuffle-LDS gather + maxpool path.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..common.deep_fused_conv_pool import (
    DeepFusedConvPoolSpec,
    FusedConvPoolProblem,
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    deep_fused_conv_pool_signature,
    is_valid_spec,
    make_deep_fused_conv_pool_spec as _make_common_spec,
)

__all__ = [
    "FusedConvPoolProblem",
    "Gfx1201DeepFusedConvPoolSpec",
    "make_deep_fused_conv_pool_spec",
    "is_valid_spec",
    "deep_fused_conv_pool_signature",
    "deep_fused_conv_pool_grid",
    "build_deep_fused_conv_pool",
]

_GFX1201_NAME = "rocke_gfx1201_deep_fused_conv_pool"


@dataclass(frozen=True)
class Gfx1201DeepFusedConvPoolSpec(DeepFusedConvPoolSpec):
    """gfx1201 deep-fusion spec: the common spec pinned to the WMMA geometry.

    Identical fields to :class:`DeepFusedConvPoolSpec`; the gfx1201 defaults are
    ``wave_size=32`` and ``warp_tile 16x16x16`` (the RDNA4 WMMA atom) plus the
    gfx1201 kernel name.
    """

    name: str = _GFX1201_NAME
    wave_size: int = 32
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 16


def make_deep_fused_conv_pool_spec(**kwargs) -> Gfx1201DeepFusedConvPoolSpec:
    """Build a gfx1201 deep-fusion spec (wave32 WMMA 16x16x16)."""
    base = _make_common_spec(
        name=_GFX1201_NAME,
        wave_size=32,
        warp_tile_m=16,
        warp_tile_n=16,
        **kwargs,
    )
    return Gfx1201DeepFusedConvPoolSpec(
        **{f.name: getattr(base, f.name) for f in fields(DeepFusedConvPoolSpec)}
    )
