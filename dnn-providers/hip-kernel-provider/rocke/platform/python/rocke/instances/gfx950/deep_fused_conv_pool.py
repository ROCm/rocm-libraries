# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 (CDNA, wave64, MFMA) deep-fused conv + maxpool — thin arch shim.

The kernel body now lives in :mod:`rocke.instances.common.deep_fused_conv_pool`,
authored once and driven by the resolved :class:`~rocke.core.arch.MmaOp` so the
same code emits both the MFMA (gfx950) and WMMA (gfx1201) paths. This module only
pins the gfx950 geometry (wave64, 32x32x16 MFMA atom) and re-exports the common
spec/builder under the historical public names so existing harnesses, sweeps, and
manifests keep working byte-for-byte (kernel name ``rocke_gfx950_deep_fused_conv_pool``).
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
    "Gfx950DeepFusedConvPoolSpec",
    "make_deep_fused_conv_pool_spec",
    "is_valid_spec",
    "deep_fused_conv_pool_signature",
    "deep_fused_conv_pool_grid",
    "build_deep_fused_conv_pool",
]

_GFX950_NAME = "rocke_gfx950_deep_fused_conv_pool"


@dataclass(frozen=True)
class Gfx950DeepFusedConvPoolSpec(DeepFusedConvPoolSpec):
    """gfx950 deep-fusion spec: the common spec pinned to the gfx950 kernel name.

    Identical fields to :class:`DeepFusedConvPoolSpec`; only the ``name`` default
    differs (preserving the historical gfx950 kernel name / ABI). The default
    geometry (``wave_size=64``, ``warp_tile 32x32x16``) is already the common
    default, so direct construction ``Gfx950DeepFusedConvPoolSpec(problem=...)``
    yields the same MFMA kernel as before.
    """

    name: str = _GFX950_NAME


def make_deep_fused_conv_pool_spec(**kwargs) -> Gfx950DeepFusedConvPoolSpec:
    """Build a gfx950 deep-fusion spec, auto-deriving ``tile_m`` (wave64 MFMA)."""
    base = _make_common_spec(
        name=_GFX950_NAME,
        wave_size=64,
        warp_tile_m=32,
        warp_tile_n=32,
        **kwargs,
    )
    return Gfx950DeepFusedConvPoolSpec(
        **{f.name: getattr(base, f.name) for f in fields(DeepFusedConvPoolSpec)}
    )
