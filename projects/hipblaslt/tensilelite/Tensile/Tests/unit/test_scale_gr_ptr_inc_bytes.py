# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for scaleGRPtrIncBytes MX scale format gating.

Advance (emitScaleGRPtrUpdate) and MultiDU tail rewind
(_emitMultiDUTailSrdRewind) must share the same per-format byte step:

  * HostPreSwizzle / InMemorySwizzle: swizzle granule
    (lrSubtileSize * lrGlobalSubtileGrid[1])
  * NoSwizzle: canonical K-step (scaleDepthU * bpe)

Using the HostPreSwizzle granule under NoSwizzle over-rewinds / over-advances
scale SRDs (the MultiDU bug class fixed in this PR).
"""

from types import SimpleNamespace

import pytest

from Tensile.Components.Subtile.SubtileScaleEmit import scaleGRPtrIncBytes

pytestmark = pytest.mark.unit


def _ti(*, lr_subtile_size=256, lr_grid_k=2, scale_depth_u=8, bpe=1):
    """Minimal tile-info stand-in for scaleGRPtrIncBytes."""
    return SimpleNamespace(
        lrSubtileSize=lr_subtile_size,
        lrGlobalSubtileGrid=[4, lr_grid_k],
        scaleDepthU=scale_depth_u,
        bpe=bpe,
    )


@pytest.mark.parametrize("fmt", ["HostPreSwizzle", "InMemorySwizzle"])
def test_scale_gr_ptr_inc_bytes_swizzled_uses_granule(fmt):
    """Swizzled layouts advance/rewind by the HostPreSwizzle LDS granule."""
    ti = _ti(lr_subtile_size=256, lr_grid_k=2)
    assert scaleGRPtrIncBytes(ti, {"MXScaleFormat": fmt}) == 512


def test_scale_gr_ptr_inc_bytes_noswizzle_uses_canonical_k_step():
    """NoSwizzle must not use the HostPreSwizzle granule (scaleDepthU * bpe)."""
    ti = _ti(scale_depth_u=8, bpe=1)
    assert scaleGRPtrIncBytes(ti, {"MXScaleFormat": "NoSwizzle"}) == 8


def test_scale_gr_ptr_inc_bytes_default_is_noswizzle():
    """Missing MXScaleFormat defaults to NoSwizzle (canonical K-step)."""
    ti = _ti(scale_depth_u=8, bpe=1)
    assert scaleGRPtrIncBytes(ti, {}) == 8


def test_scale_gr_ptr_inc_bytes_formats_diverge_for_typical_fp4():
    """Typical FP4 values: swizzled granule != NoSwizzle K-step (bug class)."""
    ti = _ti(lr_subtile_size=256, lr_grid_k=2, scale_depth_u=8, bpe=1)
    swizzled = scaleGRPtrIncBytes(ti, {"MXScaleFormat": "HostPreSwizzle"})
    noswizzle = scaleGRPtrIncBytes(ti, {"MXScaleFormat": "NoSwizzle"})
    assert swizzled == 512
    assert noswizzle == 8
    assert swizzled != noswizzle
