# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emission helpers shared by the gfx942 and gfx950 forward attention kernels.

Each helper here was a byte-for-byte copy in both arch modules
(``kernels/gfx942/attention_*.py`` and ``kernels/gfx950/attention_*.py``).
Keeping one copy means a fix lands on both arches at once. Helpers that differ
between the arches, even by one literal, stay in the arch modules.

Mirrored by instance_attention_tiled_common_internal.h in the C++ engine.
"""

from __future__ import annotations

from rocke.helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from rocke.helpers.distribution import make_static_tile_distribution

__all__ = [
    "C32_DIST",
    "mfma_32x32_c_col",
    "mfma_32x32_c_row",
]


# CK-Tile C-accumulator warp distribution for the 32x32x16 MFMA atom. The
# hand-rolled ``mfma_32x32x16_c_row/col`` lane arithmetic (in
# ``helpers/attention.py``) is exactly the ``calculate_x`` of CK Tile's
# ``CWarpDstrEncoding`` (``warp_gemm_attribute_mfma.hpp``): for one wavefront
# the lane splits as ``(m_blk, n) = (lane // 32, lane % 32)`` and the per-lane
# ``<16 x f32>`` accumulator slot ``i`` splits row-major over the two M Y dims
# ``(kCM0PerLane, kCM1PerLane) = (4, 4)``. The distribution's ``calculate_x``
# then yields ``row = (i // 4) * 8 + m_blk * 4 + (i % 4)`` and ``col = n`` --
# the same mapping the scalar helpers produce, but emitted by the tile-
# distribution algebra instead of open-coded ``div``/``mul``/``add``. The C
# layout is dtype-independent, so the f16 atom drives it for both fp16 and
# bf16. The 3D tiled kernels still use the scalar helpers from
# ``helpers/attention.py``.
C32_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16())
)


def mfma_32x32_c_row(b, lane, elem_idx: int):
    """MFMA-local output row for a ``32x32x16`` C element ``elem_idx`` (0..15).

    Drives CK Tile's C-warp ``TileDistribution.calculate_x`` instead of the
    hand-rolled ``row = (i//4)*8 + (lane//32)*4 + (i%4)`` lane arithmetic.
    """
    if not (0 <= elem_idx < 16):
        raise ValueError(f"mfma_32x32x16 elem_idx must be 0..15, got {elem_idx}")
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    y0 = b.const_i32(elem_idx // 4)
    y1 = b.const_i32(elem_idx % 4)
    row, _col = C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]])
    return row


def mfma_32x32_c_col(b, lane, n_tile32: int = 0):
    """MFMA-local output col for ``32x32x16`` C elements in N-tile ``n_tile32``.

    The per-lane column is the N coordinate of CK Tile's C-warp distribution
    (``lane % 32``); ``n_tile32 * 32`` is the 32-column tile base added on top.
    Every per-lane accumulator slot shares this column, so the element index
    does not enter the formula.
    """
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    _row, col = C32_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(0)], ps=[[m_blk, n]]
    )
    if n_tile32 == 0:
        return col
    return b.add(b.const_i32(n_tile32 * 32), col)
