# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Output-plane band order of a conv's M tiles (``m_band``).

The M tiles of an implicit-GEMM conv cover the output pixels in (n, z, y, x)
order, so in launch order -- and in the XCD-contiguous order of
``xcd_remap``, which hands each XCD one contiguous run of tiles -- the tiles
an XCD runs at one time cover a strip about one plane (Ho*Wo pixels) wide,
whose input rows (the filter's z/y taps) span 3+ planes: more than the
XCD's L2 holds on large planes, so most input is fetched again from the
fabric (fwd s1/s2 of ScaFFold: 3x the input).

The band order walks a band of ``m_band`` tiles of one plane through every
plane before the next band, so the concurrent tiles cover a compact z-y block
whose input stays in L2. The plane need not be a whole number of tiles: the
tiles are cut into rows of ``R = round(plane / tile_m)`` tiles (a row drifts
by ``R*tile_m - plane`` pixels against the plane, i.e. a few pixels), each row
into bands of ``m_band`` tiles and a narrower last band of ``R % m_band``.
The tiles past the last whole row keep their index, so the map is a
bijection of ``[0, num_m_tiles)``.
"""

from __future__ import annotations


def emit_m_band_order(b, mt, *, num_m_tiles, plane, tile_m, m_band: int):
    """The M tile ``mt`` (logical, i32) mapped to its band-order tile.

    ``tile_m`` (pixels per tile) is a build-time int or an i32 value (e.g. a
    split-K slice's runtime width)."""
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_mb = b.const_i32(m_band)
    if isinstance(tile_m, int):
        c_tm = b.const_i32(tile_m)
        R = b.smax(b.div(b.add(plane, b.const_i32(tile_m // 2)), c_tm), c1)
    else:
        c_tm = b.smax(tile_m, c1)
        R = b.smax(b.div(b.add(plane, b.div(c_tm, b.const_i32(2))), c_tm), c1)
    rows = b.div(num_m_tiles, R)
    full = b.mul(rows, R)
    nb = b.div(R, c_mb)  # whole bands per row
    w_last = b.smax(b.mod(R, c_mb), c1)
    span = b.mul(rows, c_mb)
    in_whole = b.mul(nb, span)
    # whole bands: L = (band * rows + row) * mb + i
    band = b.div(mt, b.smax(span, c1))
    r = b.mod(mt, b.smax(span, c1))
    t_whole = b.add(b.mul(b.div(r, c_mb), R), b.add(b.mul(band, c_mb), b.mod(r, c_mb)))
    # the narrow last band: L - in_whole = row * w_last + i
    l2 = b.sub(mt, in_whole)
    t_last = b.add(b.mul(b.div(l2, w_last), R), b.add(b.mul(nb, c_mb), b.mod(l2, w_last)))
    t = b.select(b.cmp_lt(mt, in_whole), t_whole, t_last)
    ok = b.land(b.cmp_lt(mt, full), b.cmp_lt(c0, rows))
    return b.select(ok, t, mt)


def python_m_band_order(mt: int, *, num_m_tiles: int, plane: int, tile_m: int, m_band: int) -> int:
    """Pure-Python model of :func:`emit_m_band_order`."""
    R = max((plane + tile_m // 2) // tile_m, 1)
    rows = num_m_tiles // R
    if rows <= 0 or mt >= rows * R:
        return mt
    nb, w_last = R // m_band, max(R % m_band, 1)
    span = rows * m_band
    if mt < nb * span:
        band, r = divmod(mt, span)
        return (r // m_band) * R + band * m_band + r % m_band
    l2 = mt - nb * span
    return (l2 // w_last) * R + nb * m_band + l2 % w_last
