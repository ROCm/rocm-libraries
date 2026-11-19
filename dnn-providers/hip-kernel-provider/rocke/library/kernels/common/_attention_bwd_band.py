# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Inverse band, tile classifier and mask policies of the attention backward.

Family glue for the backward kernels (Python-only, no C++ twin).  The forward
band (``rocke.helpers.attention_band``) is imported, never edited: the keep
rule is the frozen one,

    keep(q, k) = q < len_q and k < len_kv
                 and (right < 0 or k <= q + off + right)
                 and (left  < 0 or k >= q + off - left)

with ``off = 0`` (top-left) or ``len_kv - len_q`` (bottom-right).  The
backward adds ``q < len_q`` to the forward cell keep (which has no row term)
and the transposed (key-major) views below.  Every function only calls
integer methods of the builder, so the same code is evaluated with a
duck-typed integer builder on the host by the exhaustive tests.

* ``q_tile_range``: the query tiles a key tile ``[k0, k0 + k_n0)`` has to
  visit.  Rows ``q`` attend some key of the tile iff
  ``q_lo <= q <= q_hi`` with ``q_lo = max(0, k0 - off - right)`` (right
  bounded) and ``q_hi = min(len_q - 1, k1 - off + left)`` (left bounded),
  ``k1 = min(k0 + k_n0 - 1, len_kv - 1)``.  The row range is exact, so the
  only slack is tile quantisation at the two ends.  A tile past ``len_kv``,
  ``len_q == 0`` or a band that misses every row gives ``n_qt == 0``.
* ``tile_inside_band`` / ``is_edge_tile`` / ``tile_class``: a (q tile, key
  tile) pair is *interior* when every row is ``< len_q``, every key is
  ``< len_kv`` and every cell is inside the band (checked at the two corners
  that bound the band); interior tiles may skip the per-cell selects.  The
  classification is exact: a tile is interior iff all its cells are kept.
* ``interior_q_range``: the interior q tiles of a key tile, a contiguous
  sub-range ``[i0, i1)`` of the visited range (rows in range bound it above,
  the right band edge below, the left edge above), so a q loop splits into
  edge / interior / edge runs without per-step classification.
* ``MaskPolicy.row_limits`` / ``keep_in``: per-row key limits ``[lo, hi]``
  (``hi = -1`` for a row that keeps nothing) and the one-compare cell keep
  ``min(max(k, lo), hi) == k``.
* ``dkv_store_rule``: which dK / dV rows of a key tile are stored, and which
  carry values (the empty-band zero-fill rule).
* Mask policies (:class:`NoMaskKTail`, :class:`RuntimeBand`) bundle these for
  one CTA; ``static_kv_tile_plan`` evaluates the same formulas on the host
  for compile-time (dense) tile classification.

All bounds and lengths must stay below ``2**30`` (i32-safe arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import IRBuilder, Value
from rocke.helpers.attention_band import (
    I32_FAR,
    AttnRuntimeBounds,
    BandEmitter,
    RowBand,
)

from ._attention_bwd_addr import as_i32

__all__ = [
    "TILE_EDGE",
    "TILE_INTERIOR",
    "HostInt",
    "MaskPolicy",
    "NoMaskKTail",
    "RuntimeBand",
    "StaticKvTilePlan",
    "dkv_store_rule",
    "interior_q_range",
    "is_edge_tile",
    "q_tile_range",
    "static_kv_tile_plan",
    "tile_class",
    "tile_inside_band",
]

IntOrValue = int | Value

TILE_INTERIOR = 0
TILE_EDGE = 1


def _bound_or_none(band: BandEmitter, which: str):
    return band.right if which == "right" else band.left


def q_tile_range(
    b: IRBuilder,
    band: BandEmitter,
    k0: Value,
    *,
    k_n0: int,
    k_m0: int,
    uniform: bool = True,
) -> tuple[Value, Value]:
    """``(qt_start, n_qt)``: query tiles of ``k_m0`` rows key tile ``k0`` visits.

    ``k0`` is the first key of the tile (i32, CTA-uniform).  With ``uniform``
    both results go through ``readfirstlane`` so the q-loop trip count lives
    in an SGPR.  ``n_qt`` is 0 for an empty band, a tile past ``len_kv`` or
    ``len_q == 0``.
    """
    if k_n0 < 1 or k_m0 < 1:
        raise ValueError("tile sizes must be positive")
    zero = b.const_i32(0)
    cm = b.const_i32(k_m0)
    k_last = b.smin(b.add(k0, b.const_i32(k_n0 - 1)), band.sk_minus1)
    sq_m1 = b.sub(band.sq, b.const_i32(1))
    off = band.off
    k_first_c = k0 if off is None else b.sub(k0, off)
    k_last_c = k_last if off is None else b.sub(k_last, off)
    right = _bound_or_none(band, "right")
    left = _bound_or_none(band, "left")
    if right is None:
        q_lo = zero
    else:
        r = as_i32(b, right)
        q_lo = b.sub(k_first_c, r)
        if not isinstance(right, int):
            q_lo = b.select(b.cmp_lt(r, zero), zero, q_lo)
        q_lo = b.smax(q_lo, zero)
    if left is None:
        q_hi = sq_m1
    else:
        lft = as_i32(b, left)
        q_hi = b.add(k_last_c, lft)
        if not isinstance(left, int):
            q_hi = b.select(b.cmp_lt(lft, zero), sq_m1, q_hi)
        q_hi = b.smin(q_hi, sq_m1)
    # Empty when the tile has no live key or the band misses every row.
    dead = b.lor(b.cmp_lt(k_last, k0), b.cmp_lt(q_hi, q_lo))
    q_hi = b.select(dead, b.const_i32(-1), q_hi)
    qt_start = b.div(q_lo, cm)
    qt_stop = b.div(b.add(b.smax(q_hi, b.const_i32(-1)), cm), cm)
    n_qt = b.smax(b.sub(qt_stop, qt_start), zero)
    if uniform:
        qt_start = b.readfirstlane(qt_start)
        n_qt = b.readfirstlane(n_qt)
    return qt_start, n_qt


def tile_inside_band(
    b: IRBuilder,
    band: BandEmitter,
    q_first: Value,
    q_last: Value,
    k_first: Value,
    k_last: Value,
) -> Value:
    """i1: every cell of ``[q_first, q_last] x [k_first, k_last]`` is in the band.

    Length terms are not included (see :func:`is_edge_tile`).  The right edge
    binds at ``(q_first, k_last)``, the left edge at ``(q_last, k_first)``.
    """
    zero = b.const_i32(0)
    inside = None
    off = band.off
    right = _bound_or_none(band, "right")
    left = _bound_or_none(band, "left")
    if right is not None:
        r = as_i32(b, right)
        centre = q_first if off is None else b.add(q_first, off)
        ok = b.cmp_le(k_last, b.add(centre, r))
        if not isinstance(right, int):
            ok = b.lor(b.cmp_lt(r, zero), ok)
        inside = ok
    if left is not None:
        lft = as_i32(b, left)
        centre = q_last if off is None else b.add(q_last, off)
        ok = b.cmp_ge(k_first, b.sub(centre, lft))
        if not isinstance(left, int):
            ok = b.lor(b.cmp_lt(lft, zero), ok)
        inside = ok if inside is None else b.land(inside, ok)
    if inside is None:
        inside = b.cmp_le(zero, zero)  # no band limit: always inside
    return inside


def is_edge_tile(
    b: IRBuilder,
    band: BandEmitter,
    qt: Value,
    k0: Value,
    *,
    k_m0: int,
    k_n0: int,
    masked: bool = True,
) -> Value:
    """i1: tile ``(qt, k0)`` needs the per-cell selects.

    Interior means: rows ``qt*k_m0 .. qt*k_m0 + k_m0 - 1`` all ``< len_q``,
    keys ``k0 .. k0 + k_n0 - 1`` all ``< len_kv`` and (``masked``) every
    cell inside the band.
    """
    q_first = b.mul(qt, b.const_i32(k_m0))
    q_last = b.add(q_first, b.const_i32(k_m0 - 1))
    k_last = b.add(k0, b.const_i32(k_n0 - 1))
    interior = b.land(b.cmp_lt(q_last, band.sq), b.cmp_lt(k_last, band.sk))
    if masked:
        interior = b.land(
            interior, tile_inside_band(b, band, q_first, q_last, k0, k_last)
        )
    return b.lnot(interior)


def tile_class(
    b: IRBuilder,
    band: BandEmitter,
    qt: Value,
    k0: Value,
    *,
    k_m0: int,
    k_n0: int,
    masked: bool = True,
    uniform: bool = True,
) -> Value:
    """i32 CTA-uniform tile class: ``TILE_INTERIOR`` (0) or ``TILE_EDGE`` (1)."""
    edge = is_edge_tile(b, band, qt, k0, k_m0=k_m0, k_n0=k_n0, masked=masked)
    cls = b.select(edge, b.const_i32(TILE_EDGE), b.const_i32(TILE_INTERIOR))
    return b.readfirstlane(cls) if uniform else cls


def _floor_div(b, a: Value, m: int) -> Value:
    """``floor(a / m)`` for a signed i32 ``a`` and a positive constant ``m``."""
    cm = b.const_i32(m)
    neg = b.sub(b.const_i32(0), b.div(b.sub(b.const_i32(m - 1), a), cm))
    return b.select(b.cmp_lt(a, b.const_i32(0)), neg, b.div(a, cm))


def interior_q_range(
    b: IRBuilder,
    band: BandEmitter,
    k0: Value,
    qt_start: Value,
    n_qt: Value,
    *,
    k_m0: int,
    k_n0: int,
    masked: bool = True,
    uniform: bool = True,
) -> tuple[Value, Value]:
    """``(i0, i1)``: the interior q tiles of key tile ``k0`` are ``[i0, i1)``.

    ``qt_start <= i0 <= i1 <= qt_start + n_qt``; every q tile of the visited
    range outside ``[i0, i1)`` is an edge tile (:func:`is_edge_tile`), every
    tile inside is interior.  The interior set is contiguous: rows in range
    bound ``qt`` above (``qt < floor(len_q / k_m0)``), the right edge bounds
    it below and the left edge above; a key tile reaching past ``len_kv`` has
    no interior tile.
    """
    zero = b.const_i32(0)
    qt_end = b.add(qt_start, n_qt)
    k_last = b.add(k0, b.const_i32(k_n0 - 1))
    lo = qt_start
    hi = b.smin(qt_end, b.div(band.sq, b.const_i32(k_m0)))
    off = band.off
    right = _bound_or_none(band, "right") if masked else None
    left = _bound_or_none(band, "left") if masked else None
    if right is not None:
        # q_first >= k_last - off - right  ->  qt >= ceil((...) / k_m0)
        r = as_i32(b, right)
        need = b.sub(k_last, r) if off is None else b.sub(b.sub(k_last, off), r)
        first = b.sub(zero, _floor_div(b, b.sub(zero, need), k_m0))
        if not isinstance(right, int):
            first = b.select(b.cmp_lt(r, zero), lo, first)
        lo = b.smax(lo, first)
    if left is not None:
        # q_last <= k0 - off + left  ->  qt <= floor((... - k_m0 + 1) / k_m0)
        lft = as_i32(b, left)
        top = b.add(k0, lft) if off is None else b.add(b.sub(k0, off), lft)
        last = _floor_div(b, b.sub(top, b.const_i32(k_m0 - 1)), k_m0)
        stop = b.add(last, b.const_i32(1))
        if not isinstance(left, int):
            stop = b.select(b.cmp_lt(lft, zero), hi, stop)
        hi = b.smin(hi, stop)
    hi = b.select(b.cmp_lt(k_last, band.sk), hi, lo)  # tile past len_kv: none
    i0 = b.smin(lo, qt_end)
    i1 = b.smax(b.smin(hi, qt_end), i0)
    if uniform:
        i0 = b.readfirstlane(i0)
        i1 = b.readfirstlane(i1)
    return i0, i1


def dkv_store_rule(
    b: IRBuilder,
    k: Value,
    len_kv: Value,
    *,
    seq_mode: str,
    s_kv_max: IntOrValue | None = None,
) -> tuple[Value, Value]:
    """``(store, live)`` for dK / dV row ``k`` in the direct epilogue.

    * batched: rows ``k < s_kv_max`` are stored; rows ``k >= len_kv`` store
      exact zeros (``live`` false) - also when the CTA's band is empty or the
      whole tile is past ``len_kv``.
    * THD: only rows ``k < len_kv`` are stored; rows outside the segment are
      never touched.

    When the band of a key tile is empty (``n_qt == 0``) the accumulators stay
    zero, so the stored rows are exact zeros as well.
    """
    live = b.cmp_lt(k, len_kv)
    if seq_mode == "batched":
        if s_kv_max is None:
            raise ValueError("batched dK/dV stores need s_kv_max")
        return b.cmp_lt(k, as_i32(b, s_kv_max)), live
    if seq_mode == "thd":
        return live, live
    raise ValueError(f"seq_mode must be 'batched' or 'thd' (got {seq_mode!r})")


# ---------------------------------------------------------------------------
# Mask policies (one instance per CTA)
# ---------------------------------------------------------------------------


class MaskPolicy:
    """Common interface of the backward mask policies.

    ``q_tile_range(b, k0)`` -> ``(qt_start, n_qt)``; ``tile_class(b, qt, k0)``
    -> i32 ``TILE_INTERIOR`` / ``TILE_EDGE``; ``row(b, q)`` -> per-row band
    facts; ``cell_keep(b, q, k)`` / ``keep(b, row, k)`` -> i1 per cell
    (edge tiles only).
    """

    masked = True

    def __init__(self, b: IRBuilder, band: BandEmitter, *, k_m0: int, k_n0: int):
        if k_m0 < 1 or k_n0 < 1:
            raise ValueError("tile sizes must be positive")
        self.band = band
        self.k_m0 = k_m0
        self.k_n0 = k_n0

    def _check(self, k_n0, k_m0) -> None:
        if (k_n0 is not None and k_n0 != self.k_n0) or (
            k_m0 is not None and k_m0 != self.k_m0
        ):
            raise ValueError("tile sizes differ from the policy's")

    def q_tile_range(
        self,
        b: IRBuilder,
        k0: Value,
        k_n0: int | None = None,
        k_m0: int | None = None,
    ) -> tuple[Value, Value]:
        self._check(k_n0, k_m0)
        return q_tile_range(b, self.band, k0, k_n0=self.k_n0, k_m0=self.k_m0)

    def tile_class(self, b: IRBuilder, qt: Value, k0: Value) -> Value:
        return tile_class(
            b, self.band, qt, k0, k_m0=self.k_m0, k_n0=self.k_n0, masked=self.masked
        )

    def interior_q_range(
        self, b: IRBuilder, k0: Value, qt_start: Value, n_qt: Value
    ) -> tuple[Value, Value]:
        """``(i0, i1)``: interior q tiles ``[i0, i1)`` of key tile ``k0``."""
        return interior_q_range(
            b, self.band, k0, qt_start, n_qt, k_m0=self.k_m0, k_n0=self.k_n0,
            masked=self.masked,
        )  # fmt: skip

    def row(self, b: IRBuilder, q: Value) -> RowBand:
        return self.band.row_band(q)

    def keep(self, b: IRBuilder, row: RowBand, k: Value) -> Value:
        """Cell keep with the backward row term ``q < len_q``."""
        cell = self.band.cell_keep(row, k, self.band.col_in(k))
        return b.land(row.row_in, cell)

    def cell_keep(self, b: IRBuilder, q: Value, k: Value) -> Value:
        return self.keep(b, self.row(b, q), k)

    # -- per-row key limits (one compare per cell) ------------------------
    def _row_lo_hi(self, b: IRBuilder, q: Value) -> tuple[Value, Value]:
        """Unclamped-validity ``(lo, hi)`` key limits of row ``q`` (``q >= 0``)."""
        if self.band.left is not None or self.band.right is not None:
            raise NotImplementedError("a banded policy defines its own row limits")
        return b.const_i32(0), self.band.sk_minus1

    def row_limits(
        self, b: IRBuilder, q: Value, ok: Value | None = None
    ) -> tuple[Value, Value]:
        """``(lo, hi)``: row ``q`` keeps exactly the keys ``lo <= k <= hi``.

        ``hi = -1`` for a row that keeps nothing (``q >= len_q``, an empty
        band row, or ``ok`` false), so :meth:`keep_in` rejects every key
        ``k >= 0``. Equals :meth:`cell_keep` for ``q >= 0`` and ``k >= 0``.
        """
        lo, hi = self._row_lo_hi(b, q)
        valid = b.land(b.cmp_lt(q, self.band.sq), b.cmp_le(lo, hi))
        if ok is not None:
            valid = b.land(valid, ok)
        return lo, b.select(valid, hi, b.const_i32(-1))

    @staticmethod
    def keep_in(b: IRBuilder, limits: tuple[Value, Value], k: Value) -> Value:
        """i1 ``lo <= k <= hi`` as one compare: ``min(max(k, lo), hi) == k``."""
        lo, hi = limits
        return b.cmp_eq(b.smin(b.smax(k, lo), hi), k)


class RuntimeBand(MaskPolicy):
    """Runtime two-sided band (causal TL/BR, sliding windows) over lengths.

    The per-cell keep uses two CTA-uniform offsets formed once:
    ``hi_off = right < 0 ? FAR : off + right`` and
    ``lo_off = left < 0 ? -FAR : off - left`` (``FAR = 2**30``), so a row's
    limits are ``q + hi_off`` / ``q + lo_off`` with no per-row bound select.
    For rows ``q >= 0`` this equals the frozen keep rule: an unbounded side
    gives a limit beyond every key (keys and rows are ``< 2**30``).
    """

    def __init__(
        self, b: IRBuilder, bounds: AttnRuntimeBounds, *, k_m0: int, k_n0: int
    ) -> None:
        super().__init__(b, BandEmitter(b, bounds), k_m0=k_m0, k_n0=k_n0)
        band = self.band
        zero = b.const_i32(0)
        self.hi_off = self._side_offset(b, band.right, band.off, +1, zero)
        self.lo_off = self._side_offset(b, band.left, band.off, -1, zero)

    @staticmethod
    def _side_offset(b, bound, off, sign, zero):
        """``off + sign * bound`` (``sign * FAR`` when unbounded), or None."""
        if bound is None:
            return None
        far = b.const_i32(sign * I32_FAR)
        if isinstance(bound, int):
            v = b.const_i32(sign * bound)
            return v if off is None else b.add(off, v)
        v = bound if sign > 0 else b.sub(zero, bound)
        if off is not None:
            v = b.add(off, v)
        return b.select(b.cmp_lt(bound, zero), far, v)

    def cell_keep(self, b: IRBuilder, q: Value, k: Value) -> Value:
        """Cell keep with the backward row term ``q < len_q`` (``q >= 0``)."""
        keep = b.land(b.cmp_lt(q, self.band.sq), b.cmp_lt(k, self.band.sk))
        if self.hi_off is not None:
            keep = b.land(keep, b.cmp_le(k, b.add(q, self.hi_off)))
        if self.lo_off is not None:
            keep = b.land(keep, b.cmp_ge(k, b.add(q, self.lo_off)))
        return keep

    def _row_lo_hi(self, b: IRBuilder, q: Value) -> tuple[Value, Value]:
        lo = b.const_i32(0)
        if self.lo_off is not None:
            lo = b.smax(b.add(q, self.lo_off), lo)
        hi = self.band.sk_minus1
        if self.hi_off is not None:
            hi = b.smin(b.add(q, self.hi_off), hi)
        return lo, hi


class NoMaskKTail(MaskPolicy):
    """No attention mask: only the q tail and the k tail are edge tiles."""

    masked = False

    def __init__(
        self, b: IRBuilder, len_q: Value, len_kv: Value, *, k_m0: int, k_n0: int
    ) -> None:
        bounds = AttnRuntimeBounds(len_q, len_kv, 0, -1, -1)
        super().__init__(b, BandEmitter(b, bounds), k_m0=k_m0, k_n0=k_n0)


# ---------------------------------------------------------------------------
# Host-side (compile-time) evaluation of the same formulas
# ---------------------------------------------------------------------------


class HostInt:
    """Integer builder on Python ints: evaluates integer emitters on the host
    (the band formulas above, the DMA lane map of :mod:`._attention_bwd_ring`).

    ``land`` / ``lor`` are bitwise, so they are the logical and / or of two
    compare results (Python bools) and the bitwise ops of two ints, like the
    IR builder's ``land`` / ``lor`` on i1 and i32 values.
    """

    def const_i32(self, v):
        return int(v)

    def add(self, x, y):
        return x + y

    def sub(self, x, y):
        return x - y

    def mul(self, x, y):
        return x * y

    def div(self, x, y):
        q = abs(x) // abs(y)
        return q if (x >= 0) == (y > 0) else -q  # sdiv truncates toward zero

    def smax(self, x, y):
        return max(x, y)

    def smin(self, x, y):
        return min(x, y)

    def select(self, c, x, y):
        return x if c else y

    def cmp_lt(self, x, y):
        return x < y

    def cmp_le(self, x, y):
        return x <= y

    def cmp_ge(self, x, y):
        return x >= y

    def land(self, x, y):
        return x & y

    def lor(self, x, y):
        return x | y

    def xor(self, x, y):
        return x ^ y

    def lshr(self, x, y):
        return x >> y

    def lnot(self, x):
        return not x

    def readfirstlane(self, x):
        return x


@dataclass(frozen=True)
class StaticKvTilePlan:
    """Host-side plan of one key tile: q tiles to visit and which are edges."""

    k0: int
    qt_start: int
    n_qt: int
    edge_tiles: tuple[int, ...]  # absolute q-tile indices that are edge tiles

    @property
    def interior_tiles(self) -> tuple[int, ...]:
        edges = set(self.edge_tiles)
        return tuple(
            qt
            for qt in range(self.qt_start, self.qt_start + self.n_qt)
            if qt not in edges
        )


def static_kv_tile_plan(
    *,
    s_q: int,
    s_kv: int,
    k0: int,
    k_m0: int,
    k_n0: int,
    left: int = -1,
    right: int = -1,
    bottom_right: bool = False,
    masked: bool = True,
) -> StaticKvTilePlan:
    """Evaluate ``q_tile_range`` and ``tile_class`` for compile-time shapes."""
    if left < -1 or right < -1:
        raise ValueError("band bounds must be >= -1")
    hb = HostInt()
    off = s_kv - s_q if bottom_right else 0
    bounds = AttnRuntimeBounds(
        s_q, s_kv, off, left if masked else -1, right if masked else -1
    )
    band = BandEmitter(hb, bounds)
    qt_start, n_qt = q_tile_range(hb, band, k0, k_n0=k_n0, k_m0=k_m0, uniform=False)
    edges = tuple(
        qt
        for qt in range(qt_start, qt_start + n_qt)
        if is_edge_tile(hb, band, qt, k0, k_m0=k_m0, k_n0=k_n0, masked=masked)
    )
    return StaticKvTilePlan(k0=k0, qt_start=qt_start, n_qt=n_qt, edge_tiles=edges)
