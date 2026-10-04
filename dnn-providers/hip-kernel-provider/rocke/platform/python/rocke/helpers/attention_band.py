# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime two-sided diagonal band for attention shells.

A shell-side utility: the attention inner bodies never call it.  A shell builds
one ``BandEmitter`` and uses it inside its score hook (per-cell keep) and
epilogue hook (per-row validity), and to derive a per-tile K-loop range.

* ``AttnRuntimeBounds`` -- per-sequence valid lengths and a two-sided
  diagonal band whose limits may be compile-time ints or runtime i32 values.
* ``BandEmitter`` -- emits the per-row band, per-cell keep predicate,
  per-row validity and k-tile range from those bounds.

Band semantics (the keep rule)::

    keep(q, k) = k < seqlen_k
             and (right < 0 or k <= q + diag_offset + right)
             and (left  < 0 or k >= q + diag_offset - left)

``-1`` means unbounded. ``diag_offset`` is ``0`` for a top-left diagonal and
``seqlen_k - seqlen_q`` for a bottom-right one. A row is *valid* when it is
inside ``seqlen_q`` and its band, clipped to ``[0, seqlen_k)``, is non-empty;
this is decided from the bounds, never from the softmax denominator (the masked
fill value of the body mask is finite, so a fully masked row can still
accumulate a positive sum).  Runtime bounds must stay below 2**30; callers
normalise "unbounded" to -1.  The constructor emits constants, so construct the
emitter once, inside the shell only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..core.ir import IRBuilder, Value

__all__ = ["AttnRuntimeBounds", "BandEmitter", "RowBand"]

# Stand-in for "no limit" when a bound is only known at run time.
_FAR = 1 << 30

IntOrValue = Union[int, Value]


def _is_static_unbounded(x: IntOrValue) -> bool:
    return isinstance(x, int) and x < 0


@dataclass(frozen=True)
class AttnRuntimeBounds:
    """Valid lengths and diagonal band of one sequence (see module docstring).

    Each field is either a Python int (known when the kernel is built; a static
    ``-1`` bound emits no IR at all) or an i32 SSA value (known at launch).
    ``seqlen_q`` / ``seqlen_k`` are the per-sequence valid counts, not the
    allocated extents.
    """

    seqlen_q: IntOrValue
    seqlen_k: IntOrValue
    diag_offset: IntOrValue = 0
    left_bound: IntOrValue = -1
    right_bound: IntOrValue = -1
    # Derive the k-tile loop range from the band so tiles that no row of the
    # q-tile can attend are never loaded. Ignored when the caller passes an
    # explicit ``k_tile_start`` / ``k_tile_stop``.
    skip_tiles: bool = True

    def __post_init__(self) -> None:
        for name in ("left_bound", "right_bound"):
            v = getattr(self, name)
            if isinstance(v, int) and v < -1:
                raise ValueError(f"{name} must be >= -1 (got {v})")


@dataclass(frozen=True)
class RowBand:
    """Loop-invariant band facts of one query row (computed once per slot)."""

    q_pos: Value
    row_in: Value
    valid: Value
    lo: Optional[Value]  # smallest attendable key position (None: no left limit)
    hi: Optional[Value]  # largest attendable key position (None: no right limit)


class BandEmitter:
    """Emit band arithmetic for one :class:`AttnRuntimeBounds`.

    The constructor emits the shared limits (``seqlen - 1``), so build it once.
    """

    def __init__(self, b: IRBuilder, bounds: AttnRuntimeBounds) -> None:
        self.b = b
        self.bounds = bounds
        self.sq = self._i32(bounds.seqlen_q)
        self.sk = self._i32(bounds.seqlen_k)
        one = b.const_i32(1)
        self._zero = b.const_i32(0)
        self.sk_minus1 = b.sub(self.sk, one)  # -1 when the key sequence is empty
        self.off = (
            None
            if isinstance(bounds.diag_offset, int) and bounds.diag_offset == 0
            else self._i32(bounds.diag_offset)
        )
        self.left = (
            None if _is_static_unbounded(bounds.left_bound) else bounds.left_bound
        )
        self.right = (
            None if _is_static_unbounded(bounds.right_bound) else bounds.right_bound
        )
        self._far = None

    def _i32(self, x: IntOrValue) -> Value:
        return self.b.const_i32(x) if isinstance(x, int) else x

    def _far_const(self) -> Value:
        if self._far is None:
            self._far = self.b.const_i32(_FAR)
        return self._far

    # -- per-row / per-cell predicates -----------------------------------------

    def row_band(self, q_pos: Value) -> RowBand:
        b = self.b
        row_in = b.cmp_lt(q_pos, self.sq)
        centre = q_pos if self.off is None else b.add(q_pos, self.off)
        lo = hi = None
        if self.right is not None:
            r = self._i32(self.right)
            hi = b.add(centre, r)
            if not isinstance(self.right, int):
                hi = b.select(b.cmp_lt(r, self._zero), self._far_const(), hi)
        if self.left is not None:
            lft = self._i32(self.left)
            lo = b.sub(centre, lft)
            if not isinstance(self.left, int):
                lo = b.select(
                    b.cmp_lt(lft, self._zero), b.sub(self._zero, self._far_const()), lo
                )
        lo_eff = self._zero if lo is None else b.smax(lo, self._zero)
        hi_eff = self.sk_minus1 if hi is None else b.smin(hi, self.sk_minus1)
        valid = b.land(row_in, b.cmp_le(lo_eff, hi_eff))
        return RowBand(q_pos=q_pos, row_in=row_in, valid=valid, lo=lo, hi=hi)

    def col_in(self, k_pos: Value) -> Value:
        return self.b.cmp_lt(k_pos, self.sk)

    def cell_keep(self, rb: RowBand, k_pos: Value, col_in: Value) -> Value:
        b = self.b
        keep = col_in
        if rb.hi is not None:
            keep = b.land(keep, b.cmp_le(k_pos, rb.hi))
        if rb.lo is not None:
            keep = b.land(keep, b.cmp_ge(k_pos, rb.lo))
        return keep

    # -- k-tile range ---------------------------------------------------------

    def k_tile_range(
        self, q_pos_first: Value, block_m: int, block_k: int
    ) -> Tuple[Value, Value]:
        """``[start, stop)`` k-tile indices any row of the q-tile can attend.

        ``q_pos_first`` is the in-sequence row of the tile's first row. The range
        is a superset of every row's band (each row still applies its own cell
        keep), and is empty (``stop <= start``) for a tile with nothing to read.
        """
        b = self.b
        ck = b.const_i32(block_k)
        centre0 = q_pos_first if self.off is None else b.add(q_pos_first, self.off)
        if self.left is None:
            start = self._zero
        else:
            lft = self._i32(self.left)
            lo0 = b.sub(centre0, lft)
            if not isinstance(self.left, int):
                lo0 = b.select(b.cmp_lt(lft, self._zero), self._zero, lo0)
            start = b.div(b.smax(lo0, self._zero), ck)
        if self.right is None:
            hi_last = self.sk_minus1
        else:
            r = self._i32(self.right)
            hi_raw = b.add(b.add(centre0, b.const_i32(block_m - 1)), r)
            if not isinstance(self.right, int):
                hi_raw = b.select(b.cmp_lt(r, self._zero), self._far_const(), hi_raw)
            hi_last = b.smin(hi_raw, self.sk_minus1)
        # (hi + block_k) // block_k == hi // block_k + 1 for hi >= 0 and 0 for
        # an empty range (hi == -1); clamp so very negative values stay empty.
        stop = b.div(b.add(b.smax(hi_last, b.const_i32(-1)), ck), ck)
        return start, stop

    def k_tile_count_ceil(self, block_k: int) -> Value:
        b = self.b
        ck = b.const_i32(block_k)
        return b.div(b.add(self.sk, b.const_i32(block_k - 1)), ck)
