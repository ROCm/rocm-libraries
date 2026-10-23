# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DeadRegisterLease -- a pool of register windows proven dead at a program point.

Two mechanisms in the subtile store borrow registers that are provably dead at the epilogue:

  * the weave operand lend (``_operandLendVgprs``) frees the dead K=0 A/B/SA/SB operand
    tiles so the fused store's batch can grow into those holes; and
  * (coming) the DPP store-fold's ValuC-slot reuse writes batchB's packed data and the
    blended store source into batchA's dead accumulator slots.

Both borrow the same thing -- "registers nothing reads from here on" -- but each previously
carried its own hand-rolled liveness/disjointness bookkeeping, a raw ``[(base, size)]`` list
with the safety argument in prose.  ``DeadRegisterLease`` centralizes it:

  * it VALIDATES at construction that the dead windows do not overlap (a lent window that
    aliased another would corrupt the store silently) -- fail-loud, not fail-quiet;
  * it is backward-compatible with the raw list for batch-sizing consumers (iterate it as
    ``(base, size)``, ``len()`` / ``bool()`` it); and
  * it hands out DISJOINT sub-windows to explicit-reuse consumers via :meth:`lease` -- the
    reuse site asks the pool for a window instead of asserting disjointness by hand, so two
    leases can never alias by construction.

A *window* is ``(base, size)``: the registers ``[base, base + size)``.
"""

from typing import Iterable, List, Tuple

Window = Tuple[int, int]


class DeadRegisterLease:
    """A validated pool of dead register windows, leasable as disjoint sub-windows."""

    def __init__(self, windows: Iterable[Window]):
        ws: List[Window] = sorted((int(b), int(s)) for b, s in windows)
        for (b0, s0), (b1, s1) in zip(ws, ws[1:]):
            assert s0 > 0 and s1 > 0, f"DeadRegisterLease: non-positive window size in {ws}"
            assert b0 + s0 <= b1, \
                f"DeadRegisterLease: overlapping dead windows ({b0},{s0}) and ({b1},{s1})"
        self._free: List[Window] = ws   # sorted, disjoint, non-empty-size

    # -- list-compatible view: weave batch sizing iterates/len/bools the raw windows -------
    def __iter__(self):
        return iter(self._free)

    def __len__(self) -> int:
        return len(self._free)

    def __bool__(self) -> bool:
        return bool(self._free)

    def __repr__(self) -> str:
        return f"DeadRegisterLease({self._free!r})"

    def total(self) -> int:
        """Total count of dead registers still in the pool."""
        return sum(s for _, s in self._free)

    # -- explicit reuse: carve a disjoint window out of the pool ---------------------------
    def lease(self, size: int, align: int = 1) -> int:
        """Carve a ``size``-wide, ``align``-aligned window out of the dead pool and return
        its base, removing it so no later lease can alias it.  Raise if none fits.

        This is the call that replaces hand-written disjointness asserts at reuse sites:
        a second ``lease`` physically cannot return a window overlapping the first, and if
        the pool cannot satisfy the request it fails loud instead of handing back an unsafe
        register.
        """
        assert size > 0, f"lease size must be positive, got {size}"
        for i, (b, s) in enumerate(self._free):
            start = ((b + align - 1) // align) * align   # round base up to alignment
            if start + size <= b + s:
                remainder: List[Window] = []
                if start > b:
                    remainder.append((b, start - b))
                if start + size < b + s:
                    remainder.append((start + size, (b + s) - (start + size)))
                self._free = self._free[:i] + remainder + self._free[i + 1:]
                return start
        raise AssertionError(
            f"DeadRegisterLease: no {size}-reg window (align {align}) available in {self._free}")
