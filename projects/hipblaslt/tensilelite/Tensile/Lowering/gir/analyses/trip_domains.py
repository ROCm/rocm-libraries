# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
TripDomains -- the trip counts each CFG edge admits.

Two arms created by different scaffold passes are correlated when they test the same trip symbol:
`prologue` guards the peel on `T == 1` and `prologue_join` enters the steady loop on `T > 2`, so a
path through the PGR1 arm cannot reach `steady` at all.  Keying on the SYMBOL rather than on the
predicate is what lets the two passes stay ignorant of each other -- the domains intersect, and an
empty intersection is the exclusion.

The domain is a bounded enumeration of trip counts, matching what StinkyTofu already reasons in,
so the frame contract ships the answer rather than the condition.
"""

from __future__ import annotations

from ..analysis import Analysis
from ..nodes import CondChain, CondGoto, Goto, LoopBack

#: Small enough to enumerate, large enough that every peel/drain literal has room above it.
TRIP_FLOOR = 4
TRIP_HEADROOM = 3


def _preds_of(block):
    """Every `(Pred, target)` the terminator guards, in arm order, plus its unguarded default."""
    term = block.term
    if isinstance(term, CondGoto):
        return [(term.pred, term.t_target)], term.f_target
    if isinstance(term, CondChain):
        return list(term.arms), term.default
    return [], None


def trip_symbol(prog):
    """The one trip symbol the domains range over, or "" when the program names none."""
    for blk in prog.blocks.values():
        if isinstance(blk.term, LoopBack) and blk.term.trips.var:
            return blk.term.trips.var
    for blk in prog.blocks.values():
        for pred, _t in _preds_of(blk)[0]:
            if pred.lhs:
                return pred.lhs
    return ""


def trip_limit(prog):
    """The enumeration bound: every literal a terminator tests, plus headroom."""
    limit = TRIP_FLOOR
    for blk in prog.blocks.values():
        term = blk.term
        if isinstance(term, LoopBack) and term.trips.var:
            # Include one-trip and repeated-trip cases after the loop's subtraction/division.
            limit = max(limit, int(term.trips.sub) + 2 * int(term.trips.div) + TRIP_HEADROOM)
        for pred, _t in _preds_of(blk)[0]:
            if not pred.rhs.var:
                limit = max(limit, int(pred.rhs.const) + TRIP_HEADROOM)
    return limit


def _holds(pred, symbol, trip):
    """Does `pred` hold at this concrete trip count?  None when it is not a pure test of `symbol`.

    A symbol-relative bound relates two unknowns, so it narrows nothing and stays conservative."""
    if pred.lhs != symbol or pred.rhs.var:
        return None
    rhs = int(pred.rhs.const)
    return {">=": trip >= rhs, ">": trip > rhs, "==": trip == rhs,
            "<": trip < rhs, "<=": trip <= rhs, "!=": trip != rhs}[pred.op]


class TripDomainSet:
    """`{(src, dst): frozenset(trips)}`, plus the universe every domain is a subset of."""

    def __init__(self, universe, edges):
        self.universe = frozenset(universe)
        self._edges = dict(edges)

    def admits(self, src, dst):
        """The trips that can take this edge; the universe when nothing narrows it."""
        return self._edges.get((src, dst), self.universe)


class TripDomains(Analysis):
    """See module docstring.  Pure; returns a `TripDomainSet`."""

    def run(self, prog, am):
        symbol = trip_symbol(prog)
        universe = frozenset(range(1, trip_limit(prog) + 1))
        edges = {}
        if not symbol:
            return TripDomainSet(universe, edges)

        for lab, blk in prog.blocks.items():
            if isinstance(blk.term, LoopBack) and blk.term.trips.var == symbol:
                trips = blk.term.trips
                counts = {
                    trip: (trip - int(trips.sub)) // max(1, int(trips.div))
                    for trip in universe
                }
                # One trip reaches only the exit. Two or more trips take the back edge before
                # eventually taking that same exit, so these domains intentionally overlap.
                _narrow(edges, lab, blk.term.body,
                        {trip for trip, count in counts.items() if count > 1})
                _narrow(edges, lab, blk.term.exit_target,
                        {trip for trip, count in counts.items() if count > 0})
                continue
            arms, default = _preds_of(blk)
            if not arms:
                continue
            # An ordered chain takes the FIRST satisfied arm, so an arm also requires that every
            # earlier one failed, and the default requires that all of them did.
            unmatched = set(universe)
            for pred, target in arms:
                taken = set()
                for trip in sorted(unmatched):
                    held = _holds(pred, symbol, trip)
                    if held is None:
                        taken = None
                        break
                    if held:
                        taken.add(trip)
                if taken is None:
                    unmatched = None
                    break
                _narrow(edges, lab, target, taken)
                unmatched -= taken
            if unmatched is None:
                continue
            if default is not None:
                _narrow(edges, lab, default, unmatched)
            elif isinstance(blk.term, CondGoto):
                _narrow(edges, lab, blk.term.f_target, unmatched)
        return TripDomainSet(universe, edges)


def _narrow(edges, src, dst, domain):
    """A repeated `(src, dst)` -- two arms reaching one target -- admits the union."""
    if dst is None or isinstance(dst, Goto):
        return
    key = (src, dst)
    edges[key] = frozenset(domain) | edges.get(key, frozenset())
