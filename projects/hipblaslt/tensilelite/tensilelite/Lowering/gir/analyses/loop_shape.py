# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LoopShape -- DERIVE a summation loop's test position and trip count from the CFG, and check
that the steady region and the drain together cover the summation exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd as _gcd

from ..nodes import Move, Mma, CondGoto, LoopBack, LoopCopy
from ..nodes import terminator_targets
from ..analysis import Analysis
from .cfg import BackEdges
from .cfg import successors


PRE, POST = "pre-test", "post-test"


# A count linear in the trip symbol: `(coeff * T + const) / div`.  Exact, so the covering check is
# an equality on integers rather than a comparison of rendered strings.  `div` is what an unrolled
# body needs: it runs `(T - M) / n` times, and scaling by its own `n` cancels back to `T - M`.
@dataclass(frozen=True)
class Linear:
    coeff: int
    const: int
    div: int = 1

    def __add__(self, k):
        return Linear(self.coeff, self.const + k * self.div, self.div)

    def scaled(self, k):
        common = _gcd(int(k), self.div) or 1
        return Linear(self.coeff * (k // common), self.const * (k // common), self.div // common)

    def render(self, sym="T"):
        if not self.coeff:
            base = str(self.const)
        else:
            s = sym if self.coeff == 1 else f"{self.coeff}*{sym}"
            base = s if not self.const else f"{s} {'+' if self.const > 0 else '-'} {abs(self.const)}"
        return base if self.div == 1 else f"({base}) // {self.div}"


@dataclass(frozen=True)
class Loop:
    header:    str        # block the back edge targets
    latch:     str        # block whose terminator carries the back edge
    tester:    str        # block whose CondGoto leaves the loop -- NOT always the latch
    position:  str        # PRE | POST -- derived, see `_test_position`
    trips:     Linear     # how many times the loop's work runs
    per_trip:  int        # chunks one trip consumes (the back edge's GenXfer.adv)
    covers:    Linear     # chunks the loop covers = trips * per_trip
    exit_to:   str


def _does_work(blk):
    """Does this block perform data movement or compute?  A block that only tests does not."""
    return any(isinstance(i, (Move, Mma)) for i in blk.body)


def _loop_blocks(prog, header, latch):
    """Blocks of the natural loop: reachable from `header` and able to reach `latch`."""
    succ = successors(prog)

    def reach(start):
        seen, stack = set(), [start]
        while stack:
            x = stack.pop()
            for y in succ.get(x, ()):
                if y in prog.blocks and y not in seen:
                    seen.add(y); stack.append(y)
        return seen

    fwd = reach(header) | {header}
    return {b for b in fwd if b == latch or latch in reach(b)}


def _exit_tester(prog, blocks):
    """The block whose terminator leaves the loop -- and it is NOT always the latch."""
    for lab in sorted(blocks):
        t = prog.blocks[lab].term
        tgts = terminator_targets(t)
        if tgts and any(x not in blocks for x in tgts):
            return lab
    return None


def _test_position(prog, tester):
    """PRE or POST, from whether the block carrying the exit test also does work.

    If it does, that work happens BEFORE the test on every iteration -- post-test.  If it only
    tests, the work is in the blocks it guards -- pre-test.  Structural: no phase name, no flag,
    nothing declared."""
    return POST if _does_work(prog.blocks[tester]) else PRE


def _trips_from_node(t):
    """The loop's trip count -- READ, not derived.

    `div` is how many chunks one back edge consumes, so the BACK EDGE count carries it: an
    unrolled body runs fewer times, each time covering more.
    """
    return Linear(1 if t.trips.var else 0, -t.trips.sub, max(1, int(t.trips.div)))


class LoopShape(Analysis):
    """See module docstring.  Pure; returns `[Loop]`, one per back edge, in CFG order."""

    def run(self, prog, am):
        out = []
        for be in am.get(BackEdges(), prog):
            blocks = _loop_blocks(prog, be.header, be.src)
            tester = _exit_tester(prog, blocks)
            if tester is None:
                continue                       # an infinite loop has no exit test to read
            latch_term = prog.blocks[be.src].term
            if isinstance(latch_term, LoopCopy):
                # Every copy tests the counter, so the final cycle may be partial. `trips`
                # already counts chunks; dividing by the unroll width would drop the remainder.
                trips = _trips_from_node(latch_term)
                out.append(Loop(be.header, be.src, tester, _test_position(prog, tester),
                                trips, 1, trips, latch_term.exit_target))
                continue
            term = prog.blocks[tester].term
            if not isinstance(term, LoopBack):
                continue              # not a counted loop: nothing to state a trip count about
            position = _test_position(prog, tester)
            trips = _trips_from_node(term)
            # chunks per trip is what the back edge itself advances the generation by; a
            # multi-block body advances by the number of copies, and that IS the fact.
            per_trip = term.trips.div
            if be.xfers:
                adv = max(xf.adv for xf in be.xfers)
                if per_trip != adv:
                    raise RuntimeError(
                        f"loop {be.header!r}: the trip count divides by {per_trip} but the back "
                        f"edge advances the generation by {adv} -- two statements of "
                        f"chunks-per-trip that disagree")
            exit_to = term.exit_target
            out.append(Loop(be.header, be.src, tester, position, trips, per_trip,
                            trips.scaled(per_trip), exit_to))
        return out


def reduction_coverage_violations(prog, loops):
    """[] or a list of human-readable violations of the covering invariant.

    Returns rather than raises so a caller can decide (the verifier raises; a report-only harness
    prints).  Scoped to the summation loop -- the one whose exit leads into the drain chain -- since
    that is the only loop whose coverage the drain completes."""
    M = prog.meta.get("peel_depth")
    if not M:
        return []                              # unpipelined: there is no peel to split with
    drains = [lab for lab, b in prog.blocks.items() if str(b.phase).startswith("drain")]
    # With a drain the loop stops M chunks short and exits into it; with the drain suppressed it
    # owns every chunk and exits straight to the epilogue.
    required = Linear(1, -M) if drains else Linear(1, 0)
    exit_prefix = "drain" if drains else "end"
    out = []
    for lp in loops:
        if not str(lp.exit_to).startswith(exit_prefix):
            continue                           # not the loop the drain completes (e.g. the tail)
        if lp.covers != required:
            out.append(
                f"loop {lp.header!r} ({lp.position}, {lp.per_trip} chunk/trip) covers chunks "
                f"[0, {lp.covers.render()}) but the {len(drains)} drain steps cover "
                f"[T-{M if drains else 0}, T), so the two "
                f"{'OVERLAP' if lp.covers.const > required.const else 'LEAVE A GAP'} by "
                f"{abs(lp.covers.const - required.const)} chunk(s) -- they must together be exactly "
                f"[0, T)")
    return out
