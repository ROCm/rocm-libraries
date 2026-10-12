# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ValuePlacement -- where the defs of a single-register resource go, as two real fixpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from ..nodes import BLOCK_ENTRY, BLOCK_EXIT


BOTTOM = "bottom"      # nothing reaches / demands here yet
TOP    = "top"         # a genuine conflict -- reported, never silently resolved


def meet(a, b):
    """Lattice meet: BOTTOM is the identity, equal values agree, disagreement is TOP."""
    if a is BOTTOM:
        return b
    if b is BOTTOM:
        return a
    if a is TOP or b is TOP:
        return TOP
    return a if a == b else TOP


@dataclass(frozen=True)
class RequiredValue:
    """One required value at a program point, in that block's own frame."""
    block: str
    node:  object
    value: int


@dataclass(frozen=True)
class Placement:
    """A def that must happen, and the window it may occupy inside ONE block.

    `after`/`before` are opaque anchors (a body node, or BLOCK_ENTRY / BLOCK_EXIT).  Choosing the
    point inside the window is POLICY and belongs to PlacementPass, not here.
    """
    block:      str
    after:      object
    before:     object
    from_value: int
    to_value:   int
    at_exit:    bool = False       # True: sits at the block's EXIT, so it is on EVERY edge leaving
                                   # that block (one register, branch comes after).  False: an
                                   # in-block window between two accesses.


class ValuePlacementSolver:
    """Solve def placement for one register.

    `accesses`   -- {block label: [RequiredValue]} in program order.
    `preds`/`succs` -- the CFG to solve over, ALREADY filtered (back edges included, unmodeled paths
                   excluded).  Back edges must appear in both, or the loop-carried def is lost.
    """

    def __init__(self, accesses, preds, succs, edge_delta, entry, entry_value, modulus=None):
        self._acc = accesses
        self._preds = preds
        self._succs = succs
        self._delta = edge_delta
        self._entry = entry
        self._entry_value = entry_value
        self._mod = modulus

    # ------------------------------------------------------------------ domain
    def _norm(self, v):
        if v is BOTTOM or v is TOP or self._mod is None:
            return v
        return v % self._mod

    def _fwd(self, v, p, b):
        """Re-express a value crossing p->b in b's frame."""
        return v if (v is BOTTOM or v is TOP) else self._norm(v - self._delta(p, b))

    def _bwd(self, v, b, s):
        """Re-express a successor's demand back into b's frame."""
        return v if (v is BOTTOM or v is TOP) else self._norm(v + self._delta(b, s))

    # ------------------------------------------------------------------ fixpoints
    def antic(self):
        """antic_in[b] -- the value demanded ENTERING b.  Independent of placement: a block's own
        first access fixes it, and an accessless block inherits the meet of its successors."""
        inn = {b: BOTTOM for b in self._acc}
        for b, acc in self._acc.items():
            if acc:
                inn[b] = self._norm(acc[0].value)
        changed = True
        while changed:
            changed = False
            for b in self._acc:
                if self._acc[b]:
                    continue
                cur = BOTTOM
                for s in self._succs.get(b, ()):
                    cur = meet(cur, self._bwd(inn[s], b, s))
                if cur != inn[b]:
                    inn[b] = cur
                    changed = True
        return inn

    def _exit_want(self, b, antic):
        """The single value all of b's successors demand, in b's frame -- or None.

        None means either "no successor demands anything" or "they disagree", and in both cases one
        def at b's exit cannot serve them: there is ONE register and the branch comes after it."""
        wants = set()
        for s in self._succs.get(b, ()):
            w = antic.get(s, BOTTOM)
            if w is BOTTOM:
                continue
            if w is TOP:
                return None
            wants.add(self._bwd(w, b, s))
        return wants.pop() if len(wants) == 1 else None

    def avail(self, antic):
        """Forward fixpoint over what LEAVES each block."""
        a_in   = {b: BOTTOM for b in self._acc}
        body   = {b: BOTTOM for b in self._acc}
        leave  = {b: BOTTOM for b in self._acc}
        changed = True
        while changed:
            changed = False
            for b in self._acc:
                cur = self._norm(self._entry_value) if b == self._entry else BOTTOM
                for p in self._preds.get(b, ()):
                    cur = meet(cur, self._fwd(leave[p], p, b))
                nb = self._norm(self._acc[b][-1].value) if self._acc[b] else cur
                want = self._exit_want(b, antic)
                nl = nb if (want is None or nb is BOTTOM) else want
                if (cur, nb, nl) != (a_in[b], body[b], leave[b]):
                    a_in[b], body[b], leave[b] = cur, nb, nl
                    changed = True
        return a_in, body, leave

    # ------------------------------------------------------------------ placement
    def solve(self, on_conflict):
        """[Placement], deterministically ordered.  `on_conflict(kind, detail)` reports a TOP meet
        or an unsplittable critical edge; it is expected to raise.

        Three def kinds fall out of the two fixpoints, and they are disjoint by construction:
          intra-block -- between consecutive accesses in b whose requirements differ;
          on-entry   -- where a block's own first requirement differs from what reaches it;
          on-edge    -- where two predecessors deliver different values to one successor.
        """

        antic = self.antic()
        a_in, body, leave = self.avail(antic)
        out = []

        for b in sorted(self._acc):
            acc = self._acc[b]
            for prev, nxt in zip(acc, acc[1:]):
                if self._norm(prev.value) != self._norm(nxt.value):
                    out.append(Placement(b, prev.node, nxt.node,
                                         self._norm(prev.value), self._norm(nxt.value)))

        for b in sorted(self._acc):
            # HEAD def: what arrives vs what this block demands.
            arrive, want = a_in[b], antic.get(b, BOTTOM)
            if arrive is TOP and want is not BOTTOM:
                on_conflict("join", (b, None, arrive, want))
            elif (arrive is not BOTTOM and want is not BOTTOM and want is not TOP
                  and arrive != want):
                preds = self._preds.get(b, ())
                if len(preds) <= 1:
                    first = self._acc[b][0].node if self._acc.get(b) else BLOCK_EXIT
                    out.append(Placement(b, BLOCK_ENTRY, first, arrive, want))
                else:
                    on_conflict("critical-edge", (preds[0], b, arrive, want))

            # EXIT def: the agreed successor demand vs what the block already holds.
            want_x = self._exit_want(b, antic)
            have_x = body[b]
            if want_x is not None and have_x is not BOTTOM and have_x != want_x:
                last = self._acc[b][-1].node if self._acc.get(b) else BLOCK_ENTRY
                out.append(Placement(b, last, BLOCK_EXIT, have_x, want_x, at_exit=True))
        return out
