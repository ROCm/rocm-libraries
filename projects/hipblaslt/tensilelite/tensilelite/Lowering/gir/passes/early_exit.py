# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
EarlyExitPass (G1/G3) -- give the CFG the `T < M` entries the scaffold actually takes.
"""

from __future__ import annotations

from ..nodes import CondGoto, CondChain, Pred, Bound
from ..nodes import terminator_targets
from .base import Pass


def _drain_chain(prog):
    """The drain block LABELS in step order, or () when there is no chain to enter partway into.

    Keyed on `Block.phase`, the structural fact, never on the label: a Program may label its blocks
    anything (the hand-built test fixtures use `d0`/`d1`), and the phase is what says "this is drain
    step k" -- the same rule the rest of this package follows for the tail region."""
    steps = {}
    for lab, blk in prog.blocks.items():
        ph = getattr(blk, "phase", "") or ""
        if ph.startswith("drain") and ph[5:].isdigit():
            steps[int(ph[5:])] = lab
    return tuple(steps[k] for k in sorted(steps))


class EarlyExitPass(Pass):
    """See module docstring.  Adds the `T < M` entry edges; touches no body."""

    def run(self, prog, am):
        drains = _drain_chain(prog)
        M = len(drains)
        if M < 2:
            return                          # no partially-drained variant to enter
        changed = False
        for lab, blk in list(prog.blocks.items()):
            t = blk.term
            # only the peel-validity guard: a two-way branch whose false arm is the chain head and
            # whose taken arm is the steady loop.  Identified STRUCTURALLY (targets), never by label
            if not isinstance(t, CondGoto) or t.f_target != drains[0]:
                continue
            if t.t_target == lab:
                continue                    # a back edge, not the entry guard
            arms = tuple((Pred(lhs=t.pred.lhs, op="==", rhs=Bound(const=k)), drains[M - k])
                         for k in range(1, M))
            arms += ((t.pred, t.t_target),)
            blk.term = CondChain(arms=arms, default=drains[0])
            blk.succs = tuple(tgt for _p, tgt in arms) + (drains[0],)
            # ONE RULE, STATED BY THE FOLD: every edge from this guard lands in the arm's frame.
            head = prog.blocks[drains[0]]
            base = head.path_chunk_base.get(lab, head.chunk_base)
            for k in range(1, M):
                prog.blocks[drains[M - k]].path_chunk_base[lab] = base
            changed = True
        if not changed:
            return
        # preds are a DECLARED fact (G-TERM): recompute from the rewritten terminators.
        computed = {l: set() for l in prog.blocks}
        for l, b in prog.blocks.items():
            for tgt in terminator_targets(b.term):
                if tgt in prog.blocks:
                    computed[tgt].add(l)
        for l, b in prog.blocks.items():
            if b.preds or computed[l]:
                b.preds = tuple(sorted(computed[l]))
        prog.bump()


