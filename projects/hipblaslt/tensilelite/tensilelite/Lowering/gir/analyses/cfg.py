# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
CFG analyses : dominators and back-edges.
"""

from __future__ import annotations

from ..nodes import successor_labels as _succ_labels
from ..analysis import Analysis


def successors(prog):
    """{label: [succ labels]} over existing blocks only (the 'end' sink is dropped)."""
    return {lab: [s for s in _succ_labels(blk) if s in prog.blocks]
            for lab, blk in prog.blocks.items()}


def reachable(prog):
    """{label: set(labels reachable from it, excluding itself unless via a cycle)}.

    Reachability, not dominance: some execution may run one block before another even where the
    first does not dominate the second."""
    succ = successors(prog)
    out = {}
    for start in prog.blocks:
        seen, stack = set(), list(succ.get(start, ()))
        while stack:
            lab = stack.pop()
            if lab in seen:
                continue
            seen.add(lab)
            stack.extend(succ.get(lab, ()))
        out[start] = seen
    return out


class Dominators(Analysis):
    """Classic iterative dominator-set computation -> {label: set(dominator labels)}.
    dom(entry) = {entry}; dom(n) = {n} | (& dom(p) for reachable preds p)."""

    def run(self, prog, am):
        succ = successors(prog)
        preds = {lab: [] for lab in prog.blocks}
        for lab, ss in succ.items():
            for s in ss:
                preds[s].append(lab)

        entry = prog.entry
        labels = list(prog.blocks.keys())
        all_set = set(labels)
        dom = {lab: ({lab} if lab == entry else set(all_set)) for lab in labels}

        changed = True
        while changed:
            changed = False
            for lab in labels:
                if lab == entry:
                    continue
                rp = [p for p in preds[lab] if p in dom]
                if rp:
                    inter = set(dom[rp[0]])
                    for p in rp[1:]:
                        inter &= dom[p]
                else:
                    inter = set()
                new = {lab} | inter
                if new != dom[lab]:
                    dom[lab] = new
                    changed = True
        return dom


class BackEdge:
    """A back-edge (src -> header) whose header dominates src. Carries the source block's
 GenXfer list so frame_map keys the transfer per back-edge (not xfers[0])."""

    def __init__(self, src, header, xfers):
        self.src = src            # block LABEL the back-edge leaves from
        self.header = header      # block LABEL the back-edge targets (loop header)
        self.xfers = xfers        # list[GenXfer] on the src block


class BackEdgeSet:
    """The back-edges of one program."""
    def __init__(self, edges):
        self._edges = list(edges)

    def __iter__(self):
        return iter(self._edges)

    def is_back_edge(self, src, target):
        return any(be.src == src and be.header == target for be in self._edges)


class BackEdges(Analysis):
    """Enumerate back-edges by DOMINANCE. A persistent second loop is admitted with
 no analysis change -- the R0 obligation."""

    def run(self, prog, am):
        dom = am.get(Dominators(), prog)
        edges = []
        for lab, blk in prog.blocks.items():
            for s in _succ_labels(blk):
                if s in prog.blocks and s in dom.get(lab, set()):
                    edges.append(BackEdge(lab, s, list(blk.xfers)))
        return BackEdgeSet(edges)
