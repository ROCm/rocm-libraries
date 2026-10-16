# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
BarrierUniformity -- every wave must reach every cross-wave fence.

A fence at block or cluster scope is a rendezvous: the waves that arrive wait for the waves that
have not. So it must sit where ALL of them run. A wave that skips it does not stall -- the ones
that did arrive do, forever.

The check walks each proc-scoped fence back to the entry block and reports any block on the way
that some waves can skip: one restricted to a subset by its `role`, or one entered through a
guard whose predicate reads a symbol that varies per wave.

This is a liveness property. The ledger only proves the ORDERING obligations were discharged, so
it is satisfied by a fence no wave reaches; only this says the fence is reachable.
`tool/barrier_uniformity_asm.py` asks the same question of the emitted instruction stream.
"""

from __future__ import annotations

from ..nodes import Mark, CondGoto, CondChain, LoopBack
from ..analysis import Analysis
from .cfg import successors


# The scopes that name a CROSS-WAVE rendezvous.  A selector at any other scope is discharged on
# one wave, so program order carries it and there is nothing to be uniform about.
PROC_SCOPES = ("block", "cluster")

# The role value meaning "every wave runs this block".  GIR has no role ALGEBRA yet, so
# anything else is treated as a restriction rather than guessed at.
ROLE_ALL = "all"


class BarrierViolation:
    """One fence some wave of its scope can skip, and the block that lets it."""

    __slots__ = ("block", "index", "scope", "reason", "where")

    def __init__(self, block, index, scope, reason, where):
        self.block = block        # label of the block holding the fence Mark
        self.index = index        # position in that block's body
        self.scope = scope        # the fence's scope -- the wave set that must rendezvous
        self.reason = reason      # 'role' (a subset runs the block) | 'guard' (a per-wave branch)
        self.where = where        # the offending block / predicate, rendered

    def __repr__(self):
        return (f"{self.block}[{self.index}] scope={self.scope}: {self.reason} at {self.where}")


def _preds_of(term):
    """Every `Pred` a terminator tests.  `LoopBack` carries a trip COUNT, not a comparison,
    and a `Goto`/`Return` tests nothing -- neither can discriminate waves."""
    if isinstance(term, CondGoto):
        return [term.pred]
    if isinstance(term, CondChain):
        return [p for p, _t in term.arms]
    return []


def _pred_symbols(pred):
    """The symbols a predicate reads: its counter and its bound's variable (a bare constant bound
    contributes nothing)."""
    out = [pred.lhs]
    if getattr(pred.rhs, "var", ""):
        out.append(pred.rhs.var)
    return [s for s in out if s]


def _reachable_from(succ, start):
    seen, stack = set(), list(succ.get(start, ()))
    while stack:
        lab = stack.pop()
        if lab in seen:
            continue
        seen.add(lab)
        stack.extend(succ.get(lab, ()))
    return seen

def _proc_scoped_fences(blk):
    """(slot, scope) for each proc-scoped fence in `blk` -- the cross-wave rendezvous points.

    A block no backend emits is skipped: no wave reaches the selector, so no wave can be
    brought to it and not another.
    """
    if blk.model_only:
        return
    for i, node in enumerate(blk.body):
        if isinstance(node, Mark) and node.kind == "fence":
            scope = (node.at or {}).get("scope")
            if scope in PROC_SCOPES:
                yield i, scope


def _path_discriminators(on_path, lab, i, scope, varying):
    """Blocks on the way to a rendezvous that some waves can skip -- by role, or by a guard
    that varies per wave."""
    out = []
    for b in on_path:
        if b.role != ROLE_ALL:
            out.append(BarrierViolation(lab, i, scope, "role",
                                          "%s role=%r" % (b.label, b.role)))
        for p in _preds_of(b.term):
            bad = [s for s in _pred_symbols(p) if s in varying]
            if bad:
                out.append(BarrierViolation(lab, i, scope, "guard",
                                            "%s '%s' varies on %s" % (b.label, p.render(), bad)))
    return out


class BarrierUniformity(Analysis):
    """See module docstring.  Pure; returns `[BarrierViolation]` (empty = every wave arrives)."""

    def run(self, prog, am):
        succ = successors(prog)
        varying = set(prog.meta.get("agent_varying_symbols") or ())
        fwd = {lab: _reachable_from(succ, lab) for lab in prog.blocks}
        from_entry = {prog.entry} | fwd.get(prog.entry, set())

        viol = []
        for lab, blk in prog.blocks.items():
            for i, scope in _proc_scoped_fences(blk):
                # every block that lies on SOME entry -> this-block path
                on_path = [b for lb, b in prog.blocks.items()
                           if lb in from_entry and (lb == lab or lab in fwd.get(lb, ()))]
                viol += _path_discriminators(on_path, lab, i, scope, varying)
        return viol
