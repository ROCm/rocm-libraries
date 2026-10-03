# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ScaffoldMapPass (the LoopIR->TensileLite-scaffold mapping) -- the ONE place that maps the GENERIC,
backend-agnostic control flow (the peel-validity guard, the steady back-edge, the drain chain) onto
TensileLite's favored scaffold LABELS (`toPGR1`, `LoopEndL`, `NoGlobalLoadLoop_k`).
"""

from __future__ import annotations

from dataclasses import replace

from ..nodes import CondGoto, CondChain, LoopBack
from .early_exit import _drain_chain
from ..analyses.cfg import BackEdges
from .base import Pass


# generic CFG-role -> TensileLite scaffold label.  ONE table; no per-kernel literal, no name test.
_SCAFFOLD_LABEL = {
    "peel_validity": "toPGR1",     # prologue -> steady entry guard, TWO-WAY shape (M == 1): the
                                   # false arm IS the single drain, which is what TensileLite's
                                   # toPGR1 finalization does, so the name fits the whole guard.
    "short_entry":   "toPGR1",     # ...and with a CHAIN (M > 1) the name belongs to the arm that
                                   # skips every NGLL for the deepest drain step, which is the
                                   # branch TensileLite actually spells `toPGR1`.
    "back_edge":     "LoopEndL",   # steady loop-exit compare
    # a multi-exit header's guarded exits: typed early-exits taken before the loop is
    # entered, each landing in a progressively-more-drained variant.  Indexed by ARM POSITION.
    "early_exit":    "NoGlobalLoadLoop",
    "tail_loop":     "TailLoopBegin",  # the K % DepthU remainder region's own back-edge
    # the peeled generation a short trip skips; TensileLite spells the pair skipPGR<PGR>_1/_2 and
    # `open/closePrefetchGlobalRead2orMore` own both labels.
    "prefetch_guard": "skipPGR",
}


def _label_guard_arms(prog, back_headers):
    """Give each unlabelled guard arm the scaffold's own name for it; True if any changed."""
    for lab, blk in prog.blocks.items():
        t = blk.term
        if isinstance(t, CondChain):
            if any(p.label for p, _tgt in t.arms) or t.default_label:
                continue                            # already labelled
            drains = _drain_chain(prog)
            deepest = drains[-1] if drains else None
            base = _SCAFFOLD_LABEL["early_exit"]
            arms = []
            for p, tgt in t.arms:
                if tgt == deepest and len(drains) > 1:
                    lbl = _SCAFFOLD_LABEL["short_entry"]
                elif tgt in drains:
                    lbl = f"{base}_{drains.index(tgt)}"
                else:
                    lbl = ""                        # fall-through into the loop body
                arms.append((replace(p, label=lbl) if lbl else p, tgt))
            # The chain's default is the T == M direct entrance into the first drain stage.
            # TensileLite spells that branch to the same LoopEndL scaffold label used by the
            # steady loop's eventual exit; it is a distinct incoming edge even though both land
            # on the same physical block.
            default_label = (_SCAFFOLD_LABEL["back_edge"]
                             if drains and t.default == drains[0] else "")
            blk.term = CondChain(tuple(arms), t.default, default_label)
            changed = True
            continue
        if isinstance(t, LoopBack):
            # A counted loop states its TRIP COUNT, not a comparison, so there is no
            # `Pred` to hang the hint on -- the label lives on the terminator itself.  Its role
            if not t.label:
                role = "tail_loop" if blk.phase == "tail" else "back_edge"
                lbl = _SCAFFOLD_LABEL.get(role)
                if lbl:
                    blk.term = LoopBack(t.trips, t.body, t.exit_target, lbl)
                    changed = True
            continue
        if not isinstance(t, CondGoto) or t.pred.label:
            continue                                # only label a bare (label-free) CondGoto
        # a CondGoto whose taken-target is this block itself is the steady back-edge; otherwise
        # (taken-target is another block, e.g. steady) it is the peel-validity entry guard.
        if lab in back_headers:
            role = "tail_loop" if blk.phase == "tail" else "back_edge"
        else:
            role = "peel_validity"
        label = _SCAFFOLD_LABEL.get(role)
        if label:
            blk.term = CondGoto(replace(t.pred, label=label), t.t_target, t.f_target)
            changed = True
    return changed


class ScaffoldMapPass(Pass):
    """Label each guard arm with the name TensileLite's scaffold already uses for it."""
    def run(self, prog, am):
        back = am.get(BackEdges(), prog)
        back_headers = {be.src for be in back}          # blocks whose terminator is the back-edge
        if _label_guard_arms(prog, back_headers):
            prog.bump()
