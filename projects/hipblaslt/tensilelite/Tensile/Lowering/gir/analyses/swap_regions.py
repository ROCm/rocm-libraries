# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SwapRegions -- physical-pointer swap placement, as a REAL forward
dataflow analysis over the GIR CFG.
"""

from __future__ import annotations

from ..nodes import Move, Mark, CondGoto, Goto
from ..nodes import (copy_unit, descriptor_unit, first_shared_ref, read_operand, Region,
                     PendingMark, BLOCK_ENTRY, BLOCK_EXIT, EARLIEST, MIDPOINT)
from .value_placement import RequiredValue, ValuePlacementSolver
from ..analysis import Analysis
from .cfg import BackEdges, successors


# The pointer value the scaffold establishes before the kernel body runs: both the TDM descriptor
# and LocalReadAddr are initialized to the first buffer.  A modeled fact, named rather than a bare
ENTRY_GENERATION = 0

# --------------------------------------------------------------------- hop access helpers
def _hop_access(prog, inst, hop):
    """`(unit, shared_ref)` if `inst` is a Move on `hop` ('read'|'copy'), else None.

 `unit` is the POINTER identity, and it is not the same kind of thing on the two hops: a read has
 its own LocalReadAddr per operand, while every member of a Phi group shares one TDM descriptor --
 so a copy's identity is the descriptor, not the members this particular movement carries."""
    if not isinstance(inst, Move):
        return None
    if hop == "read":
        op = read_operand(inst)
        if op is not None:
            return (op, first_shared_ref(inst.srcs))
    elif hop == "copy":
        members, refs = copy_unit(inst)
        if members is not None:
            return (descriptor_unit(prog, members), refs[0])
    return None


def _ring_for_unit(prog, hop, unit):
    """The buffer ring depth for this (hop, unit) -- from the Gen a Ref names."""
    for blk in prog.blocks.values():
        for inst in blk.body:
            acc = _hop_access(prog, inst, hop)
            if acc is not None and acc[0] == unit:
                g = getattr(acc[1], "gen", None)
                if g is not None:
                    return max(1, g.ring)
    return 1


def _requirements(prog, blk, hop, unit, ring):
    """Ordered [(inst, required_generation)] for `blk`, in the BLOCK'S OWN chunk frame.

 A Ref's `gdelta` is an offset from `blk.gen_rel`, so subtracting it yields the
 frame-free carry -- the number that is comparable with any other block's once the edge's chunk
 delta is applied. An absolute (peel) Ref names its generation outright."""
    out = []
    for inst in blk.body:
        # A `buffer_pin` is a requirement with no access behind it: a conditionally-executed
        # successor uses the pointer, so the swap that serves it belongs on the common path.
        if isinstance(inst, Mark) and inst.kind == "buffer_pin":
            if inst.at.get("hop") == hop and tuple(inst.at.get("unit") or ()) == tuple(unit):
                out.append((inst, int(inst.at["gen"]) % ring))
            continue
        acc = _hop_access(prog, inst, hop)
        if acc is None or acc[0] != unit:
            continue
        ref = acc[1]
        if getattr(ref, "gen", None) is not None:
            req = (ref.gdelta - (blk.gen_rel or 0)) % ring
        elif getattr(ref, "abs_gen", None) is not None:
            req = ref.abs_gen % ring
        else:
            continue                     # a ring-less access carries no pointer requirement
        out.append((inst, req))
    return out


def _units(prog, hop):
    """The distinct pointer identities on `hop`, in first-seen order."""
    units = []
    for blk in prog.blocks.values():
        for inst in blk.body:
            acc = _hop_access(prog, inst, hop)
            if acc is not None and acc[0] not in units:
                units.append(acc[0])
    return units


# --------------------------------------------------------------------- CFG edges
def _unemitted_edges(prog, headers=None):
    """Edges into a block NO BACKEND EMITS -- the only paths the placement problem may exclude.
 REPLACES `_loop_bypass_edges`, which recognised "a `CondGoto` whose taken side
 enters a loop header" and dropped the other side. Two things were wrong with that.
 
    """
    return {(lab, s) for lab, blk in prog.blocks.items() for s in (blk.succs or ())
            if s in prog.blocks and prog.blocks[s].model_only}


def _topo_order_swap(prog, fwd_succs, fwd_preds):
    """Topological order of the CFG with back edges (and the unmodeled short path) removed.

    Forward propagation needs every predecessor's exit value settled before a block is
    visited, which a back edge cannot provide.
    """
    indeg = {lab: len(fwd_preds[lab]) for lab in prog.blocks}
    ready = [lab for lab in prog.blocks if indeg[lab] == 0]
    order, seen = [], set()
    while ready:
        lab = ready.pop(0)
        if lab in seen:
            continue
        seen.add(lab)
        order.append(lab)
        for s in fwd_succs[lab]:
            indeg[s] -= 1
            if indeg[s] <= 0 and s not in seen:
                ready.append(s)
    order += [lab for lab in prog.blocks if lab not in seen]
    return order


def _edge_delta(prog, src, dst, back_pairs):
    """Chunks the summation timeline advances crossing `src`->`dst`."""
    dst_base = prog.blocks[dst].path_chunk_base.get(src, prog.blocks[dst].chunk_base)
    d = dst_base - prog.blocks[src].chunk_base
    if (src, dst) in back_pairs:
        adv = max((xf.adv for xf in prog.blocks[src].xfers), default=1)
        d += adv
    return d


class SwapRegions(Analysis):
    def run(self, prog, am):
        back = am.get(BackEdges(), prog)
        succ = successors(prog)
        headers = {be.header for be in back}
        back_pairs = {(be.src, be.header) for be in back}
        bypass = _unemitted_edges(prog, headers)

        # forward edges = every CFG edge that is neither a back edge nor an edge into a block
        # nothing emits (the KEPT short arm; a FOLDED one has no such blocks).
        fwd_preds = {lab: [] for lab in prog.blocks}
        fwd_succs = {lab: [] for lab in prog.blocks}
        for p, ss in succ.items():
            for s in ss:
                if (p, s) in bypass or (p, s) in back_pairs:
                    continue
                fwd_preds[s].append(p)
                fwd_succs[p].append(s)
        order = _topo_order_swap(prog, fwd_succs, fwd_preds)

        # The FULL CFG the solver runs over: back edges INCLUDED (a loop-carried rotation is a real
        # edge and its def is the trip-bottom swap), edges into unemitted blocks excluded.
        all_preds = {lab: [] for lab in prog.blocks}
        all_succs = {lab: [] for lab in prog.blocks}
        for p, ss in succ.items():
            for t in ss:
                if (p, t) in bypass:
                    continue
                all_preds[t].append(p)
                all_succs[p].append(t)

        pending = []
        for hop in ("read", "copy"):
            for unit in _units(prog, hop):
                pending += self._flow(prog, hop, unit, all_preds, all_succs, back_pairs)
        return pending

    def _flow(self, prog, hop, unit, all_preds, all_succs, back_pairs):
        """Place the swaps for ONE physical pointer, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register -- which instructions access it,
        what generation each demands, how a value re-frames across an edge, and the ring modulus.
        WHERE the defs go is `value_placement`, so the rule cannot drift from GrIncrementRegions'.
        """
        ring = _ring_for_unit(prog, hop, unit)
        reqs = {lab: _requirements(prog, blk, hop, unit, ring)
                for lab, blk in prog.blocks.items()}
        if ring < 2 or not any(reqs.values()):
            return []                                  # in-place ring: the pointer never moves

        accesses = {lab: [RequiredValue(lab, inst, req) for inst, req in reqs[lab]]
                    for lab in prog.blocks}

        def on_conflict(kind, detail):
            blk, other, have, want = detail
            if kind == "join":
                raise RuntimeError(
                    f"SwapRegions: block {blk!r} is a JOIN whose predecessors leave the {hop} "
                    f"pointer for {unit!r} at different generations, so no single placement in "
                    f"{blk!r} is correct for both paths -- the critical edge must be split. "
                    f"Reported, not resolved by preferring a predecessor.")
            raise RuntimeError(
                f"SwapRegions: the {hop} pointer for {unit!r} must change from {have!r} to "
                f"{want!r} on the edge {blk!r}->{other!r}, but {blk!r} has other successors and "
                f"{other!r} has other predecessors, so neither end can host the swap without "
                f"executing it on a path that must not have it. Split the edge. "
                f"Reported, not resolved by preferring a side.")

        solver = ValuePlacementSolver(accesses, all_preds, all_succs,
                                      lambda a, b: _edge_delta(prog, a, b, back_pairs),
                                      prog.entry, ENTRY_GENERATION, modulus=ring)
        # The Mark names the pointer the way its hop names pointers: a read swap belongs to an
        # operand (`operand`), a copy swap to the Phi movement (`unit`, a member tuple).  Carrying
        key = "operand" if hop == "read" else "unit"
        return [PendingMark(Mark("swap", {"hop": hop, "gen_from": pl.from_value,
                                          "gen_to": pl.to_value, key: unit,
                                          "steps": (pl.to_value - pl.from_value) % ring}),
                            Region(pl.block, after=pl.after, before=pl.before,
                                   policy=(MIDPOINT if hop == "read" else EARLIEST)))
                for pl in solver.solve(on_conflict)]
