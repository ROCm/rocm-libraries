# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GrIncrementRegions -- the global-read-address advance, as a REAL forward
dataflow over the GIR CFG (the same shape as SwapRegions).
"""

from __future__ import annotations

from ..nodes import Move, Mark, successor_labels
from ..nodes import (copy_unit, Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT, EARLIEST)
from .value_placement import RequiredValue, ValuePlacementSolver
from ..analysis import Analysis
from .cfg import BackEdges, successors
from .swap_regions import _unemitted_edges
# IMPORTED, not re-derived: a second reading of how many regions a movement has is exactly
# how this analysis and the walk that produces the copies would drift apart.
from .region_increment import _n_regions, _canonical_unit, walk_ref


def _copy_of(inst, prog):
    """`(descriptor, walk ref)` if `inst` is a global->shared copy Move, else None.

 Keyed on the DESCRIPTOR, never on the members that move: a copy that dropped an absent member
 names a subset of its group, and counting that as its own movement miscounts the chunk advance."""
    if not isinstance(inst, Move):
        return None
    if not any(s.tile.space == "global" for s in inst.srcs):
        return None
    members, refs = copy_unit(inst)
    if members is None:
        return None
    return (_canonical_unit(prog, members), walk_ref(prog, refs))


def _chunk_of(blk, ref, peel_seq=None):
    """The absolute summation chunk a copy fetches, on the global chunk timeline.

    Loop-carried ref: `blk.chunk_base + (gdelta - blk.gen_rel)` -- frame-free, comparable across
    blocks.
    """
    if getattr(ref, "gen", None) is not None:
        return blk.chunk_base + (ref.gdelta - (blk.gen_rel or 0))
    if getattr(ref, "abs_gen", None) is not None:
        return peel_seq          # positional index among this operand's peel copies
    return None


class GrIncrementRegions(Analysis):
    """Forward dataflow over the GIR CFG for one global-read address per Phi MOVEMENT.

 Per movement, not per operand: a fused group issues one cooperative load off one
 descriptor, so it owns one address that advances once. Unfused, a movement is a lone operand
 and the two readings coincide -- which is why the distinction stayed invisible until multi-wave.
    """

    @property
    def cache_key(self):
        return "GrIncrementRegions"

    def run(self, prog, am):
        back = am.get(BackEdges(), prog)
        succ = successors(prog)
        back_pairs = {(be.src, be.header) for be in back}
        bypass = _unemitted_edges(prog, {be.header for be in back})
        fwd_preds, fwd_succs = {l: [] for l in prog.blocks}, {l: [] for l in prog.blocks}
        for p_, ss in succ.items():
            for t in ss:
                if (p_, t) not in back_pairs and (p_, t) not in bypass:
                    fwd_preds[t].append(p_)
                    fwd_succs[p_].append(t)
        order = _topo_order_gr(prog, fwd_succs, fwd_preds)
        # The FULL CFG the solver runs over: back edges INCLUDED (the per-trip advance IS the back
        # edge's def), edges into unemitted blocks excluded.
        all_preds = {l: [] for l in prog.blocks}
        all_succs = {l: [] for l in prog.blocks}
        for p_, ss in succ.items():
            for t in ss:
                if (p_, t) in bypass:
                    continue
                all_preds[t].append(p_)
                all_succs[p_].append(t)

        pending = []
        for unit in self._units(prog, order):
            pending += self._flow(prog, order, all_preds, all_succs, back, back_pairs, unit)
        return pending

    @staticmethod
    def _units(prog, order):
        """The distinct global-read addresses -- one per Phi movement -- in first-seen order."""
        units = []
        for lab in order:
            for inst in prog.blocks[lab].body:
                acc = _copy_of(inst, prog)
                if acc is not None and acc[0] not in units:
                    units.append(acc[0])
        return units

    @staticmethod
    def _reqs(prog, lab, unit, npeel=0):
        """`([(inst, chunk)], npeel)` for `unit`'s copies in `lab`, program order.

        `npeel` counts this unit's peel loads ACROSS blocks, not within one: a peel copy names its
        chunk by position, so a counter that restarts per block renumbers the copies of any block
        the peel is split across.

        A `chunk_pin` Mark is a requirement with no copy behind it: a conditionally-executed
        successor consumes the address, so the advance that serves it must be established here,
        on the path both arms take.
        """
        blk = prog.blocks[lab]
        nreg = max(1, _n_regions(prog, unit))
        out = []
        for inst in blk.body:
            if isinstance(inst, Mark) and inst.kind == "chunk_pin":
                if tuple(inst.at.get("unit") or ()) == tuple(unit):
                    out.append((inst, int(inst.at["chunk"])))
                continue
            acc = _copy_of(inst, prog)
            if acc is None or acc[0] != unit:
                continue
            c = _chunk_of(blk, acc[1], peel_seq=npeel // nreg)
            if getattr(acc[1], "gen", None) is None:
                npeel += 1                       # this unit's next peel LOAD (see above)
            if c is not None:
                out.append((inst, c))
        return out, npeel

    def _flow(self, prog, order, all_preds, all_succs, back, back_pairs, unit):
        """Place the advances for ONE global-read address, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register: which instructions consume the
        descriptor, the absolute chunk each fetches, and how a chunk re-frames across the back edge.
        WHERE the advances go is `value_placement`, the same module SwapRegions uses.
        """
        reqs, _npeel = {}, 0
        for lab in order:
            reqs[lab], _npeel = self._reqs(prog, lab, unit, _npeel)
        first = next((c for lab in order for _i, c in reqs[lab]), None)
        if first is None:
            return []

        accesses = {lab: [RequiredValue(lab, inst, chunk) for inst, chunk in reqs.get(lab, ())]
                    for lab in prog.blocks}
        adv_of = {be.src: max((xf.adv for xf in be.xfers), default=1) for be in back}
        back_set = {(be.src, be.header) for be in back}

        def delta(a, b):
            # Requirements are ABSOLUTE chunks, so forward edges need no re-framing.  The back edge
            # re-enters the same block one trip on, so the value leaving must be `adv` further along
            return adv_of.get(a, 1) if (a, b) in back_set else 0

        def on_conflict(kind, detail):
            blk, other, have, want = detail
            if kind == "join":
                raise RuntimeError(
                    f"GrIncrementRegions: block {blk!r} is a JOIN whose predecessors leave the "
                    f"global-read address for {unit!r} at different chunks; no single increment "
                    f"placement in {blk!r} is correct for both paths -- the edge must be split. "
                    f"Reported, not resolved by preferring a predecessor.")
            raise RuntimeError(
                f"GrIncrementRegions: the address for {unit!r} must advance {have!r}->{want!r} "
                f"on {blk!r}->{other!r}, but neither end can host it without advancing on a path "
                f"that must not. Split the edge. Reported, not resolved by preferring a side.")

        solver = ValuePlacementSolver(accesses, all_preds, all_succs, delta,
                                      prog.entry, first, modulus=None)
        return [PendingMark(Mark("gr_increment", {"unit": unit,
                                                  "chunks": pl.to_value - pl.from_value,
                                                  "to_chunk": pl.to_value}),
                            Region(pl.block, after=pl.after, before=pl.before,
                                   policy=EARLIEST))
                for pl in solver.solve(on_conflict)]


def _topo_order_gr(prog, fwd_succs, fwd_preds):
    """Topological order with back edges cut -- a block after every predecessor that feeds it."""
    indeg = {l: len(fwd_preds[l]) for l in prog.blocks}
    ready = [l for l in prog.blocks if indeg[l] == 0]
    order, seen = [], set()
    while ready:
        lab = ready.pop(0)
        if lab in seen:
            continue
        seen.add(lab)
        order.append(lab)
        for t in fwd_succs[lab]:
            indeg[t] -= 1
            if indeg[t] <= 0 and t not in seen:
                ready.append(t)
    return order + [l for l in prog.blocks if l not in seen]
