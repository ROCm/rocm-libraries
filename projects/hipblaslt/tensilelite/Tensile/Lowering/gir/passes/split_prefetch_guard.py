# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
SplitPrefetchGuardPass -- the peeled prefetch generation a short trip never consumes becomes its
own block, so the branch point is in the CFG rather than implied.
"""

from __future__ import annotations

from ..nodes import (Block, Bound, CondGoto, Goto, Mark, Move, Pred, copy_unit,
                     descriptor_unit)
from .base import Pass
from .fold_short_path import folded
from .scaffold_map import _SCAFFOLD_LABEL


#: Only the last peeled generation is guardable, so the two new blocks have fixed names; which
#: generation they hold is `meta['prefetch_guard']['gen']`.
_GUARDED_BLOCK, _SKIP_BLOCK, _JOIN_BLOCK = "prefetch_peel", "prefetch_skip", "prologue_join"


def peel_generation(inst):
    """The absolute peel generation a copy Move fetches, or None if `inst` is not a peel copy."""
    if not isinstance(inst, Move):
        return None
    members, dst_refs = copy_unit(inst)
    if members is None or not dst_refs:
        return None
    return dst_refs[0].abs_gen


def _guarded_generation(blk):
    """The last peel generation, when the block fills more than one; else None.

    Only the last is guardable: TensileLite's ladder tests one trip count, and a deeper peel needs
    the chain that #150 owns.
    """
    gens = {g for g in (peel_generation(i) for i in blk.body) if g is not None}
    return max(gens) if len(gens) == 2 else None


class SplitPrefetchGuardPass(Pass):
    """See module docstring.  Moves the last generation's copies -- and ONLY those -- into a
    guarded block; their `gr_increment` and copy-hop swaps stay on the path both arms take, so
    the two predecessors of the join leave the descriptor on the same chunk.
    """

    def run(self, prog, am):
        # A KEPT `T < M` arm is a second, scaffold-owned exit from the peel, and its unemitted
        # edges leave the placement solver free to route a value out and back through the guarded
        # block.  Only a folded program has the single peel exit this split assumes.
        if not folded(prog):
            return
        entry = prog.blocks.get(prog.entry)
        if entry is None or entry.phase != "prologue":
            return
        gen = _guarded_generation(entry)
        if gen is None:
            return

        guarded = [i for i in entry.body if peel_generation(i) == gen]
        first = next(n for n, i in enumerate(entry.body) if peel_generation(i) == gen)
        keep = {id(i) for i in guarded}
        head, rest = entry.body[:first], [i for i in entry.body[first:] if id(i) not in keep]

        # Both arms must enter the join with the same address AND buffer, and neither may host the
        # step the other misses -- so pin both before the branch and again at the join, keyed on
        # the DESCRIPTOR, which is what the two analyses walk.
        descs = list(dict.fromkeys(descriptor_unit(prog, copy_unit(i)[0]) for i in guarded))

        def _pins():
            return ([Mark("chunk_pin", {"unit": d, "chunk": gen}) for d in descs] +
                    [Mark("buffer_pin", {"hop": "copy", "unit": d, "gen": gen}) for d in descs])

        head, rest = head + _pins(), _pins() + rest

        pgr_lab, skip_lab, join_lab = _GUARDED_BLOCK, _SKIP_BLOCK, _JOIN_BLOCK
        term, succs = entry.term, entry.succs

        join = Block(phase=join_lab, body=rest, term=term, role=entry.role,
                     preds=(skip_lab, pgr_lab), succs=succs,
                     gen_rel=entry.gen_rel, chunk_base=entry.chunk_base,
                     path_chunk_base=dict(entry.path_chunk_base))
        # `role` names WHICH AGENTS run the block, and this guard is the scalar trip count: every
        # wave takes the same arm.  Narrowing it would read as a path some waves skip.
        pgr = Block(phase=pgr_lab, body=guarded, term=Goto(join_lab), role=entry.role,
                    preds=(entry.label,), succs=(join_lab,),
                    gen_rel=entry.gen_rel, chunk_base=entry.chunk_base)
        skip = Block(phase=skip_lab, body=[], term=Goto(join_lab), role=entry.role,
                     preds=(entry.label,), succs=(join_lab,),
                     gen_rel=entry.gen_rel, chunk_base=entry.chunk_base)

        # Taken == the short trip, which skips the generation -- the sense TensileLite's ladder uses.
        pred = Pred("T", "==", Bound(const=gen), _SCAFFOLD_LABEL["prefetch_guard"])
        entry.body, entry.term = head, CondGoto(pred, skip_lab, pgr_lab)
        entry.succs = (skip_lab, pgr_lab)

        for lab in succs:
            blk = prog.blocks.get(lab)
            if blk is None:
                continue
            blk.preds = tuple(join_lab if p == entry.label else p for p in blk.preds)
            if entry.label in blk.path_chunk_base:
                blk.path_chunk_base[join_lab] = blk.path_chunk_base.pop(entry.label)

        prog.add_block(pgr)
        prog.add_block(skip)
        prog.add_block(join)
        prog.meta["prefetch_guard"] = {
            "gen": gen, "block": pgr_lab, "skip": skip_lab,
            "join": join_lab, "copies": len(guarded)}
        prog.bump()
