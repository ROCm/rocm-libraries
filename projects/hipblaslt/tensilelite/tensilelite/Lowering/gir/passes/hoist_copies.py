# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
HoistCopiesPass -- issue each movement at its earliest hazard-legal slot.
"""

from __future__ import annotations

from ..nodes import Move
from ..analyses.frame_hazards import SchedulingFrameHazards, RAW
from ..analyses.reg_hazards import RegHazards
from .base import Pass


def _is_copy(inst):
    """A Move that FILLS shared storage -- the instruction whose issue point we are choosing."""
    return isinstance(inst, Move) and any(r.tile.space == "shared" for r in inst.dsts)


def _is_read(inst):
    """A Move that fills REGISTERS from shared: the other half of the pipeline."""
    return isinstance(inst, Move) and any(r.tile.space == "register" for r in inst.dsts)


def _read_operand(inst):
    """The operand this read fills -- the FIFO a read orders against.

    One operand's register ring is refilled in read order, so its own reads may not pass one
    another. A read of a DIFFERENT operand shares no ring and no LDS token with them, so it is
    free to cross and land on its own earliest register hazard point."""
    ref = next((ref for ref in inst.dsts if ref.tile.space == "register"), None)
    return ref.tile.operand if ref is not None else None


def _is_split_copy(inst):
    """A shared fill pinned to a split-region coordinate."""
    return _is_copy(inst) and any(ref.tile.coord for ref in inst.dsts
                                  if ref.tile.space == "shared")


def _earliest(inst, label, *hazard_sets):
    """The first index `inst` may occupy in its block, over every hazard set that binds it.

    Only a same-trip edge already in program order constrains it: that is a reader of the storage
    this movement overwrites, and it must stay behind that reader.  An edge at a trip distance is
    discharged by the trip boundary, not by this movement's position."""
    return max((h.producer.pos + 1 for hazards in hazard_sets for h in hazards
                if h.consumer.inst is inst and h.consumer.block == label
                and not h.cross_block and h.same_trip and h.in_program_order),
               default=0)


def _copy_earliest(inst, label, lds, reg):
    """A copy's earliest slot.

    Every shared refill follows the same-trip LDS readers it overwrites. When one of those reads
    also produces registers consumed later in this block, keeping the refill behind its first
    consumer lets that consumer's completion wait discharge the WAR as well. This is derived only
    from hazards; operand names and prefetch policy are irrelevant.
    """
    floors = []
    for hazard in lds:
        if (hazard.consumer.inst is not inst or hazard.consumer.block != label
                or hazard.cross_block or not hazard.same_trip or not hazard.in_program_order):
            continue
        ordinary = hazard.producer.pos + 1
        consumers = [
            edge.consumer.pos + 1
            for edge in reg
            if edge.kind == RAW
            and edge.producer.inst is hazard.producer.inst
            and edge.consumer.block == label
            and not edge.cross_block
            and edge.same_trip
            and edge.in_program_order
        ]
        floors.append(max(ordinary, min(consumers)) if consumers else ordinary)

    return max(floors, default=0)


def _read_runs(body, floors):
    """One operand's reads, cut where the floor jumps clear of the previous read.

    A trip's reads and the next trip's are one operand but not one span: the later generation
    cannot start until the ring slot it reuses is free, which is most of a block away. Spreading
    them as one run would average that gap and pin the whole generation at the jump."""
    runs = {}
    for i in sorted(floors):
        operand = runs.setdefault(_read_operand(body[i]), [[]])
        if operand[-1] and floors[i] > operand[-1][-1]:
            operand.append([])
        operand[-1].append(i)
    return [run for operand in runs.values() for run in operand]


def _floor_packed(indices, floors):
    """Every read at the earliest slot its own floor and its predecessors leave it.

    This is the hoist on its own: it finds how far forward the run reaches, which is the one thing
    the spread cannot work out for itself."""
    packed, floor = {}, 0
    for i in indices:
        floor = max(floors[i], floor)
        packed[i] = min(i, floor)
        floor = packed[i] if packed[i] < i else i + 1
    return packed


def _spread_reads(body, floors):
    """A slot for every read: packed to its floor, then spread over the span that opens up.

    The two halves are one derivation, not a choice between them. Packing alone answers how early
    the run can start but bunches every member there; the span it reaches -- that start to where
    the run already ended -- is then divided by the member count, so each read sits one even gap
    along, against an mma that carries no read yet. The packed slot stays each read's floor and its
    own index stays the ceiling, so the spread never pushes a read below the hoist or past where it
    started."""
    slots = {}
    for indices in _read_runs(body, floors):
        packed = _floor_packed(indices, floors)
        start, count = packed[indices[0]], len(indices)
        span = max(0, indices[-1] - start)
        for position, i in enumerate(indices):
            slots[i] = min(i, max(packed[i], start + position * span // count))
    return slots


def hoisted(blk, lds, reg=(), group=True, reads=False, copies=True):
    """`blk.body` with the movements moved up, or None if none moves.

    `group` sends unsplit copies to one point, the latest of their individual earliests, so they
    keep sharing a barrier. A region-pinned TDM copy keeps its own hazard floor so split movement
    follows the WMMA traversal instead of being batched across regions. `reads` SPREADS the
    register reads over the slack each one has instead of stacking them on their floor, keeping
    one operand's reads in order and letting a read cross another operand's."""
    body = blk.body
    at = ({i: _copy_earliest(inst, blk.label, lds, reg)
           for i, inst in enumerate(body) if _is_copy(inst)}
          if copies else {})
    if group and at:
        grouped = {i: slot for i, slot in at.items() if not _is_split_copy(body[i])}
        if grouped:
            slot = max(grouped.values())
            if slot < min(grouped):
                at.update({i: slot for i in grouped})
            else:
                for i in grouped:
                    at.pop(i)
    copies_at = sorted(at)
    if reads:
        # EVERY generation participates. A next-generation read has no producer in this block --
        # its source was filled a trip ago -- so its floor is the register WAR alone, and holding
        # it back is what piles the trip's reads into a burst at the bottom of the loop.
        floors = {i: _earliest(inst, blk.label, lds, reg)
                  for i, inst in enumerate(body) if _is_read(inst)}
        at.update(_spread_reads(body, floors))
    # COPY ORDER IS NOT FREE, INCLUDING A COPY THAT DOES NOT MOVE.  Split copies have individual
    # floors while an unsplit movement may stay put; without putting that stationary copy into the
    # ordering scan, a later split copy can hoist across it and reverse one peel generation
    # (MX,A0,A1 -> A0,A1,MX). No downstream token can recover FIFO order after that reversal.
    copy_movers = set(copies_at)
    floor = -1
    for i, inst in enumerate(body):
        if not _is_copy(inst):
            continue
        if i not in copy_movers:
            floor = i + 1
            continue
        at[i] = max(at[i], floor)
        floor = i + 1 if at[i] >= i else at[i]
    at = {i: s for i, s in at.items() if s < i}
    if not at:
        return None
    # Indices are read off the ORIGINAL body: every producer precedes the movement it blocks, so
    # lifting the movements out does not shift any of them.  Sorted, so two landing on one slot
    # keep their original order.
    pending = {}
    for i in sorted(at):
        pending.setdefault(at[i], []).append(body[i])
    out = []
    for i, inst in enumerate(body):
        out.extend(pending.pop(i, ()))
        if i not in at:
            out.append(inst)
    for s in sorted(pending):
        out.extend(pending[s])
    return out


class HoistCopiesPass(Pass):
    """See module docstring.  Runs before the tokens so every later analysis sees this order."""

    def __init__(self, group=True, reads=False, copies=True):
        self.group = group
        self.reads = reads
        self.copies = copies

    def run(self, prog, am):
        def apply(*, copies, reads):
            lds, reg = am.get(SchedulingFrameHazards(), prog), am.get(RegHazards(), prog)
            moved = False
            for blk in prog.blocks.values():
                order = hoisted(blk, lds, reg, self.group, reads, copies)
                if order is not None:
                    blk.body[:] = order
                    moved = True
            if moved:
                prog.bump()
                am.invalidate()
            return moved

        # A hoisted read shortens the non-read run before a RAW fence, which is what the old
        # whole-program revert measured and refused -- on every kernel, so the read arm never
        # shipped. The fence is StinkyTofu's to place now (`_raw_fence_profile` says so), and it
        # places it from the frame contract, not from how much work precedes the consumer.
        if self.reads:
            apply(copies=False, reads=True)
        if self.copies:
            apply(copies=True, reads=False)
