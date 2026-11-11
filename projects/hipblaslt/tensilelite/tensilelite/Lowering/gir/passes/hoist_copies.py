# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
HoistCopiesPass -- issue each copy at its earliest hazard-legal slot, and each read at the point
in the wmma stream its own consumer reaches.
"""

from __future__ import annotations

from ..nodes import Mma, Move
from ..analyses.frame_hazards import SchedulingFrameHazards, RAW
from ..analyses.reg_hazards import RegHazards
from .base import Pass


#: WMMA issued between a read's hazard floor and its slot. The floor only ORDERS the read behind
#: the last wmma holding the register; this covers that wmma's own source read.
WAR_SLACK = 8
#: Reads issued back to back, and the wmma issued between one such burst and the next.
BURST, BURST_STRIDE = 2, 2


def _is_copy(inst):
    """A Move that FILLS shared storage -- the instruction whose issue point we are choosing."""
    return isinstance(inst, Move) and any(r.tile.space == "shared" for r in inst.dsts)


def _is_read(inst):
    """A Move that fills REGISTERS from shared: the other half of the pipeline."""
    return isinstance(inst, Move) and any(r.tile.space == "register" for r in inst.dsts)


def _read_ring(inst):
    """The register ring this read refills -- the FIFO a read orders against.

    One ring is refilled in read order, so its own reads may not pass one another. A read of a
    DIFFERENT ring shares no register and no LDS token with them, so it is free to cross. The ring
    is operand plus slot, not operand alone: one operand's slots are consumed by different wmma
    groups and so are refilled at different points."""
    ref = next((ref for ref in inst.dsts if ref.tile.space == "register"), None)
    return (ref.tile.operand, ref.group, ref.slot) if ref is not None else None


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


def _latest(inst, label, limit, *hazard_sets):
    """The last index `inst` may occupy in its block; `limit` when nothing in the block binds it.

    The dual of `_earliest`, and what lets a read SINK: the edges OUT of the movement -- the wmma
    that consumes the register it fills, the refill that overwrites that register, the copy that
    overwrites the LDS it reads."""
    return min((h.consumer.pos for hazards in hazard_sets for h in hazards
                if h.producer.inst is inst and h.producer.block == label
                and not h.cross_block and h.same_trip and h.in_program_order),
               default=limit)


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
    """One ring's reads, cut where the floor jumps clear of the previous read.

    A trip's reads and the next trip's are one ring but not one span: the later generation cannot
    start until the slot it reuses is free, which is most of a block away. Keeping them as one run
    would place the whole generation off the first member's floor."""
    runs = {}
    for i in sorted(floors):
        ring = runs.setdefault(_read_ring(body[i]), [[]])
        if ring[-1] and floors[i] > ring[-1][-1]:
            ring.append([])
        ring[-1].append(i)
    return [run for ring in runs.values() for run in ring]


def _wmma_ruler(body):
    """Every wmma's index, and the wmma count before each index.

    A read is placed in wmma issued rather than in body positions, so the ruler does not move as
    the reads and copies around it do."""
    at, before, seen = [], [], 0
    for i, inst in enumerate(body):
        before.append(seen)
        if isinstance(inst, Mma):
            at.append(i)
            seen += 1
    before.append(seen)
    return at, before


def _place_reads(body, floors, ceilings):
    """A slot for every read: its ring's turn, `BURST` at a time, `WAR_SLACK` wmma past its floor.

    Rings are refilled in the order their first consumer runs, so what a read fetches is what the
    next wmma wants, and within a ring the reads keep their own order.  The floor only ORDERS a
    read behind the wmma still holding the register, which leaves the wmma's own source read
    uncovered -- hence the slack, which is what keeps the WAR off a wait.  Returned in placement
    order, so reads landing on one slot issue in the order they were chosen."""
    wmma, before = _wmma_ruler(body)
    if not wmma:
        return {}
    runs = _read_runs(body, floors)
    runs.sort(key=lambda run: (min(ceilings[i] for i in run), run[0]))
    slots, cursor, issued = {}, None, 0
    for i in [i for run in runs for i in run]:
        start = before[floors[i]] + WAR_SLACK
        if cursor is None or issued >= BURST or start > cursor:
            cursor, issued = start if cursor is None else max(start, cursor + BURST_STRIDE), 0
        issued += 1
        slot = wmma[cursor] if cursor < len(wmma) else len(body)
        slots[i] = min(max(slot, floors[i]), ceilings[i])
    return slots


def hoisted(blk, lds, reg=(), group=True, reads=False, copies=True):
    """`blk.body` with the movements re-issued, or None if none moves.

    `group` sends unsplit copies to one point, the latest of their individual earliests, so they
    keep sharing a barrier. A region-pinned TDM copy keeps its own hazard floor so split movement
    follows the WMMA traversal instead of being batched across regions. `reads` re-issues the
    register reads in the order the wmma consume them, which a copy's floor cannot do and which
    moves a read EITHER WAY within its own hazard window."""
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
    issue = list(copies_at)
    if reads:
        # EVERY generation participates. A next-generation read has no producer in this block --
        # its source was filled a trip ago -- so its floor is the register WAR alone, and holding
        # it back is what piles the trip's reads into a burst at the bottom of the loop.
        floors = {i: _earliest(inst, blk.label, lds, reg)
                  for i, inst in enumerate(body) if _is_read(inst)}
        ceilings = {i: _latest(body[i], blk.label, len(body), lds, reg) for i in floors}
        placed = _place_reads(body, floors, ceilings)
        at.update(placed)
        issue.extend(placed)
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
    at = {i: s for i, s in at.items() if s != i}
    if not at:
        return None
    # Indices are read off the ORIGINAL body, which holds for a sink as well: a movement lifted
    # out leaves the slots it is measured against where they were.  Ordered by `issue`, so two
    # landing on one slot keep the order they were chosen in.
    rank = {i: n for n, i in enumerate(issue)}
    pending = {}
    for i in sorted(at, key=lambda i: (at[i], rank.get(i, i))):
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
