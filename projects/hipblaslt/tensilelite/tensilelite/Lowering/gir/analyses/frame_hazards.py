# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FrameHazards -- the LDS hazard edges, per rotation frame.

Aliasing is a CONCRETE storage-id intersection, not a `(gdelta_a - gdelta_b) % ring` residue, and
the trip gap is a property of the frame graph rather than a number recovered from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nodes import Move
from ..analysis import Analysis
from .frame_map import FrameMap
from .lds_buffers import LdsBufferIds

#: A hazard is an ordered pair of accesses to one location, at least one a write.
RAW = "RAW"        # write -> read   (residency)
WAR = "WAR"        # read  -> write  (rotation anti-dep)
WAW = "WAW"        # write -> write  (rotation output-dep)


@dataclass(frozen=True)
class SharedTouch:
    """One shared-space touch by one instruction."""
    block:    str
    pos:      int          # index in the block body -- program order within the block
    inst:     object
    ref:      object
    is_write: bool
    operand:  str
    regions:  tuple = ()   # per region axis, the SET of region values this access MAY touch

    @property
    def storage(self):
        """What LDS storage this access names: `(operand, regions)`."""
        return (self.operand, self.regions)


@dataclass(frozen=True)
class Hazard:
    """One ordered pair that touches the same storage, and how far apart the two instances run."""
    kind:        str        # RAW | WAR | WAW
    producer:    SharedTouch
    consumer:    SharedTouch
    ring:        int        # the buffer count this pair rotates through
    gap:         int        # 0 = one execution of the pair; >=1 = a later instance of the consumer
    cross_agent: bool
    cross_block: bool = False

    @property
    def same_trip(self) -> bool:
        return self.gap == 0

    @property
    def in_program_order(self) -> bool:
        """Is the producer emitted before the consumer within the block?

        Only meaningful for a `same_trip` pair, and it is what tells a same-wave in-place refill
        from a violation of it: the steady body puts copies last so the vacating read precedes."""
        return (self.producer.block == self.consumer.block
                and self.producer.pos < self.consumer.pos)


class FrameHazardSet:
    """Query API over the hazard result."""

    def __init__(self, edges):
        self._edges = tuple(edges)

    def __iter__(self):
        return iter(self._edges)

    def edges(self, *, kind=None, block=None, cross_agent=None):
        """Hazards, optionally filtered.  `block` matches either endpoint's block."""
        out = self._edges
        if kind is not None:
            out = [h for h in out if h.kind == kind]
        if block is not None:
            out = [h for h in out if block in (h.producer.block, h.consumer.block)]
        if cross_agent is not None:
            out = [h for h in out if h.cross_agent == cross_agent]
        return tuple(out)

    def needing_fence(self):
        """The edges a proc-scoped fence must cover -- the cross-wave ones.

        A same-wave edge is discharged by program order plus the completion counter, so fencing it
        would be pure cost.  This is the ONLY place the fence/no-fence decision is made."""
        return self.edges(cross_agent=True)


def hazard_kind(producer: SharedTouch, consumer: SharedTouch):
    if producer.is_write and consumer.is_write:
        return WAW
    if producer.is_write:
        return RAW
    if consumer.is_write:
        return WAR
    return None                                  # read -> read is not a hazard


def _disjoint_storage(a, b) -> bool:
    """Do these two accesses provably touch DIFFERENT LDS storage?

    Different operand: different tensor.  Same operand: disjoint iff some region axis has
    non-overlapping possible values.  Sets, not equality, so an unpinned access is disjoint from
    nothing.  A precision check only -- the pairing itself is a storage-id intersection."""
    if a.operand != b.operand:
        return True
    return any(not (ra & rb) for ra, rb in zip(a.regions, b.regions))


def _accesses(prog, geometry, scheduling=False):
    """Every shared-space touch, by block, in program order."""
    out = {}
    for blk in prog.blocks.values():
        acc = []
        for pos, inst in enumerate(blk.body):
            if not isinstance(inst, Move):
                continue
            for ref, is_write in ([(r, False) for r in inst.srcs]
                                  + [(r, True) for r in inst.dsts]):
                if ref.tile.space != "shared":
                    continue
                regions = (geometry.frame_regions_of(ref, is_write) if scheduling
                           else geometry.workgroup_regions_of(ref, is_write))
                acc.append(SharedTouch(block=blk.label, pos=pos, inst=inst, ref=ref,
                                       is_write=is_write, operand=ref.tile.operand,
                                       regions=regions))
        out[blk.label] = acc
    return out


def _walk(node, here, fm, state, reg, emit=None, scheduling=False):
    """Run one `(block, frame)` in program order; return what is live on its way out.

    The state is PER FRAME -- `{frame: {slot: (writes, reads)}}` -- and every record carries HOW
    MANY FRAME STEPS AGO it ran.  A slot is absolute, so an access pairs against what every frame
    recorded for it; the frame decides which slot a ref names and where this access is filed.  A
    write kills the slot in every frame: the bytes are gone whoever wrote them."""
    _, frame = node
    st = {f: {s: (dict(w), dict(r)) for s, (w, r) in per.items()} for f, per in state.items()}
    for touch in here:
        me = (frame, id(touch))
        touched = (fm.frame_touches(touch.ref, frame, touch.is_write) if scheduling
                   else fm.workgroup_touches(touch.ref, frame, touch.is_write))
        for s in touched:
            if emit is not None:
                for per in st.values():
                    w, r = per.get(s, ({}, {}))
                    # `r` are the readers OF `w` -- this record's own, not a slot-wide union.
                    for (_frame, t), d in (r.items() if touch.is_write else ()):
                        emit(reg[t], touch, d)
                    # EVERY read owes the write its own RAW, not just the first.  An earlier read
                    # being ISSUED does not retire a completion counter, so the copy can still be
                    # in flight -- read-after-read discharges a register dependence, never this.
                    for (_frame, t), d in w.items():
                        emit(reg[t], touch, d)
            if touch.is_write:
                for per in st.values():
                    per.pop(s, None)
                st.setdefault(frame, {})[s] = ({me: 0}, {})
            else:
                filed = False
                for per in st.values():
                    if s in per:
                        per[s][1][me] = 0
                        filed = True
                if not filed:
                    st.setdefault(frame, {})[s] = ({}, {me: 0})
    return st


def _step(out):
    """`out` one frame further away: every record ages by one edge of the frame graph."""
    return {f: {s: ({k: d + 1 for k, d in w.items()}, {k: d + 1 for k, d in r.items()})
                for s, (w, r) in per.items()}
            for f, per in out.items()}


def _reaching(prog, fm, acc, reg, scheduling=False):
    """The per-frame records live at the ENTRY of each `(block, frame)`, with their distance.

    Walk from the entry, record what each frame did, carry it into the successor one step older,
    record that frame's own, and so on.  The meet is union keeping the SMALLEST distance, so a
    record's number is the frame steps to its NEAREST occurrence -- which is what makes the
    fixpoint terminate even though going round the loop keeps ageing it."""
    ins = {n: {} for n in fm.nodes()}
    work = list(fm.nodes())
    while work:
        node = work.pop()
        aged = _step(_walk(node, acc.get(node[0], ()), fm, ins[node], reg,
                            scheduling=scheduling))
        for succ in fm.successors(node):
            if succ not in ins:
                continue
            merged, changed = {}, False
            for f in set(ins[succ]) | set(aged):
                dst, src = ins[succ].get(f, {}), aged.get(f, {})
                per = {}
                for s in set(dst) | set(src):
                    dw, dr = dst.get(s, ({}, {}))
                    sw, sr = src.get(s, ({}, {}))
                    w, r = dict(dw), dict(dr)
                    for have, add in ((w, sw), (r, sr)):
                        for k, d in add.items():
                            if k not in have or d < have[k]:
                                have[k] = d
                                changed = True
                    per[s] = (w, r)
                merged[f] = per
            if changed:
                ins[succ] = merged
                work.append(succ)
    return ins


class _FrameHazards(Analysis):
    scheduling = False

    def run(self, prog, am):
        storage = am.get(LdsBufferIds(), prog)
        fm = am.get(FrameMap(), prog)
        distributed = prog.meta.get("agent_distributed", {}) or {}
        acc = _accesses(prog, storage.geometry, self.scheduling)

        edges, index = [], set()
        #: Every read of shared storage.
        reads = tuple(t for touches in acc.values() for t in touches if not t.is_write)

        def _already_ordered(a, x, ring) -> bool:
            """Do the RAW and WAR through an interposed read already order these two writes?

            Only on a ring deep enough to put them in DIFFERENT trips.  At `S == 1` the refill hits
            the buffer being read in its own trip, so the read's own edges are same-trip and order
            nothing across the wrap -- that is the phase machine's blind spot, and the WAW is the
            only edge that sees it.
            """
            return ring > 1 and any(not _disjoint_storage(a, r) and not _disjoint_storage(x, r)
                                    for r in reads)

        def emit(a, x, gap):
            kind = hazard_kind(a, x)
            ring = max(1, storage.depth_of(a.operand))
            if kind is None or (kind == WAW and _already_ordered(a, x, ring)):
                return
            key = (kind, id(a.ref), id(x.ref), gap > 0)
            if key not in index:
                index.add(key)
                edges.append(Hazard(
                    kind=kind, producer=a, consumer=x, ring=ring, gap=gap,
                    cross_agent=bool(distributed.get(a.operand) or distributed.get(x.operand)),
                    cross_block=a.block != x.block))

        reg = {id(t): t for touches in acc.values() for t in touches}
        live = _reaching(prog, fm, acc, reg, self.scheduling)
        for node in fm.nodes():
            _walk(node, acc.get(node[0], ()), fm, live[node], reg, emit,
                  scheduling=self.scheduling)
        return FrameHazardSet(edges)


class FrameHazards(_FrameHazards):
    """Workgroup-complete hazards used by GIR waits and fences.

    A static instruction executes on every agent, so an agent-relative read contributes every
    physical region those agents select. This is the completion/fence footprint, not a
    representative scheduling address.
    """


class SchedulingFrameHazards(_FrameHazards):
    """Exact representative-agent address hazards used only to constrain scheduling."""

    scheduling = True
