# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FrameMap -- which storage each access works on, under which rotation phase.

A FRAME is a phase vector, one entry per Gen; a block entered several ways runs under several.  So
the frame set is a forward dataflow over SETS of phase vectors whose meet is UNION -- nothing is
collapsed onto an invented base.
"""

from __future__ import annotations

from ..analysis import Analysis
from ..nodes import LoopBack, LoopCopy
from .cfg import BackEdges, successors
from .lds_buffers import LdsBufferIds


class Frame:
    """An immutable rotation phase vector: `{gen id: phase}`, absent entries reading 0."""

    __slots__ = ("phases",)

    def __init__(self, phases=()):
        self.phases = tuple(sorted(phases.items() if isinstance(phases, dict) else phases))

    def of(self, gen_id) -> int:
        for g, p in self.phases:
            if g == gen_id:
                return p
        return 0

    def as_dict(self) -> dict:
        return dict(self.phases)

    def __eq__(self, other):
        return isinstance(other, Frame) and self.phases == other.phases

    def __hash__(self):
        return hash(self.phases)

    def __lt__(self, other):
        return self.phases < other.phases

    def __repr__(self):
        return ("entry" if not self.phases
                else "phase(" + ", ".join(f"ring{g}={p}" for g, p in self.phases) + ")")


def _rings(prog):
    """Every Gen the program rotates through: `{gen id: depth}`."""
    out = {}
    for blk in prog.blocks.values():
        for phi in blk.phis:
            out[phi.gen.id] = max(1, phi.gen.ring)
        for xf in blk.xfers:
            out[xf.gen.id] = max(1, xf.ring)
        for edge_xfers in (blk.edge_xfers or {}).values():
            for xf in edge_xfers:
                out[xf.gen.id] = max(1, xf.ring)
        for inst in blk.body:
            for ref in tuple(getattr(inst, "srcs", ())) + tuple(getattr(inst, "dsts", ())):
                gen = getattr(ref, "gen", None)
                if gen is not None:
                    out[gen.id] = max(1, gen.ring)
    return out


def _edge_values(prog, back):
    """Per edge, `{gen id: (value, relative)}` -- the table `GirFrameAnalysis` applies: an edge
    either ASSIGNS a phase or ADVANCES one.

    A back edge advances by the steady trip's transfer. On the exit, drain references are stated
    in the T-relative frame `gdelta = i - M`; convert that frame exactly once by adding `M`
    (`-dst.gen_rel`) to the trip transfer. Using the trip transfer alone is correct accidentally
    when M is a multiple of the ring (PGR2/ring2), but loses the final copy at PGR1/ring2."""
    out = {}
    for lab, blk in prog.blocks.items():
        for succ_lab in (blk.succs or ()):
            dst = prog.blocks.get(succ_lab)
            if dst is None:
                continue
            table = {}
            is_back = back.is_back_edge(lab, succ_lab)
            is_loop_exit = isinstance(blk.term, (LoopBack, LoopCopy)) \
                and succ_lab == blk.term.exit_target
            if is_back or is_loop_exit:
                edge_xfers = (blk.edge_xfers or {}).get(succ_lab)
                for xf in (edge_xfers if edge_xfers is not None else blk.xfers):
                    advance = int(xf.adv)
                    if is_loop_exit:
                        advance -= int(dst.gen_rel or 0)
                    table[xf.gen.id] = (advance % max(1, int(xf.ring)), True)
            for phi in dst.phis:
                incoming = dict(phi.incomings)
                if incoming:
                    if lab not in incoming:
                        raise RuntimeError(
                            "FrameMap: phi for Gen %d in %s has no input from %s"
                            % (phi.gen.id, dst.label, lab)
                        )
                    value = incoming[lab]
                else:
                    value = phi.entry_val
                if value is None:
                    continue          # forwarded: this edge's own transfer already states it
                table[phi.gen.id] = (int(value) % max(1, int(phi.gen.ring)), False)
            out[(lab, succ_lab)] = table
    return out


def _advance(frame, table, rings):
    """`frame` across one edge, by the table above: assign, or advance within the ring."""
    phases = frame.as_dict()
    for gen_id, (value, relative) in table.items():
        ring = max(1, int(rings.get(gen_id, 1)))
        phases[gen_id] = ((phases.get(gen_id, 0) + value) % ring) if relative else (value % ring)
    return Frame(phases)


class FrameMapping:
    """Query API: the frames a block runs under, and the storage an access touches in each."""

    def __init__(self, frames, entrances, edges, storage, gen_by_region=()):
        self._frames = {b: tuple(sorted(fs)) for b, fs in frames.items()}
        self._entrances = {b: _classes(fs) for b, fs in entrances.items() if fs}
        self._edges = dict(edges)            # node -> tuple[node]
        self._storage = storage
        self._gen_by_region = dict(gen_by_region)   # (operand, region) -> gen id
        self._frame_touch_cache = {}
        self._workgroup_touch_cache = {}

    def frames(self, block):
        """EVERY phase this block is entered on -- not a representative."""
        return self._frames.get(block, ())

    def entrances(self, block):
        """One frame per DISTINCT entrance picture.

        A back edge is not an entrance -- it is this block one rotation on.  Neither are two
        forward edges that differ by a UNIFORM rotation of every ring: that is the same relation
        with the ids relabelled, which the token already survives.  Only a non-uniform difference
        is a second picture."""
        return self._entrances.get(block, (Frame(),))

    def nodes(self):
        """Every `(block, frame)` the program can execute."""
        return tuple((b, f) for b in self._frames for f in self._frames[b])

    def successors(self, node):
        """The `(block, frame)` nodes one edge on from `node`."""
        return self._edges.get(node, ())

    def generation(self, ref, frame, coords=None):
        """The concrete rotation phase `ref` names in `frame`, or None if it carries none.

        `coords` names ONE region, and that region's own `Gen` sets the phase.  A Ref carries a
        single `gen` while `_loop_carried_gens` makes one per `(operand, region)`, so reading the
        Ref's alone pins every region to the leader's rotation -- a region whose reader and writer
        then sit on different generations has no rotation between them at all."""
        ring = max(1, self._storage.depth_of(ref.tile.operand))
        gen = getattr(ref, "gen", None)
        if gen is not None:
            gen_id = gen.id
            if coords is not None and len(coords) == 1:
                gen_id = self._gen_by_region.get((ref.tile.operand, int(coords[0])), gen_id)
            return (frame.of(gen_id) + int(ref.gdelta)) % ring
        abs_gen = getattr(ref, "abs_gen", None)
        return None if abs_gen is None else int(abs_gen) % ring

    def frame_touches(self, ref, frame, is_write):
        """Concrete storage selected by this access's frame and agent-relative address."""
        key = (id(ref), frame, bool(is_write))
        hit = self._frame_touch_cache.get(key)
        if hit is None:
            hit = self._storage.frame_at(
                ref, lambda coords: self.generation(ref, frame, coords), is_write)
            self._frame_touch_cache[key] = hit
        return hit

    def workgroup_touches(self, ref, frame, is_write):
        """Concrete storage touched across every physical agent executing this access."""
        key = (id(ref), frame, bool(is_write))
        hit = self._workgroup_touch_cache.get(key)
        if hit is None:
            hit = self._storage.workgroup_frame_at(
                ref, lambda coords: self.generation(ref, frame, coords), is_write)
            self._workgroup_touch_cache[key] = hit
        return hit

def _classes(frames):
    """`frames` with those differing by one uniform rotation of every ring collapsed to one."""
    keep = []
    for f in sorted(frames):
        shifts = {(p - q) for (g, p), (_h, q) in zip(f.phases, keep[0].phases)} if keep else None
        if keep and len(shifts) == 1:
            continue
        keep.append(f)
    return tuple(keep)


class FrameMap(Analysis):
    """See module docstring.  Pure; returns a `FrameMapping`."""

    def run(self, prog, am):
        storage = am.get(LdsBufferIds(), prog)
        back = am.get(BackEdges(), prog)
        succ = successors(prog)

        # EVERY frame carries EVERY ring, so two frames are equal iff they denote the same phase.
        # A partial vector would make the entry frame and the all-zero loop frame distinct objects
        # for one phase, splitting the node set and inventing cross-frame edges between them.
        rings = _rings(prog)
        values = _edge_values(prog, back)
        base = Frame({g: 0 for g in rings})

        order = [blk.label for blk in prog.walk_rpo()]
        frames = {lab: set() for lab in prog.blocks}
        entrances = {lab: set() for lab in prog.blocks}
        frames[prog.entry].add(base)
        entrances[prog.entry].add(base)

        work = [prog.entry]
        while work:
            lab = work.pop(0)
            src = prog.blocks[lab]
            for s in succ.get(lab, ()):
                is_back = back.is_back_edge(lab, s)
                table = values.get((lab, s), {})
                new = {_advance(f, table, rings) for f in frames[lab]}
                if not is_back:
                    entrances[s] |= {_advance(f, table, rings)
                                     for f in entrances[lab]} or new
                fresh = new - frames[s]
                if not fresh:
                    continue
                frames[s] |= fresh
                if s not in work:
                    work.append(s)
            work.sort(key=lambda b: order.index(b) if b in order else len(order))

        edges = {}
        for lab, blk in prog.blocks.items():
            for f in frames[lab]:
                edges[(lab, f)] = tuple((s, _advance(f, values.get((lab, s), {}), rings))
                                        for s in succ.get(lab, ()))

        generation_regions = (prog.meta or {}).get("generation_regions", {}) or {}
        gen_by_region = {(str(fact.get("operand")), int(fact.get("region", 0))): int(gen_id)
                         for gen_id, fact in generation_regions.items()}
        return FrameMapping(frames, entrances, edges, storage, gen_by_region)
