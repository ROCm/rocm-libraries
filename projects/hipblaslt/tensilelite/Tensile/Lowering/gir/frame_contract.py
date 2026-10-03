# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the GIR frame facts handed to StinkyTofu."""

from __future__ import annotations

from dataclasses import dataclass, field

from .analysis import AnalysisManager
from .analyses.frame_map import FrameMap
from .analyses.lds_buffers import LdsBufferIds
from .analyses.cfg import reachable
from .analyses.trip_domains import TripDomains
from .emit_plan import plan_program
from .nodes import Move, successor_labels


# --- the facts ---------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """A rotating buffer: how deep it is, where it enters, how far each trip advances it."""
    id: int
    ring: int
    entry: int = 0
    advance: int = 0


@dataclass(frozen=True)
class Action:
    """One schedulable unit, and the action whose position its block is pinned to."""
    id: int
    kind: str
    anchor: int


@dataclass(frozen=True)
class SharedAccess:
    """One shared-memory touch an action makes."""
    action: int
    is_write: bool
    operand: int               # id of the (operand, region) pair this touch names
    ring: int
    gen: int = -1              # -1 = not bound to a generation
    gdelta: int = 0
    absolute: int = -1         # -1 = the generation is relative, not pinned
    cross_agent: bool = False


@dataclass(frozen=True)
class Edge:
    """A generation's phase arriving at `dst` from `src`.

    `relative` distinguishes the two ways GIR states it: a `transfer` forwards the predecessor's
    phase shifted by `delta`, an `incoming` pins it to an absolute `value`.
    """
    dst: int
    src: int
    gen: int
    value: int
    relative: bool


@dataclass(frozen=True)
class Requires:
    """An edge is takeable only when `gen` carries one of `values`.

    The mirror of `Edge`: `incoming` ASSIGNS a phase on an edge, `requires` CONSTRAINS one. A
    guard generation records which arm of a correlated branch ran, so a consuming edge can refuse
    the arms that cannot reach it.  Absence of a record constrains nothing.
    """
    dst:    int
    src:    int
    gen:    int
    values: tuple


@dataclass
class Contract:
    generations: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    accesses: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    requires: list = field(default_factory=list)


# --- deriving the facts from a program ---------------------------------------


def _shared_refs(action):
    """The shared-memory refs one action touches, each with whether it writes."""
    source = action.source
    if not isinstance(source, Move):
        return ()
    if action.kind == "read":
        return tuple((ref, False) for ref in source.srcs if ref.tile.space == "shared")
    if action.kind == "copy":
        return tuple((ref, True) for ref in source.dsts if ref.tile.space == "shared")
    return ()


def _generations(prog) -> dict:
    """Every rotating buffer the program names, with its ring, entry phase and per-trip advance."""
    facts = {}

    def ensure(gen, ring=None):
        if gen is None:
            return None
        depth = max(1, int(ring if ring is not None else gen.ring))
        fact = facts.setdefault(int(gen.id), {"ring": depth, "entry": None, "advance": None})
        if fact["ring"] != depth:
            raise RuntimeError("frame contract: inconsistent ring for Gen %d" % gen.id)
        return fact

    for block in prog.blocks.values():
        for phi in block.phis:
            fact = ensure(phi.gen)
            if phi.entry_val is None:
                continue
            value = int(phi.entry_val) % fact["ring"]
            if fact["entry"] not in (None, value):
                raise RuntimeError("frame contract: inconsistent phi entry for Gen %d" % phi.gen.id)
            fact["entry"] = value
        for xfer in block.xfers:
            fact = ensure(xfer.gen, xfer.ring)
            value = int(xfer.adv) % fact["ring"]
            if fact["advance"] not in (None, value):
                raise RuntimeError("frame contract: inconsistent transfer for Gen %d" % xfer.gen.id)
            fact["advance"] = value
        for inst in block.body:
            if isinstance(inst, Move):
                for ref in tuple(inst.srcs) + tuple(inst.dsts):
                    ensure(getattr(ref, "gen", None))

    return {gid: Generation(gid, fact["ring"], fact["entry"] or 0, fact["advance"] or 0)
            for gid, fact in facts.items()}


def _anchor_actions(prog, plans, label, first, seen=()):
    """The action ids that stand for `label` on an incoming (`first`) or outgoing edge.

    The FIRST action is a block's canonical identity: ST scheduling may reorder physical
    instructions, so an outgoing frame must never depend on whichever action is emitted last.
    An empty block forwards the question to its neighbours.
    """
    if label in seen:
        return ()
    actions = plans.get(label, ())
    if actions:
        return (actions[0].action_id,)
    neighbours = prog.blocks[label].succs if first else prog.blocks[label].preds
    out = set()
    for neighbour in neighbours:
        if neighbour in prog.blocks:
            out.update(_anchor_actions(prog, plans, neighbour, first, tuple(seen) + (label,)))
    return tuple(sorted(out))


def _forward_delta(frames, predecessor, destination, gen) -> int:
    """The affine phase transfer of one forwarding phi input.

    `None` in a GenPhi does not mean "no edge semantics": it forwards the value after GIR's edge
    transfer. Deriving that from FrameMap is what stops the serialized contract substituting ST's
    loop-edge convention for GIR's exact dataflow.
    """
    ring = max(1, int(gen.ring))
    deltas = {
        (target.of(gen.id) - source.of(gen.id)) % ring
        for source in frames.frames(predecessor)
        for label, target in frames.successors((predecessor, source))
        if label == destination
    }
    if len(deltas) != 1:
        raise RuntimeError(
            "frame contract: phi transfer %s->%s for Gen %d is not one affine delta: %r"
            % (predecessor, destination, gen.id, sorted(deltas)))
    return next(iter(deltas))


def _region_of(operand, regions):
    """The one region a frame instance selects, or -1 for the whole operand.

    `frame_region_instances_of` already enumerates one instance per physical agent, so an instance
    names a single region. A MAY-set would have to widen every consumer, so refuse instead."""
    axes = [tuple(sorted(axis)) for axis in (regions or ())]
    if not axes:
        return -1
    if len(axes) > 1 or len(axes[0]) != 1:
        raise RuntimeError(
            "frame contract: operand %r selects %r, but one instance must name one region. "
            "Widen the contract to a region set rather than dropping the extra values."
            % (operand, axes))
    return int(axes[0][0])


def _gens_by_region(prog):
    """`(operand, region) -> gen id`, the inverse of `generation_regions`."""
    return {(str(fact.get("operand")), int(fact.get("region", 0))): int(gen_id)
            for gen_id, fact in ((prog.meta or {}).get("generation_regions", {}) or {}).items()}


def _storage_id(operand, region, storage_ids):
    """The id of one `(operand, region)` pair, allocated on first sight.

    Whether an operand is region-split is a property of `region_axes`, so a region of -1 and a
    concrete region never name the same operand and an id compared for equality says everything
    the pair did."""
    return storage_ids.setdefault((operand, region), len(storage_ids))


def _accesses_of(action, storage, distributed, by_region, storage_ids):
    """Every shared touch of one action, one `SharedAccess` per frame-region instance.

    An instance names ITS OWN region's generation.  A `Ref` carries one `gen`, so a read spanning
    every region stamped them all with its own; a region whose reader and writer then sat on
    different generations had no rotation between them at all, which reads as a same-trip WAR
    against the write that recycles the very buffer being read."""
    out = []
    for ref, is_write in _shared_refs(action):
        operand = str(ref.tile.operand)
        gen = getattr(ref, "gen", None)
        absolute = getattr(ref, "abs_gen", None)
        for regions in storage.geometry.frame_region_instances_of(ref, is_write):
            region = _region_of(operand, regions)
            gen_id = -1 if gen is None else int(gen.id)
            if gen is not None and region >= 0:
                gen_id = by_region.get((operand, region), gen_id)
            out.append(SharedAccess(
                action=action.action_id,
                is_write=is_write,
                operand=_storage_id(operand, region, storage_ids),
                ring=max(1, int(storage.depth_of(ref.tile.operand))),
                gen=gen_id,
                gdelta=int(getattr(ref, "gdelta", 0) or 0),
                absolute=-1 if absolute is None else int(absolute),
                cross_agent=bool(distributed.get(ref.tile.operand))))
    return out


def _edges_of(prog, plans, frames):
    """Every phi input, as the phase one block's generation receives from another."""
    out = set()
    for block in prog.blocks.values():
        destinations = _anchor_actions(prog, plans, block.label, True)
        for phi in block.phis:
            for pred, value in phi.incomings:
                sources = _anchor_actions(prog, plans, pred, False)
                if not sources or not destinations:
                    raise RuntimeError(
                        "frame contract: phi input %s->%s has no physical action anchor"
                        % (pred, block.label))
                relative = value is None
                phase = (_forward_delta(frames, pred, block.label, phi.gen) if relative
                         else int(value) % max(1, int(phi.gen.ring)))
                for source in sources:
                    for destination in destinations:
                        out.add(Edge(int(destination), int(source), int(phi.gen.id),
                                     phase, relative))
    return sorted(out, key=lambda e: (e.relative, e.dst, e.src, e.gen, e.value))


#: A guard generation is not a buffer -- it never rotates, and 0 means the arm is undecided.
GUARD_UNDECIDED = 0


def _guards_of(prog, plans, domains, first_free_gen):
    """Guard generations for the branches whose ARM CHOICE some later edge can refuse.

    A branch whose arms all reach every consumer carries no information, so it gets no generation;
    allocating one per correlated branch is what lets the passes that build them stay ignorant of
    each other -- they agree on a generation id, never on a predicate. Counted-loop edge domains
    let an earlier branch choice constrain later back and exit edges by the same general rule.
    """
    def arms_of(label):
        out = []
        for target in dict.fromkeys(successor_labels(prog.blocks[label])):
            if target in prog.blocks:
                out.append((target, domains.admits(label, target)))
        return out

    everywhere = [(lab, arms_of(lab)) for lab in prog.blocks]
    consuming = [(src, dst, domains.admits(src, dst))
                 for src, dst, in ((s, d) for s in prog.blocks for d in dict.fromkeys(
                     successor_labels(prog.blocks[s])) if d in prog.blocks)]

    generations, incomings, requires = {}, [], []
    for label, arms in everywhere:
        if len(arms) < 2 or all(d == domains.universe for _t, d in arms):
            continue
        # Only worth a generation if some edge admits one arm and refuses another -- and only
        # where the arm has already been taken, since an edge before the branch reads `undecided`
        # and would carry a constraint it can never fail.
        downstream = reachable(prog)
        after = {t for t, _d in arms} | {r for t, _d in arms for r in downstream.get(t, ())}
        refusers = [(src, dst, edge) for src, dst, edge in consuming
                    if src in after
                    and any(not (d & edge) for _t, d in arms) and any(d & edge for _t, d in arms)
                    and (src, dst) not in {(label, t) for t, _d in arms}]
        if not refusers:
            continue
        gen_id = first_free_gen + len(generations)
        generations[label] = (gen_id, arms)
        for index, (target, _domain) in enumerate(arms, start=1):
            for source in _anchor_actions(prog, plans, label, False):
                for destination in _anchor_actions(prog, plans, target, True):
                    incomings.append(Edge(int(destination), int(source), gen_id, index, False))
                    # The source is undecided before its first branch. If this edge is revisited
                    # after an arm value has propagated around the CFG, only that same arm may
                    # take it. This is a general guard rule; scaffold labels play no role here.
                    requires.append(Requires(
                        int(destination), int(source), gen_id,
                        (GUARD_UNDECIDED, index)))
        for src, dst, edge in refusers:
            allowed = tuple(sorted(
                [GUARD_UNDECIDED] + [i for i, (_t, d) in enumerate(arms, start=1) if d & edge]))
            for source in _anchor_actions(prog, plans, src, False):
                for destination in _anchor_actions(prog, plans, dst, True):
                    requires.append(Requires(int(destination), int(source), gen_id, allowed))
    guard_gens = {gen_id: Generation(gen_id, len(arms) + 1, GUARD_UNDECIDED, 0)
                  for gen_id, arms in generations.values()}
    return guard_gens, incomings, requires


def build_contract(prog) -> Contract:
    """The frame facts of `prog`, with no GIR CFG topology in them."""
    analyses = AnalysisManager()
    storage = analyses.get(LdsBufferIds(), prog)
    if storage.unresolved:
        raise RuntimeError("frame contract: unresolved shared references %r"
                           % (storage.unresolved[:4],))
    frames = analyses.get(FrameMap(), prog)
    plans = plan_program(prog)

    actions = [action for block_actions in plans.values() for action in block_actions]
    anchors = {action.action_id: block_actions[0].action_id
               for block_actions in plans.values() if block_actions
               for action in block_actions}

    distributed = prog.meta.get("agent_distributed", {}) or {}
    by_region = _gens_by_region(prog)
    storage_ids = {}
    contract = Contract(generations=_generations(prog))
    for action in actions:
        contract.actions[action.action_id] = Action(
            action.action_id, action.kind, anchors[action.action_id])
        contract.accesses.extend(
            _accesses_of(action, storage, distributed, by_region, storage_ids))
    contract.edges = _edges_of(prog, plans, frames)
    domains = analyses.get(TripDomains(), prog)
    first_free = max(contract.generations, default=-1) + 1
    guard_gens, guard_incomings, guard_requires = _guards_of(prog, plans, domains, first_free)
    contract.generations.update(guard_gens)
    contract.edges = list(contract.edges) + guard_incomings
    contract.requires = guard_requires
    return contract


def install_frame_contract(prog, st_module):
    """Hand the contract to StinkyTofu as a STRUCT -- the one call the kernel writer makes.

    Only what belongs to no instruction travels: the generation table, the phi edges and the guard
    constraints.  Actions and the storage they touch ride on the instructions, and nothing is
    encoded, so there is no text format for a writer and a reader to disagree about.
    """
    from rocisa import GirFrameContract

    contract = build_contract(prog)
    out = GirFrameContract()
    for generation in sorted(contract.generations.values(), key=lambda g: g.id):
        out.addGeneration(id=generation.id, ring=generation.ring, entry=generation.entry)
    for edge in contract.edges:
        out.addIncoming(dst=edge.dst, src=edge.src, gen=edge.gen, value=edge.value,
                        relative=edge.relative)
    for need in contract.requires:
        out.addRequires(dst=need.dst, src=need.src, gen=need.gen, values=list(need.values))
    st_module.setGirFrameContract(out)
    return contract
