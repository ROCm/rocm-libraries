# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every derived decision: how many buffers, how far ahead, where each reload sits."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

from . import traversal as geometry
from .ir import COPY, Expr, Placement, REVERSE, Space
from .theta import group_labels, path_direction
from .traversal import (CLOBBER, _divisors, _inner_steps, _summation_ring_axis,
                        axis_strides, broadcast_width, chunks_crossed,
                        coverage_tile_cap, free_axes, group_live_peak,
                        group_reuse_floor, group_ring_size, is_uniform_over,
                        lds_buffers, operand_level, position_terms,
                        prefetch_axis_name, prefetch_distance_for,
                        presence_axes, read_coverage, readahead_level,
                        register_reuse_verdict, reload_modes, reload_positions,
                        reloads_whole_set, requested_read_ahead, ring_slot,
                        transfer_extents)

def readahead_reach(theta, depths, plans) -> int:
    """How many whole reduction chunks the prefetch crosses into, over every group."""
    deepest = 0
    for operand in theta.operands:
        if operand.trajectory.shared_read is None:
            continue
        prefetch_axis = prefetch_axis_name(theta, operand)  # PER operand: the axis it varies on
        if prefetch_axis is None:
            continue
        prefetch_steps = plans.want(operand)
        if prefetch_steps <= 0:
            continue
        for group in operand.fragment.groups():
            steps = plans.steps(operand, group, prefetch_steps)
            deepest = max(deepest, chunks_crossed(theta, operand, steps))
    return deepest


def copy_must_be_first(theta, operand, depths, plans=None):
    plans = Schedule(theta, depths) if plans is None else plans
    outer = theta.summation_chunk_name()
    if outer is None:
        return False
    prefetch_steps = plans.want(operand)
    crossed = max((chunks_crossed(theta, operand, plans.steps(operand, group, prefetch_steps))
                   for group in operand.fragment.groups()), default=0)
    copy_offset = theta.off_at(operand.name, COPY, outer)
    # Equality can lead only when the copy lands in a different LDS slot.  At S_shared <= off it
    # wraps onto the slab still being read and remains a WAR that must trail the reads.
    distinct_slot = lds_buffers(theta, operand) > copy_offset
    return crossed >= 1 and copy_offset == crossed and distinct_slot


def copy_first_operands(theta, depths, plans=None):
    """{operand name} whose shared copy must lead this trip's reads -- the reorder input."""
    plans = Schedule(theta, depths) if plans is None else plans
    return frozenset(operand.name for operand in theta.operands
                     if operand.trajectory.shared is not None
                     and copy_must_be_first(theta, operand, depths, plans))


def chunk_crossing_violations(theta, depths, plans=None):
    """The coupling `off(copy, iter) >= r + 1`, per operand -- [] when theta is emittable."""
    plans = Schedule(theta, depths) if plans is None else plans
    outer = theta.summation_chunk_name()
    if outer is None:
        return []
    bad = []
    for operand in theta.operands:
        if operand.trajectory.shared is None:
            continue
        # The DERIVED depth, as `copy_must_be_first` reads it: the raw `off` charges a full-inner
        # operand a crossing its clamped depth never makes, and refuses the kernel for it.
        prefetch_steps = plans.want(operand)
        for group in operand.fragment.groups():
            steps = plans.steps(operand, group, prefetch_steps)
            crossed = chunks_crossed(theta, operand, steps)
            if crossed <= 0:
                continue
            copy_offset = theta.off_at(operand.name, COPY, outer)
            need = crossed if copy_must_be_first(theta, operand, depths, plans) else crossed + 1
            if copy_offset < need:
                bad.append((operand.name, steps, crossed, copy_offset, need))
                break  # one report per operand is enough
    return bad


@dataclass(frozen=True)
class PeelDepths:
    """The peel depth per level: how many iterations each level lifts out of the loop."""
    per_level: dict
    offsets: dict
    requested_steps: int
    chunk: object
    reach: int = 0
    deferrals: dict = field(default_factory=dict)

    def depth(self, level) -> int:
        """How many steps of `level` are lifted out of the loop; 0 means it is not peeled."""
        return int(self.per_level.get(level, 0))

    @property
    def chunk_depth(self) -> int:
        """The peel at the reduction chunk -- the one level the emitter currently peels."""
        return self.depth(self.chunk)

    def peeled_levels(self) -> list:
        """Every level with `M_l > 0`, outer->inner."""
        return list(self.per_level)

    def copy_off(self) -> dict:
        return {name: depth for (name, role, level), depth in self.offsets.items()
                if role == COPY and level == self.chunk}


def _read_offsets(theta):
    """Every non-zero `off` theta carries, split by sign: forward peels and rejected deferrals."""
    offsets, deferrals = {}, {}
    for operand in theta.operands:
        if operand.movements and path_direction(operand) == REVERSE:
            raise RuntimeError(
                f"operand {operand.name!r} has a REVERSE trajectory "
                f"({[f'{h.src}->{h.dst}' for h in operand.movements]}): its hops run register->shared"
                f"->global, and the mirrored peel that shape needs is not emitted yet. Refusing "
                f"rather than peeling it forward, because the hop role still says 'read' for the "
                f"read-back leg, so a role-keyed peel would look correct and run backwards.")
        for hop in operand.movements:
            for axis in theta.ord:
                depth = theta.off_at(operand, hop.role, axis.name)
                if depth > 0:
                    offsets[(operand.name, hop.role, axis.name)] = depth
                elif depth < 0:
                    deferrals[(operand.name, hop.role, axis.name)] = depth
    return offsets, deferrals


def _deepest_read_ahead(theta, depths=None):
    """The largest read-ahead any shared->register read asks for, in its own axis's steps."""
    asked = [requested_read_ahead(theta, operand, depths) for operand in theta.operands
             if operand.trajectory.shared_read is not None]
    return max(0, max(asked, default=0))


def peel_depths(theta, depths, plans=None):
    """The peel depth of every level, and the offsets it was derived from."""
    plans = Schedule(theta, depths) if plans is None else plans
    chunk = theta.summation_chunk_name()
    offsets, deferrals = _read_offsets(theta)

    per_level = {}
    for (_name, _role, level), depth in offsets.items():
        per_level[level] = max(depth, per_level.get(level, 0))

    prefetch_steps = max((plans.want(operand) for operand in theta.operands
                          if operand.trajectory.shared_read is not None), default=0)
    reach = readahead_reach(theta, depths, plans) if prefetch_steps else 0
    if reach and chunk is not None:
        per_level[chunk] = max(reach, per_level.get(chunk, 0))
    per_level = {axis.name: per_level[axis.name] for axis in theta.ord if axis.name in per_level}
    return PeelDepths(per_level=per_level, offsets=offsets, requested_steps=prefetch_steps,
                      chunk=chunk, reach=reach, deferrals=deferrals)


@dataclass(frozen=True)
class BoundaryHoist:
    """One cross-level boundary term."""
    op: str
    role: str
    level: str
    home: str
    delta: int
    coords: tuple


def _leading_inner_coords(home, delta):
    return tuple({home: i} for i in range(max(0, delta)))


def boundary_hoists(theta, pd):
    """Every cross-level boundary term `theta` carries (see `BoundaryHoist`), outer level first."""
    names = [axis.name for axis in theta.ord]
    outer = {axis.name for axis in theta.outer_axes()}
    out = []
    for (opname, role, level), delta in pd.offsets.items():
        if level not in outer:
            continue  # (1) inner level -> no boundary to cross
        operand = theta.op(opname)
        home = operand_level(theta, operand, role)
        if home is None or home not in names:
            continue
        if names.index(home) <= names.index(level):
            continue  # (2) home is at or outer to level
        if home not in outer:
            continue  # (3) unrolled home -> ordinary prefetch
        out.append(BoundaryHoist(op=opname, role=role, level=level, home=home, delta=delta,
                                 coords=_leading_inner_coords(home, delta)))
    return sorted(out, key=lambda hop: (names.index(hop.level), hop.op, hop.role))


def _all_varying_coords(theta, operand, varying):
    """Every coordinate the operand varies over -- the whole set, in loop order."""
    axes = [axis for axis in theta.inner_axes() if axis.name in varying]
    names = [axis.name for axis in axes]
    return [dict(zip(names, values))
            for values in product(*[range(max(1, axis.extent)) for axis in axes])]

def preloaded_tiles(theta, operand, prefetch_steps, depths=None, plans=None):
    """The leading physical tiles covered by the realized operand prefetch."""
    if not prefetch_steps:
        return []
    varying = set(geometry.presence(theta, operand))
    if not varying:
        return []
    if depths is not None and plans is None:
        plans = Schedule(theta, depths)
    steps = plans.want(operand) if plans is not None else requested_read_ahead(theta, operand)
    if steps < 1:
        return []
    coords = _all_varying_coords(theta, operand, varying)
    read = operand.trajectory.fragment_fill
    factors = geometry.coverage_factors(theta, operand, getattr(read, "coverage", None)) \
        if read is not None else {}
    leaders = [coord for coord in coords
               if all(int(coord.get(axis, 0)) % max(1, int(factor)) == 0
                      for axis, factor in factors.items())]
    count = steps * geometry.group_prefetch_unit_positions(theta, operand)
    return leaders[:min(len(leaders), count)]


def _prefetching_groups(operand):
    return tuple(group for group in operand.fragment.groups()
                 if operand.fragment.policy_of(group) != "inplace")


def prefetch_steps_for(theta, operand, group, requested, depths) -> int:
    """Realized lead for one group; the request governs it, not the refill policy.

    `inplace` says WHEN the refill happens, not whether there is a lead, so it does not zero one.
    Neither does `regions_time_share`: regions landing on one slot is WHERE the refill goes, and a
    refill placed after the region's last use overwrites nothing that is still live.
    """
    if max(1, depths.get(operand.name, group)) < group_reuse_floor(theta, operand, group):
        return 0
    return min(max(0, int(requested)), requested_read_ahead(theta, operand, depths))


BAND_EMPTY = "band-empty"
CAPACITY = "capacity"
ASSIGNMENT = "assignment"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PrefetchRefusal:
    """Why one group could not read as far ahead as PrefetchLocalRead asked."""
    operand: str
    group: str
    requested: int
    derived: int
    reason: str
    values: dict

    def __getitem__(self, index):
        """Positional access, so the render tables can stay column-oriented."""
        return (self.operand, self.group, self.requested, self.derived, self.reason,
                self.values)[index]


def prefetch_refusal(theta, operand, group, prefetch_steps, depths):
    """The first depth the derivation would not grant, and what stopped it."""
    if not prefetch_steps:
        return None
    derived = prefetch_steps_for(theta, operand, group, prefetch_steps, depths)
    if derived >= prefetch_steps:
        return None
    buffers = max(1, depths.get(operand.name, group))
    refused = derived + 1

    def refusal(reason, **values):
        return PrefetchRefusal(operand.name, group, prefetch_steps, derived, reason, values)

    if refused > min(prefetch_steps, max(1, buffers - 1)):
        return refusal(BAND_EMPTY, depth=refused, buffers=buffers,
                       rate=group_ring_size(theta, operand, group),
                       fan=broadcast_width(theta, operand))
    peak = group_live_peak(theta, operand, group, _inner_steps(theta), off=refused)
    if peak > buffers:
        return refusal(CAPACITY, depth=refused, buffers=buffers, peak=peak)
    distance = prefetch_distance_for(theta, operand, refused)
    clobbered = [other for other in operand.fragment.groups()
                 if register_reuse_verdict(theta, operand, other,
                                           max(1, depths.get(operand.name, other)),
                                           distance) == CLOBBER]
    if clobbered:
        return refusal(ASSIGNMENT, depth=refused, buffers=buffers, distance=distance,
                       groups=clobbered, fan=broadcast_width(theta, operand))
    return refusal(UNKNOWN, depth=refused, buffers=buffers)


def prefetch_refusals(theta, depths, prefetch_steps=None):
    """Every (op, group) whose PLR request the derivation could not honour."""
    if prefetch_steps is None:
        try:
            prefetch_steps = peel_depths(theta, depths).requested_steps
        except Exception:
            return ()
    out = []
    for operand in theta.operands:
        if getattr(operand, "is_output", False):
            continue
        operand_steps = min(prefetch_steps, requested_read_ahead(theta, operand, depths))
        for group in operand.fragment.groups():
            refusal = prefetch_refusal(theta, operand, group, operand_steps, depths)
            if refusal is not None:
                out.append(refusal)
    return tuple(out)


def _readahead_depth(theta, operand, prefetch_steps, depths, groups=None, plans=None) -> int:
    """The shallowest depth across the selected groups: one instruction carries one advance."""
    selected = operand.fragment.groups() if groups is None else tuple(groups)
    if not selected:
        return 0
    plans = Schedule(theta, depths) if plans is None else plans
    return min(plans.steps(operand, group, prefetch_steps) for group in selected)


def _readahead_shift(theta, operand, prefetch_steps, depths, groups=None, plans=None):
    """The load-order offset the read-ahead becomes, with the strides it was measured in."""
    extents = transfer_extents(theta, operand)
    strides, reload_span = axis_strides(theta, reload_modes(theta, operand), extents)
    steps = _readahead_depth(theta, operand, prefetch_steps, depths, groups, plans)
    # `prefetch_distance_for` is the ONE answer -- re-testing the level against `strides` here is
    # how an operand invariant over it came back with no read-ahead at all.
    # An unrestricted read serves EVERY group, so it walks the whole fan rather than one residue.
    serves = tuple(groups) if groups else tuple(operand.fragment.groups())
    distance = prefetch_distance_for(theta, operand, steps, serves) if steps else 0
    return distance, strides, reload_span, extents


def loads_in_place(theta, operand, prefetch_steps, depths, groups=None, plans=None) -> bool:
    """Does the read-ahead land back on the value it started from, so the refill is in place?"""
    shift, _strides, reload_span, _extents = _readahead_shift(theta, operand, prefetch_steps,
                                                              depths, groups, plans)
    return bool(shift) and shift % max(1, reload_span) == 0


def _read_placement(theta, operand, depths, prefetch_steps=0, groups=None, plans=None):
    shift, strides, reload_span, extents = _readahead_shift(theta, operand, prefetch_steps,
                                                            depths, groups, plans)
    slots = []
    all_groups = operand.fragment.groups()
    selected = all_groups if groups is None else tuple(g for g in all_groups if g in groups)
    for group in selected:
        buffers = geometry.group_ring_depth(theta, operand, group, depths)
        label = group if len(all_groups) > 1 else ""
        slot = ring_slot(theta, operand, group, buffers, advance=shift)
        slots.append((label, slot))
    src_slot = None
    if operand.trajectory.shared_read is not None:
        lds_depth = depths.shared(operand.name)
        outer_var = theta.summation_chunk_name()
        # The source generation is the chunk the read's position falls in, whatever the ring
        # rotates on: a shift past the reload span reads the NEXT chunk's LDS buffer.
        carried = position_terms(strides, read_coverage(theta, operand))
        src_slot = Expr(var=outer_var, mod=max(1, lds_depth),
                        carry=(carried, shift, reload_span))
    return Placement(space=Space.REGISTER, slots=tuple(slots), src_slot=src_slot)


def resolve_coverage(theta, depths=None) -> dict:
    return {operand.name: operand.fragment.coverage
            for operand in theta.operands
            if operand.fragment is not None and operand.fragment.coverage is not None}


def _generation_map(theta, operand, depths, prefetch_steps):
    placement = _read_placement(theta, operand, depths, prefetch_steps=prefetch_steps)
    varying = list(presence_axes(theta, operand))
    chunk = theta.summation_chunk_name()
    out = {}
    for combo in product(*[range(max(1, axis.extent)) for axis in varying]):
        env = {chunk: 0}
        env.update({axis.name: v for axis, v in zip(varying, combo)})
        source = placement.src_slot
        generation = source.eval(env) if hasattr(source, "eval") else source
        slots = tuple(slot.eval(env) if hasattr(slot, "eval") else slot
                      for _group, slot in placement.slots)
        out[tuple(zip((axis.name for axis in varying), combo))] = (generation, slots)
    return out, varying


def transfer_coverage_axes(theta, operand, hop, depths) -> tuple:
    """The axes one instruction of this hop covers, and by what factor (the movement coverage)."""
    cap = coverage_tile_cap(theta, operand, hop)
    if cap <= 1:
        return ()
    free = [axis for axis in free_axes(theta, operand) if axis.extent > 1]
    if not free:
        return ()
    steps = theta.off_of(operand, hop, prefetch_axis_name(theta, operand))
    generations, varying = _generation_map(theta, operand, depths, steps)
    best, covered = (), 1
    for depth in range(1, len(free) + 1):
        whole = free[-(depth - 1):] if depth > 1 else []  # outer axes, spanned whole
        inner = free[-depth]  # the one that may take a factor
        outer_width = 1
        for axis in whole:
            outer_width *= max(1, axis.extent)
        if outer_width > cap:
            break
        room = cap // outer_width
        factor = max((f for f in _divisors(inner.extent) if f <= room), default=1)
        if factor <= 1 and not whole:
            break  # nothing to merge at this depth
        candidate = tuple([(axis.name, axis.extent) for axis in whole]
                          + ([(inner.name, factor)] if factor > 1 else []))
        if is_uniform_over(generations, candidate) and outer_width * factor > covered:
            best, covered = candidate, outer_width * factor
        if factor < inner.extent:
            break
    return best


# --- schedule --------------------------------------------------------------

@dataclass(frozen=True)
class OperandPlan:
    """What we decided for one operand's register group."""

    #: how many copies of this operand's registers exist, so a load never overwrites a live one
    register_buffers: int
    #: which loop axis the lookahead is counted along, derived per operand from the loop order
    prefetch_axis: str
    #: how many steps along that axis a load runs ahead; 0 means load right before use
    prefetch_steps: int
    #: the same lookahead as an offset in this operand's own load order, which is what the
    #: address arithmetic and the drain bound consume
    prefetch_distance: int
    #: the loop position a reload must follow; empty means it may sit at first use
    reload_after: dict
    #: tiles loaded before the loop starts, so the first iteration has data
    preloaded_tiles: tuple
    #: how many LDS copies the global-to-LDS side keeps
    lds_buffers: int
    #: why `prefetch_steps` came out below what was requested, or None
    prefetch_refused: object

    @property
    def loads_ahead(self) -> bool:
        return self.prefetch_steps >= 1


class Schedule:
    """The decision table for one theta, computed on demand and cached."""

    def __init__(self, theta, depths, requested_depth=None):
        self.theta = theta
        self.depths = depths                     # the register depth map
        self.requested_depth = (_deepest_read_ahead(theta, depths) if requested_depth is None
                                else requested_depth)
        self._plans = {}
        self._steps = {}   # a plan needs EVERY group's steps, so this layer resolves first
        #: `(theta, depths)` are fixed here, so the request is one number per operand
        self._want = {operand.name: requested_read_ahead(theta, operand, depths)
                      for operand in theta.operands}

    def want(self, operand) -> int:
        """The selected scheme's one realized internal PLR for this operand."""
        return self._want[operand.name]

    def steps(self, operand, group, want) -> int:
        """How far ahead this group actually reads, at the depth `want` asks for.

        Kept below `plan`, not inside it: a plan's `preloaded_tiles` spans every group of the
        fragment, so deriving one plan asks for the others' steps. Two layers, no cycle.
        """
        key = (operand.name, group, want)
        if key not in self._steps:
            self._steps[key] = prefetch_steps_for(self.theta, operand, group, want, self.depths)
        return self._steps[key]

    def plan(self, operand, group, depth=None) -> OperandPlan:
        """The plan for one (operand, group), at `depth` or theta's requested depth."""
        want = self.want(operand) if depth is None else min(depth, self.want(operand))
        key = (operand.name, group, want)
        if key not in self._plans:
            self._plans[key] = self._derive(operand, group, want)
        return self._plans[key]

    def _derive(self, operand, group, want) -> OperandPlan:
        theta, depths = self.theta, self.depths
        width = max(1, depths.get(operand.name, group))
        steps = self.steps(operand, group, want)
        asked = min(want, self.want(operand))
        distance = prefetch_distance_for(theta, operand, steps)
        anchors = reload_positions(theta, operand, group, width, distance) or {}
        reload_after = {}
        for coord in anchors.values():
            reload_after.update(coord)
        level = readahead_level(theta, operand)
        return OperandPlan(
            register_buffers=width,
            prefetch_axis=level[0] if level else None,
            prefetch_steps=steps,
            prefetch_distance=distance,
            reload_after=reload_after,
            preloaded_tiles=tuple(map(_freeze,
                                      preloaded_tiles(theta, operand, want, depths, self))),
            lds_buffers=lds_buffers(theta, operand),
            prefetch_refused=prefetch_refusal(theta, operand, group, want, depths),
        )

    def operands(self):
        """(op, group) for every register-resident operand, in theta order."""
        for operand in self.theta.operands:
            if getattr(operand, "is_output", False) or not getattr(operand, "fragment", None):
                continue
            for group in operand.fragment.groups():
                yield operand, group


def _freeze(coord):
    return tuple(sorted(coord.items())) if isinstance(coord, dict) else coord


# ---  ------------------------------------------------------------------------------------------------

def group_width(theta, operand, group, steps) -> int:
    """Rotation-unit width selected for one group."""
    read = operand.trajectory.fragment_fill
    if read is not None and not presence_axes(theta, operand, hop=read):
        return 1
    R = group_ring_size(theta, operand, group)
    serves = _prefetching_groups(operand)
    distance = prefetch_distance_for(
        theta, operand, requested_read_ahead(theta, operand),
        serves if group in serves else (group,))
    # A COVERED read's floor answers a collision between a live value and its refill, so with no
    # lead nothing is in flight and it cannot bind.  An uncovered operand is retained across a
    # sweep and owes its floor either way.
    covered = read is not None and read.coverage is not None
    L_floor = max(group_live_peak(theta, operand, group, steps),
                  1 if (covered and not distance)
                  else group_reuse_floor(theta, operand, group))
    L_prefetch = group_live_peak(theta, operand, group, steps, off=distance)
    choice = operand.fragment.policy_of(group)
    if choice == "inplace":
        return L_floor
    if isinstance(choice, int):
        return max(L_floor, choice)
    if choice == "unroll":
        return max(L_floor, L_prefetch, R)
    if choice == "pipeline":
        need = max(L_floor, L_prefetch)
        if reloads_whole_set(theta, operand):
            return need
        return next((width for width in range(need, R + 1) if R % width == 0), R)
    return max(L_floor, L_prefetch)

@dataclass(frozen=True)
class BufferDepths:
    """Read-only access to depths owned by placements."""
    theta: object
    include_shared: bool = True

    def get(self, operand, group, default=1, region=None):
        try:
            fragment = self.theta.op(operand).fragment
        except RuntimeError:
            return default
        return int((fragment.ring_depths or {}).get(group, default))

    def groups_of(self, operand, region=None):
        try:
            fragment = self.theta.op(operand).fragment
        except RuntimeError:
            return {}
        return dict(fragment.ring_depths or {})

    def shared(self, operand, default=None):
        try:
            placement = self.theta.op(operand).trajectory.shared
        except RuntimeError:
            return default
        if placement is None:
            return default
        return int(placement.ring_depth)

    def register_items(self):
        for operand in self.theta.operands:
            if operand.fragment is None:
                continue
            for group, depth in (operand.fragment.ring_depths or {}).items():
                yield operand, group, int(depth)

    def __repr__(self):
        rows = []
        for operand in self.theta.operands:
            groups = self.groups_of(operand.name)
            if groups:
                rows.append("%s=[%s]" % (
                    operand.name,
                    ", ".join("%s:%s" % item for item in sorted(groups.items()))))
        return " | ".join(rows)


def derive_S(theta) -> BufferDepths:
    """Derive register-ring depths onto fragment placements."""
    steps = _inner_steps(theta)
    for operand in theta.operands:
        if operand.is_output:
            continue
        operand.fragment.ring_depths = {
            group: group_width(theta, operand, group, steps)
            for group in operand.fragment.groups()
        }
    return BufferDepths(theta, include_shared=False)

def build_S(theta):
    needs_depths = any(
        not operand.is_output and operand.fragment.ring_depths is None
        for operand in theta.operands)
    floor = derive_S(theta) if needs_depths else BufferDepths(theta, include_shared=False)
    for operand in theta.operands:
        shared = operand.trajectory.shared
        if shared is not None and shared.ring_depth <= 0:
            shared.ring_depth = lds_buffers(theta, operand)
    return BufferDepths(theta), floor


_GROUPING_SCHEMES = (
    (1, ("pipeline",)),
    (2, ("pipeline", "pipeline")),
    (2, ("pipeline", "inplace")),
    (4, ("pipeline",) * 4),
    (4, ("pipeline", "inplace") * 2),
    (8, ("pipeline",) * 8),
    (8, ("pipeline", "inplace") * 4),
    (1, ("inplace",)),
)


class RegisterDepthSelector:
    """Choose fragment groups and ring depths within a register budget."""

    def __init__(self, theta, budget):
        self.theta = theta
        self.budget = budget
        self.splittable = [
            operand for operand in theta.operands
            if operand.movements
            and operand.fragment.grouping_mode is not None
            and geometry.grouping_extent(theta, operand) > 1
        ]
        self.original = {operand.name: operand.fragment for operand in self.splittable}
        self.blocked = None

    def admissible(self, operand, groups) -> bool:
        fan = geometry.grouping_extent(self.theta, operand)
        return (
            groups >= 1
            and fan % groups == 0
            and fan // groups >= 1
            and (groups == 1 or geometry.grouping_preserves_coverage(
                self.theta, operand, groups))
        )

    def apply(self, operand, groups, policies) -> bool:
        if not self.admissible(operand, groups):
            return False
        labels = group_labels(groups)
        operand.fragment = replace(
            self.original[operand.name],
            parts=groups,
            labels=labels,
            group_policy=dict(zip(labels, policies)),
            ring_depths=None)
        return True

    def order(self, depths) -> list:
        names = [operand.name for operand in self.theta.operands]
        positions = {axis.name: i for i, axis in enumerate(self.theta.inner_axes())}
        return sorted(
            self.splittable,
            key=lambda operand: (
                operand.name.startswith("MXS"),
                -positions.get(operand.fragment.grouping_mode, -1),
                not geometry.reloads_whole_set(self.theta, operand),
                geometry.operand_emitted_regs(self.theta, operand, depths),
                names.index(operand.name),
            ))

    def unserved(self, operand, depths):
        want = requested_read_ahead(self.theta, operand, depths)
        if operand.is_output or not want:
            return None
        for group in operand.fragment.groups():
            if operand.fragment.policy_of(group) == "inplace":
                continue
            if prefetch_steps_for(self.theta, operand, group, want, depths) < want:
                return group
        return None

    def cost(self, depths, operands=None) -> int:
        selected = self.theta.operands if operands is None else operands
        return sum(geometry.operand_emitted_regs(self.theta, operand, depths)
                   for operand in selected)

    def candidates(self, operand) -> list:
        out = []
        for groups, policies in _GROUPING_SCHEMES:
            if not self.apply(operand, groups, policies):
                continue
            try:
                depths = derive_S(self.theta)
                if self.unserved(operand, depths) is None:
                    out.append((groups, policies,
                                geometry.operand_emitted_regs(self.theta, operand, depths)))
            except RuntimeError:
                pass
        operand.fragment = self.original[operand.name]
        return out

    def choose(self, order, options, index, spent):
        if index == len(order):
            depths = derive_S(self.theta)
            return None if chunk_crossing_violations(self.theta, depths) else depths

        rest = sum(min((regs for _groups, _policies, regs in options[i]), default=0)
                   for i in range(index + 1, len(order)))
        operand = order[index]
        for groups, policies, regs in options[index]:
            if self.budget and spent + regs > self.budget:
                blocked = spent + regs + rest
                self.blocked = min(self.blocked or blocked, blocked)
                continue
            self.apply(operand, groups, policies)
            result = self.choose(order, options, index + 1, spent + regs)
            if result is not None:
                return result
        operand.fragment = self.original[operand.name]
        return None

    def build(self):
        base = derive_S(self.theta)
        order = self.order(base)
        names = {operand.name for operand in self.splittable}
        fixed = self.cost(base, [operand for operand in self.theta.operands
                                 if operand.name not in names])
        result = self.choose(order, [self.candidates(operand) for operand in order], 0, fixed)
        if result is not None:
            return result
        for operand in self.splittable:
            operand.fragment = self.original[operand.name]
        reason = (f"register budget {self.budget} is below {self.blocked}"
                  if self.blocked is not None else "no grouping carries the requested read-ahead")
        raise RuntimeError(f"no register grouping is valid: {reason}")


def select_register_depths(theta, budget=None):
    return RegisterDepthSelector(theta, budget).build()
