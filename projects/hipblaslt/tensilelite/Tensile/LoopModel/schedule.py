# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every derived decision: how many buffers, how far ahead, where each reload sits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from . import traversal as geometry
from .loopir import TENSOR_LOAD
from .traversal import (CLOBBER, axis_strides,
                        broadcast_width, chunks_crossed, group_live_peak,
                        group_reuse_floor, group_ring_size, lds_buffers, position_terms,
                        prefetch_axis_name, prefetch_distance_for,
                        presence_axes, read_coverage, readahead_level,
                        register_reuse_verdict, reload_modes, reload_positions,
                        reloads_whole_set, requested_read_ahead, ring_slot,
                        transfer_extents)

def _splits_refill(theta, operand, group, depths) -> bool:
    """Whether an invariant finite loop can split this short ring's refill."""
    buffers = geometry.group_ring_depth(theta, operand, group, depths)
    if buffers >= group_reuse_floor(
            theta, operand, group, charge_lead=False, lead=0):
        return False
    level = geometry.operand_level(theta, operand, geometry.DS_LOAD)
    order = list(theta.inner_axes())
    at = next((i for i, axis in enumerate(order) if axis.name == level), len(order))
    varying = set(geometry.varying_axes(theta, operand))
    return any(axis.name not in varying and int(axis.extent.constant_value) > 1
               for axis in order[:at])


def _placeable_read_ahead(theta, operand, depths) -> int:
    """The requested lead; physical occupancy controls placement, not realization."""
    return requested_read_ahead(theta, operand, depths)

def copy_must_be_first(theta, operand, depths, plans=None):
    plans = Schedule(theta, depths) if plans is None else plans
    outer = theta.summation_chunk_name()
    if outer is None:
        return False
    prefetch_steps = plans.want(operand)
    crossed = max((chunks_crossed(theta, operand, plans.steps(operand, group, prefetch_steps))
                   for group in operand.fragment.groups()), default=0)
    copy_offset = theta.off_at(operand.name, TENSOR_LOAD, outer)
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


def _all_varying_coords(theta, operand, varying):
    """Every coordinate the operand varies over -- the whole set, in loop order."""
    axes = [axis for axis in theta.inner_axes() if axis.name in varying]
    names = [axis.name for axis in axes]
    return [dict(zip(names, values))
            for values in product(*[
                range(max(1, int(axis.extent.constant_value)))
                for axis in axes
            ])]

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
    """Realized lead; a busy destination defers the read without reducing PLR."""
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
    peak = group_live_peak(theta, operand, group, off=refused)
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
    out = []
    for operand in theta.operands:
        if getattr(operand, "is_output", False):
            continue
        requested = requested_read_ahead(theta, operand, depths)
        operand_steps = requested if prefetch_steps is None \
            else min(max(0, int(prefetch_steps)), requested)
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


def _readahead_shift(theta, operand, prefetch_steps, depths, groups=None, plans=None,
                     split_axes=()):
    """The load-order offset the read-ahead becomes, with the strides it was measured in."""
    extents = transfer_extents(theta, operand)
    modes = set(reload_modes(theta, operand)) | set(split_axes or ())
    strides, reload_span = axis_strides(theta, modes, extents)
    steps = _readahead_depth(theta, operand, prefetch_steps, depths, groups, plans)
    # `prefetch_distance_for` is the ONE answer -- re-testing the level against `strides` here is
    # how an operand invariant over it came back with no read-ahead at all.
    # An unrestricted read serves EVERY group, so it walks the whole fan rather than one residue.
    serves = tuple(groups) if groups else tuple(operand.fragment.groups())
    distance = prefetch_distance_for(theta, operand, steps, serves) if steps else 0
    return distance, strides, reload_span, extents


def loads_in_place(theta, operand, prefetch_steps, depths, groups=None, plans=None) -> bool:
    """Does the read-ahead land back on the value it started from, so the refill is in place?"""
    shift, _strides, _reload_span, _extents = _readahead_shift(
        theta, operand, prefetch_steps, depths, groups, plans)
    selected = tuple(groups or operand.fragment.groups())
    unit = geometry.group_register_positions(theta, operand)
    return bool(shift) and all(
        shift % max(
            1,
            unit * geometry.group_ring_depth(theta, operand, group, depths),
        ) == 0
        for group in selected
    )


def _read_placement(theta, operand, depths, prefetch_steps=0, groups=None, plans=None,
                    split_axes=()):
    shift, strides, reload_span, extents = _readahead_shift(theta, operand, prefetch_steps,
                                                            depths, groups, plans, split_axes)
    slots = []
    all_groups = operand.fragment.groups()
    selected = all_groups if groups is None else tuple(g for g in all_groups if g in groups)
    for group in selected:
        buffers = geometry.group_ring_depth(theta, operand, group, depths)
        label = group if len(all_groups) > 1 else ""
        slot = ring_slot(theta, operand, group, buffers, advance=shift, strides=strides)
        slots.append((label, slot))
    src_slot = None
    if operand.trajectory.shared_read is not None:
        lds_depth = depths.shared(operand.name)
        outer_var = theta.summation_chunk_name()
        # The source generation is the chunk the read's position falls in, whatever the ring
        # rotates on: a shift past the reload span reads the NEXT chunk's LDS buffer.
        carried = position_terms(strides, read_coverage(theta, operand))
        outer_axis = next(axis for axis in theta.ord if axis.name == outer_var)
        src_slot = (
            outer_axis
            + geometry._position(theta, carried, shift) // reload_span
        ) % max(1, lds_depth)
    return tuple(slots), src_slot


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
    #: how many LDS copies the global-to-LDS side keeps
    lds_buffers: int
    #: why `prefetch_steps` came out below what was requested, or None
    prefetch_refused: object


class Schedule:
    """The decision table for one theta, computed on demand and cached."""

    def __init__(self, theta, depths):
        self.theta = theta
        self.depths = depths                     # the register depth map
        self._plans = {}
        self._steps = {}   # a plan needs EVERY group's steps, so this layer resolves first
        #: `(theta, depths)` are fixed here, so the request is one number per operand
        self._want = {operand.name: _placeable_read_ahead(theta, operand, depths)
                      for operand in theta.operands}

    def want(self, operand) -> int:
        """The selected scheme's one realized internal PLR for this operand."""
        return self._want[operand.name]

    @property
    def read_ahead_steps(self) -> int:
        """Largest selected shared-to-register lead across all operands."""

        return max(
            (
                self.want(operand)
                for operand in self.theta.operands
                if operand.trajectory.shared_read is not None
            ),
            default=0,
        )

    def steps(self, operand, group, want) -> int:
        """How far ahead this group actually reads, at the depth `want` asks for.

        Kept below `plan` so placement and diagnostics share one cached answer.
        """
        key = (operand.name, group, want)
        if key not in self._steps:
            self._steps[key] = prefetch_steps_for(self.theta, operand, group, want, self.depths)
        return self._steps[key]

    def plan(self, operand, group) -> OperandPlan:
        """The diagnostic facts for one selected operand/group schedule."""
        want = self.want(operand)
        key = (operand.name, group)
        if key not in self._plans:
            self._plans[key] = self._derive(operand, group, want)
        return self._plans[key]

    def _derive(self, operand, group, want) -> OperandPlan:
        theta, depths = self.theta, self.depths
        width = max(1, depths.get(operand.name, group))
        steps = self.steps(operand, group, want)
        distance = prefetch_distance_for(theta, operand, steps)
        anchors = reload_positions(theta, operand, group, width, distance) or {}
        reload_after = {}
        for coord in anchors.values():
            reload_after.update(coord)
        level = readahead_level(theta)
        return OperandPlan(
            register_buffers=width,
            prefetch_axis=level[0] if level else None,
            prefetch_steps=steps,
            prefetch_distance=distance,
            reload_after=reload_after,
            lds_buffers=lds_buffers(theta, operand),
            prefetch_refused=prefetch_refusal(
                theta,
                operand,
                group,
                want,
                depths,
            ),
        )

    def operands(self):
        """(op, group) for every register-resident operand, in theta order."""
        for operand in self.theta.operands:
            if getattr(operand, "is_output", False) or not getattr(operand, "fragment", None):
                continue
            for group in operand.fragment.groups():
                yield operand, group

# ---  ------------------------------------------------------------------------------------------------

def group_width(theta, operand, group) -> int:
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
    L_floor = max(group_live_peak(theta, operand, group),
                  1 if (covered and not distance)
                  else group_reuse_floor(theta, operand, group))
    L_prefetch = group_live_peak(theta, operand, group, off=distance)
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

    def get(self, operand, group, default=1):
        try:
            fragment = self.theta.op(operand).fragment
        except RuntimeError:
            return default
        return int((fragment.ring_depths or {}).get(group, default))

    def groups_of(self, operand):
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
    """Derive each fragment group's VA depth."""
    for operand in theta.operands:
        if operand.is_output:
            continue
        operand.fragment.ring_depths = {
            group: group_width(theta, operand, group)
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
