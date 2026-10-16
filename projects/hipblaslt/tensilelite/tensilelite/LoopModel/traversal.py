# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Everything true of one operand's traversal: the tile and register arithmetic, which
axes it varies over, and how its register ring turns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product as _iproduct
from math import ceil, gcd

from .loopir import DS_LOAD, REGISTER, SHARED, TENSOR_LOAD, Axis, Expr
from .theta import mode_role, readahead_level_of


@dataclass(frozen=True)
class TransferCoverage:
    """Target mapping from logical tiles to one instruction's members."""

    carrier: Expr
    slot: Expr

    def _axis(self):
        axes = self.carrier.free_axes() | self.slot.free_axes()
        if len(axes) != 1:
            raise RuntimeError("transfer coverage must use exactly one tile axis")
        return next(iter(axes))

    def carrier_of(self, term: int, env=None) -> int:
        axis = self._axis()
        return self.carrier.evaluate({**dict(env or {}), axis: term})

    def slot_of(self, term: int, env=None) -> int:
        axis = self._axis()
        return self.slot.evaluate({**dict(env or {}), axis: term})

    def group(self, term: int, extent: int, env=None) -> tuple:
        carrier = self.carrier_of(term, env)
        return tuple(
            unit for unit in range(max(1, extent))
            if self.carrier_of(unit, env) == carrier
        )

    def regs(self, term: int, extent: int, env=None) -> int:
        return len({
            self.slot_of(unit, env)
            for unit in self.group(term, extent, env)
        })


def _position(theta, terms, add=0):
    value = Expr.constant(int(add))
    for name, coefficient, divisor in terms:
        axis = name if isinstance(name, Axis) else next(
            item for item in theta.ord if item.name == name
        )
        value = value + int(coefficient) * (
            axis // max(1, int(divisor))
        )
    return value


def _digits(theta, terms, add, fields, modulus=0):
    position = _position(theta, terms, add)
    value = Expr.constant(0)
    for coefficient, divisor, extent in fields:
        digit = position // max(1, int(divisor))
        if int(extent):
            digit = digit % int(extent)
        value = value + int(coefficient) * digit
    return value % int(modulus) if int(modulus) > 1 else value


# --- tile and register arithmetic ------------------------------------------------------------

def _inner_axes(theta):
    return [
        (axis.name, int(axis.extent.constant_value))
        for axis in theta.inner_axes()
    ]


def broadcast_axes(theta, operand) -> set:
    """The intra-iteration loop axes this operand is constant over (does NOT depend on).

    An operand's OWN region axes are axes it varies over -- a scale carries its parent's split
    axis and changes with it, so riding a fused walk does not make it constant over that axis.
    """
    return set(theta.geometry(operand).broadcast)


def _coverage_candidate_axes(theta, operand):
    bcast = broadcast_axes(theta, operand)
    summation = {axis.name for axis in summation_axes(theta)}
    return [
        (axis.name, max(1, int(axis.extent.constant_value)))
        for axis in theta.inner_axes()
        if axis.name not in bcast and axis.name not in summation
    ]


def derive_transfer_coverage(movement):
    """Build coverage from instruction grouping and agent distribution."""
    merged = max(1, int(getattr(movement, "instances_per_instruction", 1) or 1))
    agents = int(getattr(movement, "agent_group_size", 0) or 0)
    if merged <= 1 and not agents:
        return None
    tile = Axis("tile", 0)
    carrier = tile // merged
    if agents > 1:
        if merged % agents:
            return None
        slot = (tile // agents) % (merged // agents)
    else:
        slot = tile % merged
    return TransferCoverage(carrier=carrier, slot=slot)

def coverage_factors(theta, operand, coverage) -> dict:
    if coverage is None:
        return {}
    axes = _coverage_candidate_axes(theta, operand)
    stride, total = {}, 1
    for name, extents in reversed(axes):
        stride[name] = total
        total *= extents
    out = {}
    for name, extents in axes:
        if extents <= 1:
            continue  # a degenerate axis is spanned by everything; say nothing
        step = stride[name]
        best = 1
        for factor in range(2, extents + 1):
            if extents % factor:
                continue  # only a uniform coverage -- regularity, no ragged group
            if all(coverage.carrier_of(tile - ((tile // step) % factor) * step)
                   == coverage.carrier_of(tile)
                   for tile in range(total)):
                best = factor
        if best > 1:
            out[name] = best
    return out


def coverage_axes(theta, operand, coverage) -> set:
    """The intra-iteration axes ONE instruction of a per-lane hop spans whole."""
    extents = dict(_coverage_candidate_axes(theta, operand))
    return {name for name, factor in coverage_factors(theta, operand, coverage).items()
            if factor >= extents.get(name, 0)}


def transfer_broadcast(theta, operand, hop) -> set:
    inner = [name for name, _ext in _inner_axes(theta)]
    if not hop.is_bulk:
        return broadcast_axes(theta, operand) | coverage_axes(theta, operand, hop.coverage)
    present = ({rm for rm in getattr(operand, "region_axes", ()) if rm in inner}
               if max(1, getattr(operand, "split", 1)) > 1 else set())
    return {name for name in inner if name not in present}


def transfer_coverage(theta, operand, hop) -> dict:
    if hop is None or getattr(hop, "is_bulk", False):
        return {}
    extents = dict(_coverage_candidate_axes(theta, operand))
    return {name: factor
            for name, factor in coverage_factors(theta, operand, hop.coverage).items()
            if 1 < factor < extents.get(name, 0)}


# ===========================================================================

def free_tiles(theta, operand) -> int:
    exts = [int(axis.extent.constant_value) for axis in free_axes(theta, operand)]
    return max(1, product(exts)) if exts else 1


def k_tiles(theta) -> int:
    exts = [int(axis.extent.constant_value) for axis in summation_axes(theta)]
    return max(1, product(exts)) if exts else 1


def summation_tiles(theta, operand) -> int:
    summation = {axis.name for axis in summation_axes(theta)}
    exts = [
        int(axis.extent.constant_value)
        for axis in presence_axes(theta, operand)
        if axis.name in summation
    ]
    return max(1, product(exts)) if exts else 1


def presence_tiles(theta, operand) -> int:
    exts = [
        int(axis.extent.constant_value)
        for axis in presence_axes(theta, operand)
    ]
    return max(1, product(exts)) if exts else 1


def wmma_grid(theta) -> tuple:
    inputs = [operand for operand in theta.operands
              if operand.is_input and any(hop.dst is REGISTER for hop in operand.movements)]
    f0 = free_tiles(theta, inputs[0]) if len(inputs) > 0 else 1
    f1 = free_tiles(theta, inputs[1]) if len(inputs) > 1 else 1
    return (f0, f1, k_tiles(theta))


# ===========================================================================

def frag_regs(theta, operand) -> int:
    elems = max(1, operand.frag_elems)
    if theta.reg_bytes % max(1, operand.elem_bytes) == 0:
        scale = theta.reg_bytes // operand.elem_bytes  # elements per register
        return max(1, recast_count(elems, scale))  # ceil(elems/scale); was recast+coshape
    return max(1, -(-elems * operand.elem_bytes // theta.reg_bytes))  # ceil, sub-register elems


# ---  ------------------------------------------------------------------------------------------------

def _inner_steps(theta):
    axes = theta.inner_axes()
    ranges = [
        [
            (axis.name, value)
            for value in range(int(axis.extent.constant_value))
        ]
        for axis in axes
    ]
    return [dict(c) for c in _iproduct(*ranges)] if ranges else [dict()]


def readahead_level(theta):
    """The shared read-ahead level of this loop nest."""
    return theta.readahead_level()


def unit_enumerator_modes(theta, operand):
    """Factor modes of the logical axis that enumerates successive units."""
    return list(theta.geometry(operand).unit_enumerator)


def original_prefetch_unit_modes(theta, operand):
    """All factors of the operand's innermost logical axis, including region factors."""
    return list(theta.geometry(operand).original_prefetch)


def prefetch_unit_modes(theta, operand):
    """The per-region prefetch unit before register grouping."""
    return list(theta.geometry(operand).prefetch_unit)


def rotation_unit_modes(theta, operand):
    """Modes whose tile product is one rotation unit before register grouping."""
    return list(theta.geometry(operand).rotation)


def grouping_modes(theta, operand):
    """The first non-region rotation mode: the one contiguous groups partition."""
    return list(theta.geometry(operand).grouping)


def grouping_mode_name(theta, operand):
    modes = grouping_modes(theta, operand)
    return modes[0][0] if modes and product(extent for _name, extent in modes) > 1 else None


def grouping_extent(theta, operand) -> int:
    return product(extent for _name, extent in grouping_modes(theta, operand))


def group_index_expr(theta, operand):
    groups = max(1, len(operand.fragment.groups()))
    modes = grouping_modes(theta, operand)
    if groups <= 1 or not modes:
        return Expr.constant(0)
    name, extent = modes[0]
    if extent % groups:
        raise RuntimeError(
            f"{operand.name}: grouping extent {extent} is not divisible by {groups} groups")
    # Contiguous ranges, repeated independently in every TDMSplit region.
    return (
        next(axis for axis in theta.ord if axis.name == name)
        // (extent // groups)
    ) % groups


def group_index(theta, operand, coord) -> int:
    return group_index_expr(theta, operand).evaluate(dict(coord))


def _flatten_digits(digits, at, group_wrap) -> tuple:
    """Mixed-radix value of `digits` under the coordinate `at`, and the extent they span."""
    value, total = 0, 1
    for name, extent, divisor, wrap in digits:
        raw = int(at.get(name, 0) or 0)
        if wrap:
            raw %= max(1, group_wrap if wrap is True else int(wrap))
        radix = max(1, -(-extent // divisor))
        value = value * radix + raw // divisor
        total *= radix
    return value, total


def _rotation_block_digits(theta, operand):
    """The digits one PU-sized ring slot distinguishes, with its grouping context."""
    groups = max(1, len(operand.fragment.groups()))
    grouping = grouping_mode_name(theta, operand)
    group_extent = max(1, grouping_extent(theta, operand) // groups)
    region_axes = getattr(operand, "region_axes", ()) or ()
    movement = operand.trajectory.shared_read
    coverage = getattr(movement, "coverage", None) if movement is not None else None
    modes = prefetch_unit_modes(theta, operand)
    if coverage is not None:
        ring = {name for name, _extent in ring_axes(
            theta, operand, operand.fragment.groups()[0])}
        modes = [(name, extent) for name, extent in theta.geometry(operand).varying
                 if name not in ring]
    digits = [
        (name, group_extent if name == grouping else extent, 1, name == grouping)
        for name, extent in modes
        if name not in region_axes
    ]
    # VG MOVED THE AXIS'S TOP FACTOR INTO THE SLOT, so the block keeps only what is left of it
    # and wraps there -- the same digit in two places would address past the allocation.
    split = vgpr_group_axis(theta, operand)
    if split is not None:
        name, radix = split
        digits = [(dname, max(1, dextent // radix) if dname == name else dextent, divisor,
                   max(1, dextent // radix) if dname == name else wrap)
                  for dname, dextent, divisor, wrap in digits]
    return digits, grouping, group_extent


def _issue_order(keys) -> list:
    """The distinct carrier keys in first-appearance order -- the order the reads issue in."""
    order = []
    for key in keys:
        if key not in order:
            order.append(key)
    return order


def unit_tile_index(theta, operand, coord) -> int:
    """Dense covered-register index inside this group's physical rotation block."""
    at = dict(coord)
    digits, grouping, group_extent = _rotation_block_digits(theta, operand)
    movement = operand.trajectory.shared_read
    coverage = getattr(movement, "coverage", None) if movement is not None else None
    if coverage is None:
        return _flatten_digits(digits, at, group_extent)[0]

    reduction = set(summation_names(theta))
    free_value, free_total = _flatten_digits(
        [d for d in digits if d[0] not in reduction], at, group_extent)
    reduction_value, reduction_total = _flatten_digits(
        [d for d in digits if d[0] in reduction], at, group_extent)

    keys = [(coverage.carrier_of(tile), coverage.slot_of(tile))
            for tile in range(max(1, free_total))]
    order = _issue_order(keys)
    free_index = order.index(keys[free_value])
    return free_index * reduction_total + reduction_value


def original_prefetch_unit_tiles(theta, operand) -> int:
    """Physical tiles in the global PLR unit, before region or register grouping."""
    return product(extent for _name, extent in original_prefetch_unit_modes(theta, operand))


def prefetch_unit_tiles(theta, operand) -> int:
    """Physical tiles in one region's prefetch unit, before register grouping."""
    return product(extent for _name, extent in prefetch_unit_modes(theta, operand))


def group_prefetch_unit_tiles(theta, operand) -> int:
    """Physical tiles in one group's prefetch unit.

    Grouping changes the prefetch unit only when it partitions the prefetch axis itself.
    """
    unit = prefetch_unit_tiles(theta, operand)
    grouping = grouping_mode_name(theta, operand)
    if grouping and grouping in {name for name, _extent in prefetch_unit_modes(theta, operand)}:
        groups = max(1, len(operand.fragment.groups()))
        if unit % groups:
            raise RuntimeError(
                f"{operand.name}: prefetch unit {unit} is not divisible by {groups} groups")
        unit //= groups
    vg_axis = vgpr_group_axis(theta, operand)
    if vg_axis is not None:
        unit //= vg_axis[1]
    return max(1, unit)


def _covered_positions(theta, operand, modes, grouped=False) -> int:
    """Movement positions represented by physical ``modes`` after read coverage."""
    hop = operand.trajectory.shared_read
    factors = (coverage_factors(theta, operand, getattr(hop, "coverage", None))
               if hop is not None else {})
    grouping = grouping_mode_name(theta, operand) if grouped else None
    groups = max(1, len(operand.fragment.groups()))
    vg_axis = vgpr_group_axis(theta, operand) if grouped else None
    total = 1
    for name, extent in modes:
        if grouping == name:
            extent //= groups
        if vg_axis is not None and name == vg_axis[0]:
            extent //= vg_axis[1]
        total *= max(1, extent // max(1, int(factors.get(name, 1))))
    return max(1, total)


def group_prefetch_unit_positions(theta, operand) -> int:
    """Read-order positions in one grouped prefetch unit."""
    return _covered_positions(theta, operand, prefetch_unit_modes(theta, operand), grouped=True)


def rotation_unit_tiles(theta, operand) -> int:
    return product(extent for _name, extent in rotation_unit_modes(theta, operand))


def rotation_unit_positions(theta, operand) -> int:
    """Read-order positions in the ungrouped rotation unit."""
    return _covered_positions(theta, operand, rotation_unit_modes(theta, operand))


def group_rotation_unit_tiles(theta, operand) -> int:
    """Tiles in one region of one register group's rotation unit."""
    regions = set(getattr(operand, "region_axes", ()) or ())
    modes = [(name, extent) for name, extent in rotation_unit_modes(theta, operand)
             if name not in regions]
    unit = product(extent for _name, extent in modes)
    grouping = grouping_mode_name(theta, operand)
    if grouping and grouping in {name for name, _extent in modes}:
        groups = max(1, len(operand.fragment.groups()))
        if unit % groups:
            raise RuntimeError(
                f"{operand.name}: rotation unit {unit} is not divisible by {groups} groups")
        unit //= groups
    return max(1, unit)


def grouping_preserves_coverage(theta, operand, groups) -> bool:
    """Whether contiguous ``groups`` keep every read carrier preimage whole."""
    groups = max(1, int(groups))
    if groups <= 1:
        return True
    modes = grouping_modes(theta, operand)
    if not modes:
        return False
    mode, extent = modes[0]
    if extent % groups:
        return False
    hop = operand.trajectory.shared_read
    coverage = getattr(hop, "coverage", None) if hop is not None else None
    if coverage is None:
        return True

    axes = _coverage_candidate_axes(theta, operand)
    if mode not in {name for name, _extent in axes}:
        return False
    block = extent // groups
    carriers = {}
    ranges = [range(max(1, axis_extent)) for _name, axis_extent in axes]
    for tile, values in enumerate(_iproduct(*ranges)):
        coord = dict(zip((name for name, _extent in axes), values))
        group = int(coord[mode]) // block
        carrier = coverage.carrier_of(tile)
        previous = carriers.setdefault(carrier, group)
        if previous != group:
            return False
    return True


def reloads_whole_set(theta, operand) -> bool:
    """Whether this operand is invariant over the global outermost logical role."""
    return theta.geometry(operand).reloads_whole_set


def group_unit_tile_count(theta, operand) -> int:
    """Tiles in one region of one group's rotation unit."""
    return group_rotation_unit_tiles(theta, operand)


def group_unit_tiles(theta, operand, group) -> int:
    """Tiles in one rotation unit of this group."""
    return group_unit_tile_count(theta, operand)


def group_owned_tiles(theta, operand) -> int:
    """Tiles from one chunk assigned to one group."""
    return group_unit_tiles(theta, operand, operand.fragment.groups()[0]) \
        * group_fan_reloads(theta, operand, operand.fragment.groups()[0])


def group_fan_reloads(theta, operand, group) -> int:
    """Distinct rotation units this group visits in one chunk."""
    if reloads_whole_set(theta, operand):
        return 1
    reduction_ring = _reduction_prefetch_ring_modes(theta, operand)
    if reduction_ring:
        return product(extent for _name, extent in reduction_ring)
    return product(extent for _name, extent in unit_enumerator_modes(theta, operand))


def group_ring_size(theta, operand, group) -> int:
    """Rotation units this group visits in one chunk."""
    return group_fan_reloads(theta, operand, group)


def concurrent_regions(theta, operand) -> int:
    """Regions live at once: those the operand cycles through INSIDE an axis it is invariant over.

    A region outside every invariant axis is visited once and retired; one nested inside one is
    revisited on every pass, so all of them are in flight together.
    """
    order = [axis.name for axis in theta.inner_axes()]
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in theta.inner_axes()
    }
    varying = set(varying_axes(theta, operand))
    invariant = [i for i, name in enumerate(order)
                 if name not in varying and max(1, extents.get(name, 1)) > 1]
    if not invariant:
        return 1
    outermost = min(invariant)
    span = getattr(operand, "region_span", {}) or {}
    live = 1
    for name in (getattr(operand, "region_axes", ()) or ()):
        if name in order and order.index(name) > outermost:
            live *= max(1, max(1, int(extents.get(name, 1))) // max(1, int(span.get(name, 1))))
    return max(1, live)


def group_live_peak(theta, operand, group, post_war=True, off=None) -> int:
    """PU-sized ring slots simultaneously live in one group."""
    if off is None:
        if post_war:
            return concurrent_regions(theta, operand)
        off = prefetch_distance_for(
            theta, operand, requested_read_ahead(theta, operand), (group,))
    if off <= 0:
        return 1
    stride = group_register_positions(theta, operand)
    return (1 + -(-int(off) // max(1, stride))) * concurrent_regions(theta, operand)


def derived_group_reuse_floor(theta, operand, group) -> int:
    """Reuse floor in prefetch-unit buffers, before the adapter records VA."""
    cache = getattr(theta, "_derived_group_reuse_floor_cache", None)
    if cache is None:
        cache = {}
        setattr(theta, "_derived_group_reuse_floor_cache", cache)
    cache_key = (operand.name, tuple(operand.fragment.groups()),
                 getattr(operand.fragment, "grouping_mode", None), group)
    if cache_key in cache:
        return cache[cache_key]
    labels = operand.fragment.groups()
    group_number = labels.index(group) if group in labels else 0
    axes = [axis.name for axis in presence_axes(theta, operand)]
    intervals = {}
    for time, step in enumerate(_inner_steps(theta)):
        coord = {name: step.get(name, 0) for name in axes}
        if group_index(theta, operand, coord) != group_number:
            continue
        source = tuple(coord[name] for name in axes)
        first, _last = intervals.get(source, (time, time))
        intervals[source] = (first, time)

    maximum = _ring_period(theta, operand)
    for width in range(1, maximum + 1):
        slot = ring_slot(theta, operand, group, width)
        by_register = {}
        for source, live in intervals.items():
            coord = dict(zip(axes, source))
            register_key = (slot.evaluate(coord), unit_tile_index(theta, operand, coord))
            by_register.setdefault(register_key, []).append(live)
        if all(previous[1] < current[0]
               for lives in by_register.values()
               for previous, current in zip(sorted(lives), sorted(lives)[1:])):
            cache[cache_key] = width
            return width
    cache[cache_key] = maximum
    return maximum


def group_reuse_floor(theta, operand, group, charge_lead=True, lead=None) -> int:
    """Smallest ring that retains one read across every invariant-axis consumer pass."""
    if vgpr_group_axis(theta, operand) is None:
        return derived_group_reuse_floor(theta, operand, group)
    cache = getattr(theta, "_group_reuse_floor_cache", None)
    if cache is None:
        cache = {}
        setattr(theta, "_group_reuse_floor_cache", cache)
    cache_key = (operand.name, tuple(operand.fragment.groups()),
                 getattr(operand.fragment, "grouping_mode", None), group, charge_lead, lead)
    if cache_key in cache:
        return cache[cache_key]
    hop = operand.trajectory.shared_read
    labels = operand.fragment.groups()
    group_number = labels.index(group) if group in labels else 0
    axes = [axis.name for axis in presence_axes(theta, operand)]
    intervals = {}
    for time, step in enumerate(_inner_steps(theta)):
        coord = {name: step.get(name, 0) for name in axes}
        if group_index(theta, operand, coord) != group_number:
            continue
        source = tuple(coord[name] for name in axes)
        first, _last = intervals.get(source, (time, time))
        intervals[source] = (first, time)

    # The flattened walk already includes every outer digit and every VG subdivision.  Its
    # period is therefore the complete search ceiling; scaling it by regions or VG would count
    # those factors twice.
    maximum = _ring_period(theta, operand)
    # WHEN THE REFILL LANDS, counted in the operand's own read order: a read-ahead issues the
    # load `lead` sources before the value is wanted, so the slot must be free that early.  The
    # consumption windows alone are disjoint either way; what separates a shape that fits from
    # one that does not is whether the refill falls inside the previous value's window.
    read_order = sorted(intervals, key=lambda source: intervals[source][0])
    at = {source: index for index, source in enumerate(read_order)}
    # THE CALLER MAY NAME THE LEAD.  A refusal has to judge the read-ahead that will actually be
    # SCHEDULED, and `_placeable_read_ahead` lowers the request when the refill has no legal
    # position -- charging the raw request refused kernels for a pipeline depth they never get,
    # including their own DERIVED depth.
    if lead is None:
        lead = prefetch_distance_for(
            theta, operand, requested_read_ahead(theta, operand), (group,))
    # A LEAD OF `L` KEEPS `L + 1` VALUES LIVE -- the one in use and the ones in flight -- so no
    # ring shorter than that can hold them, whatever the reuse windows say.  It is also what the
    # window probe cannot see once the lead runs off the front of the read order: `at - lead`
    # clamps at zero, every width then fails, and the loop would fall through to its ceiling.
    # AND `lead` IS ALREADY IN RING POSITIONS.  Scaling this by a region count charges the same
    # traversal twice -- an all-outer split operand with completely DISJOINT windows ([0,1] [2,3]
    # [4,5] [6,7], the probe passes at width 1) was charged 6 and refused its own DERIVED depth.
    # The two kernels that are numerically wrong on hardware are a SCHEDULING defect, not an
    # under-allocation: raising this to refuse them would hide it.  See #556.
    in_flight = max(1, int(lead) + 1)
    for width in range(1, maximum + 1):
        slot = ring_slot(theta, operand, group, width)
        by_register = {}
        for source, live in intervals.items():
            coord = dict(zip(axes, source))
            register_key = (slot.evaluate(coord), unit_tile_index(theta, operand, coord))
            # CHARGING THE LEAD HERE ASSUMES AN UNANCHORED REFILL.  `reload_anchor` defers the
            # reload past the last reader of the value it overwrites, so the early landing this
            # models does not happen.  The SCHEDULER still wants the charged number -- dropping
            # it globally plans a worse pipeline -- but a REFUSAL must not use it: `VG=2, VA=3,
            # PLR=2` is charged 4 and verifies clean at 3, measured.
            refill = intervals[read_order[max(0, at[source] - lead)]][0] \
                if charge_lead else live[0]
            by_register.setdefault(register_key, []).append((live[0], live[1], refill))
        if all(previous[1] < current[2]
               for lives in by_register.values()
               for previous, current in zip(sorted(lives), sorted(lives)[1:])):
            cache[cache_key] = max(width, in_flight)
            return cache[cache_key]
    cache[cache_key] = max(maximum, in_flight)
    return cache[cache_key]


def _raw_read_ahead(theta, operand) -> int:
    """PLR as the model counts it: VG-sized buffers.

    VG changes each buffer's width; it does not divide the number of buffers requested.
    """
    if not operand.movements:
        return 0
    axis = prefetch_axis_name(theta, operand)
    if axis is None:
        return 0
    return max(0, int(theta.off_at(operand.name, DS_LOAD, axis) or 0))


def requested_read_ahead(theta, operand, depths=None) -> int:
    """Internal per-operand PLR, in this operand's grouped-prefetch units.

    The adapter normalizes VG after TDMSplit, and PLR already counts the resulting structural
    prefetch unit.  In particular, a split region changes the flattened ring offset; it does not
    multiply the requested lead. ``depths`` applies the VA capacity cap.
    """
    want = _raw_read_ahead(theta, operand)
    if not want:
        return 0

    if reloads_whole_set(theta, operand):
        outer = theta.summation_chunk_name()
        staged = (
            theta.off_at(operand.name, TENSOR_LOAD, outer)
            if outer is not None else 0
        )
        per_chunk = Fraction(rotation_unit_tiles(theta, operand),
                             group_prefetch_unit_tiles(theta, operand))
        staged_units = max(0, staged - 1) * per_chunk
        want = min(want, staged_units.numerator // staged_units.denominator)

    if depths is None:
        return want
    want = min(want, prefetch_capacity(theta, operand, depths))
    return _staged_prefetch_cap(theta, operand, want)


def prefetch_capacity(theta, operand, depths) -> int:
    """VGPR capacity expressed in this operand's grouped-prefetch units."""
    return sum(max(1, int(depths.get(operand.name, group)))
               for group in operand.fragment.groups())


def _staged_prefetch_cap(theta, operand, want) -> int:
    """Deepest operand PLR whose source chunk the copy/LDS pipeline has staged."""
    outer = theta.summation_chunk_name()
    if outer is None or not any(hop.dst is SHARED for hop in operand.movements):
        return max(0, int(want))
    copy_offset = theta.off_at(operand.name, TENSOR_LOAD, outer)
    shared = lds_buffers(theta, operand)
    for steps in range(max(0, int(want)), -1, -1):
        crossed = chunks_crossed(theta, operand, steps)
        if crossed <= 0:
            return steps
        can_lead = copy_offset == crossed and shared > copy_offset
        need = crossed if can_lead else crossed + 1
        if copy_offset >= need:
            return steps
    return 0


def _free_tile_index_seq(theta, operand, steps):
    free = set(free_names(theta, operand))
    free_axes = [axis for axis in theta.inner_axes() if axis.name in free]
    seq = []
    for step in steps:
        idx = 0
        for j, axis in enumerate(free_axes):
            inner = 1
            for m2 in free_axes[j + 1:]:
                inner *= int(m2.extent.constant_value)
            idx += step.get(axis.name, 0) * inner
        seq.append(idx)
    return seq


def resident_free_tiles(theta, operand) -> int:
    steps = _inner_steps(theta)
    T = len(steps)
    if T == 0:
        return 1
    off = requested_read_ahead(theta, operand)
    seq = _free_tile_index_seq(theta, operand, steps)
    if not any(seq):  # operand has no own free mode -> 1 tile
        return 1
    gens = {}
    niter = 3
    for s in range(niter):
        for i, value in enumerate(seq):
            step = s * T + i
            gens.setdefault((s, value), [step, step])[1] = step
    intervals = [(first - off, second) for first, second in gens.values()]
    peak = 0
    for p in range(T, 2 * T):  # sample the steady iteration
        peak = max(peak, sum(1 for first, second in intervals if first <= p <= second))
    return max(1, peak)


def lds_buffers(theta, operand) -> int:
    """S_shared: the shared ring buffer depth (per reduction chunk)/."""
    if not any(hop.dst is SHARED for hop in operand.movements):
        raise RuntimeError(
            f"{operand.name} has no shared placement (hops={[f'{hop.src}->{hop.dst}' for hop in operand.movements]}), "
            f"so S_shared is undefined for it -- the caller should have guarded on the operand "
            f"touching SHARED before asking for a shared-ring depth.")
    outer_var = theta.summation_chunk_name()
    d = (
        theta.off_at(operand, TENSOR_LOAD, outer_var)
        if outer_var is not None else 0
    )
    if operand.lds_buffers:
        return max(1, int(operand.lds_buffers))
    return max(1, max(0, d))  # unstated -> delta (see the docstring's warning)


# ---  ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MmaShape:
    """The compute grid decoded for one kiter."""
    m_tiles: int; n_tiles: int; k_tiles: int
    @property
    def count(self) -> int:
        return self.m_tiles * self.n_tiles * self.k_tiles


def group_register_positions(theta, operand) -> int:
    """Physical register positions in one ``(group, ring slot)``: exactly one PU."""
    hop = operand.trajectory.shared_read
    if hop is not None and hop.coverage is not None:
        ring = {name for name, _extent in ring_axes(
            theta, operand, operand.fragment.groups()[0])}
        modes = [(name, extent) for name, extent in theta.geometry(operand).varying
                 if name not in ring]
        grouping = grouping_mode_name(theta, operand)
        groups = max(1, len(operand.fragment.groups()))
        positions = product(
            max(1, extent // groups) if name == grouping else extent
            for name, extent in modes
        )
        active = len({
            (hop.coverage.carrier_of(tile), hop.coverage.slot_of(tile))
            for tile in range(max(1, positions))
        })
        agents = max(1, int(getattr(operand.fragment, "agent_group_size", 0) or 1))
        return max(1, active * agents)
    return group_prefetch_unit_positions(theta, operand)


def operand_footprint_regs(theta, operand, depths) -> int:
    """Logical VGPR footprint if register groups are packed independently."""
    if operand.is_output:
        return max(1, free_tiles(theta, operand) * frag_regs(theta, operand))
    positions = group_register_positions(theta, operand)
    total = sum(group_ring_depth(theta, operand, group, depths) * positions
                for group in operand.fragment.groups())
    return max(1, total) * frag_regs(theta, operand)


def operand_emitted_regs(theta, operand, depths) -> int:
    """VGPRs reserved for GIR plus the scaffold-owned tail-loop read footprint."""
    return max(operand_footprint_regs(theta, operand, depths),
               max(1, free_tiles(theta, operand) * frag_regs(theta, operand)))


def vgpr_group(theta, operand) -> int:
    """VG -- the factor this operand's prefetch unit is DIVIDED by.

    Not a group count and not a region count: it divides the current prefetch unit after structural
    TDMSplit factors have already been removed from that unit.
    """
    return max(1, int(getattr(operand.fragment, "vgpr_group", 1) or 1))


def vgpr_subdivided(theta, operand) -> bool:
    """Whether this operand's unit is smaller than its original unsplit prefetch unit.

    TDMSplit first changes the structural PU; VG independently divides that resulting PU.
    """
    return split_unit_divisor(theta, operand) > 1 or vgpr_group(theta, operand) > 1


def split_unit_divisor(theta, operand) -> int:
    """The factor TDMSplit removes from the unsplit PU, independent of VG."""
    return max(1, original_prefetch_unit_tiles(theta, operand)
               // max(1, prefetch_unit_tiles(theta, operand)))


def group_ring_depth(theta, operand, group, depths) -> int:
    """Physical PU-sized slots in this group's VA ring."""
    return max(1, int(depths.get(operand.name, group)))


def register_layout(theta, depths) -> dict:
    """Per-operand ``(group, slot) -> VGPR offset`` and total compact allocation."""
    layout = {}
    for operand in theta.operands:
        if operand.is_output or not operand.movements:
            continue
        offset, slots = 0, {}
        fragment_regs = frag_regs(theta, operand)
        unit_regs = group_register_positions(theta, operand) * fragment_regs
        for group_index, group in enumerate(operand.fragment.groups()):
            for slot in range(group_ring_depth(theta, operand, group, depths)):
                slots[(group_index, slot)] = offset
                offset += unit_regs
        layout[operand.name] = {"slots": slots, "total": offset}
    return layout


def _region_count(theta, operand) -> int:
    """Storage regions that are factors of this operand's rotation unit."""
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in theta.inner_axes()
    }
    span = getattr(operand, "region_span", {}) or {}
    rotation = {name for name, _extent in rotation_unit_modes(theta, operand)}
    regions = [max(1, int(extents.get(axis, 1))) // max(1, int(span.get(axis, 1)))
               for axis in (getattr(operand, "region_axes", ()) or ())
               if axis in extents and axis in rotation]
    return product(regions) if regions else 1


# ===========================================================================

def operand_buffer_regs(theta, operand) -> int:
    """Registers this operand loads per kiter = the axes it varies over tiles x frag_regs."""
    return presence_tiles(theta, operand) * frag_regs(theta, operand)


def load_regs(theta, operand, hop) -> int:
    """Registers moved by ONE per-lane instruction = vector width packed into regs."""
    vw = hop.vector_elems or operand.frag_elems
    return max(1, ceil(vw * operand.elem_bytes / theta.reg_bytes))


def operand_tile_bytes(theta, operand) -> int:
    return presence_tiles(theta, operand) * operand.frag_elems * operand.elem_bytes * theta.lanes


def plan_transfers(theta, operand, hop) -> int:
    """Instructions this hop issues per kiter."""
    if hop.is_bulk:
        return max(1, hop.split)
    return max(1, ceil(operand_buffer_regs(theta, operand) / load_regs(theta, operand, hop)))


# ===========================================================================

def mma_shape(theta) -> MmaShape:
    """m_tiles x n_tiles x k_tiles for one kiter, derived from wmma_grid."""
    axis, n, k = wmma_grid(theta)
    return MmaShape(axis, n, k)


# --- integer layout arithmetic ------------------------------------------------------------

def tile_split(n: int, t: int) -> tuple:
    n = max(1, n)
    t = max(1, t)
    if n % t == 0:
        return (t, n // t)
    return (min(t, n), -(-n // t))


def n_tiles(n: int, t: int) -> int:
    """The tile count along an axis = `tile_split(n,t)[1]` = n//t (divisible)."""
    return max(1, tile_split(n, t)[1])


def product(extents) -> int:
    """Size of a `Layout(tuple(extents))` = product of the extents."""
    p = 1
    for e in extents:
        p *= max(1, int(e))
    return max(1, p)


def recast_count(elems: int, scale: int) -> int:
    elems = max(1, elems)
    scale = max(1, scale)
    return max(1, -(-elems // scale))


# --- what an operand varies over -------------------------------------------------------------

def summation_axes(theta):
    """The axes the output is not present on: the summation (K) axes."""
    on_output = set()
    for operand in theta.output_operands():
        on_output.update(axis.name for axis in presence_axes(theta, operand))
    served_by_waves = theta.wave_served_axes()
    return [axis for axis in theta.inner_axes()
            if axis.name not in on_output and axis.name not in served_by_waves]


def summation_names(theta):
    return set(theta._summation_names)


def free_axes(theta, operand):
    """The operand's own free (M or N) axes: the ones it varies over that are not summation."""
    summation = summation_names(theta)
    return [axis for axis in presence_axes(theta, operand) if axis.name not in summation]


def free_names(theta, operand):
    return [axis.name for axis in free_axes(theta, operand)]


def presence(theta, operand):
    return [axis.name for axis in presence_axes(theta, operand)]


def presence_axes(theta, operand, hop=None):
    """The axes this operand's data changes over on one transfer, or over all of them."""
    if hop is None:                                   # settled at construction; no coverage
        return list(theta.geometry(operand).presence)
    constant_over = transfer_broadcast(theta, operand, hop)
    coverage = transfer_coverage(theta, operand, hop)
    return [axis if axis.name not in coverage
            else Axis(
                axis.name,
                max(
                    1,
                    int(axis.extent.constant_value) // coverage[axis.name],
                ),
            )
            for axis in theta.inner_axes() if axis.name not in constant_over]


# --- the axes it varies over --------------------------------------------------------------

def prefetch_axis_name(theta, operand=None):
    axis = readahead_level(theta)
    return axis[0] if axis else None


def operand_level(theta, operand, operation):
    chunk = theta.summation_chunk_name()
    if operation is TENSOR_LOAD:
        inner = {axis.name for axis in theta.inner_axes()}
        region = next((r for r in getattr(operand, "region_axes", ()) if r in inner), None)
        return region if (region is not None and max(1, operand.split) > 1) else chunk
    varying = varying_axes(theta, operand)
    return varying[-1] if varying else chunk


def varying_axes(theta, operand):
    hop = operand.trajectory.fragment_fill
    constant_over = (broadcast_axes(theta, operand) if hop is None
                     else transfer_broadcast(theta, operand, hop))
    return [axis.name for axis in theta.inner_axes() if axis.name not in constant_over]


def reload_modes(theta, operand):
    """Operand read-order modes within one reduction chunk."""
    return list(varying_axes(theta, operand))


def _reduction_prefetch_ring_modes(theta, operand):
    """Reduction ring used when a free prefetch-unit factor is packed into each slot."""
    return list(theta.geometry(operand).reduction_ring)


def ring_axes(theta, operand, group):
    """Outer modes converted to this operand's dense PU-ring offset."""
    geometry = theta.geometry(operand)
    hop = operand.trajectory.shared_read
    if hop is not None and hop.coverage is not None:
        return list(geometry.summation_modes)
    unit = {name for name, _extent in geometry.prefetch_unit}
    return [(name, extent) for name, extent in geometry.varying if name not in unit]


def ring_slot(theta, operand, group, buffers, advance=0, strides=None):
    """Rotation slot of the coordinate reached by ``advance`` physical tile positions."""
    if buffers <= 1:
        return Expr.constant(0)
    extents = transfer_extents(theta, operand)
    varying = varying_axes(theta, operand)
    walk_strides = strides or axis_strides(theta, varying, extents)[0]
    if reloads_whole_set(theta, operand) and len(operand.fragment.groups()) == 1:
        shifted = _shifted_ring_slot(theta, operand, group, int(advance), walk_strides, buffers,
                                     read_coverage(theta, operand))
        return shifted if shifted is not None else Expr.constant(0)
    shifted = _shifted_ring_slot(theta, operand, group, int(advance), walk_strides, buffers,
                                 read_coverage(theta, operand))
    return shifted if shifted is not None else Expr.constant(0)


def _summation_ring_axis(theta, operand, group):
    summation = summation_names(theta)
    on_summation = [(name, extent) for name, extent in ring_axes(theta, operand, group)
                    if name in summation]
    return on_summation[-1] if on_summation else None


def axis_strides(theta, modes=None, extents=None):
    axes = list(theta.inner_axes())
    if modes is not None:
        axes = [axis for axis in axes if axis.name in modes]
    strides, place = {}, 1
    for axis in reversed(axes):  # innermost axis has stride 1
        strides[axis.name] = place
        place *= (extents or {}).get(
            axis.name,
            int(axis.extent.constant_value),
        )
    return strides, place       # place is now the product over `modes`


def transfer_extents(theta, operand) -> dict:
    """`{mode: extent}` as the operand's read hop sees them -- its the axes it varies over extents."""
    hop = operand.trajectory.fragment_fill
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in theta.inner_axes()
    }
    if hop is not None:
        extents.update({
            axis.name: int(axis.extent.constant_value)
            for axis in presence_axes(theta, operand, hop)
        })
    return extents


def read_coverage(theta, operand) -> dict:
    """`{mode: q}` -- the partial coverage factor of the operand's read hop or `{}`."""
    hop = operand.trajectory.fragment_fill
    return {} if hop is None else transfer_coverage(theta, operand, hop)


def broadcast_width(theta, operand) -> int:
    varying = set(varying_axes(theta, operand))
    if not varying:
        return 1
    inner_names = [axis.name for axis in theta.inner_axes()]
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in theta.inner_axes()
    }
    outermost_varying = min(inner_names.index(name) for name in varying)
    width = 1
    for name in inner_names[:outermost_varying]:
        if name not in varying and extents.get(name, 1) > 1:
            width *= extents[name]
    return width


def position_terms(strides, fold=None):
    return tuple((name, coef, max(1, int((fold or {}).get(name, 1))))
                 for name, coef in strides.items())


def _shifted_coord(theta, axis, shift, strides, _reload_span, extents, fold=None):
    name = axis.name
    if name not in strides:
        return axis
    terms = position_terms(strides, fold)
    coverage = max(1, int((fold or {}).get(name, 1)))
    stride = max(1, strides[name])
    extent = max(1, int(extents.get(name, 1)))
    position = _position(theta, terms, shift)
    if coverage > 1:
        return coverage * ((position // stride) % extent)
    return (position // stride) % extent


def _ring_period(theta, operand, extra_axes=()) -> int:
    """PU-sized VA offsets traversed by one chunk.

    Every varying axis participates in the flattened position.  Region axes therefore advance
    the offset exactly like any other outer digit; neither VA nor the ring needs a region mode.
    """
    ring_modes = ring_axes(theta, operand, operand.fragment.groups()[0])
    ring_names = dict(ring_modes)
    extras = {getattr(axis, "name", axis) for axis in (extra_axes or ())}
    ring_modes = [
        (axis.name, ring_names.get(
            axis.name, int(axis.extent.constant_value)))
        for axis in theta.inner_axes()
        if axis.name in ring_names or axis.name in extras
    ]
    ring_positions = _covered_positions(theta, operand, ring_modes)
    ring_names = {name for name, _extent in ring_modes}
    unit_modes = [(name, extent) for name, extent in theta.geometry(operand).varying
                  if name not in ring_names]
    full_unit = _covered_positions(theta, operand, unit_modes)
    unit = group_register_positions(theta, operand)
    subdivisions = max(1, -(-full_unit // max(1, unit)))
    return max(1, ring_positions * subdivisions)


def vgpr_ring_carries(theta, operand, group, buffers, extra_axes=()) -> bool:
    """Whether one trip advances this VA ring to a different starting phase."""
    return _ring_period(theta, operand, extra_axes) % max(1, int(buffers)) != 0


def vgpr_group_axis(theta, operand):
    """The axis VG subdivides and by how much -- ``(name, radix)`` -- or ``None``.

    VG divides the unit, so a unit the enumerator counts once is now VG buffers; the factor that
    tells them apart is the TOP factor of the current innermost prefetch-unit axis. TDMSplit has
    already been removed from that unit and is not divided out again here.
    """
    radix = vgpr_group(theta, operand)
    modes = [(name, extent) for name, extent in prefetch_unit_modes(theta, operand)
             if max(1, extent) % radix == 0]
    return (modes[-1][0], radix) if radix > 1 and modes else None


def ring_positions(theta, operand, group, buffers, extra_axes=()) -> int:
    """Rotation positions one trip walks -- the PERIOD of the slot expression.

    Where it is a multiple of the depth the ring closes every trip; where it is not -- four
    positions over three buffers -- the slot carries into the next one.
    """
    return _ring_period(theta, operand, extra_axes)


def ring_trip_copies(theta, depths) -> int:
    """Trips before EVERY ring is back where it started -- the steady body's copy count.

    A ring of `VA` buffers walked `P` positions a trip returns after `VA/gcd(VA, P)` of them, so
    the body is one fixed piece of code only when replicated that many times.  `VA=3` over four
    positions is the 1.5-buffer rotation and asks for three; everything that divides asks for one.
    """
    copies = 1
    for operand in theta.operands:
        if operand.is_output or not (operand.movements and operand.fragment):
            continue
        for group in operand.fragment.groups():
            buffers = group_ring_depth(theta, operand, group, depths)
            period = ring_positions(theta, operand, group, buffers)
            turns = max(1, buffers // gcd(buffers, max(1, period)))
            copies = copies * turns // gcd(copies, turns)
    return max(1, copies)


def _shifted_ring_slot(theta, operand, group, shift, strides, buffers, fold=None):
    """The PU-sized ring slot the prefetched read writes, as an Expr."""
    extents = transfer_extents(theta, operand)
    position = _position(theta, position_terms(strides, fold), int(shift))

    base_ring = dict(ring_axes(theta, operand, group))
    varying = {name for name, _extent in theta.geometry(operand).varying}
    ring_modes = [
        (axis.name, base_ring.get(
            axis.name, int(axis.extent.constant_value)))
        for axis in theta.inner_axes()
        if axis.name in base_ring
        or (axis.name in strides and axis.name not in varying)
    ]
    ring_value = Expr.constant(0)
    for name, extent in ring_modes:
        if name not in strides:
            continue
        digit = (position // max(1, int(strides[name]))) % max(
            1, int(extents.get(name, extent)))
        ring_value = max(1, int(extent)) * ring_value + digit

    ring_names = {name for name, _extent in ring_modes}
    unit_modes = [(name, extent) for name, extent in theta.geometry(operand).varying
                  if name not in ring_names]
    unit_width = group_register_positions(theta, operand)
    full_unit = _covered_positions(theta, operand, unit_modes)
    subdivisions = max(1, -(-full_unit // max(1, unit_width)))
    unit_value = Expr.constant(0)
    for name, extent in unit_modes:
        if name not in strides:
            continue
        digit = (position // max(1, int(strides[name]))) % max(
            1, int(extents.get(name, extent)))
        unit_value = max(1, int(extent)) * unit_value + digit
    subunit = (unit_value // max(1, unit_width)) % subdivisions
    walk_span = max(
        (
            max(1, int(stride)) * max(1, int(extents.get(name, 1)))
            for name, stride in strides.items()
        ),
        default=1,
    )
    generation = position // max(1, walk_span)
    return (
        generation * _ring_period(theta, operand)
        + subdivisions * ring_value
        + subunit
    ) % max(1, buffers)


# --- registers -------------------------------------------------------------

SAFE = "safe"
IN_PLACE = "inplace"
CLOBBER = "clobber"

TRIPS_WALKED = 3


def chunks_crossed(theta, operand, prefetch_steps):
    """How many reduction chunks this read-ahead reaches into -- the DISTANCE over the chunk.

 Asked of `prefetch_distance_for`, never re-derived: a second distance here is how the whole-set
 arm went unseen, so the staging bound passed a schedule needing a chunk the copy had not loaded
 and it surfaced as a clobber in the ring walk instead of a refusal at the adapter."""
    if prefetch_steps <= 0:
        return 0
    distance = prefetch_distance_for(theta, operand, prefetch_steps)
    if distance <= 0:
        return 0
    _strides, span = axis_strides(theta, reload_modes(theta, operand),
                                  transfer_extents(theta, operand))
    span = max(1, span)
    return (span - 1 + distance) // span


def prefetch_distance_for(theta, operand, prefetch_steps, groups=None) -> int:
    if not prefetch_steps:
        return 0
    return int(prefetch_steps) * group_prefetch_unit_positions(theta, operand)


@dataclass(frozen=True, order=True)
class Generation:
    """Which load put a value in a register, and the coordinates that load did not move."""
    trip: int
    position: int
    pinned: tuple


@dataclass(frozen=True, order=True)
class Event:
    """A register is loaded, or read, at one point in the walk."""
    time: int
    is_load: bool
    generation: Generation


@dataclass(frozen=True)
class RegisterTimeline:
    """Per register, every load and every read, over three trips of the loop body."""
    events: dict
    steps: tuple
    steps_per_trip: int

    def is_steady(self, time) -> bool:
        return self.steps_per_trip <= time < 2 * self.steps_per_trip


@dataclass(frozen=True)
class _RingWalk:
    """The fixed geometry one register-timeline walk needs, so the walk itself is just a loop."""
    varying: tuple
    varying_strides: dict
    reload_strides: dict
    reload_span: int
    extents: dict
    fan_axes: tuple      # the axes that name a distinct register, not a slot in the ring
    pinned_axes: tuple   # the axes a reload does not move
    ring: tuple
    buffers: int

    def register_of(self, coord):
        """Which register this coordinate lives in: its fan position plus its ring slot."""
        slot, place = 0, 1
        for name, extent in reversed(self.ring):
            slot += coord.get(name, 0) * place
            place *= extent
        return (tuple(coord.get(name, 0) for name in self.fan_axes), slot % max(1, self.buffers))

    def position(self, coord):
        return sum(self.varying_strides[name] * coord[name] for name in self.varying)

    def reload_position(self, coord):
        return sum(self.reload_strides[name] * coord[name] for name in self.reload_strides)

    def advanced(self, coord, position):
        """`coord` moved to the reload position `position`, leaving the pinned axes alone."""
        out = dict(coord)
        out.update({name: (position // self.reload_strides[name])
                    % max(1, self.extents.get(name, 1)) for name in self.reload_strides})
        return out


def _ring_walk(theta, operand, group, buffers, varying):
    reload_axes = reload_modes(theta, operand) or varying
    varying_strides, _ = axis_strides(theta, varying)
    reload_strides, reload_span = axis_strides(theta, reload_axes)
    summation = set(summation_names(theta))
    ring = tuple(ring_axes(theta, operand, group))
    ring_names = {name for name, _extent in ring}
    return _RingWalk(
        varying=tuple(varying), varying_strides=varying_strides,
        reload_strides=reload_strides, reload_span=reload_span,
        extents={
            axis.name: int(axis.extent.constant_value)
            for axis in theta.inner_axes()
        },
        fan_axes=tuple(name for name in varying if name not in summation
                       and name not in ring_names),
        pinned_axes=tuple(name for name in varying if name not in reload_strides),
        ring=ring, buffers=buffers)


def _build_register_timeline(
    theta,
    operand,
    group,
    buffers,
    distance,
    at,
):
    """Build one register timeline with trip-invariant placement computed once."""

    varying = varying_axes(theta, operand)
    if not varying or not distance:
        return None
    walk = _ring_walk(theta, operand, group, buffers, varying)
    steps = _inner_steps(theta)
    per_trip = len(steps)
    labels = operand.fragment.groups()
    group_number = labels.index(group) if group in labels else 0
    loaded_positions = set()
    static_steps = []
    for index, step in enumerate(steps):
        coord = {
            name: step.get(name, 0)
            for name in walk.varying
        }
        pinned = tuple(coord[name] for name in walk.pinned_axes)
        reload_position = walk.reload_position(coord)
        read = (
            (walk.register_of(coord), reload_position, pinned)
            if group_index(theta, operand, coord) == group_number
            else None
        )
        load = None
        if not at or all(
            step.get(name, 0) == value
            for name, value in at.items()
        ):
            position = walk.position(coord)
            if position not in loaded_positions:
                loaded_positions.add(position)
                reached = reload_position + distance
                target = walk.advanced(coord, reached)
                if group_index(theta, operand, target) == group_number:
                    load = walk.register_of(target), reached, pinned
        static_steps.append((index, read, load))

    events = {}
    reload_span = max(1, walk.reload_span)
    for trip in range(TRIPS_WALKED):
        for index, read, load in static_steps:
            time = trip * per_trip + index
            if read is not None:
                register, position, pinned = read
                events.setdefault(register, []).append(Event(
                    time,
                    False,
                    Generation(trip, position, pinned),
                ))
            if load is not None:
                register, reached, pinned = load
                events.setdefault(register, []).append(Event(
                    time,
                    True,
                    Generation(
                        trip + reached // reload_span,
                        reached % reload_span,
                        pinned,
                    ),
                ))
    return RegisterTimeline(
        {
            register: tuple(register_events)
            for register, register_events in events.items()
        },
        tuple(steps),
        per_trip,
    )


def _register_timeline(theta, operand, group, buffers, distance, at=None):
    """Return the cached timeline for one fixed Theta placement request."""

    key = (
        operand.name,
        group,
        int(buffers),
        int(distance),
        tuple(sorted((at or {}).items())),
    )
    if key not in theta._register_timeline_cache:
        theta._register_timeline_cache[key] = _build_register_timeline(
            theta,
            operand,
            group,
            buffers,
            distance,
            at,
        )
    return theta._register_timeline_cache[key]


def _uses_before_load(timeline):
    """Steady-trip reads that see a generation no load put there."""
    stale = {}
    for register, events in timeline.events.items():
        loaded = None
        for event in sorted(events):
            if event.is_load:
                loaded = event.generation
            elif loaded is not None and timeline.is_steady(event.time) \
                    and loaded != event.generation:
                stale.setdefault(register, []).append((event.time, event.generation, loaded))
    return stale


def _overwrites_live_register(timeline):
    """Steady-trip loads that land on a generation still read afterwards."""
    conflicts = {}
    for register, events in timeline.events.items():
        loads = sorted(event for event in events if event.is_load)
        reads = [event for event in events if not event.is_load]
        for order, load in enumerate(loads):
            if not order or not timeline.is_steady(load.time):
                continue
            overwritten = loads[order - 1].generation
            if overwritten == load.generation:
                continue
            still_read = sorted(read.time for read in reads
                                if read.generation == overwritten and read.time >= load.time)
            if still_read:
                conflicts.setdefault(register, []).append((load.time, still_read))
    return conflicts


def register_reuse_verdict(theta, operand, group, buffers, distance, at=None) -> str:
    """Can this many buffers carry a read-ahead of `distance`: SAFE, IN_PLACE, or CLOBBER?"""
    timeline = _register_timeline(theta, operand, group, buffers, distance, at=at)
    if timeline is None:
        return SAFE
    if _uses_before_load(timeline):
        return CLOBBER
    verdict = SAFE
    for _register, conflicts in _overwrites_live_register(timeline).items():
        for load_time, still_read in conflicts:
            if any(read_time > load_time for read_time in still_read):
                return CLOBBER
            verdict = IN_PLACE
    return verdict


def reload_anchor(theta, operand, group, buffers, distance):
    """Where a prefetching reload sits -- the ONE answer, searched or derived.

 `reload_positions` looks for the earliest slot the ring admits: one that follows the last reader
 of the value being overwritten, which only exists when there are buffers to spare.  When there are
 not, the END of every invariant pass is legal unconditionally, because every consumer in that pass
 has already run.  None only when even the latest position clobbers.
    """
    found = reload_positions(theta, operand, group, buffers, distance)
    if found is not None:
        return found
    extents = {
        axis.name: int(axis.extent.constant_value)
        for axis in theta.inner_axes()
    }
    invariant = {axis.name for axis in theta.inner_axes()} - set(varying_axes(theta, operand))
    at = {name: extents[name] - 1 for name in invariant if max(1, extents.get(name, 1)) > 1}
    if not at or register_reuse_verdict(theta, operand, group, buffers, distance,
                                        at=at) == CLOBBER:
        return None
    return {None: at}


def reload_positions(theta, operand, group, buffers, distance):
    """Where each reload must sit so it follows the last read of the value it overwrites.

    None means no such position exists, so this width cannot carry this read-ahead.
    """
    timeline = _register_timeline(theta, operand, group, buffers, distance)
    if timeline is None:
        return {}
    steps, steps_per_trip = timeline.steps, timeline.steps_per_trip

    after = {}
    for register, conflicts in _overwrites_live_register(timeline).items():
        for _load_time, still_read in conflicts:
            last_read = still_read[-1]
            if last_read >= 2 * steps_per_trip:
                return None
            after[register] = dict(steps[last_read % steps_per_trip])

    invariant_axes = ({axis.name for axis in theta.inner_axes()}
                      - set(varying_axes(theta, operand))) | set(read_coverage(theta, operand) or {})
    at = {}
    for name in invariant_axes:
        values = {coord[name] for coord in after.values() if name in coord}
        if len(values) > 1:
            return None
        if values:
            at[name] = next(iter(values))

    if at and register_reuse_verdict(theta, operand, group, buffers, distance, at=at) == CLOBBER:
        return None
    return after
