# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Everything true of one operand's traversal: the tile and register arithmetic, which
axes it varies over, and how its register ring turns."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product as _iproduct
from math import ceil

from .ir import COPY, Expr, READ, Space, TransferCoverage, COVERAGE_VAR, cst
from .theta import Axis, mode_role, readahead_level_of


# --- tile and register arithmetic ------------------------------------------------------------

# ===========================================================================

def axis_tiles(macro_elems: int, atom_elems: int) -> int:
    return max(1, n_tiles(macro_elems, atom_elems))


# ===========================================================================

def _inner_axes(theta):
    return [(axis.name, axis.extent) for axis in theta.inner_axes()]


def broadcast_axes(theta, operand) -> set:
    """The intra-iteration loop axes this operand is constant over (does NOT depend on).

    An operand's OWN region axes are axes it varies over -- a scale carries its parent's split
    axis and changes with it, so riding a fused walk does not make it constant over that axis.
    """
    return set(theta.geometry(operand).broadcast)


def _coverage_candidate_axes(theta, operand):
    bcast = broadcast_axes(theta, operand)
    summation = {axis.name for axis in summation_axes(theta)}
    return [(axis.name, max(1, axis.extent)) for axis in theta.inner_axes()
            if axis.name not in bcast and axis.name not in summation]


def derive_transfer_coverage(movement):
    """Build coverage from instruction grouping and agent distribution."""
    merged = max(1, int(getattr(movement, "instances_per_instruction", 1) or 1))
    agents = int(getattr(movement, "agent_group_size", 0) or 0)
    if merged <= 1 and not agents:
        return None
    tile = COVERAGE_VAR
    carrier = Expr(digits=(((tile, 1),), 0, ((1, merged, 0),)))
    if agents > 1:
        if merged % agents:
            return None
        slot = Expr(digits=(((tile, 1),), 0, ((1, agents, merged // agents),)))
    else:
        slot = Expr(digits=(((tile, 1),), 0, ((1, 1, merged),)))
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
    exts = [axis.extent for axis in free_axes(theta, operand)]
    return max(1, product(exts)) if exts else 1


def k_tiles(theta) -> int:
    exts = [axis.extent for axis in summation_axes(theta)]
    return max(1, product(exts)) if exts else 1


def summation_tiles(theta, operand) -> int:
    summation = {axis.name for axis in summation_axes(theta)}
    exts = [axis.extent for axis in presence_axes(theta, operand) if axis.name in summation]
    return max(1, product(exts)) if exts else 1


def presence_tiles(theta, operand) -> int:
    exts = [axis.extent for axis in presence_axes(theta, operand)]
    return max(1, product(exts)) if exts else 1


def wmma_grid(theta) -> tuple:
    inputs = [operand for operand in theta.operands
              if operand.is_input and any(hop.dst == Space.REGISTER for hop in operand.movements)]
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
    ranges = [[(axis.name, value) for value in range(axis.extent)] for axis in axes]
    return [dict(c) for c in _iproduct(*ranges)] if ranges else [dict()]


def readahead_level(theta, op=None):
    """The level is a property of `ord`, not of an operand: every read walks the same nest.

 `op` is accepted so call sites read the same either way, and ignored: an operand INVARIANT over
 the level is the whole-set case, which `reloads_whole_set` answers, not a different level."""
    return theta.readahead_level()


def _varying_modes(theta, operand):
    """The operand's physical tile modes in ``ord`` order.

    A read coverage changes how many instructions move those tiles; it does not change the
    prefetch or rotation unit that owns them.  Keep the physical extents here and apply the
    coverage only when naming a covered register position.
    """
    return list(theta.geometry(operand).varying)


def global_outer_role(theta):
    """Outermost logical role in ``ord``: a TDMSplit region factor selects storage, not order."""
    return theta.global_outer_role()


def _role_modes(theta, operand):
    """The operand's varying modes that order its rotation: a split axis selects storage."""
    return list(theta.geometry(operand).role_modes)


def operand_outer_role(theta, operand):
    """Outermost logical role this operand varies on."""
    return theta.geometry(operand).outer_role


def operand_inner_role(theta, operand):
    """Innermost logical role this operand varies on: the prefetch-unit axis."""
    return theta.geometry(operand).inner_role


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


def _mode_terms(theta, operand, modes):
    terms, stride = [], 1
    fold = read_coverage(theta, operand)
    for name, extent in reversed(modes):
        terms.append((name, stride, max(1, int(fold.get(name, 1)))))
        stride *= extent
    return tuple(reversed(terms))


def grouping_terms(theta, operand):
    """The one axis whose contiguous ranges form register groups."""
    return _mode_terms(theta, operand, grouping_modes(theta, operand))


def unit_enumerator_terms(theta, operand):
    return _mode_terms(theta, operand, unit_enumerator_modes(theta, operand))


def group_index_expr(theta, operand):
    groups = max(1, len(operand.fragment.groups()))
    modes = grouping_modes(theta, operand)
    if groups <= 1 or not modes:
        return cst(0)
    name, extent = modes[0]
    if extent % groups:
        raise RuntimeError(
            f"{operand.name}: grouping extent {extent} is not divisible by {groups} groups")
    # Contiguous ranges, repeated independently in every TDMSplit region.
    return Expr(carry=(((name, 1),), 0, extent // groups), mod=groups)


def shifted_group_index_expr(theta, operand, advance=0):
    """Register-group index of the read-order coordinate reached by ``advance``."""
    if not advance:
        return group_index_expr(theta, operand)
    groups = max(1, len(operand.fragment.groups()))
    modes = grouping_modes(theta, operand)
    if groups <= 1 or not modes:
        return cst(0)
    name, extent = modes[0]
    if extent % groups:
        raise RuntimeError(
            f"{operand.name}: grouping extent {extent} is not divisible by {groups} groups")
    extents = transfer_extents(theta, operand)
    strides, _span = axis_strides(theta, varying_axes(theta, operand), extents)
    terms = position_terms(strides, read_coverage(theta, operand))
    divisor = max(1, int(strides.get(name, 1))) * max(1, extent // groups)
    return Expr(digits=(terms, int(advance), ((1, divisor, groups),)))


def group_index(theta, operand, coord) -> int:
    return group_index_expr(theta, operand).eval(dict(coord))


def grouping_value(theta, operand, coord) -> int:
    return _flat_modes(theta, operand, grouping_modes(theta, operand), coord)


def unit_enumerator_value(theta, operand, coord) -> int:
    return _flat_modes(theta, operand, unit_enumerator_modes(theta, operand), coord)


def _flat_modes(theta, operand, modes, coord) -> int:
    at = dict(coord)
    fold = read_coverage(theta, operand)
    value, stride = 0, 1
    for name, extent in reversed(modes):
        value += (int(at.get(name, 0) or 0) // max(1, int(fold.get(name, 1)))) * stride
        stride *= extent
    return value


def _region_span_of(operand, name) -> int:
    return max(1, int((getattr(operand, "region_span", {}) or {}).get(name, 1)))


def _flatten_digits(digits, at, group_wrap) -> tuple:
    """Mixed-radix value of `digits` under the coordinate `at`, and the extent they span."""
    value, total = 0, 1
    for name, extent, divisor, is_grouping in digits:
        raw = int(at.get(name, 0) or 0)
        if is_grouping:
            raw %= max(1, group_wrap)
        radix = max(1, -(-extent // divisor))
        value = value * radix + raw // divisor
        total *= radix
    return value, total


def _rotation_block_digits(theta, operand):
    """The digits one rotation block distinguishes, outer to inner, with its grouping context.

    A region factor belonging to this rotation unit is packed inside its (group, slot) block; a
    reduction region outside the unit is an outer enumerator reusing the same names.
    """
    groups = max(1, len(operand.fragment.groups()))
    grouping = grouping_mode_name(theta, operand)
    group_extent = max(1, grouping_extent(theta, operand) // groups)
    region_axes = getattr(operand, "region_axes", ()) or ()
    rotation = rotation_unit_modes(theta, operand)

    digits = [(name, group_extent if name == grouping else extent, 1, name == grouping)
              for name, extent in rotation if name not in region_axes]

    named = {name for name, _extent, _divisor, _is_grouping in digits}
    ring_names = {name for name, _extent in ring_axes(
        theta, operand, operand.fragment.groups()[0])}
    for name, extent in prefetch_unit_modes(theta, operand):
        if name in named or name in region_axes or name in ring_names:
            continue
        digits.append((name, group_extent if name == grouping else extent, 1, name == grouping))
        named.add(name)
    return digits, grouping, group_extent


def _scale_presence_digits(theta, operand, grouping, group_extent):
    """An MX scale's digits: every axis it is present on, minus the region the slot selects."""
    region_axes = ((getattr(operand, "region_axes", ()) or ())
                   if _region_count(theta, operand) > 1 else ())
    return [(axis.name, group_extent if axis.name == grouping else axis.extent, 1,
             axis.name == grouping)
            for axis in presence_axes(theta, operand) if axis.name not in region_axes]


def _issue_order(keys) -> list:
    """The distinct carrier keys in first-appearance order -- the order the reads issue in."""
    order = []
    for key in keys:
        if key not in order:
            order.append(key)
    return order


def _realized_ring_depth(theta, operand, coord) -> int:
    """Slots the group holding `coord` actually rotates through, read off the fragment."""
    labels = operand.fragment.groups()
    group = labels[group_index(theta, operand, coord)] if labels else None
    width = int((operand.fragment.ring_depths or {}).get(group, 1))
    return max(1, width) * max(1, _region_count(theta, operand))


def unit_tile_index(theta, operand, coord) -> int:
    """Dense covered-register index inside this group's physical rotation block."""
    at = dict(coord)
    digits, grouping, group_extent = _rotation_block_digits(theta, operand)
    movement = operand.trajectory.shared_read
    coverage = getattr(movement, "coverage", None) if movement is not None else None
    if coverage is None:
        return _flatten_digits(digits, at, group_extent)[0]

    is_scale = operand.name.startswith("MXS")
    if is_scale:
        digits = _scale_presence_digits(theta, operand, grouping, group_extent)
    reduction = set(summation_names(theta))
    free_value, free_total = _flatten_digits(
        [d for d in digits if d[0] not in reduction], at, group_extent)
    reduction_value, reduction_total = _flatten_digits(
        [d for d in digits if d[0] in reduction], at, group_extent)

    # A scale's carrier is (carrier, slot): the fold distributes one instruction over sub-agents,
    # so two tiles sharing a carrier still occupy distinct slots.
    keys = [(coverage.carrier_of(tile), coverage.slot_of(tile)) if is_scale
            else coverage.carrier_of(tile) for tile in range(max(1, free_total))]
    order = _issue_order(keys)
    free_index = order.index(keys[free_value])
    if is_scale:
        # The ring already names the reduction values it has slots for; the block carries only
        # what is left, so a slot-deep ring does not encode the same digit twice.
        carried = reduction_value // _realized_ring_depth(theta, operand, at)
        return carried * len(order) + free_index
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
    return max(1, unit)


def _covered_positions(theta, operand, modes, grouped=False) -> int:
    """Movement positions represented by physical ``modes`` after read coverage."""
    hop = operand.trajectory.shared_read
    factors = (coverage_factors(theta, operand, getattr(hop, "coverage", None))
               if hop is not None else {})
    grouping = grouping_mode_name(theta, operand) if grouped else None
    groups = max(1, len(operand.fragment.groups()))
    total = 1
    for name, extent in modes:
        if grouping == name:
            extent //= groups
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
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
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


def group_live_peak(theta, operand, group, steps, post_war=True, off=None) -> int:
    """Rotation units simultaneously live in one group."""
    if off is None:
        if post_war:
            return concurrent_regions(theta, operand)
        off = prefetch_distance_for(
            theta, operand, requested_read_ahead(theta, operand), (group,))
    if off <= 0:
        return 1
    stride = rotation_unit_positions(theta, operand)
    return (1 + -(-int(off) // max(1, stride))) * concurrent_regions(theta, operand)


def group_reuse_floor(theta, operand, group) -> int:
    """Smallest ring that retains one read across every invariant-axis consumer pass."""
    cache = getattr(theta, "_group_reuse_floor_cache", None)
    if cache is None:
        cache = {}
        setattr(theta, "_group_reuse_floor_cache", cache)
    cache_key = (operand.name, tuple(operand.fragment.groups()),
                 getattr(operand.fragment, "grouping_mode", None), group)
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

    maximum = max(1, group_ring_size(theta, operand, group))
    regions = max(1, _region_count(theta, operand))
    for width in range(1, maximum + 1):
        slot = ring_slot(theta, operand, group, width * regions)
        by_register = {}
        for source, live in intervals.items():
            coord = dict(zip(axes, source))
            register_key = (slot.eval(coord), unit_tile_index(theta, operand, coord))
            by_register.setdefault(register_key, []).append(live)
        if all(previous[1] < current[0]
               for lives in by_register.values()
               for previous, current in zip(sorted(lives), sorted(lives)[1:])):
            cache[cache_key] = width
            return width
    cache[cache_key] = maximum
    return maximum


def _raw_read_ahead(theta, operand) -> int:
    if not operand.movements:
        return 0
    axis = prefetch_axis_name(theta, operand)
    if axis is None:
        return 0
    return max(0, int(theta.off_at(operand.name, READ, axis) or 0))


def _side(operand):
    return "A" if operand.name in ("A", "MXSA") else \
           "B" if operand.name in ("B", "MXSB") else None


def _direct_input(theta, side):
    """The primary A/B operand on ``side``."""
    name = "A" if side == "A" else "B"
    return next((op for op in theta.operands if op.name == name), None)


def requested_read_ahead(theta, operand, depths=None) -> int:
    """Internal per-operand PLR, in this operand's grouped-prefetch units.

    Every operand with a split region converts the global request from its own region-local unit
    to the selected grouped unit. Split axes select storage/agents and never multiply PLR,
    including for an all-inner operand. An unsplit all-inner operand retains the direct side's
    literal tile lead. ``depths`` applies the selected VGPR scheme's capacity cap.
    """
    raw = _raw_read_ahead(theta, operand)
    if not raw:
        return 0

    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    has_split_region = any(max(1, int(extents.get(axis, 1))) > 1
                           for axis in (getattr(operand, "region_axes", ()) or ()))
    if reloads_whole_set(theta, operand) and not has_split_region:
        side = _side(operand)
        peer = _direct_input(theta, "B" if side == "A" else "A")
        want = raw if peer is None else \
            _raw_read_ahead(theta, peer) * prefetch_unit_tiles(theta, peer)
    else:
        original = product(extent for _name, extent
                           in original_prefetch_unit_modes(theta, operand))
        grouped = group_prefetch_unit_tiles(theta, operand)
        if original % grouped:
            raise RuntimeError(
                f"{operand.name}: original prefetch unit {original} is not divisible by "
                f"group unit {grouped}")
        want = raw * (original // grouped)

    if reloads_whole_set(theta, operand):
        outer = theta.summation_chunk_name()
        staged = theta.off_at(operand.name, COPY, outer) if outer is not None else 0
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
    punit = group_prefetch_unit_tiles(theta, operand)
    runit = group_rotation_unit_tiles(theta, operand)
    capacity = sum(Fraction(max(1, int(depths.get(operand.name, group))) * runit, punit)
                   for group in operand.fragment.groups())
    return max(0, capacity.numerator // capacity.denominator)


def _staged_prefetch_cap(theta, operand, want) -> int:
    """Deepest operand PLR whose source chunk the copy/LDS pipeline has staged."""
    outer = theta.summation_chunk_name()
    if outer is None or not any(hop.dst == Space.SHARED for hop in operand.movements):
        return max(0, int(want))
    copy_offset = theta.off_at(operand.name, COPY, outer)
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


_reg_read_off = requested_read_ahead      # the old name, kept for the sweep that still uses it


def _free_tile_index_seq(theta, operand, steps):
    free = set(free_names(theta, operand))
    free_axes = [axis for axis in theta.inner_axes() if axis.name in free]
    seq = []
    for step in steps:
        idx = 0
        for j, axis in enumerate(free_axes):
            inner = 1
            for m2 in free_axes[j + 1:]:
                inner *= m2.extent
            idx += step.get(axis.name, 0) * inner
        seq.append(idx)
    return seq


def resident_free_tiles(theta, operand) -> int:
    steps = _inner_steps(theta)
    T = len(steps)
    if T == 0:
        return 1
    off = _reg_read_off(theta, operand)
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
    if not any(hop.dst == Space.SHARED for hop in operand.movements):
        raise RuntimeError(
            f"{operand.name} has no shared placement (hops={[f'{hop.src}->{hop.dst}' for hop in operand.movements]}), "
            f"so S_shared is undefined for it -- the caller should have guarded on the operand "
            f"touching SHARED before asking for a shared-ring depth.")
    outer_var = theta.summation_chunk_name()
    d = theta.off_at(operand, COPY, outer_var) if outer_var is not None else 0
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


# ===========================================================================

def operand_generations(operand, depths) -> int:
    """sum over register rate-groups of the group's depth (generation count) -- from derive_S."""
    groups = depths.groups_of(operand.name)
    return sum(groups.values()) if groups else 1


def group_regs(theta, operand, group) -> int:
    """Register width of ONE reuse group = its share of the fragment's registers."""
    nslots = operand.fragment.total_slots() or 1
    return max(1, frag_regs(theta, operand) * operand.fragment.size_of(group) // nslots)


def group_register_positions(theta, operand) -> int:
    """Physical register positions in one (group, rotation-slot) block: ONE region's tiles.

    Regions are extra rotation, so their multiplicity is slot COUNT, not block size.
    """
    tiles = group_unit_tiles(theta, operand, operand.fragment.groups()[0])
    rotation = {name for name, _extent in rotation_unit_modes(theta, operand)}
    regions = set(getattr(operand, "region_axes", ()) or ())
    ring_names = {name for name, _extent in ring_axes(
        theta, operand, operand.fragment.groups()[0])}
    tiles *= product(extent for name, extent in prefetch_unit_modes(theta, operand)
                     if name not in rotation and name not in regions and name not in ring_names)
    hop = operand.trajectory.shared_read
    coverage = getattr(hop, "coverage", None) if hop is not None else None
    if coverage is None:
        return max(1, tiles)
    if operand.name.startswith("MXS"):
        regions = set(getattr(operand, "region_axes", ()) or ())
        region_span = getattr(operand, "region_span", {}) or {}
        grouping = grouping_mode_name(theta, operand)
        groups = max(1, len(operand.fragment.groups()))
        free_positions = reduction_positions = 1
        reduction = set(summation_names(theta))
        for axis in presence_axes(theta, operand):
            extent = max(1, axis.extent)
            if axis.name in regions:
                extent = max(1, extent // max(1, int(region_span.get(axis.name, 1))))
            if axis.name == grouping:
                extent = max(1, extent // groups)
            if axis.name in reduction:
                reduction_positions *= extent
            else:
                free_positions *= extent
        pairs = {(coverage.carrier_of(tile), coverage.slot_of(tile))
                 for tile in range(max(1, free_positions))}
        return max(1, reduction_positions * len(pairs))
    return max(1, len({(coverage.carrier_of(tile), coverage.slot_of(tile))
                       for tile in range(max(1, tiles))}))


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


def group_ring_depth(theta, operand, group, depths) -> int:
    """Slots one group rotates through: its own buffers once per storage region."""
    return max(1, int(depths.get(operand.name, group))) * max(1, _region_count(theta, operand))


def register_layout(theta, depths) -> dict:
    """Per-operand ``(group, slot) -> VGPR offset`` and total compact allocation."""
    layout = {}
    for operand in theta.operands:
        if operand.is_output or not operand.movements:
            continue
        offset, slots = 0, {}
        fragment_regs = frag_regs(theta, operand)
        for group_index, group in enumerate(operand.fragment.groups()):
            unit_regs = group_register_positions(theta, operand) * fragment_regs
            for slot in range(group_ring_depth(theta, operand, group, depths)):
                slots[(group_index, slot)] = offset
                offset += unit_regs
        layout[operand.name] = {"slots": slots, "total": offset}
    return layout


def _region_count(theta, operand) -> int:
    """Storage regions that are factors of this operand's rotation unit."""
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
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


def per_tile(n: int, t: int) -> int:
    """The per-tile extent = `tile_split(n,t)[0]` = t (divisible)."""
    return max(1, tile_split(n, t)[0])


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
            else Axis(axis.name, max(1, axis.extent // coverage[axis.name]))
            for axis in theta.inner_axes() if axis.name not in constant_over]


# --- the axes it varies over --------------------------------------------------------------

def prefetch_axis_name(theta, operand=None):
    axis = readahead_level(theta, operand)
    return axis[0] if axis else None


def operand_level(theta, operand, role):
    chunk = theta.summation_chunk_name()
    if role == COPY:
        inner = {axis.name for axis in theta.inner_axes()}
        region = next((r for r in getattr(operand, "region_axes", ()) if r in inner), None)
        return region if (region is not None and max(1, operand.split) > 1) else chunk
    varying = varying_axes(theta, operand)
    return varying[-1] if varying else chunk


def group_block_coverage(theta, operand) -> int:
    """Level values one read covers, so the partition interleaves BLOCKS a read cannot straddle."""
    grouping_mode = getattr(getattr(operand, "fragment", None), "grouping_mode", None)
    if grouping_mode is None:
        return 1
    return max(1, int((read_coverage(theta, operand) or {}).get(grouping_mode, 1) or 1))


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
    """Modes enumerating rotation units for a streamed operand.

    The group is the operand's, not this argument's: every group of one fragment enumerates the
    same modes, so `group` says which ring, never which axes.
    """
    geometry = theta.geometry(operand)
    hop = operand.trajectory.shared_read
    if operand.name.startswith("MXS") and hop is not None and hop.coverage is not None:
        return list(geometry.summation_modes)
    if geometry.reduction_ring:
        return list(geometry.reduction_ring)
    if geometry.reloads_whole_set:
        if len(operand.fragment.groups()) == 1:
            return list(geometry.summation_modes)
        return []
    return unit_enumerator_modes(theta, operand)


def ring_slot(theta, operand, group, buffers, advance=0):
    """Rotation slot of the coordinate reached by ``advance`` physical tile positions."""
    if buffers <= 1:
        return cst(0)
    extents = transfer_extents(theta, operand)
    hop = operand.trajectory.shared_read
    if operand.name.startswith("MXS") and hop is not None and hop.coverage is not None:
        # Decode the reduction digit from the FULL covered read-order position.  Filtering the
        # free carrier axes first changes their radix: on K.M with four M tiles folded by two,
        # advance=2 is one K step, not two.
        varying = varying_axes(theta, operand)
        strides, _span = axis_strides(theta, varying, extents)
        shifted = _shifted_ring_slot(theta, operand, group, int(advance), strides, buffers,
                                     read_coverage(theta, operand))
        return shifted if shifted is not None else cst(0)
    varying = varying_axes(theta, operand)
    strides, _span = axis_strides(theta, varying, extents)
    if reloads_whole_set(theta, operand) and len(operand.fragment.groups()) == 1:
        shifted = _shifted_ring_slot(theta, operand, group, int(advance), strides, buffers,
                                     read_coverage(theta, operand))
        return shifted if shifted is not None else cst(0)
    shifted = _shifted_ring_slot(theta, operand, group, int(advance), strides, buffers,
                                 read_coverage(theta, operand))
    return shifted if shifted is not None else cst(0)


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
        place *= (extents or {}).get(axis.name, axis.extent)
    return strides, place       # place is now the product over `modes`


def transfer_extents(theta, operand) -> dict:
    """`{mode: extent}` as the operand's read hop sees them -- its the axes it varies over extents."""
    hop = operand.trajectory.fragment_fill
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    if hop is not None:
        extents.update({axis.name: axis.extent for axis in presence_axes(theta, operand, hop)})
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
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    outermost_varying = min(inner_names.index(name) for name in varying)
    width = 1
    for name in inner_names[:outermost_varying]:
        if name not in varying and extents.get(name, 1) > 1:
            width *= extents[name]
    return width


def position_terms(strides, fold=None):
    return tuple((name, coef, max(1, int((fold or {}).get(name, 1))))
                 for name, coef in strides.items())


def _shifted_coord(axis, shift, strides, _reload_span, extents, fold=None):
    if axis not in strides:
        return Expr(var=axis)
    terms = position_terms(strides, fold)
    coverage = max(1, int((fold or {}).get(axis, 1)))
    stride = max(1, strides[axis])
    if coverage > 1:
        return Expr(digits=(terms, shift, ((coverage, stride, max(1, extents.get(axis, 1))),)))
    return Expr(carry=(terms, shift, stride), mod=max(1, extents.get(axis, 1)))


def _regions_ride_the_ring(theta, operand, group) -> bool:
    """Is a region axis read ahead across INSIDE a ring axis, so two regions are in flight?"""
    if not requested_read_ahead(theta, operand):
        return False
    order = [axis.name for axis in theta.inner_axes()]
    ring = [order.index(name) for name, _extent in ring_axes(theta, operand, group)
            if name in order]
    outermost = min(ring) if ring else len(order)
    return any(name in order and order.index(name) > outermost
               for name in (getattr(operand, "region_axes", ()) or ()))


def ring_block_width(theta, operand, group, buffers) -> int:
    """Slots the enumerator owns before the region digits begin -- its own span, normally.

    A region owns a block, so the enumerator wraps inside its share of the depth rather than
    spanning it whole, which would land the region digit where the modulus is 0.  Its share is
    reserved whenever there IS more than one region -- `group_ring_depth` counts them
    unconditionally, so gating the reservation on a read-ahead erased the digit at PLR=0.
    """
    span = 1
    for _name, extent in ring_axes(theta, operand, group):
        span *= max(1, extent)
    regions = max(1, _region_count(theta, operand))
    if regions <= 1:
        return max(1, span)
    return max(1, min(span, max(1, int(buffers)) // regions))


def regions_time_share(theta, operand, group, buffers) -> bool:
    """Do this group's regions land on the same slots, rather than each owning its own?

    They start at the ring's own width, so they separate only when that width is not a multiple
    of the depth; when they share, the value a refill overwrites stays live across the sweep.
    """
    if _region_count(theta, operand) <= 1:
        return False
    return ring_block_width(theta, operand, group, buffers) % max(1, buffers) == 0


def _shifted_ring_slot(theta, operand, group, shift, strides, buffers, fold=None):
    """The rotation slot the prefetched read writes, as an Expr."""
    ring = [(name, extent) for name, extent in ring_axes(theta, operand, group)
            if name in strides]
    width = ring_block_width(theta, operand, group, buffers)
    digits, place = [], 1
    terms = position_terms(strides, fold)
    for name, extent in reversed(ring):  # the innermost axis is the fastest radix
        if place >= width:
            break                        # the rest of the enumerator wraps inside the ring
        radix = min(max(1, extent), width // place)
        if radix <= 1:
            continue                     # an extent-1 axis enumerates nothing, it does not end the ring
        digits.append((place, max(1, strides[name]), radix))
        place *= radix
    # A region is extra rotation, so it can carry the ring alone when the enumerator is empty.
    digits.extend(_region_digits(theta, operand, strides, width))
    if not digits:
        return None
    return Expr(digits=(terms, shift, tuple(reversed(digits))), mod=max(1, buffers))


def _region_digits(theta, operand, strides, place):
    """The region's digits in the ring: the outermost radix of the read order.

    It rides the shifted position like every other digit, so a lead that reaches the end of a
    region carries into the next one -- which is how a time-shared slot gets refilled.
    """
    inner = {axis.name: axis.extent for axis in theta.inner_axes()}
    axes = [name for name in (getattr(operand, "region_axes", ()) or ())
            if name in inner and name in strides]
    if not axes or _region_count(theta, operand) <= 1:
        return []
    digits = []
    for name in reversed(axes):
        width = _region_span_of(operand, name)
        count = max(1, max(1, int(inner.get(name, 1))) // width)
        digits.append((place, max(1, strides[name]) * width, count))
        place *= count
    return digits


def coverage_tile_cap(theta, operand, hop) -> int:
    if hop.is_bulk or not hop.vector_elems:
        return 1
    per_fragment = max(1, int(operand.frag_elems))
    return max(1, int(hop.vector_elems) // per_fragment)


def _divisors(value: int):
    value = max(1, int(value))
    return [d for d in range(value, 0, -1) if value % d == 0]


def is_uniform_over(generations, axes) -> bool:
    """Is the generation map constant within each block of `axes`?"""
    block = dict(axes)
    seen = {}
    for coord, generation in generations.items():
        key = tuple((name, value // block[name] if name in block else value)
                    for name, value in coord)
        if seen.setdefault(key, generation) != generation:
            return False
    return True


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
        extents={axis.name: axis.extent for axis in theta.inner_axes()},
        fan_axes=tuple(name for name in varying if name not in summation
                       and name not in ring_names),
        pinned_axes=tuple(name for name in varying if name not in reload_strides),
        ring=ring, buffers=buffers)


def _register_timeline(theta, operand, group, buffers, distance, at=None):
    """Walk TRIPS_WALKED trips, recording when each register is read and when it is loaded."""
    varying = varying_axes(theta, operand)
    if not varying or not distance:
        return None
    walk = _ring_walk(theta, operand, group, buffers, varying)
    steps = _inner_steps(theta)
    per_trip = len(steps)
    events = {}
    labels = operand.fragment.groups()
    group_number = labels.index(group) if group in labels else 0
    for trip in range(TRIPS_WALKED):
        loaded_positions = set()
        for index, step in enumerate(steps):
            time = trip * per_trip + index
            coord = {name: step.get(name, 0) for name in walk.varying}
            pinned = tuple(coord[name] for name in walk.pinned_axes)
            reload_position = walk.reload_position(coord)
            if group_index(theta, operand, coord) == group_number:
                events.setdefault(walk.register_of(coord), []).append(
                    Event(time, False, Generation(trip, reload_position, pinned)))

            if at and any(step.get(name, 0) != value for name, value in at.items()):
                continue
            position = walk.position(coord)
            if position in loaded_positions:
                continue  # first touch of this position issues the one read
            loaded_positions.add(position)

            reached = reload_position + distance
            target = walk.advanced(coord, reached)
            if group_index(theta, operand, target) != group_number:
                continue
            events.setdefault(walk.register_of(target), []).append(
                Event(time, True, Generation(trip + reached // max(1, walk.reload_span),
                                             reached % max(1, walk.reload_span), pinned)))
    return RegisterTimeline(events, tuple(steps), per_trip)


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
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
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
    if buffers > 1 and not (reloads_whole_set(theta, operand)
                            and len(operand.fragment.groups()) == 1):
        return {}
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
