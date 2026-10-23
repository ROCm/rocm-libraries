# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit a transform graph's coordinates, each access's split element address, and its validity.

Builder first, IR side effects.

- ``emit_accesses(b, graph, tensor, accesses, vector_axis=, vector_width=)`` -- the entry: one
  ``Access`` per access, its address and its validity (every side that can fail against the
  tensor's own bounds), over one shared table.
- ``emit_coords(b, graph, coords)`` -- every named coordinate of the graph as one emitted value.
- ``emit_addresses(b, graph, tensor, accesses)`` -- one ``ElementAddress`` per access: the
  element address split into wave-uniform terms (``UniformTerm``), one per-lane part and a
  Python-int constant, so the delivery side can put each part where it is cheapest.
- ``emit_access_validity(b, graph, tensor, accesses)`` -- each access's validity alone.
- ``emit_bound_check(b, value, lo, hi)`` -- the one predicate: ``lo <= value < hi`` for one plain
  value, its per-wave side folded on the scalar unit.
- ``emit_accesses_except_buffer_checked`` -- ``emit_accesses`` leaving out of ``valid`` the sides
  the buffer hardware checks; for the tiling layer's buffer delivery only (not on the door).

Each coordinate of an access is a plain i32 IR value or an int, one per graph input in
``inputs(...)`` order. The IR range walk splits it into a constant, wave-uniform parts and a
per-lane part, each with its proven range and residue; the author states nothing the IR shows.
Tiling-layer code that already holds the parts passes an ``AffineCoord`` instead.

How the parts move through the graph (one shared table per call, so a sub-expression that two
accesses share is emitted once, in first-use order):

- A per-lane part never starts below 0: one whose range starts at ``lo < 0`` is rebased to
  ``lane - lo`` once per call, and ``lo`` moves into the constant, so the value is unchanged.
- ``embed`` / ``merge`` / ``offset`` / ``slice_window`` / ``freeze`` keep the three parts apart:
  constants are folded in Python, uniform parts are scaled and kept per input set, the per-lane
  parts are scaled and added.
- ``unmerge`` splits the uniform side, the per-lane part and the constant separately and
  recombines them with exact 0 / 1 carries, innermost piece first: one division of the uniform
  side per level, at most one shared division of the per-lane part per level, and one compare +
  select per access. Only the outermost piece keeps a uniform part.
- ``xor`` first adds its reads' parts (``(uniform + lane) + const``) and is per-lane when a
  per-lane part reached it.
- The stride fold multiplies each output's parts by its tensor stride: one uniform term per
  distinct set of graph inputs (i64 unless its range fits i32), one per-lane sum in i32, and the
  constant in Python.

Integer widths follow the unconditioned ranges (what every lane computes, valid or not), so no
emitted ``nsw`` operation can overflow even in a lane whose coordinates are out of bounds.

How a bound is checked (one function body for every caller). The coordinate is ``s + lane``:
``s`` its per-wave side (uniform parts plus constant), ``lane`` its per-lane part in ``[0, L)``
(the constant 0 and ``L = 1`` when it has none). The ends fold on per-wave values and saturate::

    a = clamp(lo - s, 0, L)       # the lowest valid lane value     (no lower end: not emitted)
    c = clamp(hi - s, 0, L)       # one past the highest one        (no upper end: not emitted)
    no lower end:  lane u< c
    no upper end:  (lane - a) u< (L - a)
    both ends:     (lane - a) u< smax(c - a, 0)

For every lane in ``[0, L)``, ``lo <= lane + s < hi`` exactly when ``a <= lane < c``; a negative
``lane - a`` reads as a large unsigned value, so it fails. The fold is i32 when every range it
touches is proven inside i32; otherwise each end is pre-clamped once in i64 per distinct
(per-wave side, end) over the call and finished per access in i32, so no subtraction can wrap
whatever the per-wave values hold. One case takes a single compare instead: a lone i32 per-wave
value ``u`` with no constant and no per-lane part against ``[0, hi)``, ``hi`` below ``2^31``,
is ``u u< hi`` (a negative ``u`` reads as ``>= 2^31``).

A vector (``vector_width = w > 1`` along graph input ``vector_axis``) is addressed by its first
element, which must be contiguous in memory. ``valid`` says every element is valid: nothing when
statically valid, the first element's checks when every check has one answer for the whole
vector, else the first and the last element's checks plus ``valid_span``, the valid elements
``[first, end)``, from the same folded ends (every checked coordinate moves by a constant step
along the vector).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import TYPE_CHECKING

from rocke.core.dtypes import dtype_info
from rocke.core.ir import I32, I64, Type, Value
from rocke.helpers.magic_division import calculate_magic_numbers, do_magic_division

from .address import _require_nonnegative_strides
from .graph import Node, TransformGraph, TransformType, _require_outputs_match_rank
from .ranges import (
    AffineCoord,
    AxisBound,
    BoundCheck,
    CheckSide,
    Congruence,
    End,
    Interval,
    RuntimeValue,
    Side,
    UniformPart,
    VectorRun,
    _hi_proven,
    _lo_proven,
    _numeric_hi,
    _numeric_lo,
    _required_checks,
)
from .runs import _check_uniform, _contiguity_blocker, _first_non_monotone, _run_steps
from .walk import split_coordinate

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder

    from ..descriptors import TensorDesc

__all__ = [
    "Access",
    "Coordinate",
    "ElementAddress",
    "UniformTerm",
    "emit_access_validity",
    "emit_accesses",
    "emit_accesses_except_buffer_checked",
    "emit_addresses",
    "emit_bound_check",
    "emit_coords",
]

# One coordinate of an access: a plain i32 IR value or a Python int.
Coordinate = Value | int

_I32_LO, _I32_END = -(1 << 31), 1 << 31

# The residue of a value nothing is known about.
_ANY_RESIDUE = Congruence(1, 0)


# --- results ------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class UniformTerm:
    """One wave-uniform address term, in elements, and the graph inputs it derives from.

    ``value`` is i32 or i64 by its range; ``range`` is its half-open ``(lo, hi)`` pair. Terms of
    different input sets are never summed here, so the delivery side can decide per term where
    it goes (for example a term that does not change inside a loop).
    """

    value: Value
    range: tuple[End, End]
    inputs: frozenset[int]


@dataclass(frozen=True, eq=False)
class ElementAddress:
    """One access's element address: ``sum(uniform) + lane + const``, in elements.

    ``uniform`` holds one term per distinct set of graph inputs (``()`` when there is none);
    ``lane`` is the i32 per-lane part with its unconditioned ``lane_range`` (starting at 0 or
    above), or ``None``; ``const`` is a Python int that may be negative (a halo). ``elem_bits``
    is the element size in bits, so byte proofs also hold for packed sub-byte types.
    ``uniform_delta`` (with its range) is this access's own wave-uniform term on runtime-stride
    axes; it is ``None`` for a tensor whose strides are all ints.
    """

    uniform: tuple[UniformTerm, ...]
    lane: Value | None
    lane_range: tuple[int, int] | None
    const: int
    elem_bits: int
    uniform_delta: Value | None = None
    uniform_delta_range: tuple[End, End] | None = None


@dataclass(frozen=True, eq=False)
class Access:
    """One access's element address and its validity.

    ``valid`` is the i1 "every element of the access is inside the tensor and every domain"
    (``None``: statically always valid, nothing emitted). ``valid_span`` is set only for a
    vector whose validity can differ between its elements: ``(first, end)``, i32 element indices
    inside the vector; elements ``first <= e < end`` are valid (none when ``end <= first``).
    """

    address: ElementAddress
    valid: Value | None
    valid_span: tuple[Value, Value] | None = None


# --- the entries --------------------------------------------------------------------------------


def emit_accesses(
    b: IRBuilder,
    graph: TransformGraph,
    tensor: TensorDesc,
    accesses: Sequence[tuple[Coordinate, ...]],
    *,
    vector_axis: int | None = None,
    vector_width: int = 1,
) -> tuple[Access, ...]:
    """One ``Access`` per access of ``tensor`` through ``graph``: its split element address and
    its validity, over one shared table.

    Each access is a tuple with one coordinate per graph input, in ``inputs(...)`` order (a
    plain i32 IR value or an int). ``valid`` covers every side that can fail against the
    tensor's own bounds (``0 <= coordinate < length`` per axis) and the graph's input and read
    domains, so the access is safe with a plain global load. With ``vector_width = w > 1`` each
    access stands for ``w`` consecutive values of graph input ``vector_axis`` (an input index,
    not a tile-axis index): its address is the first element's, and ``valid`` says every element
    is valid (with ``valid_span`` when that can differ between elements).
    """
    return _emit_access_entry(
        b, graph, tensor, accesses, vector_axis, vector_width, buffer_checked_sides=None
    )


def emit_accesses_except_buffer_checked(
    b: IRBuilder,
    graph: TransformGraph,
    tensor: TensorDesc,
    accesses: Sequence[tuple[Coordinate, ...]],
    *,
    buffer_checked_sides: frozenset[CheckSide],
    vector_axis: int | None = None,
    vector_width: int = 1,
) -> tuple[Access, ...]:
    """``emit_accesses`` with the sides in ``buffer_checked_sides`` left out of ``valid``.

    For the tiling layer's buffer delivery only. The caller MUST have the buffer hardware check
    every side it names: the returned ``valid`` alone is NOT safe. Each named side must be one
    ``required_checks`` reports as failing for some access of this call.
    """
    return _emit_access_entry(
        b,
        graph,
        tensor,
        accesses,
        vector_axis,
        vector_width,
        buffer_checked_sides=frozenset(buffer_checked_sides),
    )


def emit_access_validity(
    b: IRBuilder,
    graph: TransformGraph,
    tensor: TensorDesc,
    accesses: Sequence[tuple[Coordinate, ...]],
) -> tuple[Value | None, ...]:
    """Each access's validity: the AND of the checks it needs against the tensor's own bounds and
    the graph's domains (``None``: statically always valid, nothing emitted). Accesses share every
    compare they have in common."""
    _check_accesses(graph, accesses, tensor=tensor)
    split = [_split_access(b, access) for access in accesses]
    plans = _plan_validity(graph, tensor, split, None, None)
    table = _CallTable(b)
    envs = [_emit_coord_table(table, graph, coords) for coords in split]
    return tuple(valid for valid, _ in _emit_validity(table, envs, plans, None))


def emit_bound_check(
    b: IRBuilder, value: Value | int, lo: End, hi: End
) -> Value | None:
    """The i1 ``lo <= value < hi`` for one plain i32 value (or int), each end an int, a
    ``RuntimeValue`` or ``None`` (that side unchecked).

    The value is split by the IR range walk; its per-wave side is folded on per-wave values and
    one unsigned compare decides. A side the value's walked range already proves emits nothing,
    and ``None`` is returned when both are proven. Check a value BEFORE stating a fact about it:
    on the marked value the walk reads the stated fact back and proves the check away.
    """
    if not _is_plain(value):
        raise TypeError(
            f"emit_bound_check takes a plain i32 IR value or an int, got {_type_text(value)}"
        )
    name = value.name if isinstance(value, Value) else str(value)
    _require_ends(name, lo, hi)
    coord = split_coordinate(b, value)
    lo, hi = _unproven(coord, lo, hi)
    if lo is None and hi is None:
        return None
    table = _CallTable(b)
    coord = _nonnegative_lane(table, coord, name)
    return _check_value(table, _table_coord(coord), lo, hi, name, None)


def _emit_bound_check(
    b: IRBuilder, coord: AffineCoord, lo: End, hi: End
) -> Value | None:
    """``emit_bound_check`` on a coordinate the tiling layer already holds split (its per-lane
    part must start at 0 or above: the access entries and ``emit_bound_check`` rebase it).
    """
    name = _coord_text(coord)
    _require_ends(name, lo, hi)
    if coord.lane is not None:
        r = coord.lane_range
        lane_lo = _numeric_lo(r.lo)
        if lane_lo is None or lane_lo < 0:
            raise ValueError(
                f"_emit_bound_check: lane part {coord.lane} has range {r}, starting below 0 -- "
                "pass the coordinate through _emit_coord_table or emit_bound_check, which rebase "
                "it to start at 0"
            )
        if not _fits_lane(r):
            raise _lane_range_error(name, r)
    lo, hi = _unproven(coord, lo, hi)
    if lo is None and hi is None:
        return None
    return _check_value(_CallTable(b), _table_coord(coord), lo, hi, name, None)


def _fold_bound_check(b: IRBuilder, coord: AffineCoord, lo: End, hi: End) -> Value:
    """The saturating fold of the bound check, always: the step the single-compare case of
    ``emit_bound_check`` bypasses (both answer the same predicate)."""
    name = _coord_text(coord)
    _require_ends(name, lo, hi)
    tcoord = _table_coord(coord)
    return _fold_check(_CallTable(b), tcoord, lo, hi, name, _lane_extent(tcoord), None)


# --- the toolbox --------------------------------------------------------------------------------


def emit_coords(
    b: IRBuilder, graph: TransformGraph, coords: tuple[Coordinate, ...]
) -> dict[str, Value]:
    """Every named coordinate of ``graph`` as one emitted value, for ``coords`` (one per graph
    input, in ``inputs(...)`` order).

    Each value is its parts summed, ``(uniform + lane) + const``: i32, or i64 when its range
    needs it. Keyed by the names the graph was built with, inputs and intermediates included.
    """
    _check_accesses(graph, (coords,))
    table = _CallTable(b)
    env = _emit_coord_table(table, graph, _split_access(b, coords))
    return {name: table.value_of(coord) for name, coord in env.items()}


def emit_addresses(
    b: IRBuilder,
    graph: TransformGraph,
    tensor: TensorDesc,
    accesses: Sequence[tuple[Coordinate, ...]],
) -> tuple[ElementAddress, ...]:
    """One ``ElementAddress`` per access of ``tensor`` through ``graph`` (graph output ``i`` is
    tensor axis ``i``).

    Each access is a tuple with one coordinate per graph input, in ``inputs(...)`` order. All
    accesses share one table: a sub-expression they have in common is emitted once.
    """
    _check_accesses(graph, accesses, tensor=tensor)
    _require_nonnegative_strides(tensor)
    split = [_split_access(b, access) for access in accesses]
    _, addresses = _emit_addresses(_CallTable(b), graph, tensor, split)
    return addresses


def _emit_access_entry(
    b: IRBuilder,
    graph: TransformGraph,
    tensor: TensorDesc,
    accesses: Sequence[tuple[Coordinate, ...]],
    vector_axis: int | None,
    vector_width: int,
    *,
    buffer_checked_sides: frozenset[CheckSide] | None,
) -> tuple[Access, ...]:
    """The two access entries' one body: every check and error first, then one table for the
    addresses and the validity (the same steps as the two fundamentals)."""
    _check_accesses(
        graph,
        accesses,
        tensor=tensor,
        vector_axis=vector_axis,
        vector_width=vector_width,
    )
    _require_nonnegative_strides(tensor)
    run = VectorRun(vector_axis, vector_width) if vector_width > 1 else None
    split = [_split_access(b, access) for access in accesses]
    plans = _plan_validity(graph, tensor, split, run, buffer_checked_sides)
    table = _CallTable(b)
    envs, addresses = _emit_addresses(table, graph, tensor, split)
    validity = _emit_validity(table, envs, plans, run)
    return tuple(
        Access(address, valid, span)
        for address, (valid, span) in zip(addresses, validity)
    )


def _emit_addresses(
    table: _CallTable,
    graph: TransformGraph,
    tensor: TensorDesc,
    split: Sequence[tuple[AffineCoord, ...]],
) -> tuple[list[dict[str, _TableCoord]], tuple[ElementAddress, ...]]:
    """Each access's coordinate table and element address, access by access."""
    elem_bits = _elem_bits(tensor)
    envs, addresses = [], []
    for coords in split:
        env = _emit_coord_table(table, graph, coords)
        envs.append(env)
        addresses.append(_element_address(table, graph, tensor, env, elem_bits))
    return envs, tuple(addresses)


# --- the access checks --------------------------------------------------------------------------


def _check_accesses(
    graph: TransformGraph,
    accesses: Sequence[tuple[Coordinate, ...]],
    *,
    tensor: TensorDesc | None = None,
    vector_axis: int | None = None,
    vector_width: int = 1,
) -> None:
    """Refuse a malformed access before anything is emitted (the one helper every entry calls)."""
    names = tuple(graph.inputs)
    for i, access in enumerate(accesses):
        if not isinstance(access, tuple):
            raise TypeError(
                f"access {i} is a {type(access).__name__}, not a tuple -- give each access as a "
                "tuple of coordinates in inputs(...) order"
            )
        if len(access) != len(names):
            raise ValueError(
                f"access {i} gives {len(access)} coordinates, the graph has {len(names)} inputs "
                f"{names} -- give one coordinate per input, in inputs(...) order"
            )
        for j, coord in enumerate(access):
            if not _is_coordinate(coord):
                raise TypeError(
                    f"coordinate {j} ('{names[j]}') of access {i} is a {_type_text(coord)} -- a "
                    "coordinate is a plain i32 IR value or an int"
                )
    if tensor is not None:
        _require_outputs_match_rank(graph, len(tensor.strides))
    if vector_width < 1:
        raise ValueError(f"vector_width must be >= 1, got {vector_width}")
    if vector_width > 1 and vector_axis is None:
        raise ValueError(
            f"vector_width {vector_width} needs vector_axis -- give the index, in inputs(...) "
            "order, of the input the vector runs along"
        )
    if vector_axis is not None and not 0 <= vector_axis < len(names):
        raise ValueError(
            f"vector_axis {vector_axis} is not an input index -- the graph has {len(names)} "
            f"inputs {names}"
        )


def _is_coordinate(coord: object) -> bool:
    return _is_plain(coord) or isinstance(coord, AffineCoord)


def _is_plain(value: object) -> bool:
    """A plain i32 IR value or an int."""
    if isinstance(value, Value):
        return value.type == I32
    return isinstance(value, int) and not isinstance(value, bool)


def _type_text(coord: object) -> str:
    if isinstance(coord, Value):
        return f"Value of type {coord.type.name}"
    return type(coord).__name__


def _split_access(
    b: IRBuilder, access: tuple[Coordinate | AffineCoord, ...]
) -> tuple[AffineCoord, ...]:
    """Each coordinate split by the IR range walk; an ``AffineCoord`` is already split."""
    return tuple(
        c if isinstance(c, AffineCoord) else split_coordinate(b, c) for c in access
    )


def _elem_bits(tensor: TensorDesc) -> int:
    return dtype_info(getattr(tensor.dtype, "name", tensor.dtype)).encoded_bits


def _lane_range_error(name: str, r: Interval) -> ValueError:
    return ValueError(
        f"lane part of '{name}' has range {r}, outside 32 bits -- per-lane offsets must fit i32; "
        "move the large part into the uniform coordinate"
    )


def _fits_i32(r: Interval) -> bool:
    return r.within(_I32_LO, _I32_END)


def _fits_lane(r: Interval) -> bool:
    """A per-lane part's range: at least 0 and below ``2^31``."""
    return r.within(0, _I32_END)


def _sum_type(
    types: Sequence[Type], ranges: Sequence[Interval], const: int = 0
) -> tuple[Type, Interval]:
    """The width of a left-fold sum of values of ``types`` with ``ranges``, plus ``const``, and
    its range: i32 when every value is i32 and every partial sum is proven inside i32 (so no
    ``nsw`` add can overflow), else i64."""
    partials = [ranges[0]]
    for r in ranges[1:]:
        partials.append(partials[-1].plus(r))
    total = partials[-1].shifted(const)
    narrow = all(t == I32 for t in types) and all(
        _fits_i32(r) for r in [*partials, total]
    )
    return (I32 if narrow else I64), total


# --- the per-call table -------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Term:
    """The wave-uniform side of a coordinate from one set of graph inputs: its parts, summed
    only where a consumer first needs one value for them, in part order."""

    inputs: frozenset[int]
    parts: tuple[UniformPart, ...]

    @property
    def range(self) -> Interval:
        return AffineCoord(0, self.parts).uniform_range

    @property
    def congruence(self) -> Congruence:
        return AffineCoord(0, self.parts).congruence


@dataclass(frozen=True, eq=False)
class _TableCoord:
    """One graph coordinate in the table: a constant, uniform terms keyed by input set (in
    input order), and one per-lane part (range starting at 0 or above)."""

    const: int
    terms: tuple[_Term, ...] = ()
    lane: Value | None = None
    lane_range: Interval | None = None
    lane_congruence: Congruence = _ANY_RESIDUE

    def as_affine(self) -> AffineCoord:
        """The same coordinate with its input keys dropped (the home of its range rules)."""
        parts = tuple(p for term in self.terms for p in term.parts)
        return AffineCoord(
            self.const, parts, self.lane, self.lane_range, self.lane_congruence
        )

    @property
    def inputs(self) -> frozenset[int]:
        return frozenset().union(*(term.inputs for term in self.terms))

    def shifted(self, k: int) -> _TableCoord:
        return _TableCoord(
            self.const + k,
            self.terms,
            self.lane,
            self.lane_range,
            self.lane_congruence,
        )


def _keyed_terms(terms: Sequence[_Term]) -> tuple[_Term, ...]:
    """Terms of one input set joined (parts in order), the sets in input order."""
    parts: dict[frozenset[int], list[UniformPart]] = {}
    for term in terms:
        parts.setdefault(term.inputs, []).extend(term.parts)
    order = sorted(parts, key=lambda key: tuple(sorted(key)))
    return tuple(_Term(key, tuple(parts[key])) for key in order)


@dataclass(frozen=True, eq=False)
class _Lane:
    """A per-lane i32 value on its way into a coordinate, with its range and residue."""

    value: Value
    range: Interval
    congruence: Congruence


class _CallTable:
    """The one table of a call: every op it emits is built once per distinct operands, so the
    accesses of a call share every identical sub-expression (emitted in first-use order).

    Every op here is side-effect free (arithmetic, casts, compares, ``umul_hi``), so a uniform
    value's roots are exactly the uniform parts the caller passed in.
    """

    def __init__(self, b: IRBuilder) -> None:
        self.b = b
        self._memo: dict[tuple, Value] = {}

    def _once(self, key: tuple, emit: Callable[[], Value]) -> Value:
        if key not in self._memo:
            self._memo[key] = emit()
        return self._memo[key]

    # -- single ops ------------------------------------------------------------------------

    def const(self, value: int, ty: Type = I32) -> Value:
        b = self.b
        if ty == I64:
            return self._once(("i64", value), lambda: b.const_i64(value))
        return self._once(("i32", value), lambda: b.const_i32(value))

    def as_type(self, v: Value, ty: Type) -> Value:
        """``v`` as ``ty``: an i32 value sign-extended to i64 once."""
        if v.type == ty:
            return v
        return self._once(("sext", id(v)), lambda: self.b.sext(v, I64))

    def add(self, x: Value, y: Value) -> Value:
        return self._once(("add", id(x), id(y)), lambda: self.b.add(x, y))

    def add_const(self, x: Value, k: int) -> Value:
        return self._once(
            ("add_const", id(x), k), lambda: self.b.add(x, self.const(k, x.type))
        )

    def sub(self, x: Value, y: Value) -> Value:
        return self._once(("sub", id(x), id(y)), lambda: self.b.sub(x, y))

    def mul_const(self, x: Value, k: int) -> Value:
        """``x * k``; ``k == 1`` emits nothing."""
        if k == 1:
            return x
        return self._once(
            ("mul", id(x), k), lambda: self.b.mul(x, self.const(k, x.type))
        )

    def at_least(self, x: Value, k: int) -> Value:
        return self._once(("ge", id(x), k), lambda: self.b.cmp_ge(x, self.const(k)))

    def flag_value(self, flag: Value) -> Value:
        """An i1 compare as the i32 carry 0 / 1."""
        return self._once(("zext", id(flag)), lambda: self.b.zext(flag, I32))

    def length_if(self, flag: Value, k: int) -> Value:
        return self._once(
            ("select", id(flag), k),
            lambda: self.b.select(flag, self.const(k), self.const(0)),
        )

    def quotient(self, x: Value, k: int) -> Value:
        """``x // k`` for an i32 ``x`` in ``[0, 2^31)``: a shift for a power of two, else magic
        division."""
        if k & (k - 1) == 0:
            return self._once(
                ("lshr", id(x), k),
                lambda: self.b.lshr(x, self.const(k.bit_length() - 1)),
            )
        multiplier, shift = calculate_magic_numbers(k)
        return self._once(
            ("div", id(x), k),
            lambda: do_magic_division(self.b, x, multiplier, shift),
        )

    def remainder(self, x: Value, q: Value, k: int) -> Value:
        """``x mod k`` for the ``q = x // k`` above: a mask for a power of two, else
        ``x - q * k``."""
        if k & (k - 1) == 0:
            return self.low_bits(x, k)
        return self.sub(x, self.mul_const(q, k))

    def xor(self, x: Value, y: Value) -> Value:
        return self._once(("xor", id(x), id(y)), lambda: self.b.xor(x, y))

    def low_bits(self, x: Value, k: int) -> Value:
        """``x mod k`` for a power of two ``k``, every sign of ``x``: ``and (k - 1)``."""
        return self._once(
            ("and", id(x), k), lambda: self.b.land(x, self.const(k - 1, x.type))
        )

    def at_most(self, x: Value, k: int) -> Value:
        """``min(x, k)``: ``smin`` on i32, compare + select on i64 (the IR's ``smin`` is i32)."""
        bound = self.const(k, x.type)
        if x.type == I32:
            return self._once(("smin", id(x), k), lambda: self.b.smin(x, bound))
        return self._once(
            ("min64", id(x), k),
            lambda: self.b.select(self.b.cmp_gt(x, bound), bound, x),
        )

    def no_less(self, x: Value, k: int) -> Value:
        """``max(x, k)``: ``smax`` on i32, compare + select on i64."""
        bound = self.const(k, x.type)
        if x.type == I32:
            return self._once(("smax", id(x), k), lambda: self.b.smax(x, bound))
        return self._once(
            ("max64", id(x), k),
            lambda: self.b.select(self.b.cmp_lt(x, bound), bound, x),
        )

    def low_word(self, x: Value) -> Value:
        """An i64 value proven inside i32, as i32."""
        return self._once(("trunc", id(x)), lambda: self.b.trunc(x, I32))

    def larger(self, x: Value, y: Value) -> Value:
        """``smax(x, y)``."""
        return self._once(("smax2", id(x), id(y)), lambda: self.b.smax(x, y))

    def smaller(self, x: Value, y: Value) -> Value:
        """``smin(x, y)``."""
        return self._once(("smin2", id(x), id(y)), lambda: self.b.smin(x, y))

    def below(self, x: Value, y: Value) -> Value:
        """The unsigned compare ``x u< y``."""
        return self._once(("ult", id(x), id(y)), lambda: self.b.cmp_ult(x, y))

    def both(self, x: Value, y: Value) -> Value:
        """``x and y`` of two i1 values."""
        return self._once(("both", id(x), id(y)), lambda: self.b.land(x, y))

    def ceil_quotient(self, x: Value, k: int) -> Value:
        """``ceil(x / k)`` for an i32 ``x`` with ``x + k - 1`` in ``[0, 2^31)``."""
        if k == 1:
            return x
        return self.quotient(self.add_const(x, k - 1), k)

    # -- sums ------------------------------------------------------------------------------

    def sum_of(
        self, items: Sequence[tuple[Value, Interval]], const: int = 0
    ) -> tuple[Value | int, Interval]:
        """``((items[0] + items[1]) + ...) + const``: a left fold, the first value starting it
        and the constant added last (a zero constant emits nothing). i32 when every value is
        i32 and every partial sum is proven inside i32, else i64."""
        if not items:
            return const, Interval(const, const + 1)
        ty, total = _sum_type([v.type for v, _ in items], [r for _, r in items], const)
        acc = self.as_type(items[0][0], ty)
        for v, _ in items[1:]:
            acc = self.add(acc, self.as_type(v, ty))
        if const:
            acc = self.add_const(acc, const)
        return acc, total

    def term_value(self, term: _Term) -> Value:
        """A uniform term's parts summed into one value."""
        value, _ = self.sum_of([(p.value, p.range) for p in term.parts])
        return value

    def uniform_value(self, coord: _TableCoord) -> tuple[Value, Interval]:
        """Every uniform term of ``coord`` summed (in input order) into one value."""
        return self.sum_of([(self.term_value(t), t.range) for t in coord.terms])

    def value_of(self, coord: _TableCoord) -> Value:
        """The whole coordinate as one value: ``(uniform + lane) + const``."""
        items = [(self.term_value(t), t.range) for t in coord.terms]
        if coord.lane is not None:
            items.append((coord.lane, coord.lane_range))
        value, total = self.sum_of(items, coord.const)
        if isinstance(value, int):
            return self.const(value, I32 if _fits_i32(total) else I64)
        return value

    # -- the parts of a coordinate ---------------------------------------------------------

    def scaled_term(self, term: _Term, k: int) -> _Term:
        """``term * k`` (``k >= 1``): one part, i32 when the product is proven inside i32."""
        if k == 1:
            return term
        value, r = self.term_value(term), term.range.scaled(k)
        ty = I32 if value.type == I32 and _fits_i32(r) else I64
        product = self.mul_const(self.as_type(value, ty), k)
        return _Term(term.inputs, (UniformPart(product, r, term.congruence.scaled(k)),))

    def scaled_lane(self, lane: _Lane, k: int, name: str) -> _Lane:
        """``lane * k`` (``k >= 1``) in i32."""
        r = lane.range.scaled(k)
        if not _fits_lane(r):
            raise _lane_range_error(name, r)
        return _Lane(self.mul_const(lane.value, k), r, lane.congruence.scaled(k))

    def lane_sum(self, lanes: Sequence[tuple[_Lane, str]]) -> _Lane | None:
        """The per-lane values added in order, each partial sum proven to fit a lane part (the
        error names the coordinate whose addition leaves it)."""
        if not lanes:
            return None
        total, congruence = lanes[0][0].range, lanes[0][0].congruence
        for index, (lane, name) in enumerate(lanes):
            if index:
                total, congruence = total.plus(lane.range), congruence.plus(
                    lane.congruence
                )
            if not _fits_lane(total):
                raise _lane_range_error(name, total)
        value, _ = self.sum_of([(lane.value, lane.range) for lane, _ in lanes])
        return _Lane(value, total, congruence)

    def wrap_once(
        self, t: Value, t_range: Interval, k: int
    ) -> tuple[Value, Value | int]:
        """``t`` in ``[0, 2k)`` brought into ``[0, k)``: ``(t - select(t >= k, k, 0), t >= k)``;
        nothing when ``t < k`` is proven (the carry is then the int 0)."""
        if t_range.within(None, k):
            return t, 0
        flag = self.at_least(t, k)
        return self.sub(t, self.length_if(flag, k)), flag


def _nonnegative_lane(table: _CallTable, coord: AffineCoord, name: str) -> AffineCoord:
    """``coord`` with its per-lane part rebased to start at 0 (the one home of the rule).

    A lane part with range ``[lo, hi)``, ``lo < 0``, becomes ``lane - lo`` (one add, once per
    call) and ``lo`` moves into the constant, so the coordinate's value and residue are
    unchanged. A lane part must fit i32 after the rebase.
    """
    if coord.lane is None:
        return coord
    r = coord.lane_range
    lo = _numeric_lo(r.lo)
    if lo is None or not _fits_lane(r.shifted(-min(lo, 0))):
        raise _lane_range_error(name, r)
    if lo >= 0:
        return coord
    return AffineCoord(
        coord.const + lo,
        coord.uniform,
        table.add_const(coord.lane, -lo),
        r.shifted(-lo),
        coord.lane_congruence.shifted(-lo),
    )


def _emit_coord_table(
    table: _CallTable, graph: TransformGraph, coords: tuple[AffineCoord, ...]
) -> dict[str, _TableCoord]:
    """Every graph coordinate of one access, keyed by name, over the call's table.

    Input ``i``'s uniform parts are keyed by ``{i}``; each step's rule is in the module
    docstring. Emits only what the steps need; a uniform term is summed where a consumer first
    needs one value for it.
    """
    env: dict[str, _TableCoord] = {}
    for i, (name, coord) in enumerate(zip(graph.inputs, coords)):
        coord = _nonnegative_lane(table, coord, name)
        terms = (_Term(frozenset({i}), coord.uniform),) if coord.uniform else ()
        env[name] = _TableCoord(
            coord.const, terms, coord.lane, coord.lane_range, coord.lane_congruence
        )
    for node in graph.nodes:
        env.update(_emit_node(table, graph, node, env))
    return env


def _emit_node(
    table: _CallTable, graph: TransformGraph, node: Node, env: dict[str, _TableCoord]
) -> dict[str, _TableCoord]:
    reads = [env[r] for r in node.read]
    kind, params = node.kind, node.params
    if kind is TransformType.EMBED:
        return {node.write[0]: _linear(table, reads, params.coeffs, node.write[0])}
    if kind is TransformType.MERGE:
        weights = tuple(
            prod(params.lengths[i + 1 :]) for i in range(len(params.lengths))
        )
        return {node.write[0]: _linear(table, reads, weights, node.write[0])}
    if kind is TransformType.UNMERGE:
        return _unmerge(table, reads[0], node)
    if kind is TransformType.OFFSET:
        return {node.write[0]: reads[0].shifted(params.shift)}
    if kind is TransformType.SLICE:
        return {node.write[0]: reads[0].shifted(params.begin)}
    if kind is TransformType.FREEZE:
        return {node.write[0]: _TableCoord(params.value)}
    if kind is TransformType.XOR:
        return _xor(table, reads, node, graph.edge(node.write[1]).domain)
    return {}  # BROADCAST writes nothing


def _lane_of(coord: _TableCoord) -> _Lane | None:
    if coord.lane is None:
        return None
    return _Lane(coord.lane, coord.lane_range, coord.lane_congruence)


def _with_lane(const: int, terms: tuple[_Term, ...], lane: _Lane | None) -> _TableCoord:
    if lane is None:
        return _TableCoord(const, terms)
    return _TableCoord(const, terms, lane.value, lane.range, lane.congruence)


def _linear(
    table: _CallTable, reads: Sequence[_TableCoord], coeffs: Sequence[int], into: str
) -> _TableCoord:
    """``into = sum(read_i * coeff_i)`` (embed, merge): constants in Python, uniform terms
    scaled and kept per input set, per-lane parts scaled and added in read order."""
    const, terms, lanes = 0, [], []
    for read, k in zip(reads, coeffs):
        if k == 0:
            continue  # a zero part emits nothing
        const += read.const * k
        terms.extend(table.scaled_term(term, k) for term in read.terms)
        lane = _lane_of(read)
        if lane is not None:
            lanes.append((table.scaled_lane(lane, k, into), into))
    return _with_lane(const, _keyed_terms(terms), table.lane_sum(lanes))


def _xor(
    table: _CallTable, reads: Sequence[_TableCoord], node: Node, length: int
) -> dict[str, _TableCoord]:
    """``into[1] = reads[1] xor (reads[0] mod length)``; ``into[0] = reads[0]``.

    Each read's parts are added first (``(uniform + lane) + const``); the result is per-lane
    when a per-lane part reached it, else a uniform term keyed by both reads' inputs. XOR with a
    value in ``[0, length)`` changes only the low bits, so the result stays in the
    ``length``-aligned block of ``reads[1]``. A per-lane result whose block starts below 0 (a
    halo read) is rebased like every per-lane part; the bound check masks its negative lanes.
    """
    first, second = reads
    into0, into1 = node.write
    r1 = second.as_affine().range
    lo, hi = _numeric_lo(r1.lo), _numeric_hi(r1.hi)
    result_range = Interval(
        None if lo is None else lo - lo % length,
        None if hi is None else -(-hi // length) * length,
    )
    is_lane = first.lane is not None or second.lane is not None
    if is_lane:
        for r in (first.as_affine().range, r1, result_range):
            if not _fits_i32(r):
                raise _lane_range_error(into1, r)
    if (
        not first.terms
        and first.lane is None
        and not second.terms
        and second.lane is None
    ):
        return {
            into0: first,
            into1: _TableCoord(second.const ^ (first.const & (length - 1))),
        }
    v0, v1 = (_folded(table, read) for read in reads)
    mask = v0 & (length - 1) if isinstance(v0, int) else table.low_bits(v0, length)
    value = _xor_values(table, v1, mask)
    if is_lane and value.type != I32:
        raise _lane_range_error(into1, result_range)
    if is_lane:
        lane = _nonnegative_lane(
            table, AffineCoord(0, (), value, result_range, _ANY_RESIDUE), into1
        )
        return {
            into0: first,
            into1: _TableCoord(
                lane.const, (), lane.lane, lane.lane_range, lane.lane_congruence
            ),
        }
    inputs = first.inputs | second.inputs
    part = UniformPart(value, result_range, _ANY_RESIDUE)
    return {into0: first, into1: _TableCoord(0, (_Term(inputs, (part,)),))}


def _folded(table: _CallTable, coord: _TableCoord) -> Value | int:
    """``(uniform + lane) + const`` as one value (the int itself for a constant)."""
    if not coord.terms and coord.lane is None:
        return coord.const
    return table.value_of(coord)


def _xor_values(table: _CallTable, x: Value | int, y: Value | int) -> Value:
    """``x xor y`` with at least one value; an int 0 emits nothing."""
    if isinstance(y, int) and y == 0:
        return x
    if isinstance(x, int) and x == 0:
        return y
    ty = I64 if I64 in (getattr(x, "type", I32), getattr(y, "type", I32)) else I32
    vx = table.const(x, ty) if isinstance(x, int) else table.as_type(x, ty)
    vy = table.const(y, ty) if isinstance(y, int) else table.as_type(y, ty)
    return table.xor(vx, vy)


def _dividend_error(read: str, r: Interval) -> ValueError:
    return ValueError(
        f"unmerge of '{read}' needs its dividend in [0, 2^31) for magic division, but its range "
        f"is {r} -- bound the origin with exit_if_tile_outside(b, origin, length), or state a "
        "trusted range with runtime_value(b, v, at_most=...)"
    )


def _unmerge(
    table: _CallTable, read: _TableCoord, node: Node
) -> dict[str, _TableCoord]:
    """Split ``read`` into ``node.write`` (outermost first) with the carry rule.

    The read is ``s + L + c``: ``s`` its uniform side folded to one value, ``L`` its per-lane
    part, ``c`` its constant. Innermost level first, each is divided separately (``s`` once per
    call on the scalar unit, ``L`` at most once per call by the method its range allows, ``c``
    in Python) and the remainders are recombined with exact 0 / 1 carries: one scalar compare +
    select per distinct constant, one per-lane compare + select per access, never a per-lane
    division per access. Only the outermost piece keeps a uniform part.
    """
    names, lengths = node.write, node.params.lengths
    if len(lengths) == 1:
        return {names[0]: read}
    affine = read.as_affine()
    inputs = read.inputs
    s: Value | int = 0
    s_range, s_congruence = Interval(0, 1), _ANY_RESIDUE
    lane, c = _lane_of(read), read.const
    if read.terms:
        u_range = affine.uniform_range
        if _is_division_operand(read, u_range):
            s, _ = table.uniform_value(read)
            s_range = u_range
            s_congruence = AffineCoord(0, affine.uniform).congruence
        else:
            # The uniform side may be negative (a halo origin) or unbounded: split the whole
            # coordinate as one value instead.
            whole = affine.range
            if not whole.within(0, _I32_END):
                raise _dividend_error(node.read[0], whole)
            folded = table.value_of(read)
            c = 0
            if lane is None:
                s, s_range, s_congruence = folded, whole, affine.congruence
            else:
                lane = _Lane(folded, whole, affine.congruence)

    pieces: dict[str, _TableCoord] = {}
    level_range, level_congruence = affine.range, affine.congruence
    e_s: Value | int = 0  # the scalar carry into the next level
    e_l: Value | int = 0  # the per-lane carry into the next level
    for i in range(len(lengths) - 1, 0, -1):
        k = lengths[i]
        piece_range = level_range.floor_mod(k)
        piece_congruence = level_congruence.floor_mod(k)
        level_range = level_range.floor_div(k)
        level_congruence = level_congruence.floor_div(k)

        # 1. the uniform side, once per call
        qs: Value | int = 0
        rs: Value | int = 0
        if isinstance(s, Value):
            qs = table.quotient(s, k)
            if s_congruence.modulus % k or s_congruence.residue % k:
                rs = table.remainder(s, qs, k)
            s_range, s_congruence = s_range.floor_div(k), s_congruence.floor_div(k)
        # 2. the per-lane part, once per call
        q_lane: _Lane | None = None
        r_lane: _Lane | None = None
        if lane is not None:
            lane_hi = _numeric_hi(lane.range.hi)
            if lane_hi <= k:
                r_lane = lane
            elif lane_hi <= 2 * k:
                flag = table.at_least(lane.value, k)
                q_lane = _Lane(table.flag_value(flag), Interval(0, 2), _ANY_RESIDUE)
                r_lane = _Lane(
                    table.sub(lane.value, table.length_if(flag, k)),
                    Interval(0, k),
                    _ANY_RESIDUE,
                )
            else:
                q = table.quotient(lane.value, k)
                q_lane = _Lane(q, lane.range.floor_div(k), lane.congruence.floor_div(k))
                r_lane = _Lane(
                    table.remainder(lane.value, q, k),
                    Interval(0, k),
                    _ANY_RESIDUE,
                )
        # 3. the constant, in Python
        qc, rc = divmod(c, k)
        # 4. the scalar remainder with the scalar carry, per distinct (rc, e_s)
        r_s: Value | int = rc
        f_s: Value | int = 0
        if isinstance(rs, Value) or isinstance(e_s, Value):
            items = []
            if isinstance(rs, Value):
                items.append((rs, Interval(0, k)))
            if isinstance(e_s, Value):
                items.append((e_s, Interval(0, 2)))
            t, t_range = table.sum_of(items, rc)
            r_s, f_s = table.wrap_once(t, t_range, k)
        # 5. the piece, per access
        f_l: Value | int = 0
        if r_lane is None and not isinstance(e_l, Value):
            pieces[names[i]] = _piece(
                r_s, piece_range, piece_congruence, inputs, lane=False
            )
        else:
            items = []
            if isinstance(r_s, Value):
                items.append((r_s, Interval(0, k)))
            if r_lane is not None:
                items.append((r_lane.value, r_lane.range))
            if isinstance(e_l, Value):
                items.append((e_l, Interval(0, 2)))
            t, t_range = table.sum_of(items, r_s if isinstance(r_s, int) else 0)
            piece, f_l = table.wrap_once(t, t_range, k)
            pieces[names[i]] = _piece(
                piece, piece_range, piece_congruence, inputs, lane=True
            )
        # 6. the next level
        s, lane, c = qs, q_lane, qc
        e_s = table.flag_value(f_s) if isinstance(f_s, Value) else 0
        e_l = table.flag_value(f_l) if isinstance(f_l, Value) else 0

    # The outermost piece: (s + e_s) + (L + e_l) + c.
    parts = []
    if isinstance(s, Value):
        parts.append(UniformPart(s, s_range, s_congruence))
    if isinstance(e_s, Value):
        parts.append(UniformPart(e_s, Interval(0, 2), _ANY_RESIDUE))
    terms = (_Term(inputs, tuple(parts)),) if parts else ()
    lanes = [] if lane is None else [(lane, names[0])]
    if isinstance(e_l, Value):
        lanes.append((_Lane(e_l, Interval(0, 2), _ANY_RESIDUE), names[0]))
    pieces[names[0]] = _with_lane(c, terms, table.lane_sum(lanes))
    return {name: pieces[name] for name in names}


def _is_division_operand(read: _TableCoord, u_range: Interval) -> bool:
    """The uniform side sums in i32 and is proven in ``[0, 2^31)``: the scalar division's
    operand (magic division and the shift both need a non-negative i32)."""
    if not u_range.within(0, _I32_END):
        return False
    term_types = [
        _sum_type([p.value.type for p in t.parts], [p.range for p in t.parts])[0]
        for t in read.terms
    ]
    return _sum_type(term_types, [t.range for t in read.terms])[0] == I32


def _piece(
    value: Value | int,
    r: Interval,
    congruence: Congruence,
    inputs: frozenset[int],
    *,
    lane: bool,
) -> _TableCoord:
    """An inner unmerge piece: a constant, a uniform value or a per-lane value in ``[0, k)``."""
    if isinstance(value, int):
        return _TableCoord(value)
    if lane:
        return _TableCoord(0, (), value, r, congruence)
    return _TableCoord(0, (_Term(inputs, (UniformPart(value, r, congruence),)),))


# --- the stride fold ----------------------------------------------------------------------------


def _element_address(
    table: _CallTable,
    graph: TransformGraph,
    tensor: TensorDesc,
    env: dict[str, _TableCoord],
    elem_bits: int,
) -> ElementAddress:
    """``sum(output_a * stride_a)`` in tensor-axis order, split into its parts."""
    const = 0
    lanes: list[tuple[_Lane, str]] = []
    by_inputs: dict[frozenset[int], list[tuple[Value, Interval]]] = {}
    for name, stride in zip(graph.outputs, tensor.strides):
        coord = env[name]
        const += coord.const * stride
        if stride == 0:
            continue  # a zero part emits nothing
        lane = _lane_of(coord)
        if lane is not None:
            lanes.append((table.scaled_lane(lane, stride, name), name))
        for term in coord.terms:
            scaled = table.scaled_term(term, stride)
            by_inputs.setdefault(term.inputs, []).append(
                (table.term_value(scaled), scaled.range)
            )
    uniform = []
    for key in sorted(by_inputs, key=lambda k: tuple(sorted(k))):
        value, r = table.sum_of(by_inputs[key])
        uniform.append(UniformTerm(value, r.as_pair(), key))
    lane_sum = table.lane_sum(lanes)
    return ElementAddress(
        uniform=tuple(uniform),
        lane=None if lane_sum is None else lane_sum.value,
        lane_range=(
            None
            if lane_sum is None
            else (_numeric_lo(lane_sum.range.lo), _numeric_hi(lane_sum.range.hi))
        ),
        const=const,
        elem_bits=elem_bits,
    )


# --- the bound check ------------------------------------------------------------------------------

# A per-wave or per-lane scalar on its way through the fold: an emitted value (or a Python int,
# folded at build time) and its numeric range (int or ``None`` ends).
_Scalar = tuple[Value | int, Interval]

# Per distinct (per-wave side, side, end) of a call on the wide fold: the least access constant,
# the spread of the constants above it, and the widest per-lane extent among those accesses.
_WideGroups = dict[tuple, tuple[int, int, int]]


def _require_ends(name: str, lo: End, hi: End) -> None:
    for end in (lo, hi):
        is_int = isinstance(end, int) and not isinstance(end, bool)
        if end is not None and not is_int and not isinstance(end, RuntimeValue):
            raise TypeError(
                f"bound check of '{name}': an end is an int, a RuntimeValue or None, got "
                f"{_type_text(end)}"
            )
    if lo is None and hi is None:
        raise ValueError(
            f"emit_bound_check of '{name}' got lo=None and hi=None -- that checks nothing; pass "
            "the end(s) to check, or emit no check"
        )
    if isinstance(lo, int) and isinstance(hi, int) and lo >= hi:
        raise ValueError(
            f"bound of '{name}' is empty: [{lo}, {hi}) -- an always-empty window cannot be read"
        )


def _unproven(coord: AffineCoord, lo: End, hi: End) -> tuple[End, End]:
    """The ends whose side the coordinate's range does not already prove (``None`` otherwise)."""
    r = coord.range
    return (
        None if lo is None or _lo_proven(r.lo, lo) else lo,
        None if hi is None or _hi_proven(r.hi, hi) else hi,
    )


def _coord_text(coord: AffineCoord) -> str:
    """A split coordinate in plain words: its parts' IR names and its constant."""
    names = [part.value.name for part in coord.uniform]
    if coord.lane is not None:
        names.append(coord.lane.name)
    text = " + ".join(names) if names else str(coord.const)
    if names and coord.const:
        text += f" + {coord.const}" if coord.const > 0 else f" - {-coord.const}"
    return text


def _table_coord(coord: AffineCoord) -> _TableCoord:
    """A split coordinate as a table coordinate (its uniform parts one term, keyed by nothing)."""
    terms = (_Term(frozenset(), coord.uniform),) if coord.uniform else ()
    return _TableCoord(
        coord.const, terms, coord.lane, coord.lane_range, coord.lane_congruence
    )


def _lane_extent(coord: _TableCoord) -> int:
    """``L``: the per-lane part lies in ``[0, L)`` (1 for a coordinate with none)."""
    if coord.lane is None:
        return 1
    return _numeric_hi(coord.lane_range.hi)


def _check_value(
    table: _CallTable,
    coord: _TableCoord,
    lo: End,
    hi: End,
    name: str,
    groups: _WideGroups | None,
) -> Value:
    """``lo <= coord < hi`` for every lane: the single compare when it applies, else the fold."""
    u = _single_compare(coord, lo, hi)
    if u is not None:
        end = hi.value if isinstance(hi, RuntimeValue) else table.const(hi)
        return table.below(u, end)
    return _fold_check(table, coord, lo, hi, name, _lane_extent(coord), groups)


def _single_compare(coord: _TableCoord, lo: End, hi: End) -> Value | None:
    """``u`` when the check is ``u u< hi``: one i32 per-wave part ``u``, no constant, no
    per-lane part, a lower end of 0 (the int 0, or no lower end because ``u``'s range starts at
    0 or above) and ``hi`` an int in ``[1, 2^31)`` or a runtime length whose stated range lies in
    ``[0, 2^31)``. A negative ``u`` reads as ``>= 2^31`` unsigned."""
    parts = [part for term in coord.terms for part in term.parts]
    if coord.lane is not None or coord.const != 0 or len(parts) != 1:
        return None
    u = parts[0].value
    lower_is_zero = isinstance(lo, int) and not isinstance(lo, bool) and lo == 0
    least = _numeric_lo(parts[0].range.lo)
    lower_proven = lo is None and least is not None and least >= 0
    if u.type != I32 or not (lower_is_zero or lower_proven):
        return None
    if isinstance(hi, RuntimeValue):
        return u if hi._plus == 0 and hi._envelope.within(0, _I32_END) else None
    return u if hi is not None and 1 <= hi < _I32_END else None


def _fold_check(
    table: _CallTable,
    coord: _TableCoord,
    lo: End,
    hi: End,
    name: str,
    extent: int,
    groups: _WideGroups | None,
) -> Value:
    """The saturating fold for one element (the per-lane part over ``[0, extent)``)."""
    a, c = _folded_ends(table, coord, lo, hi, name, extent, groups)
    return _lane_compare(table, coord, a, c, extent, 0)


def _folded_ends(
    table: _CallTable,
    coord: _TableCoord,
    lo: End,
    hi: End,
    name: str,
    extent: int,
    groups: _WideGroups | None,
) -> tuple[_Scalar | None, _Scalar | None]:
    """``a = clamp(lo - s, 0, extent)`` and ``c = clamp(hi - s, 0, extent)`` (``None`` for an
    unchecked side), ``s`` the coordinate's per-wave side."""
    return tuple(
        (
            None
            if end is None
            else _folded_end(table, coord, end, side, extent, name, groups)
        )
        for side, end in (("lo", lo), ("hi", hi))
    )


def _folded_end(
    table: _CallTable,
    coord: _TableCoord,
    end: int | RuntimeValue,
    side: str,
    extent: int,
    name: str,
    groups: _WideGroups | None,
) -> _Scalar:
    """``clamp(end - s, 0, extent)`` as an i32 scalar, at the width its ranges allow."""
    k = coord.const
    if not coord.terms and isinstance(end, int):
        _require_spread(name, extent, 0)
        return _clamp(table, _exact(end - k), 0, extent)
    u_value, u_range = table.uniform_value(coord) if coord.terms else (None, None)
    u_type = None if u_value is None else u_value.type
    if _fold_fits_i32(u_type, u_range, end, k):
        _require_spread(name, extent, 0)
        if isinstance(end, int):
            d = _minus(table, _exact(end - k), (u_value, _numeric(u_range)))
        else:
            d = _plus_const(
                table, _runtime_minus(table, end, u_value, u_range, I32), end._plus - k
            )
        return _clamp(table, d, 0, extent)
    # The wide fold: one i64 pre-clamp per distinct (per-wave side, end) over the call, measured
    # from the least constant, then an i32 finish per access constant.
    cmin, span, widest = (groups or {}).get(
        _group_key(coord, side, end), (k, 0, extent)
    )
    _require_spread(name, widest, span)
    u64 = None if u_value is None else table.as_type(u_value, I64)
    if isinstance(end, int):
        x = _minus(table, _exact(end - cmin), (u64, _numeric(u_range)))
    else:
        x = _plus_const(
            table, _runtime_minus(table, end, u64, u_range, I64), end._plus - cmin
        )
    d_value, d_range = _clamp(table, x, 0, widest + span)
    if isinstance(d_value, Value):
        d_value = table.low_word(d_value)
    return _clamp(table, _minus(table, (d_value, d_range), _exact(k - cmin)), 0, extent)


def _runtime_minus(
    table: _CallTable,
    end: RuntimeValue,
    u: Value | None,
    u_range: Interval | None,
    ty: Type,
) -> _Scalar:
    """``end.value - u`` as ``ty`` (``end.value`` itself when there is no ``u``)."""
    value = table.as_type(end.value, ty)
    if u is None:
        return value, _numeric(end._envelope)
    return table.sub(value, u), _runtime_difference(end, u_range)


def _runtime_difference(end: RuntimeValue, u_range: Interval | None) -> Interval:
    """The range of ``end.value - u`` for ``u`` in ``u_range``. A ``u`` end over the same
    runtime value bounds the difference by identity (``u < M + p`` gives ``M - u > -p``);
    any other end counts with its envelope's extreme value."""
    envelope = end._envelope
    if u_range is None:
        return _numeric(envelope)
    u_lo, u_hi = u_range.lo, u_range.hi
    if isinstance(u_hi, RuntimeValue) and u_hi.value is end.value:
        lo = 1 - u_hi._plus
    else:
        most = _numeric_hi(u_hi)
        lo = None if envelope.lo is None or most is None else envelope.lo - most + 1
    if isinstance(u_lo, RuntimeValue) and u_lo.value is end.value:
        hi = 1 - u_lo._plus
    else:
        least = _numeric_lo(u_lo)
        hi = None if envelope.hi is None or least is None else envelope.hi - least
    if lo is not None and hi is not None and lo >= hi:
        return Interval(lo, lo + 1)  # a range no value can take: any width proof holds
    return Interval(lo, hi)


def _fold_fits_i32(
    u_type: Type | None, u_range: Interval | None, end: int | RuntimeValue, k: int
) -> bool:
    """Every value the i32 fold of ``end - (u + k)`` computes is proven inside i32 (``u``: the
    per-wave side, of ``u_type`` with ``u_range``; ``None`` when there is none)."""
    if u_type is not None and u_type != I32:
        return False
    if isinstance(end, int):
        start = _exact(end - k)[1]
        u = Interval(0, 1) if u_range is None else _numeric(u_range)
        return _fits_i32(start) and _fits_i32(start.plus(u.scaled(-1)))
    t = _runtime_difference(end, u_range)
    return (
        _fits_i32(_numeric(end._envelope))
        and _fits_i32(t)
        and _fits_i32(t.shifted(end._plus - k))
    )


def _require_spread(name: str, lane_hi: int, span: int) -> None:
    if lane_hi + span >= _I32_END:
        raise ValueError(
            f"bound check of '{name}' needs lane range + constant spread below 2^31, got "
            f"{lane_hi} + {span} -- narrow the lane range or split the accesses into separate "
            "calls"
        )


def _group_key(coord: _TableCoord, side: str, end: int | RuntimeValue) -> tuple:
    parts = tuple(id(part.value) for term in coord.terms for part in term.parts)
    if isinstance(end, RuntimeValue):
        return parts, side, "runtime", id(end.value), end._plus
    return parts, side, "int", end


def _lane_compare(
    table: _CallTable,
    coord: _TableCoord,
    a: _Scalar | None,
    c: _Scalar | None,
    extent: int,
    offset: int,
) -> Value:
    """The one per-lane compare for the element ``offset`` along the per-lane part."""
    x = _plus_const(table, _lane_scalar(coord), offset)
    if a is None:
        left, right = x, c
    elif c is None:
        left, right = _minus(table, x, a), _minus(table, _exact(extent), a)
    else:
        left, right = _minus(table, x, a), _clamp(table, _minus(table, c, a), 0, None)
    return table.below(_materialized(table, left), _materialized(table, right))


def _lane_scalar(coord: _TableCoord) -> _Scalar:
    if coord.lane is None:
        return _exact(0)
    return coord.lane, Interval(0, _lane_extent(coord))


# --- scalar steps of the fold (an int operand folds at build time) --------------------------------


def _exact(k: int) -> _Scalar:
    return k, Interval(k, k + 1)


def _numeric(r: Interval) -> Interval:
    """``r`` with every runtime end replaced by its envelope's extreme value."""
    lo, hi = _numeric_lo(r.lo), _numeric_hi(r.hi)
    if lo is not None and hi is not None and lo >= hi:
        return Interval(lo, lo + 1)  # a range no value can take: any width proof holds
    return Interval(lo, hi)


def _materialized(table: _CallTable, s: _Scalar) -> Value:
    value, _ = s
    return table.const(value) if isinstance(value, int) else value


def _minus(table: _CallTable, x: _Scalar, y: _Scalar | None) -> _Scalar:
    """``x - y`` (``y`` ``None`` is 0), at the type of the emitted operand."""
    if y is None:
        return x
    (xv, xr), (yv, yr) = x, y
    r = xr.plus(yr.scaled(-1))
    if isinstance(xv, int) and isinstance(yv, int):
        return xv - yv, r
    if isinstance(yv, int) and yv == 0:
        return x
    ty = xv.type if isinstance(xv, Value) else yv.type
    left = table.const(xv, ty) if isinstance(xv, int) else xv
    right = table.const(yv, ty) if isinstance(yv, int) else yv
    return table.sub(left, right), r


def _plus_const(table: _CallTable, s: _Scalar, k: int) -> _Scalar:
    value, r = s
    if k == 0:
        return s
    if isinstance(value, int):
        return _exact(value + k)
    return table.add_const(value, k), r.shifted(k)


def _clamp(table: _CallTable, s: _Scalar, lo: int, hi: int | None) -> _Scalar:
    """``s`` saturated into ``[lo, hi]`` (``hi`` ``None``: no upper limit); a side its range
    already proves emits nothing."""
    value, r = s
    if isinstance(value, int):
        clamped = max(value, lo) if hi is None else min(max(value, lo), hi)
        return _exact(clamped)
    r_lo, r_hi = r.lo, r.hi
    if r_hi is not None and r_hi - 1 <= lo:
        return _exact(lo)
    if hi is not None and r_lo is not None and r_lo >= hi:
        return _exact(hi)
    if r_lo is None or r_lo < lo:
        value, r_lo = table.no_less(value, lo), lo
    if hi is not None and (r_hi is None or r_hi - 1 > hi):
        value, r_hi = table.at_most(value, hi), hi + 1
    return value, Interval(r_lo, r_hi)


# --- the validity of accesses ---------------------------------------------------------------------


@dataclass(frozen=True)
class _PlannedCheck:
    """One check of one access: ``step`` is its coordinate's step along the vector (0 for a
    single element), ``uniform`` whether it has one answer for every element."""

    check: BoundCheck
    step: int
    uniform: bool


def _plan_validity(
    graph: TransformGraph,
    tensor: TensorDesc,
    split: Sequence[tuple[AffineCoord, ...]],
    run: VectorRun | None,
    buffer_checked_sides: frozenset[CheckSide] | None,
) -> list[tuple[_PlannedCheck, ...]]:
    """Every access's checks, decided before anything is emitted (pure Python).

    Each input's range is its split coordinate's, the vector's input widened to its last
    element. The vector errors and the buffer-checked-side errors are raised here.
    """
    bounds = tuple(AxisBound.of(0, length) for length in tensor.lengths)
    checks = []
    for coords in split:
        ranges = [coord.range for coord in coords]
        if run is not None:
            ranges[run.axis] = ranges[run.axis].plus(Interval(0, run.width))
        checks.append(_required_checks(graph, tuple(ranges), bounds))
    if buffer_checked_sides is not None:
        checks = _without_buffer_checked(checks, buffer_checked_sides)
    if run is None:
        return [tuple(_PlannedCheck(c, 0, True) for c in access) for access in checks]
    plans = []
    for coords, access in zip(split, checks):
        steps = _run_steps(graph, run, coords)
        blocker = _contiguity_blocker(graph, tensor, run, steps)
        if blocker is not None:
            raise ValueError(_vector_text(graph, run, blocker))
        bent = _first_non_monotone(access, steps)
        if bent is not None:
            axis = run.axis
            raise ValueError(
                f"vector along input {axis} ('{tuple(graph.inputs)[axis]}') is not monotone at "
                f"checked coordinate '{bent}' -- its validity cannot be decided from the first "
                "and last element; emit per element (vector_width=1)"
            )
        plans.append(
            tuple(
                _PlannedCheck(
                    c,
                    steps[c.coordinate].step,
                    _check_uniform(c, steps[c.coordinate], run.width),
                )
                for c in access
            )
        )
    return plans


def _vector_text(
    graph: TransformGraph, run: VectorRun, blocker: tuple[str, str]
) -> str:
    coordinate, reason = blocker
    axis, width = run.axis, run.width
    return (
        f"vector along input {axis} ('{tuple(graph.inputs)[axis]}') of width {width} is not "
        f"contiguous at coordinate '{coordinate}' ({reason}) -- use a narrower vector, or start "
        f"it at a multiple of {width} (the walk reads that fact from how the coordinate was "
        "computed; a runtime value the IR cannot show it for is stated with "
        "runtime_value(..., multiple_of=...))"
    )


def _without_buffer_checked(
    checks: list[tuple[BoundCheck, ...]], sides: frozenset[CheckSide]
) -> list[tuple[BoundCheck, ...]]:
    """The checks with ``sides`` dropped; each side must be one some access reports."""
    if not sides:
        raise ValueError("buffer_checked_sides is empty -- call emit_accesses")
    reported = set()
    for access in checks:
        for check in access:
            if check.lo is not None:
                reported.add(CheckSide(check.coordinate, Side.LOWER))
            if check.hi is not None:
                reported.add(CheckSide(check.coordinate, Side.UPPER))
    for side in sorted(sides, key=lambda s: (s.coordinate, s.side.value)):
        if side not in reported:
            raise ValueError(
                f"buffer_checked_sides names the {side.side.name.lower()} side of "
                f"'{side.coordinate}', which required_checks did not report as failing for any "
                "access of this call -- only a side that can fail can be left to the buffer"
            )
    kept = []
    for access in checks:
        remaining = []
        for check in access:
            lo = None if CheckSide(check.coordinate, Side.LOWER) in sides else check.lo
            hi = None if CheckSide(check.coordinate, Side.UPPER) in sides else check.hi
            if lo is not None or hi is not None:
                remaining.append(replace(check, lo=lo, hi=hi))
        kept.append(tuple(remaining))
    return kept


def _emit_validity(
    table: _CallTable,
    envs: Sequence[dict[str, _TableCoord]],
    plans: Sequence[tuple[_PlannedCheck, ...]],
    run: VectorRun | None,
) -> list[tuple[Value | None, tuple[Value, Value] | None]]:
    """Each access's ``(valid, valid_span)`` over the call's table, access by access."""
    width = 1 if run is None else run.width
    groups = _wide_groups(envs, plans, width)
    results = []
    for env, plan in zip(envs, plans):
        valid: Value | None = None
        straddles = any(not pc.uniform for pc in plan)
        firsts: list[_Scalar] = []
        ends: list[_Scalar] = []
        for pc in plan:
            name, check = pc.check.coordinate, pc.check
            coord = env[name]
            if pc.uniform:
                ok = _check_value(table, coord, check.lo, check.hi, name, groups)
                valid = _all_of(table, valid, ok)
                if straddles:
                    ends.append((table.length_if(ok, width), Interval(0, width + 1)))
                continue
            reach = (width - 1) * pc.step
            extent = _lane_extent(coord) + reach
            a, c = _folded_ends(table, coord, check.lo, check.hi, name, extent, groups)
            valid = _all_of(table, valid, _lane_compare(table, coord, a, c, extent, 0))
            valid = _all_of(
                table, valid, _lane_compare(table, coord, a, c, extent, reach)
            )
            lane = _lane_scalar(coord)
            if a is not None:
                firsts.append(
                    _element_index(table, _minus(table, a, lane), pc.step, width)
                )
            if c is not None:
                ends.append(
                    _element_index(table, _minus(table, c, lane), pc.step, width)
                )
        span = None
        if straddles:
            first = _materialized(table, _extreme(table, firsts, 0, take_max=True))
            end = _materialized(table, _extreme(table, ends, width, take_max=False))
            span = (first, end)
        results.append((valid, span))
    return results


def _wide_groups(
    envs: Sequence[dict[str, _TableCoord]],
    plans: Sequence[tuple[_PlannedCheck, ...]],
    width: int,
) -> _WideGroups:
    """Per distinct (per-wave side, side, end) on the wide fold: the least constant, the spread
    above it and the widest per-lane extent, over every access of the call."""
    members: dict[tuple, list[tuple[int, int]]] = {}
    for env, plan in zip(envs, plans):
        for pc in plan:
            coord = env[pc.check.coordinate]
            if not coord.terms:
                continue
            u_type, u_range = _uniform_type(coord)
            extent = _lane_extent(coord) + (0 if pc.uniform else (width - 1) * pc.step)
            for side, end in (("lo", pc.check.lo), ("hi", pc.check.hi)):
                if end is None or _fold_fits_i32(u_type, u_range, end, coord.const):
                    continue
                if (
                    pc.uniform
                    and _single_compare(coord, pc.check.lo, pc.check.hi) is not None
                ):
                    continue
                key = _group_key(coord, side, end)
                members.setdefault(key, []).append((coord.const, extent))
    groups = {}
    for key, found in members.items():
        consts = [k for k, _ in found]
        groups[key] = (min(consts), max(consts) - min(consts), max(e for _, e in found))
    return groups


def _uniform_type(coord: _TableCoord) -> tuple[Type, Interval]:
    """The type and range ``uniform_value`` gives the coordinate's per-wave side (no IR)."""
    term_types = [
        _sum_type([p.value.type for p in t.parts], [p.range for p in t.parts])[0]
        for t in coord.terms
    ]
    ty, _ = _sum_type(term_types, [t.range for t in coord.terms])
    return ty, coord.as_affine().uniform_range


def _all_of(table: _CallTable, valid: Value | None, ok: Value) -> Value:
    return ok if valid is None else table.both(valid, ok)


def _element_index(table: _CallTable, x: _Scalar, step: int, width: int) -> _Scalar:
    """``clamp(ceil(x / step), 0, width)``: the first element whose per-lane value reaches
    ``x``, for a coordinate moving by ``step`` per element."""
    value, _ = _clamp(table, x, 0, width * step)
    if isinstance(value, int):
        return _exact(-(-value // step))
    return table.ceil_quotient(value, step), Interval(0, width + 1)


def _extreme(
    table: _CallTable, items: list[_Scalar], start: int, *, take_max: bool
) -> _Scalar:
    """The max (or min) of ``start`` and ``items`` (each in ``[0, width]``), in order; ints fold
    at build time and a bound an item's range already passes emits nothing."""
    acc: _Scalar = _exact(start)
    for item in items:
        (av, ar), (iv, ir) = acc, item
        if isinstance(av, int) and isinstance(iv, int):
            acc = _exact(max(av, iv) if take_max else min(av, iv))
        elif isinstance(av, int) or isinstance(iv, int):
            bound, other = (av, item) if isinstance(av, int) else (iv, acc)
            acc = (
                _clamp(table, other, bound, None)
                if take_max
                else _clamp(table, other, 0, bound)
            )
        else:
            value = table.larger(av, iv) if take_max else table.smaller(av, iv)
            acc = value, (ar.maximum(ir) if take_max else ar.minimum(ir))
    return acc
