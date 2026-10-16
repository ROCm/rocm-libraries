# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit a transform graph's coordinates, and each access's element address split into parts.

Builder first, IR side effects. Two toolbox functions, for tests and custom memory code:

- ``emit_coords(b, graph, coords)`` -- every named coordinate of the graph as one emitted value.
- ``emit_addresses(b, graph, tensor, accesses)`` -- one ``ElementAddress`` per access: the
  element address split into wave-uniform terms (``UniformTerm``), one per-lane part and a
  Python-int constant, so the delivery side can put each part where it is cheapest.

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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from rocke.core.dtypes import dtype_info
from rocke.core.ir import I32, I64, Type, Value
from rocke.helpers.magic_division import calculate_magic_numbers, do_magic_division

from .address import _require_nonnegative_strides
from .graph import Node, TransformGraph, TransformType, _require_outputs_match_rank
from .ranges import (
    AffineCoord,
    Congruence,
    End,
    Interval,
    UniformPart,
    _numeric_hi,
    _numeric_lo,
)
from .walk import split_coordinate

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder

    from ..descriptors import TensorDesc

__all__ = [
    "Coordinate",
    "ElementAddress",
    "UniformTerm",
    "emit_addresses",
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
    elem_bits = _elem_bits(tensor)
    table = _CallTable(b)
    return tuple(
        _element_address(
            table,
            graph,
            tensor,
            _emit_coord_table(table, graph, _split_access(b, access)),
            elem_bits,
        )
        for access in accesses
    )


# --- the access checks --------------------------------------------------------------------------


def _check_accesses(
    graph: TransformGraph,
    accesses: Sequence[tuple[Coordinate, ...]],
    *,
    tensor: TensorDesc | None = None,
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


def _is_coordinate(coord: object) -> bool:
    if isinstance(coord, Value):
        return coord.type == I32
    if isinstance(coord, bool):
        return False
    return isinstance(coord, (int, AffineCoord))


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
