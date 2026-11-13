# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Value ranges and required bound checks: which side of which coordinate can fail.

Two questions, one home each:

- ``value_ranges`` -- for given input ranges, the interval every coordinate of a graph can take
  (the values every lane computes, valid or not);
- ``required_checks`` -- at build time, which side of which coordinate can leave the domain it
  must stay in, and therefore needs a run-time check. A side the ranges prove needs none.

A coordinate needs a check exactly when its range can leave one of:

1. its INPUT DOMAIN -- an input whose supplied range is not inside ``[0, declared length)``;
2. a READ DOMAIN -- a range a step's read is not inside (an ``unmerge`` read past the product
   of its pieces: a K tail, derived from the graph, never written by hand);
3. its TENSOR BOUND -- an output whose range is not inside the tensor's ``[0, length)``.

Checks are AND-ed into one validity, so each check may assume the checks before it passed: a
coordinate's range is narrowed by its own domains (the "conditioned" range) before the steps that
read it compute theirs. ``value_ranges`` reports the unconditioned ranges, because an invalid lane
still computes an address from them. Whether an axis is checked follows from data (ranges against
domains and bounds), never from a per-axis switch.

The toolbox (on the package door) takes and returns plain ``(lo, hi)`` pairs, half-open, one per
graph input in ``inputs(...)`` order: each end an int, a ``Value`` (an end at a runtime integer,
such as a kernel parameter or a Value ``gpu_checked_value`` returned), or ``None`` (unbounded on
that side). ``(None, None)`` is "not known" and is always checked; there is no "trusted" default.

The internal sibling surface is the arithmetic itself: ``Interval`` and ``Congruence`` hold one
value's range and residue, and their methods are the one home of every range-combining rule
(``plus``, ``scaled``, ``floor_div``, ``minimum``, ``intersect``, ``snapped``, ...). Everything that
needs a range -- these checks, the address split, the IR range walk -- composes them rather than
restating a rule. ``_value_ranges`` and ``_required_checks`` are the same rules on ``Interval`` s
with explicit ``AxisBound`` s. ``AffineCoord`` (with ``UniformPart``) is one coordinate split into a
constant, wave-uniform parts and a per-lane part, each with its range and residue: the IR range
walk produces it, the emitters consume it.

Runtime ends prove relations to a runtime length by IDENTITY: an upper side bounded by a runtime
length is proven only by a computed end over the same IR value object (``length`` itself, or
``length`` shifted down). Inside a range such an end is the private ``_RuntimeEnd(value, plus)``;
it holds no facts and reads its numbers from ``stated_facts_of(value)``, so a length's facts have
one home (its parameter declaration or the marker that produced it) and no restated copy can break
the proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd, prod
from typing import TYPE_CHECKING

from rocke.core.facts import stated_facts_of
from rocke.core.ir import I32, Value

from .graph import Node, TransformGraph, TransformType, _require_outputs_match_rank

if TYPE_CHECKING:
    from ..descriptors import TensorDesc

__all__ = [
    "UNBOUNDED",
    "AffineCoord",
    "AxisBound",
    "BoundCheck",
    "CheckReason",
    "CheckSide",
    "Congruence",
    "End",
    "Interval",
    "Side",
    "UniformPart",
    "VectorRun",
    "graph_injective",
    "required_checks",
    "value_ranges",
]


# --- the values ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Congruence:
    """``value == residue (mod modulus)``, ``modulus >= 1``: a multiple-of fact.

    The residue is kept in ``[0, modulus)``. A constant is an ``int``, never a ``Congruence``:
    no modulus states "exactly c", so constants enter through ``shifted`` / ``scaled`` /
    ``join(int)``.
    """

    modulus: int
    residue: int = 0

    def __post_init__(self) -> None:
        if self.modulus < 1:
            raise ValueError(f"Congruence modulus must be >= 1, got {self.modulus}")
        object.__setattr__(self, "residue", self.residue % self.modulus)

    def __str__(self) -> str:
        return f"{self.residue} mod {self.modulus}"

    def plus(self, other: Congruence) -> Congruence:
        """Residue of ``x + y``."""
        return Congruence(
            gcd(self.modulus, other.modulus), self.residue + other.residue
        )

    def shifted(self, k: int) -> Congruence:
        """Residue of ``x + k``."""
        return Congruence(self.modulus, self.residue + k)

    def scaled(self, k: int) -> Congruence:
        """Residue of ``x * k``, ``k != 0`` (a product by 0 is the int 0; the caller folds it)."""
        if k == 0:
            raise ValueError(
                "Congruence.scaled(0): the product is the constant 0 -- fold it as the int 0 "
                "instead of a congruence"
            )
        return Congruence(self.modulus * abs(k), self.residue * k)

    def floor_div(self, k: int) -> Congruence:
        """Residue of ``x // k``, ``k >= 1``: exact when ``k`` divides the modulus, else none."""
        _require_divisor("Congruence", "floor_div", k)
        if self.modulus % k:
            return Congruence(1, 0)
        return Congruence(self.modulus // k, self.residue // k)

    def floor_mod(self, k: int) -> Congruence:
        """Residue of ``x % k``, ``k >= 1``."""
        _require_divisor("Congruence", "floor_mod", k)
        return Congruence(gcd(self.modulus, k), self.residue)

    def join(self, other: Congruence | int) -> Congruence:
        """The residue both operands share: the congruence of ``min(x, y)`` or ``max(x, y)``.

        The result is one of the two operands, so it keeps ``gcd(m, n, s - r)``; an int ``c`` is
        an exact value and keeps ``gcd(m, c - r)`` (a cap congruent to the residue keeps the
        congruence whole).
        """
        if isinstance(other, Congruence):
            modulus = gcd(self.modulus, other.modulus, other.residue - self.residue)
        else:
            modulus = gcd(self.modulus, other - self.residue)
        return Congruence(modulus, self.residue)


def _stated_interval(value: Value) -> Interval:
    """The range ``value`` itself is stated to lie in (``None`` for a side no fact states),
    read from ``stated_facts_of``: never derived, never restated."""
    facts = stated_facts_of(value)
    return Interval(facts.lower, None if facts.upper is None else facts.upper + 1)


@dataclass(frozen=True, eq=False)
class _RuntimeEnd:
    """A range end ``value + plus`` relative to a runtime integer: ``T`` or ``T - 63``.

    Private to range arithmetic. Authors pass the plain ``Value`` (an end with ``plus == 0``);
    a range keeps every runtime end in this form, so an end shifted by a constant keeps the
    identity of its value. It holds no facts: its numbers come from ``stated_facts_of(value)``
    whenever one is needed, so the facts keep their one home (the parameter's declaration or the
    marker that produced the value). Two ends are equal exactly when they are over the same IR
    value object with the same ``plus``: the identity a proof keys on.
    """

    value: Value
    plus: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _RuntimeEnd):
            return NotImplemented
        return self.value is other.value and self.plus == other.plus

    def __hash__(self) -> int:
        return hash((id(self.value), self.plus))

    def __str__(self) -> str:
        """``L``, ``L + 3`` or ``L - 3``: the end in plain words, by its IR value's name."""
        if self.plus > 0:
            return f"{self.value.name} + {self.plus}"
        if self.plus < 0:
            return f"{self.value.name} - {-self.plus}"
        return self.value.name

    def shifted(self, k: int) -> _RuntimeEnd:
        """The end ``k`` further: ``value + plus + k``."""
        return _RuntimeEnd(self.value, self.plus + k)

    @property
    def envelope(self) -> Interval:
        """The stated range of ``value`` itself (without ``plus``)."""
        return _stated_interval(self.value)

    @property
    def congruence(self) -> Congruence:
        """The stated residue of ``value`` itself (without ``plus``)."""
        facts = stated_facts_of(self.value)
        return Congruence(facts.multiple_of, facts.residue)


# A range end at the package door: an int, a Value (an end at a runtime integer) or None.
End = int | Value | None
# A range end inside a range: a Value end is kept as a ``_RuntimeEnd``.
_End = int | _RuntimeEnd | None
# Either form: the end rules below take both.
_AnyEnd = int | Value | _RuntimeEnd | None


def _require_i32_end(end: Value, what: str) -> None:
    """Refuse a door end ``Value`` that is not an i32: a runtime length or end in this package
    is an i32 integer (the type every range compare and fold is emitted in). ``what`` opens the
    message and names the rule the caller states."""
    if end.type != I32:
        raise TypeError(f"{what}, got Value {end.name} of type {end.type.name}")


def _runtime(end: object) -> _RuntimeEnd | None:
    """``end`` as a runtime end (a plain ``Value`` is the end at it), or ``None`` for an int or
    ``None`` end."""
    if isinstance(end, _RuntimeEnd):
        return end
    if isinstance(end, Value):
        return _RuntimeEnd(end)
    return None


def _door_end(end: _End, side: str) -> End:
    """An end as the package door shows it: a runtime end with no shift is its ``Value``; a
    shifted one (``T - 1``) is the number its value's stated facts give on that side, because
    the shifted form is private to range arithmetic."""
    if not isinstance(end, _RuntimeEnd):
        return end
    if end.plus == 0:
        return end.value
    return _numeric_lo(end) if side == "lo" else _numeric_hi(end)


@dataclass(frozen=True)
class Interval:
    """The half-open range ``[lo, hi)`` of one value; ``None`` is unbounded on that side.

    Each end is an int, a runtime end (an end relative to a runtime integer: a ``Value`` given
    here is kept as the private ``_RuntimeEnd`` at it) or ``None``. Int ends need ``lo < hi``: an
    empty range states no fact. Every method is pure (no IR) and returns a new value; the rules
    for ends that are not ints are in ``_numeric_lo`` / ``_numeric_hi`` and the pair rules below
    them.
    """

    lo: _End
    hi: _End

    def __post_init__(self) -> None:
        for side in ("lo", "hi"):
            end = getattr(self, side)
            if isinstance(end, Value):
                _require_i32_end(end, "a range end is an int, an i32 Value or None")
                object.__setattr__(self, side, _RuntimeEnd(end))
        if isinstance(self.lo, int) and isinstance(self.hi, int) and self.lo >= self.hi:
            raise ValueError(
                f"Interval [{self.lo}, {self.hi}) is empty -- int ends need lo < hi; an empty "
                "range states no fact (to ask whether two ranges meet, use Interval.overlaps)"
            )

    def __str__(self) -> str:
        return f"[{_end_text(self.lo, 'lo')}, {_end_text(self.hi, 'hi')})"

    @classmethod
    def from_pair(cls, pair: tuple[End, End]) -> Interval:
        """The ``Interval`` of a public ``(lo, hi)`` pair (the toolbox form)."""
        lo, hi = pair
        return cls(lo, hi)

    def as_pair(self) -> tuple[End, End]:
        """The public ``(lo, hi)`` pair: a public value never carries an ``Interval``. An end
        relative to a runtime value is shown as that ``Value``, or, when shifted by a constant,
        as the number its stated facts give (``_door_end``)."""
        return (_door_end(self.lo, "lo"), _door_end(self.hi, "hi"))

    def plus(self, other: Interval) -> Interval:
        """Range of ``x + y``."""
        lo = _add_ends(self.lo, other.lo, "lo")
        hi = _add_ends(self.hi, other.hi, "hi")
        return Interval(lo, _shift_end(hi, -1))

    def shifted(self, k: int) -> Interval:
        """Range of ``x + k``."""
        return Interval(_shift_end(self.lo, k), _shift_end(self.hi, k))

    def scaled(self, k: int) -> Interval:
        """Range of ``x * k`` for any sign of ``k`` (``k == 0`` gives ``[0, 1)``)."""
        if k == 1:
            return self
        if k == 0:
            return Interval(0, 1)
        lo, last = _numeric_lo(self.lo), _numeric_last(self.hi)
        if k > 0:
            return Interval(_mul(lo, k), _shift_end(_mul(last, k), 1))
        return Interval(_mul(last, k), _shift_end(_mul(lo, k), 1))

    def floor_div(self, k: int) -> Interval:
        """Range of ``x // k``, ``k >= 1`` (floor semantics for every sign)."""
        _require_divisor("Interval", "floor_div", k)
        lo, last = _numeric_lo(self.lo), _numeric_last(self.hi)
        return Interval(
            None if lo is None else lo // k, None if last is None else last // k + 1
        )

    def floor_mod(self, k: int) -> Interval:
        """Range of ``x % k``, ``k >= 1`` (floor semantics): exact inside one period."""
        _require_divisor("Interval", "floor_mod", k)
        lo, last = _numeric_lo(self.lo), _numeric_last(self.hi)
        if lo is not None and last is not None and lo // k == last // k:
            return Interval(lo % k, last % k + 1)
        return Interval(0, k)

    def minimum(self, other: Interval) -> Interval:
        """Range of ``min(x, y)``; symmetric in its operands.

        The ``lo`` must bound both operands (the looser end when the two cannot be ordered); the
        ``hi`` may come from either (the tighter end). A runtime end survives only when both
        ends are over the same value, or the other side is unbounded.
        """
        return Interval(
            _pick(self.lo, other.lo, "lo", take_max=False, both_bound=True),
            _pick(self.hi, other.hi, "hi", take_max=False, both_bound=False),
        )

    def maximum(self, other: Interval) -> Interval:
        """Range of ``max(x, y)``; symmetric in its operands (``minimum`` mirrored)."""
        return Interval(
            _pick(self.lo, other.lo, "lo", take_max=True, both_bound=False),
            _pick(self.hi, other.hi, "hi", take_max=True, both_bound=True),
        )

    def overlaps(self, other: Interval) -> bool:
        """``False`` exactly when no value lies in both, for every value of each runtime end.

        Decided on the four lower-end / upper-end pairs, not on the ends ``intersect`` keeps:
        a kept runtime ``hi`` gives up the int end's numeric fact, so the kept ends alone can look
        non-empty when the inputs are disjoint (``[0, 5)`` and ``[10, L)`` keep ``[10, L)``).
        """
        return not any(
            _pair_always_empty(lo, hi)
            for lo in (self.lo, other.lo)
            for hi in (self.hi, other.hi)
        )

    def intersect(self, other: Interval) -> Interval:
        """Range of a value known to lie in both; symmetric.

        Each end is one of the two operands' ends of that side. On ``hi`` a runtime end is kept
        over an int (it can prove a runtime bound by identity); on ``lo`` the int is kept.
        Raises when ``overlaps`` is ``False``: two facts about one value that cannot both hold.
        """
        if not self.overlaps(other):
            raise ValueError(
                f"Interval.intersect: {self} and {other} have no value in common -- the two "
                "ranges state contradictory facts; check where each was derived"
            )
        return Interval(
            _intersect_end(self.lo, other.lo, "lo"),
            _intersect_end(self.hi, other.hi, "hi"),
        )

    def snapped(self, congruence: Congruence) -> Interval:
        """The tightest range holding the same members of the residue class ``congruence``.

        An int ``lo`` is raised and the last value lowered onto the class. A runtime end moves
        only when its value's stated modulus is a multiple of ``congruence.modulus`` (its residue
        is then known) and stays over the same value; otherwise it is kept as it is.
        """
        m, r = congruence.modulus, congruence.residue
        lo, hi = self.lo, self.hi
        if isinstance(lo, int):
            new_lo = lo + (r - lo) % m
        elif isinstance(lo, _RuntimeEnd) and lo.congruence.modulus % m == 0:
            rho = lo.congruence.residue % m
            new_lo = lo.shifted((r - rho - lo.plus) % m)
        else:
            new_lo = lo
        if isinstance(hi, int):
            new_hi = hi - (hi - 1 - r) % m
        elif isinstance(hi, _RuntimeEnd) and hi.congruence.modulus % m == 0:
            rho = hi.congruence.residue % m
            new_hi = hi.shifted(-((rho + hi.plus - 1 - r) % m))
        else:
            new_hi = hi
        if isinstance(new_lo, int) and isinstance(new_hi, int) and new_lo >= new_hi:
            raise ValueError(
                f"Interval [{lo}, {hi}) has no value congruent to {r} mod {m} -- the stated "
                "range and congruence contradict each other"
            )
        return Interval(new_lo, new_hi)

    def within(self, lo: int | None, hi: int | None) -> bool:
        """Every value of the range lies in ``[lo, hi)`` (``None``: no bound on that side).

        Decided on the numeric ends: a runtime end counts with its stated extreme value.
        """
        return _lo_proven(self.lo, lo) and _hi_proven(self.hi, hi)


UNBOUNDED = Interval(None, None)


@dataclass(frozen=True)
class VectorRun:
    """An access that stands for ``width`` consecutive values of graph input ``axis``."""

    axis: int
    width: int


@dataclass(frozen=True)
class AxisBound:
    """The valid extent ``[lo, hi)`` of one tensor axis (the tensor side; emits nothing).

    Each end is an int or a ``Value`` (a runtime length): a bound is a length, not an interval
    end. Built by ``AxisBound.of``; the toolbox builds ``AxisBound.of(0, length)`` per axis from
    the tensor's lengths, the one home of a length.
    """

    lo: int | Value
    hi: int | Value

    def __post_init__(self) -> None:
        for side in ("lo", "hi"):
            end = getattr(self, side)
            if isinstance(end, Value):
                _require_i32_end(end, "an AxisBound end is an int or an i32 Value")
            if isinstance(end, _RuntimeEnd):
                if end.plus != 0:
                    raise ValueError(
                        f"AxisBound end {end} is a length shifted by {end.plus} -- a bound is "
                        "a length; an end relative to a length belongs in an Interval"
                    )
                object.__setattr__(self, side, end.value)

    @classmethod
    def of(cls, lo: int | Value, hi: int | Value) -> AxisBound:
        """The bound ``[lo, hi)`` of one tensor axis."""
        return cls(lo, hi)


@dataclass(frozen=True, eq=False)
class UniformPart:
    """One wave-uniform summand of a coordinate, with its range and residue."""

    value: Value
    range: Interval
    congruence: Congruence


@dataclass(frozen=True, eq=False)
class AffineCoord:
    """One coordinate split into a constant, wave-uniform parts and one per-lane part.

    The coordinate's value is ``const + sum(part.value for part in uniform) + lane``; nothing is
    emitted until a consumer emits it. Every part carries an explicit range (``UNBOUNDED`` when
    unknown) and residue. The IR range walk builds one from a plain value; tiling-layer code that
    already holds the parts builds one with the constructor, ``of_int`` and ``+``.
    """

    const: int
    uniform: tuple[UniformPart, ...] = ()
    lane: Value | None = None
    lane_range: Interval | None = None
    lane_congruence: Congruence = Congruence(1, 0)

    def __post_init__(self) -> None:
        if any(part.range is None for part in self.uniform):
            raise ValueError(_missing_range_text("uniform"))
        if self.lane is not None and self.lane_range is None:
            raise ValueError(_missing_range_text("lane"))

    @classmethod
    def of_int(cls, c: int) -> AffineCoord:
        """The coordinate that is the constant ``c``."""
        return cls(c)

    @property
    def uniform_range(self) -> Interval:
        """The range of the uniform side: the sum of the uniform parts (``[0, 1)`` when none)."""
        if not self.uniform:
            return Interval(0, 1)
        total = self.uniform[0].range
        for part in self.uniform[1:]:
            total = total.plus(part.range)
        return total

    @property
    def range(self) -> Interval:
        """The range of the whole coordinate: uniform side + lane part + constant."""
        total = self.uniform_range
        if self.lane is not None:
            total = total.plus(self.lane_range)
        return total.shifted(self.const)

    @property
    def congruence(self) -> Congruence:
        """The residue of the whole coordinate, combined from every part, the lane part and the
        constant. A coordinate with no part is its constant (a multiple of itself)."""
        residues = [part.congruence for part in self.uniform]
        if self.lane is not None:
            residues.append(self.lane_congruence)
        if not residues:
            return Congruence(abs(self.const) or 1, 0)
        total = residues[0]
        for residue in residues[1:]:
            total = total.plus(residue)
        return total.shifted(self.const)

    def __add__(self, other: AffineCoord) -> AffineCoord:
        """The sum, emitting nothing: consts added, uniform parts concatenated (this one's
        first), the one lane part kept."""
        if self.lane is not None and other.lane is not None:
            raise ValueError(
                f"AffineCoord + AffineCoord: both operands have a lane part ({self.lane}, "
                f"{other.lane}) -- a coordinate has one lane part; add the per-lane values first "
                "and split the sum with the walk"
            )
        lane_side = self if self.lane is not None else other
        return AffineCoord(
            self.const + other.const,
            self.uniform + other.uniform,
            lane_side.lane,
            lane_side.lane_range,
            lane_side.lane_congruence,
        )


def _missing_range_text(part: str) -> str:
    return (
        f"AffineCoord {part} part has no range -- every part needs an explicit Interval "
        "(UNBOUNDED if unknown)"
    )


class Side(Enum):
    """One side of a checked coordinate's range."""

    LOWER = 1
    UPPER = 2


@dataclass(frozen=True)
class CheckSide:
    """One side of one checked coordinate (a graph coordinate name)."""

    coordinate: str
    side: Side


class CheckReason(Enum):
    """Why a coordinate needs a run-time check."""

    INPUT_DOMAIN = 1
    READ_DOMAIN = 2
    TENSOR_BOUND = 3


@dataclass(frozen=True)
class BoundCheck:
    """One coordinate's run-time check: ``lo <= coordinate < hi`` on the sides that can fail.

    ``coordinate`` is a name the graph was built with. A side is ``None`` when it is proven
    (nothing to check there), so the two sides can be enforced separately. Each end is an int or
    a ``Value`` (a runtime length). ``reasons`` lists every reason a side was found failing.
    """

    coordinate: str
    lo: int | Value | None
    hi: int | Value | None
    reasons: tuple[CheckReason, ...]


# --- end rules -------------------------------------------------------------------------------


def _require_divisor(owner: str, operation: str, k: int) -> None:
    if k < 1:
        raise ValueError(f"{owner}.{operation} needs a divisor >= 1, got {k}")


def _end_text(end: _AnyEnd, side: str) -> str:
    if end is None:
        return "-inf" if side == "lo" else "+inf"
    runtime = _runtime(end)
    return str(end) if runtime is None else str(runtime)


def _numeric_lo(end: _AnyEnd) -> int | None:
    """The least value a lower end can take (a runtime end from its value's stated facts)."""
    runtime = _runtime(end)
    if runtime is None:
        return end
    least = stated_facts_of(runtime.value).lower
    return None if least is None else least + runtime.plus


def _numeric_hi(end: _AnyEnd) -> int | None:
    """The largest value an exclusive upper end can take (``x < L + p <= most(L) + p``)."""
    runtime = _runtime(end)
    if runtime is None:
        return end
    most = stated_facts_of(runtime.value).upper
    return None if most is None else most + runtime.plus


def _numeric_last(end: _AnyEnd) -> int | None:
    """The largest value below an exclusive upper end."""
    hi = _numeric_hi(end)
    return None if hi is None else hi - 1


def _shift_end(end: _End, k: int) -> _End:
    if end is None:
        return None
    runtime = _runtime(end)
    return end + k if runtime is None else runtime.shifted(k)


def _mul(end: int | None, k: int) -> int | None:
    return None if end is None else end * k


def _add_ends(a: _End, b: _End, side: str) -> _End:
    """``a + b`` for two ends of one side: a runtime end plus an int stays runtime, else
    numeric."""
    if a is None or b is None:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(b, int):
        return _shift_end(a, b)
    if isinstance(a, int):
        return _shift_end(b, a)
    numeric = _numeric_lo if side == "lo" else _numeric_hi
    na, nb = numeric(a), numeric(b)
    return None if na is None or nb is None else na + nb


def _same_value(a: _AnyEnd, b: _AnyEnd) -> bool:
    """Both ends are runtime ends over one IR value object."""
    ra, rb = _runtime(a), _runtime(b)
    return ra is not None and rb is not None and ra.value is rb.value


def _comparable_key(end: _AnyEnd) -> int:
    runtime = _runtime(end)
    return end if runtime is None else runtime.plus


def _pick(
    a: _AnyEnd, b: _AnyEnd, side: str, *, take_max: bool, both_bound: bool
) -> _AnyEnd:
    """One end of ``min`` / ``max`` of two ranges (the symmetric rule).

    Comparable ends (two ints, or two runtime ends over one value) give the exact min / max.
    Otherwise the end is a number from the numeric ends: when it must bound BOTH operands, the
    looser one (``None`` if either is unbounded); when it may come from EITHER operand, the
    tighter one, and a ``None`` end keeps the other end as it is.
    """
    choose = max if take_max else min
    comparable = (isinstance(a, int) and isinstance(b, int)) or _same_value(a, b)
    if comparable:
        return (
            a
            if choose(_comparable_key(a), _comparable_key(b)) == _comparable_key(a)
            else b
        )
    if both_bound:
        if a is None or b is None:
            return None
        numeric = _numeric_lo if side == "lo" else _numeric_hi
        na, nb = numeric(a), numeric(b)
        return None if na is None or nb is None else choose(na, nb)
    if a is None:
        return b
    if b is None:
        return a
    numeric = _numeric_lo if side == "lo" else _numeric_hi
    known = [n for n in (numeric(a), numeric(b)) if n is not None]
    return choose(known) if known else None


def _intersect_end(a: _AnyEnd, b: _AnyEnd, side: str) -> _AnyEnd:
    """One end of ``intersect``: which operand's end survives (both are sound)."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b) if side == "lo" else min(a, b)
    if _same_value(a, b):
        if side == "lo":
            return a if _comparable_key(a) >= _comparable_key(b) else b
        return a if _comparable_key(a) <= _comparable_key(b) else b
    if isinstance(a, int) or isinstance(b, int):
        runtime, number = (b, a) if isinstance(a, int) else (a, b)
        return number if side == "lo" else runtime
    if side == "lo":
        known = [n for n in (_numeric_lo(a), _numeric_lo(b)) if n is not None]
        return max(known) if known else None
    known = [n for n in (_numeric_hi(a), _numeric_hi(b)) if n is not None]
    return min(known) if known else None


def _pair_always_empty(lo: _AnyEnd, hi: _AnyEnd) -> bool:
    """No integer satisfies ``lo <= x < hi`` for any value of the runtime ends in their envelopes."""
    if lo is None or hi is None:
        return False
    if isinstance(lo, int) and isinstance(hi, int):
        return lo >= hi
    if _same_value(lo, hi):
        return _comparable_key(hi) <= _comparable_key(lo)
    nlo, nhi = _numeric_lo(lo), _numeric_hi(hi)
    return nlo is not None and nhi is not None and nlo >= nhi


def _lo_proven(lo: _AnyEnd, bound: _AnyEnd) -> bool:
    """A range starting at ``lo`` cannot go below ``bound`` (``None``: no lower bound)."""
    if bound is None:
        return True
    if _runtime(bound) is not None:
        return _same_value(lo, bound) and _comparable_key(lo) >= _comparable_key(bound)
    numeric = _numeric_lo(lo)
    return numeric is not None and numeric >= bound


def _hi_proven(hi: _AnyEnd, bound: _AnyEnd) -> bool:
    """A range ending below ``hi`` cannot reach ``bound`` (``None``: no upper bound)."""
    if bound is None:
        return True
    if _runtime(bound) is not None:
        return _same_value(hi, bound) and _comparable_key(hi) <= _comparable_key(bound)
    numeric = _numeric_hi(hi)
    return numeric is not None and numeric <= bound


def _always_at_most(a: _AnyEnd, b: _AnyEnd) -> bool:
    """Bound end ``a`` is ``<=`` bound end ``b`` for every value of their runtime parts."""
    largest_a, least_b = _numeric_hi(a), _numeric_lo(b)
    return largest_a is not None and least_b is not None and largest_a <= least_b


def _tightest(coordinate: str, a: _AnyEnd, b: _AnyEnd, side: str) -> _AnyEnd:
    """The tighter of two failing bound ends on one side, so one compare enforces both.

    ``a`` is ``None`` before the first failing end. Two ints, or two runtime ends over one value,
    are ordered exactly; any other pair only when the envelopes order them for every value.
    """
    if a is None:
        return b
    lower = side == "lo"
    if (isinstance(a, int) and isinstance(b, int)) or _same_value(a, b):
        key_a, key_b = _comparable_key(a), _comparable_key(b)
        a_tighter = key_a >= key_b if lower else key_a <= key_b
        return a if a_tighter else b
    if _always_at_most(b, a) if lower else _always_at_most(a, b):
        return a
    if _always_at_most(a, b) if lower else _always_at_most(b, a):
        return b
    raise ValueError(
        f"coordinate '{coordinate}' must be checked against both {_end_text(a, side)} and "
        f"{_end_text(b, side)} on its {'lower' if lower else 'upper'} side, and which of the "
        "two is tighter is not known at build time -- one check holds one end per side; bound "
        "the coordinate by one of them (an input over a runtime-length axis is declared "
        "inputs(name=None))"
    )


# --- value ranges and required checks --------------------------------------------------------


def _check_input_count(graph: TransformGraph, n: int) -> None:
    names = tuple(graph.inputs)
    if n != len(names):
        raise ValueError(
            f"input_ranges has {n} entries but the graph has {len(names)} inputs {names} -- "
            "give one (lo, hi) pair per input, in inputs(...) order; use (None, None) when a "
            "range is not known"
        )


def _written_ranges(
    graph: TransformGraph, node: Node, reads: tuple[Interval, ...]
) -> tuple[Interval, ...]:
    """The range of each coordinate ``node`` writes, from the ranges of what it reads."""
    params, node_kind = node.params, node.kind
    if node_kind is TransformType.EMBED:
        total = Interval(0, 1)
        for read, coeff in zip(reads, params.coeffs):
            total = total.plus(read.scaled(coeff))
        return (total,)
    if node_kind is TransformType.MERGE:
        total = Interval(0, 1)
        for i, read in enumerate(reads):
            total = total.plus(read.scaled(prod(params.lengths[i + 1 :])))
        return (total,)
    if node_kind is TransformType.UNMERGE:
        lengths = params.lengths
        pieces = [reads[0].floor_div(prod(lengths[1:]))]
        for i in range(1, len(lengths)):
            pieces.append(
                reads[0].floor_div(prod(lengths[i + 1 :])).floor_mod(lengths[i])
            )
        return tuple(pieces)
    if node_kind is TransformType.OFFSET:
        return (reads[0].shifted(params.shift),)
    if node_kind is TransformType.SLICE:
        return (reads[0].shifted(params.begin),)
    if node_kind is TransformType.FREEZE:
        return (Interval(params.value, params.value + 1),)
    if node_kind is TransformType.XOR:
        length = graph.edge(node.write[1]).domain
        # The bits of reads[1] above the mask pass through; the low bits stay in [0, length).
        swizzled = reads[1].floor_div(length).scaled(length).plus(Interval(0, length))
        return (reads[0], swizzled)
    return ()


def _read_domains(graph: TransformGraph) -> dict[str, list[tuple[int, int | None]]]:
    """The domain each step requires of each coordinate it reads (3 of the 8 kinds need one)."""
    required: dict[str, list[tuple[int, int | None]]] = {}
    for node in graph.nodes:
        if node.kind is TransformType.MERGE:
            for read, length in zip(node.read, node.params.lengths):
                required.setdefault(read, []).append((0, length))
        elif node.kind is TransformType.UNMERGE:
            required.setdefault(node.read[0], []).append((0, prod(node.params.lengths)))
        elif node.kind is TransformType.SLICE:
            span = node.params.end - node.params.begin
            required.setdefault(node.read[0], []).append((0, span))
        elif node.kind is TransformType.XOR:
            required.setdefault(node.read[0], []).append((0, None))
            domain = graph.edge(node.write[1]).domain
            required.setdefault(node.read[1], []).append((0, domain))
    return required


def _conditioned_walk(
    graph: TransformGraph,
    ranges: tuple[Interval, ...],
    bounds: tuple[AxisBound, ...] | None,
) -> tuple[tuple[BoundCheck, ...], dict[str, Interval]]:
    """Every coordinate's checks and its conditioned range, in graph order.

    Each coordinate's range is computed from the conditioned ranges of what it reads, checked
    against its constraints in order (its input domain, the domains its readers require, its
    tensor bound), and narrowed by each before the next and before any reader sees it: checks
    are AND-ed, so a check may assume the ones before it passed.
    """
    constraints: dict[str, list[tuple[End, End, CheckReason]]] = {}
    for name, domain in graph.inputs.items():
        if domain is not None:
            constraints.setdefault(name, []).append(
                (0, domain, CheckReason.INPUT_DOMAIN)
            )
    for name, domains in _read_domains(graph).items():
        for lo, hi in domains:
            constraints.setdefault(name, []).append((lo, hi, CheckReason.READ_DOMAIN))
    if bounds is not None:
        for name, bound in zip(graph.outputs, bounds):
            constraints.setdefault(name, []).append(
                (bound.lo, bound.hi, CheckReason.TENSOR_BOUND)
            )

    checks: list[BoundCheck] = []
    conditioned: dict[str, Interval] = {}

    def settle(name: str, computed: Interval) -> None:
        current = computed
        lo_end: _AnyEnd = None
        hi_end: _AnyEnd = None
        reasons: list[CheckReason] = []
        for lo, hi, reason in constraints.get(name, ()):
            lo_fails = not _lo_proven(current.lo, lo)
            hi_fails = not _hi_proven(current.hi, hi)
            if lo_fails:
                lo_end = _tightest(name, lo_end, lo, "lo")
            if hi_fails:
                hi_end = _tightest(name, hi_end, hi, "hi")
            if (lo_fails or hi_fails) and reason not in reasons:
                reasons.append(reason)
            region = Interval(lo, hi)
            if current.overlaps(region):
                current = current.intersect(region)
        if lo_end is not None or hi_end is not None:
            checks.append(
                BoundCheck(
                    name,
                    _door_end(lo_end, "lo"),
                    _door_end(hi_end, "hi"),
                    tuple(reasons),
                )
            )
        conditioned[name] = current

    for name, given in zip(graph.inputs, ranges):
        settle(name, given)
    for node in graph.nodes:
        reads = tuple(conditioned[r] for r in node.read)
        for name, computed in zip(node.write, _written_ranges(graph, node, reads)):
            settle(name, computed)
    return tuple(checks), conditioned


def _value_ranges(
    graph: TransformGraph, ranges: tuple[Interval, ...]
) -> dict[str, Interval]:
    """Every coordinate's unconditioned range (what every lane computes, valid or not)."""
    _check_input_count(graph, len(ranges))
    found = dict(zip(graph.inputs, ranges))
    for node in graph.nodes:
        reads = tuple(found[r] for r in node.read)
        found.update(zip(node.write, _written_ranges(graph, node, reads)))
    return found


def _required_checks(
    graph: TransformGraph, ranges: tuple[Interval, ...], bounds: tuple[AxisBound, ...]
) -> tuple[BoundCheck, ...]:
    """The checks a graph needs for ``ranges`` (one per input) against ``bounds`` (one per
    output, in tensor-axis order); one ``BoundCheck`` per coordinate that can fail."""
    _check_input_count(graph, len(ranges))
    if len(bounds) != len(graph.outputs):
        raise ValueError(
            f"bounds has {len(bounds)} entries but the tensor has rank {len(graph.outputs)} -- "
            "give one AxisBound per tensor axis"
        )
    for axis, bound in enumerate(bounds):
        if (
            isinstance(bound.lo, int)
            and isinstance(bound.hi, int)
            and bound.lo >= bound.hi
        ):
            raise ValueError(
                f"bound of axis {axis} is empty: [{bound.lo}, {bound.hi}) -- an always-empty "
                "window cannot be read"
            )
    checks, _ = _conditioned_walk(graph, ranges, bounds)
    return checks


def _intervals(
    graph: TransformGraph, input_ranges: tuple[tuple[End, End], ...]
) -> tuple[Interval, ...]:
    _check_input_count(graph, len(input_ranges))
    return tuple(Interval.from_pair(pair) for pair in input_ranges)


def value_ranges(
    graph: TransformGraph, input_ranges: tuple[tuple[End, End], ...]
) -> dict[str, tuple[End, End]]:
    """The ``(lo, hi)`` range every coordinate of ``graph`` can take, keyed by its name.

    ``input_ranges`` gives one half-open pair per graph input, in ``inputs(...)`` order. The
    ranges are unconditioned: an out-of-domain input drives what it feeds out of range too (an
    ``unmerge`` read past its domain drives the outer piece past its length), because every lane
    computes its address whether or not it is valid. An end at a runtime ``Value`` is reported
    as that ``Value``; an end shifted from one by a constant (``x + 1`` for ``x`` in ``[0, T)``)
    is reported as the number the value's stated facts give on that side.
    """
    ranges = _value_ranges(graph, _intervals(graph, input_ranges))
    return {name: interval.as_pair() for name, interval in ranges.items()}


def required_checks(
    graph: TransformGraph, tensor: TensorDesc, input_ranges: tuple[tuple[End, End], ...]
) -> tuple[BoundCheck, ...]:
    """The run-time checks ``graph`` bound to ``tensor`` needs for ``input_ranges``.

    One ``BoundCheck`` per coordinate that can leave its input domain, a domain a step requires
    of it, or (for an output) the tensor's ``[0, length)``; a side the ranges prove is ``None``.
    ``input_ranges`` gives one half-open pair per graph input, in ``inputs(...)`` order.
    """
    _require_outputs_match_rank(graph, len(tensor.lengths))
    bounds = tuple(AxisBound.of(0, length) for length in tensor.lengths)
    return _required_checks(graph, _intervals(graph, input_ranges), bounds)


def graph_injective(
    graph: TransformGraph, input_ranges: tuple[tuple[End, End], ...]
) -> bool:
    """Whether distinct in-domain inputs always reach distinct outputs (store safety).

    Decided at build time, conservatively: ``True`` only when every input can be recovered from
    the outputs, walking back through steps that lose no information on their in-domain read
    ranges. A ``False`` is either a real collision (an ``embed`` whose coefficients make two
    inputs meet, a ``broadcast`` that drops a coordinate) or one the ranges cannot rule out.
    """
    ranges = _intervals(graph, input_ranges)
    _, conditioned = _conditioned_walk(graph, ranges, None)

    known = {name for name, r in conditioned.items() if _size(r) == 1} | set(
        graph.outputs
    )
    pending = list(graph.nodes)
    progress = True
    while progress:
        progress = False
        for node in list(pending):
            if all(w in known for w in node.write) and _step_injective(
                node, conditioned
            ):
                known.update(node.read)
                pending.remove(node)
                progress = True
    return all(name in known for name in graph.inputs)


def _size(r: Interval) -> int | None:
    """How many values a range holds, or ``None`` when that is not bounded."""
    lo, last = _numeric_lo(r.lo), _numeric_last(r.hi)
    return None if lo is None or last is None else last - lo + 1


def _step_injective(node: Node, conditioned: dict[str, Interval]) -> bool:
    """Whether a step's reads are recoverable from its writes, on their in-domain ranges."""
    sizes = [_size(conditioned[r]) for r in node.read]
    if node.kind is TransformType.BROADCAST:
        return all(size == 1 for size in sizes)
    if node.kind is not TransformType.EMBED:
        # merge, unmerge, offset, slice_window and xor are bijections on their read domains;
        # freeze reads nothing.
        return True
    # An embed is injective when, ordered by coefficient, each coefficient exceeds the widest
    # span the smaller ones can cover together (a mixed radix); a read holding one value is fixed.
    varying = sorted(
        (
            (coeff, size)
            for coeff, size in zip(node.params.coeffs, sizes)
            if size is None or size > 1
        ),
        key=lambda pair: pair[0],
    )
    covered = 0
    for i, (coeff, size) in enumerate(varying):
        if coeff <= covered or coeff == 0:
            return False
        if size is None:
            return i == len(varying) - 1
        covered += coeff * (size - 1)
    return True
