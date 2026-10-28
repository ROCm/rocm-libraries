# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The IR range walk: what the IR proves about one plain i32 value, and how it splits.

Kernel authors pass plain IR values (or ints) as coordinates. Splitting a value into its
per-wave and per-lane parts, and proving each part's range, is this module's job: it reads each
``Value``'s defining op and emits nothing.

- ``value_facts(b, value)`` -- the range (``Interval``) and residue (``Congruence``) of the whole
  value, derived from how it was computed.
- ``split_coordinate(b, value)`` -- the value as an ``AffineCoord``: a constant, wave-uniform parts
  and one per-lane part, each with its ``value_facts``.
- ``runtime_value_of(value)`` -- the ``RuntimeValue`` an envelope marker states; the one
  constructor of a ``RuntimeValue`` from IR.
- ``leaf_varying_within_workgroup(b, value)`` -- the first leaf that may differ between the waves
  of a workgroup, or ``None``.

The walk owns no arithmetic: every rule maps an op onto a range operation of ``ranges.py``. Its
rule set is closed and unknown is a fact, never an abort: a leaf or op outside the set (or a rule
whose precondition fails) contributes ``UNBOUNDED`` with ``Congruence(1, 0)``, and the rows
combine as usual, so ``x * 128`` is a multiple of 128 whatever ``x`` was computed from. Interval
arithmetic assumes no i32 wrap, exactly as the ``nsw`` flags the lowering emits let LLVM assume.

A value is wave-uniform only when LLVM's uniformity analysis is known to prove it: block ids,
kernel parameters, ``readfirstlane`` results, the counter of a loop with uniform bounds, a range
marker on a uniform value, and pure arithmetic on those. A load, an intrinsic, a thread or lane id
is per-lane whatever its operands; a per-wave value loaded from memory becomes uniform by passing
it through ``b.readfirstlane``. A value the walk cannot see is uniform is per-lane: correct, at
worst slower.

Spelling rule: add per-lane terms together before adding the per-wave origin, and scale a sum
term by term (``origin * 2 + lane * 2``, not ``(origin + lane) * 2``); write ``x // 2^c`` of a
non-negative ``x`` as ``div``, not a right shift, and ``x | c`` as ``add`` when the bits are known
disjoint. Any spelling gives a correct value; only this one keeps the per-wave part on the scalar
unit and the ranges proven.

The first walk that reaches a thread id reads ``b.kernel.attrs["max_workgroup_size"]``, which the
author must set explicitly, and records it in ``b.kernel.attrs["thread_ranges_workgroup_size"]``;
every later walk refuses a kernel whose size has changed since.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rocke.core.ir import I32, Op, Value

from .ranges import (
    UNBOUNDED,
    AffineCoord,
    Congruence,
    Interval,
    RuntimeValue,
    UniformPart,
)

if TYPE_CHECKING:
    from rocke.core.ir import IRBuilder

__all__ = [
    "WORKGROUP_SIZE_RECORD",
    "leaf_varying_within_workgroup",
    "runtime_value_of",
    "split_coordinate",
    "value_facts",
]

# The kernel attribute the walk records the workgroup size it derived thread-id ranges with.
WORKGROUP_SIZE_RECORD = "thread_ranges_workgroup_size"

Facts = tuple[Interval, Congruence]

_UNKNOWN: Facts = (UNBOUNDED, Congruence(1, 0))

# Side-effect-free arithmetic: an op of this list with at least one operand, all of them uniform
# (or the same for the whole workgroup), is itself uniform (or the same for the workgroup).
_PURE_ARITH_OPS = frozenset(
    {
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.div",
        "arith.mod",
        "arith.shl",
        "arith.and",
        "arith.smin",
        "arith.smax",
        "arith.zext",
        "arith.sext",
        "arith.trunc",
        "arith.bitcast",
        "arith.cmp",
        "arith.select",
        "arith.or",
        "arith.xor",
        "arith.lshr",
    }
)

# Scalar global loads, operands [ptr, idx]: the only loads the workgroup rule can accept.
_SCALAR_GLOBAL_LOADS = frozenset({"memref.global_load", "memref.global_load_typed"})


# --- public sibling surface -------------------------------------------------------------------


def value_facts(b: IRBuilder, value: Value | int) -> Facts:
    """The range and residue of ``value``, derived from how it was computed (emits nothing)."""
    _require_coordinate(value)
    _require_unchanged_workgroup_size(b)
    return _FactWalk(b, value).facts(value)


def split_coordinate(b: IRBuilder, value: Value | int) -> AffineCoord:
    """``value`` as a constant, wave-uniform parts and a per-lane part (emits nothing).

    Rules, top-down, the first match wins: an int or a constant is the constant part; ``x + c``,
    ``c + x`` and ``x - c`` (``c`` exact) move ``c`` into the constant and split ``x``; a uniform
    value is one uniform part; an ``add`` that is not wholly uniform splits both operands and adds
    the splits, unless both carry a per-lane part, in which case the add itself is the per-lane
    part; anything else is the per-lane part. Each part's range and residue are its
    ``value_facts``.
    """
    _require_coordinate(value)
    _require_unchanged_workgroup_size(b)
    return _Splitter(b, _FactWalk(b, value)).split(value)


def runtime_value_of(value: Value | int) -> RuntimeValue | None:
    """The ``RuntimeValue`` an envelope marker states, or ``None`` when ``value`` is not one's
    result.

    An envelope marker is an ``assume_range`` marker with two int ends. The object is built from
    the marker's STATED facts -- exactly what its producer put in the object it returned -- never
    from what the walk derives, so it is looser than the derived range, never wrong, and shares
    the ``Value`` an identity proof keys on.
    """
    op = _defining_op(value)
    if (
        op is None
        or op.name != "arith.assume_range"
        or len(op.operands) != 1
        or "lo" not in op.attrs
        or "hi" not in op.attrs
    ):
        return None
    return RuntimeValue(
        value,
        0,
        Interval(op.attrs["lo"], op.attrs["hi"]),
        Congruence(op.attrs.get("multiple_of", 1), op.attrs.get("residue", 0)),
    )


def leaf_varying_within_workgroup(b: IRBuilder, value: Value | int) -> Value | None:
    """``None`` when ``value`` is the same in every wave of the workgroup; else the first leaf
    (depth first, in operand order) that may differ.

    The same for the whole workgroup: ints and constants, block ids, kernel parameters, a range
    marker, a ``readfirstlane`` or a pure arithmetic op whose operands all are, the counter of a
    loop whose bounds and step are, and a read-only load -- a scalar global load at such an index
    through a kernel parameter declared ``readonly=True`` and ``noalias=True`` (or with
    ``addr_space="constant"``), so every wave reads memory no pointer of the kernel writes. Wave-
    uniform is not enough: ``readfirstlane(thread_id_x / 64)`` differs between waves. Anything
    else may differ (a refusal, never a wrong answer); a load that is not read-only is itself the
    leaf returned.
    """
    _require_coordinate(value)
    return _first_varying_leaf(b, value, {})


# --- shared checks ----------------------------------------------------------------------------


def _require_coordinate(value: object) -> None:
    if isinstance(value, Value) and value.type != I32:
        raise TypeError(
            f"IR range walk: {value} has type {value.type} -- a coordinate is an i32 value"
        )


def _require_unchanged_workgroup_size(b: IRBuilder) -> None:
    attrs = b.kernel.attrs
    if WORKGROUP_SIZE_RECORD not in attrs:
        return
    old, new = attrs[WORKGROUP_SIZE_RECORD], attrs.get("max_workgroup_size")
    if new != old:
        raise ValueError(
            f"IR range walk: max_workgroup_size is now {new} but thread-id ranges were already "
            f"derived with {old} -- set it once, before positioning windows from thread ids"
        )


def _defining_op(value: Value | int) -> Op | None:
    return value.op if isinstance(value, Value) else None


def _is_param(b: IRBuilder, value: Value) -> bool:
    return value.op is None and any(
        b.get_param(p.name) is value for p in b.kernel.params
    )


def _loop_of_counter(value: Value) -> Op | None:
    """The ``scf.for`` whose induction variable ``value`` is (not an ``iter_args`` value or a
    loop result), or ``None``."""
    op = value.op
    if op is None or op.name != "scf.for" or value.name != op.attrs.get("iv"):
        return None
    if any(r is value for r in op.results):
        return None
    if any(arg["name"] == value.name for arg in op.attrs.get("iter_args", ())):
        return None
    return op


def _is_exact(r: Interval) -> bool:
    """A width-one int range: the value is known exactly."""
    return isinstance(r.lo, int) and isinstance(r.hi, int) and r.hi - r.lo == 1


def _constant_facts(c: int) -> Facts:
    return Interval(c, c + 1), Congruence(abs(c) or 1, 0)


def _low_mask_bits(r: Interval) -> int | None:
    """``c`` when the exact range ``r`` is the mask ``2^c - 1`` with ``c`` in ``[0, 31]``."""
    if not _is_exact(r) or r.lo < 0:
        return None
    width = (r.lo + 1).bit_length() - 1
    return width if r.lo + 1 == 1 << width and width <= 31 else None


# --- value_facts --------------------------------------------------------------------------------


class _FactWalk:
    """One walk: the range rules over one builder, each value's facts derived once."""

    def __init__(self, b: IRBuilder, root: Value | int) -> None:
        self._b = b
        self._root = root
        self._memo: dict[int, Facts] = {}

    def facts(self, value: Value | int) -> Facts:
        if isinstance(value, int):
            return _constant_facts(value)
        key = id(value)
        if key not in self._memo:
            self._memo[key] = self._derive(value)
        return self._memo[key]

    def exact(self, value: Value | int) -> int | None:
        """The int ``value`` is known to be, or ``None``."""
        r, _ = self.facts(value)
        return r.lo if _is_exact(r) else None

    def runtime_end(self, marker: Op) -> RuntimeValue:
        """A marker's runtime end ``hi_value + hi_plus``, rebuilt from ``hi_value``'s marker."""
        hi_value = marker.operands[1]
        end = runtime_value_of(hi_value)
        if end is None:
            raise ValueError(
                f"IR range walk: the marker on {marker.operands[0]} states an end over "
                f"{hi_value}, which is not a runtime value's marker result -- a runtime end is "
                "the .value of a RuntimeValue (from runtime_value)"
            )
        return end._shifted(marker.attrs.get("hi_plus", 0))

    def _derive(self, value: Value) -> Facts:
        op = value.op
        if op is None:
            return _UNKNOWN  # a kernel parameter or another value with no defining op
        if op.name == "scf.for":
            return self._loop_counter(value)
        rule = _RANGE_RULES.get(op.name)
        return _UNKNOWN if rule is None else rule(self, op)

    # -- leaves ----------------------------------------------------------------------------

    def _constant(self, op: Op) -> Facts:
        c = op.attrs.get("value")
        if not isinstance(c, int) or isinstance(c, bool):
            return _UNKNOWN
        return _constant_facts(c)

    def _thread_id(self, op: Op) -> Facts:
        attrs = self._b.kernel.attrs
        size = attrs.get("max_workgroup_size")
        if size is None:
            raise ValueError(
                f"IR range walk: {self._root} depends on gpu.thread_id but the kernel states no "
                'max_workgroup_size -- set b.kernel.attrs["max_workgroup_size"] to the launch '
                "block size right after creating the builder"
            )
        attrs.setdefault(WORKGROUP_SIZE_RECORD, size)
        return Interval(0, int(size)), Congruence(1, 0)

    def _block_id(self, op: Op) -> Facts:
        return Interval(0, None), Congruence(1, 0)

    def _readfirstlane(self, op: Op) -> Facts:
        return self.facts(op.operands[0])

    def _loop_counter(self, value: Value) -> Facts:
        """The induction variable of ``scf.for`` with an exact start and an exact step > 0."""
        loop = _loop_of_counter(value)
        if loop is None:
            return _UNKNOWN  # an iter_args value or a loop result
        start, stop, step = loop.operands[:3]
        lo, st = self.exact(start), self.exact(step)
        if lo is None or st is None or st <= 0:
            return _UNKNOWN
        congruence = Congruence(st, lo)
        never_runs = Interval(lo, lo + 1)
        u = self.exact(stop)
        if u is not None:  # (a) an exact stop: the tight end of the last trip
            if u <= lo:
                return never_runs, congruence
            return Interval(lo, u).snapped(congruence), congruence
        runtime_stop = self._runtime_stop(stop)
        if (
            runtime_stop is not None
        ):  # (b) a runtime length plus constants: proven by identity
            return Interval(lo, runtime_stop), congruence
        h = self.facts(stop)[0].hi  # (c) any other stop: its own upper end
        if h is None:
            return Interval(lo, None), congruence
        if isinstance(h, int):
            return (Interval(lo, h) if h > lo else never_runs), congruence
        counter = Interval(lo, h)  # a runtime end: never runs when its envelope says so
        return (never_runs if counter.within(None, lo + 1) else counter), congruence

    def _runtime_stop(self, stop: Value) -> RuntimeValue | None:
        """``runtime_value_of(v).shifted(c)`` when ``stop`` is an envelope-marker result ``v``
        with exact constants added or subtracted (summed into ``c``), else ``None``."""
        c = 0
        while (base := runtime_value_of(stop)) is None:
            op = _defining_op(stop)
            if (
                op is None
                or op.name not in ("arith.add", "arith.sub")
                or op.attrs.get("wrap")
            ):
                return None
            x, y = op.operands
            ky, kx = self.exact(y), self.exact(x)
            if ky is not None:
                c, stop = c + (ky if op.name == "arith.add" else -ky), x
            elif kx is not None and op.name == "arith.add":
                c, stop = c + kx, y
            else:
                return None
        return base._shifted(c)

    def _marker(self, op: Op) -> Facts:
        """An ``assume_range`` marker: the operand's facts narrowed by the stated ones."""
        x_range, x_congruence = self.facts(op.operands[0])
        attrs = op.attrs
        hi = self.runtime_end(op) if len(op.operands) == 2 else attrs.get("hi")
        stated = Interval(attrs.get("lo"), hi)
        multiple_of, residue = attrs.get("multiple_of", 1), attrs.get("residue", 0)
        congruence = (
            Congruence(multiple_of, residue) if multiple_of > 1 else x_congruence
        )
        return x_range.intersect(stated).snapped(congruence), congruence

    # -- arithmetic --------------------------------------------------------------------------

    def _add(self, op: Op) -> Facts:
        if op.attrs.get("wrap"):
            # ``add nuw`` may pass the signed maximum: the no-signed-wrap sum rule does not hold.
            return _UNKNOWN
        return self._sum(op, sign=1)

    def _sub(self, op: Op) -> Facts:
        return self._sum(op, sign=-1)

    def _sum(self, op: Op, *, sign: int) -> Facts:
        x, y = op.operands
        (rx, cx), (ry, cy) = self.facts(x), self.facts(y)
        if _is_exact(ry):  # x + c, x - c
            return rx.shifted(sign * ry.lo), cx.shifted(sign * ry.lo)
        if _is_exact(rx):  # c + x, c - x
            if sign == 1:
                return ry.shifted(rx.lo), cy.shifted(rx.lo)
            return ry.scaled(-1).shifted(rx.lo), cy.scaled(-1).shifted(rx.lo)
        if sign == -1:
            ry, cy = ry.scaled(-1), cy.scaled(-1)
        return rx.plus(ry), cx.plus(cy)

    def _mul(self, op: Op) -> Facts:
        (rx, cx), (ry, cy) = (self.facts(v) for v in op.operands)
        if not _is_exact(ry):
            (rx, cx), (ry, cy) = (ry, cy), (rx, cx)
        if not _is_exact(ry):
            return _UNKNOWN
        c = ry.lo
        if c == 0:
            return _constant_facts(0)
        return rx.scaled(c), cx.scaled(c)

    def _div_mod(self, op: Op) -> Facts:
        (rx, cx), (ry, _) = (self.facts(v) for v in op.operands)
        if not _is_exact(ry) or ry.lo <= 0 or not rx.within(0, None):
            return _UNKNOWN  # IR div / mod truncate: floor only for a dividend >= 0
        c = ry.lo
        if op.name == "arith.div":
            return rx.floor_div(c), cx.floor_div(c)
        return rx.floor_mod(c), cx.floor_mod(c)

    def _shl(self, op: Op) -> Facts:
        """``x << c``: LLVM ``shl`` without ``nsw`` wraps, so the scaled range holds only when it
        is proven inside i32; otherwise the operand is unknown first, and the low ``c`` zero bits
        still give ``(2^c, 0)`` through any wrap."""
        (rx, cx), (rc, _) = (self.facts(v) for v in op.operands)
        if not _is_exact(rc) or not 0 <= rc.lo <= 31:
            return _UNKNOWN  # a shift amount of 32 or more is poison
        k = 1 << rc.lo
        scaled = rx.scaled(k)
        if scaled.within(-(1 << 31), 1 << 31):
            return scaled, cx.scaled(k)
        return UNBOUNDED.scaled(k), Congruence(1, 0).scaled(k)

    def _and(self, op: Op) -> Facts:
        """``x & (2^c - 1)`` is the floor ``x mod 2^c`` for every i32 ``x``, negative included."""
        (rx, cx), (ry, cy) = (self.facts(v) for v in op.operands)
        bits = _low_mask_bits(ry)
        if bits is None:
            (rx, cx), bits = (ry, cy), _low_mask_bits(rx)
        if bits is None:
            return _UNKNOWN
        return rx.floor_mod(1 << bits), cx.floor_mod(1 << bits)

    def _min_max(self, op: Op) -> Facts:
        (rx, cx), (ry, cy) = (self.facts(v) for v in op.operands)
        r = rx.minimum(ry) if op.name == "arith.smin" else rx.maximum(ry)
        if _is_exact(r):
            return _constant_facts(r.lo)
        if _is_exact(rx):
            return r, cy.join(rx.lo)
        return r, cx.join(ry.lo if _is_exact(ry) else cy)


_RANGE_RULES = {
    "arith.constant": _FactWalk._constant,
    "gpu.thread_id": _FactWalk._thread_id,
    "gpu.block_id": _FactWalk._block_id,
    "tile.readfirstlane": _FactWalk._readfirstlane,
    "arith.assume_range": _FactWalk._marker,
    "arith.add": _FactWalk._add,
    "arith.sub": _FactWalk._sub,
    "arith.mul": _FactWalk._mul,
    "arith.div": _FactWalk._div_mod,
    "arith.mod": _FactWalk._div_mod,
    "arith.shl": _FactWalk._shl,
    "arith.and": _FactWalk._and,
    "arith.smin": _FactWalk._min_max,
    "arith.smax": _FactWalk._min_max,
}


# --- split_coordinate -------------------------------------------------------------------------


class _Splitter:
    """The split rules over one walk; each part's facts come from that walk."""

    def __init__(self, b: IRBuilder, walk: _FactWalk) -> None:
        self._b = b
        self._walk = walk
        self._uniform: dict[int, bool] = {}

    def split(self, value: Value | int) -> AffineCoord:
        c = value if isinstance(value, int) else self._constant_of(value)
        if c is not None:
            return AffineCoord.of_int(c)
        op = value.op
        if (
            op is not None
            and op.name in ("arith.add", "arith.sub")
            and not op.attrs.get("wrap")
        ):
            x, y = op.operands
            ky = self._walk.exact(y)
            if ky is not None:  # x + c, x - c
                moved = ky if op.name == "arith.add" else -ky
                return self.split(x) + AffineCoord.of_int(moved)
            kx = self._walk.exact(x)
            if kx is not None and op.name == "arith.add":  # c + x
                return self.split(y) + AffineCoord.of_int(kx)
        if self.is_uniform(value):
            r, congruence = self._walk.facts(value)
            return AffineCoord(0, (UniformPart(value, r, congruence),))
        if op is not None and op.name == "arith.add" and not op.attrs.get("wrap"):
            left, right = (self.split(v) for v in op.operands)
            if left.lane is None or right.lane is None:
                return left + right
        return self._lane(value)

    def _constant_of(self, value: Value) -> int | None:
        """The int of an ``arith.constant`` (the constant row of the split)."""
        if value.op is None or value.op.name != "arith.constant":
            return None
        return self._walk.exact(value)

    def _lane(self, value: Value) -> AffineCoord:
        r, congruence = self._walk.facts(value)
        return AffineCoord(0, (), value, r, congruence)

    def is_uniform(self, value: Value | int) -> bool:
        """Wave-uniform by the leaf list or as pure arithmetic on uniform values."""
        if isinstance(value, int):
            return True
        key = id(value)
        if key not in self._uniform:
            self._uniform[key] = self._derive_uniform(value)
        return self._uniform[key]

    def _derive_uniform(self, value: Value) -> bool:
        op = value.op
        if op is None:
            return _is_param(self._b, value)
        if op.name in ("arith.constant", "gpu.block_id", "tile.readfirstlane"):
            return True
        if op.name == "scf.for":
            loop = _loop_of_counter(value)
            return loop is not None and all(
                self.is_uniform(v) for v in loop.operands[:3]
            )
        if op.name == "arith.assume_range":
            return self.is_uniform(op.operands[0])
        if op.name in _PURE_ARITH_OPS:
            return bool(op.operands) and all(self.is_uniform(v) for v in op.operands)
        return False


# --- leaf_varying_within_workgroup --------------------------------------------------------------


def _first_varying_leaf(
    b: IRBuilder, value: Value | int, seen: dict[int, Value | None]
) -> Value | None:
    if isinstance(value, int):
        return None
    key = id(value)
    if key not in seen:
        seen[key] = _varying_leaf_of(b, value, seen)
    return seen[key]


def _first_of(b: IRBuilder, values, seen: dict[int, Value | None]) -> Value | None:
    for v in values:
        leaf = _first_varying_leaf(b, v, seen)
        if leaf is not None:
            return leaf
    return None


def _varying_leaf_of(
    b: IRBuilder, value: Value, seen: dict[int, Value | None]
) -> Value | None:
    op = value.op
    if op is None:
        return None if _is_param(b, value) else value
    if op.name in ("arith.constant", "gpu.block_id"):
        return None
    if op.name == "scf.for":
        loop = _loop_of_counter(value)
        return value if loop is None else _first_of(b, loop.operands[:3], seen)
    if op.name in ("arith.assume_range", "tile.readfirstlane") or (
        op.name in _PURE_ARITH_OPS and op.operands
    ):
        return _first_of(b, op.operands, seen)
    if op.name in _SCALAR_GLOBAL_LOADS and len(op.operands) == 2:
        ptr, idx = op.operands
        if _is_read_only_param(b, ptr) and _first_varying_leaf(b, idx, seen) is None:
            return None
    return value


def _is_read_only_param(b: IRBuilder, ptr: Value) -> bool:
    """A kernel parameter no pointer of the kernel writes through: ``readonly`` and ``noalias``,
    or constant memory."""
    for param in b.kernel.params:
        if b.get_param(param.name) is ptr:
            attrs = param.attrs
            return attrs.get("addr_space") == "constant" or bool(
                attrs.get("readonly") and attrs.get("noalias")
            )
    return False
