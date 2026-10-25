# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for value ranges and required bound checks (no GPU).

The range arithmetic is checked against exhaustive enumeration of the values each operand can
take; the check decisions against exhaustive enumeration of the inputs and of what the graph
computes from them, so "no check required" can never hide an out-of-domain value.
"""

from __future__ import annotations

import itertools
import random
import re
import subprocess
import sys
from dataclasses import fields

import pytest

from rocke.core.ir import F16, I32, Value
from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    BoundCheck,
    CheckReason,
    RuntimeValue,
    embed,
    graph_injective,
    inputs,
    make_transform_graph,
    merge,
    offset,
    outputs,
    required_checks,
    slice_window,
    unmerge,
    value_ranges,
    xor,
)
from rocke.helpers.tiling.tensor_transforms.graph import TransformType, identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    UNBOUNDED,
    AxisBound,
    CheckSide,
    Congruence,
    Interval,
    VectorRun,
    _required_checks,
    _value_ranges,
)


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _runtime(
    name: str,
    at_most: int | None,
    *,
    at_least: int | None = 0,
    multiple_of: int = 1,
    residue: int = 0,
) -> RuntimeValue:
    """A runtime length as its producer states it: ``at_least <= value <= at_most``."""
    hi = None if at_most is None else at_most + 1
    return RuntimeValue(
        Value(name, I32), 0, Interval(at_least, hi), Congruence(multiple_of, residue)
    )


def _tensor(*lengths: int) -> TensorDesc:
    strides = []
    stride = 1
    for length in reversed(lengths):
        strides.append(stride)
        stride *= length
    return TensorDesc(tuple(lengths), tuple(reversed(strides)), F16)


# --- Interval operations against exhaustive enumeration (int ends) -----------------------------

_ENDS = range(-5, 6)
_INTERVALS = [Interval(a, b) for a in _ENDS for b in _ENDS if a < b]
_SMALL_INTERVALS = [Interval(a, b) for a in range(-4, 4) for b in range(-3, 5) if a < b]


def _values(r: Interval) -> range:
    return range(r.lo, r.hi)


def _assert_exact(result: Interval, reached: set[int]) -> None:
    """The result holds every reached value and both of its ends are reached."""
    assert min(reached) == result.lo
    assert max(reached) == result.hi - 1


def _assert_holds(result: Interval, reached: set[int]) -> None:
    assert all(result.lo <= v < result.hi for v in reached)


def test_interval_plus_is_exact_for_every_pair():
    for a, b in itertools.product(_SMALL_INTERVALS, repeat=2):
        _assert_exact(a.plus(b), {x + y for x in _values(a) for y in _values(b)})


@pytest.mark.parametrize("k", range(-4, 5))
def test_interval_shifted_and_scaled_are_exact(k):
    for a in _INTERVALS:
        _assert_exact(a.shifted(k), {x + k for x in _values(a)})
        _assert_exact(a.scaled(k), {x * k for x in _values(a)})


def test_interval_scaled_by_a_negative_constant_swaps_the_ends():
    assert Interval(2, 5).scaled(-3) == Interval(-12, -5)


@pytest.mark.parametrize("k", range(1, 6))
def test_interval_floor_div_and_floor_mod_use_floor_semantics(k):
    for a in _INTERVALS:
        _assert_exact(a.floor_div(k), {x // k for x in _values(a)})
        _assert_holds(a.floor_mod(k), {x % k for x in _values(a)})
    # Inside one period floor_mod is exact, negative operands included.
    assert Interval(-7, -5).floor_mod(4) == Interval(1, 3)
    assert Interval(-7, -5).floor_div(4) == Interval(-2, -1)


def test_interval_floor_mod_ends_are_reached():
    for k in range(1, 6):
        for a in _INTERVALS:
            _assert_exact(a.floor_mod(k), {x % k for x in _values(a)})


def test_interval_minimum_and_maximum_are_exact_and_symmetric():
    for a, b in itertools.product(_SMALL_INTERVALS, repeat=2):
        _assert_exact(a.minimum(b), {min(x, y) for x in _values(a) for y in _values(b)})
        _assert_exact(a.maximum(b), {max(x, y) for x in _values(a) for y in _values(b)})
        assert a.minimum(b) == b.minimum(a)
        assert a.maximum(b) == b.maximum(a)


def test_interval_intersect_is_exact_and_raises_exactly_when_disjoint():
    for a, b in itertools.product(_SMALL_INTERVALS, repeat=2):
        common = set(_values(a)) & set(_values(b))
        assert a.overlaps(b) == bool(common)
        if common:
            _assert_exact(a.intersect(b), common)
            assert a.intersect(b) == b.intersect(a)
        else:
            with pytest.raises(ValueError, match="have no value in common"):
                a.intersect(b)


@pytest.mark.parametrize("modulus", range(1, 6))
def test_interval_snapped_holds_the_same_members(modulus):
    for residue in range(modulus):
        congruence = Congruence(modulus, residue)
        for a in _INTERVALS:
            members = {x for x in _values(a) if x % modulus == residue}
            if members:
                _assert_exact(a.snapped(congruence), members)
            else:
                with pytest.raises(ValueError, match="has no value congruent"):
                    a.snapped(congruence)


def test_interval_snapped_examples():
    assert Interval(0, 1000).snapped(Congruence(128, 0)) == Interval(0, 897)
    with _raises(
        ValueError,
        "Interval [1, 5) has no value congruent to 0 mod 8 -- the stated range and congruence "
        "contradict each other",
    ):
        Interval(1, 5).snapped(Congruence(8, 0))


def test_unbounded_ends_stay_unbounded():
    half = Interval(0, None)
    assert half.plus(Interval(1, 3)) == Interval(1, None)
    assert half.shifted(-4) == Interval(-4, None)
    assert half.scaled(3) == Interval(0, None)
    assert half.scaled(-3) == Interval(None, 1)
    assert half.floor_div(4) == Interval(0, None)
    assert half.floor_mod(4) == Interval(0, 4)
    assert UNBOUNDED.minimum(Interval(0, 8)) == Interval(None, 8)
    assert UNBOUNDED.maximum(Interval(0, 8)) == Interval(0, None)
    assert UNBOUNDED.intersect(Interval(0, 8)) == Interval(0, 8)


# --- Congruence operations against exhaustive enumeration --------------------------------------

_CONGRUENCES = [Congruence(m, r) for m in range(1, 7) for r in range(m)]
_SAMPLE = range(-30, 30)


def _members(c: Congruence) -> list[int]:
    return [x for x in _SAMPLE if x % c.modulus == c.residue]


def _assert_congruent(c: Congruence, values) -> None:
    assert all(v % c.modulus == c.residue for v in values)


def test_congruence_operations_hold_for_every_member():
    for c in _CONGRUENCES:
        xs = _members(c)
        for k in range(-4, 5):
            _assert_congruent(c.shifted(k), [x + k for x in xs])
            if k:
                _assert_congruent(c.scaled(k), [x * k for x in xs])
        for k in range(1, 7):
            _assert_congruent(c.floor_div(k), [x // k for x in xs])
            _assert_congruent(c.floor_mod(k), [x % k for x in xs])
        for d in _CONGRUENCES:
            ys = _members(d)
            _assert_congruent(c.plus(d), [x + y for x in xs for y in ys])
            _assert_congruent(c.join(d), [min(x, y) for x in xs for y in ys])
            _assert_congruent(c.join(d), [max(x, y) for x in xs for y in ys])
        for cap in range(-8, 9):
            _assert_congruent(c.join(cap), [min(x, cap) for x in xs])


def test_congruence_examples():
    assert Congruence(128, 0).floor_div(4) == Congruence(32, 0)
    assert Congruence(6, 4).floor_div(4) == Congruence(1, 0)
    assert Congruence(128, 0).join(256) == Congruence(128, 0)
    assert Congruence(128, 0).join(100) == Congruence(4, 0)
    assert Congruence(128, 0).join(Congruence(64, 32)) == Congruence(32, 0)
    assert Congruence(4, 7).residue == 3


# --- RuntimeValue ends -------------------------------------------------------------------------


def test_runtime_end_survives_shift_and_unit_scale_only():
    ms = _runtime("Ms", 4095, multiple_of=8)
    ks = _runtime("Ks", 4095)
    r = Interval(0, ms)
    assert r.shifted(-3) == Interval(-3, ms._shifted(-3))
    assert r.plus(Interval(0, 4)) == Interval(0, ms._shifted(3))
    assert r.scaled(1) == r
    assert r.scaled(2) == Interval(0, 8189)
    assert r.plus(Interval(0, ks)) == Interval(0, 4094 + 4094 + 1)
    shifted = ms._shifted(-63)
    assert shifted.value is ms.value
    assert shifted._envelope == ms._envelope and shifted._congruence == ms._congruence
    assert shifted._plus == -63


def test_minimum_and_maximum_with_runtime_ends():
    ms = _runtime("Ms", 4095)
    small = _runtime("Ms", 32)
    other = _runtime("Ns", 99)
    assert Interval(0, ms).minimum(Interval(0, ms._shifted(-1))).hi == ms._shifted(-1)
    assert Interval(0, ms).minimum(Interval(0, 64)).hi == 64
    assert Interval(0, small).minimum(Interval(0, 64)).hi == 32
    assert Interval(0, ms).minimum(Interval(0, other)).hi == 99
    assert Interval(0, None).minimum(Interval(0, ms)).hi == ms
    assert Interval(0, ms).maximum(Interval(0, other)).hi == 4095
    assert Interval(ms, None).maximum(Interval(other, None)).lo == 0


def _symmetry_cases() -> list[tuple[Interval, Interval]]:
    ms, ns = _runtime("Ms", 4095), _runtime("Ns", 99)
    return [
        (Interval(0, ms), Interval(0, ms._shifted(-1))),
        (Interval(0, ms), Interval(0, 64)),
        (Interval(0, ms), Interval(0, ns)),
        (Interval(0, None), Interval(0, ms)),
        (Interval(ms, None), Interval(ns, None)),
        (Interval(-3, ms), Interval(ns._shifted(-2), 7)),
        (Interval(None, ms), Interval(None, ns)),
    ]


@pytest.mark.parametrize("a, b", _symmetry_cases())
def test_minimum_maximum_and_intersect_are_symmetric(a, b):
    assert a.minimum(b) == b.minimum(a)
    assert a.maximum(b) == b.maximum(a)
    assert a.overlaps(b) == b.overlaps(a)
    if a.overlaps(b):
        assert a.intersect(b) == b.intersect(a)


def test_intersect_keeps_the_runtime_upper_end_and_the_int_lower_end():
    ms, ks, ns = _runtime("Ms", 4095), _runtime("Ks", 4095), _runtime("Ns", 99)
    assert Interval(0, 4096).intersect(Interval(0, ms._shifted(-63))) == Interval(
        0, ms._shifted(-63)
    )
    assert Interval(0, ms._shifted(-1)).intersect(Interval(0, ms._shifted(-63))).hi == (
        ms._shifted(-63)
    )
    assert Interval(-5, ms).intersect(Interval(0, ms)).lo == 0
    assert Interval(ks._shifted(-8), ms).intersect(Interval(0, ms)).lo == 0
    assert Interval(-15, 4096).intersect(Interval(None, ms._shifted(-15))) == Interval(
        -15, ms._shifted(-15)
    )
    assert Interval(0, ms).intersect(Interval(0, ns)).hi == 99
    pairs = [
        (Interval(0, 4096), Interval(0, ms._shifted(-63))),
        (Interval(0, ms._shifted(-1)), Interval(0, ms._shifted(-63))),
        (Interval(-5, ms), Interval(0, ms)),
        (Interval(ks._shifted(-8), ms), Interval(0, ms)),
        (Interval(-15, 4096), Interval(None, ms._shifted(-15))),
        (Interval(0, ms), Interval(0, ns)),
    ]
    for a, b in pairs:
        assert a.intersect(b) == b.intersect(a)
    with _raises(
        ValueError,
        "Interval.intersect: [0, 4) and [8, 16) have no value in common -- the two ranges "
        "state contradictory facts; check where each was derived",
    ):
        Interval(0, 4).intersect(Interval(8, 16))


def test_overlaps_cases():
    ls = _runtime("Ls", 127)
    ls9 = _runtime("Ls", 8)
    ls_wide = _runtime("Ls", 4095)
    cases = [
        (Interval(0, 4), Interval(8, 16), False),
        (Interval(100, 4096), Interval(0, ls._shifted(-63)), False),
        (Interval(0, 5), Interval(10, ls), False),
        (Interval(0, None), Interval(0, ls9._shifted(-15)), False),
        (Interval(0, ls_wide._shifted(-3)), Interval(0, ls_wide._shifted(-1)), True),
        (Interval(ls._shifted(2), None), Interval(0, ls._shifted(2)), False),
        (Interval(None, 3), UNBOUNDED, True),
    ]
    for a, b, expected in cases:
        assert a.overlaps(b) is expected and b.overlaps(a) is expected
    assert not Interval(0, 4).overlaps(Interval(8, 16))
    assert not Interval(100, 4096).overlaps(Interval(0, ls._shifted(-63)))
    with pytest.raises(ValueError, match="have no value in common"):
        Interval(100, 4096).intersect(Interval(0, ls._shifted(-63)))
    assert not Interval(0, 5).overlaps(Interval(10, ls))
    assert not Interval(0, None).overlaps(Interval(0, ls9._shifted(-15)))
    assert Interval(0, ls_wide._shifted(-3)).overlaps(Interval(0, ls_wide._shifted(-1)))
    assert not Interval(ls._shifted(2), None).overlaps(Interval(0, ls._shifted(2)))
    assert Interval(None, 3).overlaps(UNBOUNDED)


def _random_end(rng: random.Random, side: str, runtime: tuple[RuntimeValue, ...]):
    pick = rng.random()
    if pick < 0.15:
        return None
    if pick < 0.55:
        return rng.randint(-6, 6)
    return rng.choice(runtime)._shifted(rng.randint(-3, 3))


def _substitute(end, values: dict[int, int], side: str) -> float:
    if end is None:
        return float("-inf") if side == "lo" else float("inf")
    if isinstance(end, RuntimeValue):
        return values[id(end.value)] + end._plus
    return end


def _random_runtime_pairs(count: int):
    rng = random.Random(20261016)
    bounded = _runtime("L1", 7)
    unbounded = _runtime("L2", None, at_least=2)
    runtime = (bounded, unbounded)
    substitutions = [
        {id(bounded.value): l1, id(unbounded.value): l2}
        for l1 in range(8)
        for l2 in range(2, 24)
    ]
    pairs = []
    while len(pairs) < count:
        ends = [_random_end(rng, side, runtime) for side in ("lo", "hi", "lo", "hi")]
        try:
            a, b = Interval(ends[0], ends[1]), Interval(ends[2], ends[3])
        except ValueError:
            continue
        # A range empty for every value of its runtime ends states a contradiction; skip it.
        if a.overlaps(a) and b.overlaps(b):
            pairs.append((a, b))
    return pairs, substitutions


def _concrete(r: Interval, values: dict[int, int]) -> tuple[float, float]:
    return _substitute(r.lo, values, "lo"), _substitute(r.hi, values, "hi")


def test_overlaps_is_exact_on_random_runtime_pairs():
    """``overlaps`` is False exactly when no integer lies in both for every substitution."""
    pairs, substitutions = _random_runtime_pairs(3000)
    for a, b in pairs:
        meets = False
        for values in substitutions:
            (alo, ahi), (blo, bhi) = _concrete(a, values), _concrete(b, values)
            if max(alo, blo) < min(ahi, bhi):
                meets = True
                break
        assert a.overlaps(b) == meets, (a, b)
        assert b.overlaps(a) == meets


def test_minimum_maximum_intersect_are_sound_on_random_runtime_pairs():
    pairs, substitutions = _random_runtime_pairs(600)
    window = range(-40, 40)
    for a, b in pairs:
        results = {"min": a.minimum(b), "max": a.maximum(b)}
        assert results["min"] == b.minimum(a) and results["max"] == b.maximum(a)
        if a.overlaps(b):
            results["both"] = a.intersect(b)
            assert results["both"] == b.intersect(a)
        for values in substitutions:
            (alo, ahi), (blo, bhi) = _concrete(a, values), _concrete(b, values)
            xs = [x for x in window if alo <= x < ahi]
            ys = [y for y in window if blo <= y < bhi]
            if not xs or not ys:
                continue
            checks = {
                "min": {min(x, y) for x in xs for y in ys},
                "max": {max(x, y) for x in xs for y in ys},
                "both": set(xs) & set(ys),
            }
            for op, reached in checks.items():
                if op not in results:
                    continue
                lo, hi = _concrete(results[op], values)
                assert all(lo <= v < hi for v in reached), (op, a, b, values)


def test_snapped_moves_a_runtime_end_only_when_its_residue_is_known():
    ms128 = _runtime("Ms", 4096, multiple_of=128)
    ms64 = _runtime("Ms", 4096, multiple_of=64)
    assert Interval(0, ms128).snapped(Congruence(128, 0)).hi == ms128._shifted(-127)
    assert Interval(0, ms64).snapped(Congruence(128, 0)).hi == ms64


def test_runtime_snap_matches_concrete_lengths():
    for multiple in (8, 16):
        for residue in range(0, multiple, 3):
            length = _runtime("L", 64, multiple_of=multiple, residue=residue)
            for plus in range(-5, 6):
                for r in range(4):
                    snapped = Interval(0, length._shifted(plus)).snapped(
                        Congruence(4, r)
                    )
                    for value in range(residue, 65, multiple):
                        members = [x for x in range(value + plus) if x % 4 == r]
                        if members:
                            assert value + snapped.hi._plus == max(members) + 1


# --- value ranges against exhaustive enumeration -----------------------------------------------


def _conv_graph(n, hi, wi, c, y, x, stride, pad):
    ho = (hi + 2 * pad - y) // stride + 1
    wo = (wi + 2 * pad - x) // stride + 1
    graph = make_transform_graph(
        inputs(m=n * ho * wo, k=y * x * c),
        unmerge("m", into=("n", "ho", "wo"), lengths=(n, ho, wo)),
        unmerge("k", into=("y", "x", "c"), lengths=(y, x, c)),
        embed(("ho", "y"), into="h_dil", coeffs=(stride, 1)),
        offset("h_dil", into="hi", shift=-pad),
        embed(("wo", "x"), into="w_dil", coeffs=(stride, 1)),
        offset("w_dil", into="wi", shift=-pad),
        outputs("n", "hi", "wi", "c"),
    )
    return graph, _tensor(n, hi, wi, c)


_SWIZZLE = make_transform_graph(
    inputs(row=8, col=8),
    xor(("row", "col"), into=("row_s", "col_s")),
    outputs("row_s", "col_s"),
)

_MERGE_SLICE = make_transform_graph(
    inputs(a=3, b=4, t=6),
    merge(("a", "b"), into="ab", lengths=(3, 4)),
    slice_window("t", into="ts", begin=2, end=8),
    outputs("ab", "ts"),
)


def _reached(graph, ranges) -> dict[str, set[int]]:
    seen: dict[str, set[int]] = {}
    for coords in itertools.product(*(range(lo, hi) for lo, hi in ranges)):
        for name, value in graph.resolve(coords).items():
            seen.setdefault(name, set()).add(value)
    return seen


@pytest.mark.parametrize(
    "graph, ranges",
    [
        (_conv_graph(2, 5, 5, 3, 3, 3, 1, 1)[0], ((0, 64), (0, 32))),
        (_conv_graph(2, 7, 7, 2, 3, 3, 2, 1)[0], ((0, 32), (0, 18))),
        (_SWIZZLE, ((0, 8), (0, 8))),
        (_SWIZZLE, ((0, 8), (0, 20))),
        (_MERGE_SLICE, ((0, 3), (0, 4), (0, 6))),
        (_MERGE_SLICE, ((-2, 5), (1, 9), (-3, 4))),
    ],
)
def test_value_ranges_match_exhaustive_enumeration(graph, ranges):
    found = value_ranges(graph, ranges)
    for name, values in _reached(graph, ranges).items():
        lo, hi = found[name]
        assert min(values) == lo and max(values) == hi - 1, name


def test_value_ranges_report_the_unconditioned_outer_piece():
    """A tile-rounded K past the domain drives the outer piece past its length; the checks
    still elide only what the in-domain ranges prove."""
    graph, tensor = _conv_graph(1, 4, 4, 3, 3, 3, 1, 1)  # K = 27
    found = value_ranges(graph, ((0, 16), (0, 32)))
    assert found["y"] == (0, 4)  # 31 // 9 = 3, past the length 3
    checks = {c.coordinate for c in required_checks(graph, tensor, ((0, 16), (0, 32)))}
    assert checks == {"k", "hi", "wi"}


# --- required checks ---------------------------------------------------------------------------


def _passes(
    checks: tuple[BoundCheck, ...], resolved: dict[str, int], runtime=None
) -> bool:
    for check in checks:
        value = resolved[check.coordinate]
        if check.lo is not None and value < _end_value(check.lo, runtime):
            return False
        if check.hi is not None and value >= _end_value(check.hi, runtime):
            return False
    return True


def _end_value(end, runtime) -> int:
    if isinstance(end, RuntimeValue):
        return runtime[id(end.value)] + end._plus
    return end


def _truly_valid(graph, resolved: dict[str, int], lengths) -> bool:
    """Every input in its domain, every read in the domain its step requires, every output
    inside the tensor -- written from the kind table, independent of the checks."""
    for name, domain in graph.inputs.items():
        if domain is not None and not 0 <= resolved[name] < domain:
            return False
    for node in graph.nodes:
        reads = [resolved[r] for r in node.read]
        if node.kind is TransformType.UNMERGE:
            total = 1
            for length in node.params.lengths:
                total *= length
            if not 0 <= reads[0] < total:
                return False
        elif node.kind is TransformType.MERGE:
            if not all(0 <= v < n for v, n in zip(reads, node.params.lengths)):
                return False
        elif node.kind is TransformType.SLICE:
            if not 0 <= reads[0] < node.params.end - node.params.begin:
                return False
        elif node.kind is TransformType.XOR:
            if reads[0] < 0 or not 0 <= reads[1] < graph.edge(node.write[1]).domain:
                return False
    return all(0 <= resolved[name] < n for name, n in zip(graph.outputs, lengths))


def _assert_checks_exact(graph, lengths, ranges) -> tuple[BoundCheck, ...]:
    """The checks accept exactly the truly valid inputs: no check hides an invalid value."""
    checks = required_checks(graph, _tensor(*lengths), ranges)
    for coords in itertools.product(*(range(lo, hi) for lo, hi in ranges)):
        resolved = graph.resolve(coords)
        assert _passes(checks, resolved) == _truly_valid(
            graph, resolved, lengths
        ), coords
    return checks


def test_conv_tile_rounded_inputs_need_exactly_m_k_and_the_halo_rows_and_columns():
    graph, tensor = _conv_graph(2, 5, 5, 3, 3, 3, 1, 1)  # M = 50, K = 27
    checks = _assert_checks_exact(graph, tensor.lengths, ((0, 64), (0, 32)))
    assert {c.coordinate for c in checks} == {"m", "k", "hi", "wi"}
    by_name = {c.coordinate: c for c in checks}
    assert by_name["m"] == BoundCheck("m", None, 50, (CheckReason.INPUT_DOMAIN,))
    assert by_name["k"] == BoundCheck("k", None, 27, (CheckReason.INPUT_DOMAIN,))
    assert by_name["hi"] == BoundCheck("hi", 0, 5, (CheckReason.TENSOR_BOUND,))


def test_conv_k_aligned_graph_needs_no_check_on_y_x_c():
    graph, tensor = _conv_graph(2, 5, 5, 3, 3, 3, 1, 1)
    checks = required_checks(graph, tensor, ((0, 50), (0, 27)))
    assert {c.coordinate for c in checks} == {"hi", "wi"}


def test_stride_two_conv_takes_its_bound_from_the_tensor():
    """Hi = 224, Ho = 112: the span the graph would derive (225) undercuts nothing, because the
    bound is the tensor's 224. Tap (111, 2) is a real row; h = 224 is rejected."""
    graph = make_transform_graph(
        inputs(ho=112, r=3),
        embed(("ho", "r"), into="h_dil", coeffs=(2, 1)),
        offset("h_dil", into="h", shift=-1),
        outputs("h"),
    )
    ranges = ((0, 128), (0, 3))
    checks = _assert_checks_exact(graph, (224,), ranges)
    assert graph.evaluate((111, 2)) == (223,)
    assert _passes(checks, graph.resolve((111, 2)))
    rejected = [
        c
        for c in itertools.product(range(128), range(3))
        if graph.evaluate(c) == (224,)
    ]
    assert rejected and not any(_passes(checks, graph.resolve(c)) for c in rejected)


@pytest.mark.parametrize(
    "graph, lengths, ranges",
    [
        (_SWIZZLE, (8, 8), ((0, 8), (0, 8))),
        (_SWIZZLE, (8, 8), ((-2, 9), (0, 11))),
        (_MERGE_SLICE, (12, 10), ((0, 3), (0, 4), (0, 6))),
        (_MERGE_SLICE, (12, 7), ((-1, 4), (0, 6), (-2, 8))),
        (_conv_graph(1, 4, 4, 2, 3, 3, 2, 1)[0], (1, 4, 4, 2), ((0, 8), (-3, 24))),
    ],
)
def test_no_check_required_never_hides_an_out_of_domain_value(graph, lengths, ranges):
    _assert_checks_exact(graph, lengths, ranges)


def test_unknown_input_range_is_an_input_domain_check():
    graph = identity_graph((16,))
    checks = required_checks(graph, _tensor(16), ((None, None),))
    assert checks == (BoundCheck("axis0", 0, 16, (CheckReason.INPUT_DOMAIN,)),)


def test_toolbox_bounds_come_from_the_tensor_lengths():
    graph, tensor = _conv_graph(2, 5, 5, 3, 3, 3, 1, 1)
    ranges = ((0, 64), (0, 32))
    internal = _required_checks(
        graph,
        tuple(Interval.from_pair(r) for r in ranges),
        tuple(AxisBound.of(0, length) for length in tensor.lengths),
    )
    assert required_checks(graph, tensor, ranges) == internal


def test_runtime_lower_and_upper_bound_give_a_two_sided_check():
    lo, hi = _runtime("Lo", 1000), _runtime("Hi", 4096)
    checks = _required_checks(
        identity_graph((None,)), (UNBOUNDED,), (AxisBound.of(lo, hi),)
    )
    assert checks == (BoundCheck("axis0", lo, hi, (CheckReason.TENSOR_BOUND,)),)


def _tile_inside_graph(tile: int):
    return make_transform_graph(
        inputs(origin=None, lane=tile),
        embed(("origin", "lane"), into="row", coeffs=(1, 1)),
        outputs("row"),
    )


def test_whole_tile_inside_a_runtime_length_is_proven_by_identity():
    tile = 16
    seqlen = _runtime("seqlen", 4096, multiple_of=tile)
    graph = _tile_inside_graph(tile)
    bound = (AxisBound.of(0, seqlen),)
    inside = (Interval(0, seqlen._shifted(-(tile - 1))), Interval(0, tile))
    assert _required_checks(graph, inside, bound) == ()
    unknown = (UNBOUNDED, Interval(0, tile))
    assert _required_checks(graph, unknown, bound) == (
        BoundCheck("row", 0, seqlen, (CheckReason.TENSOR_BOUND,)),
    )
    same_number = _runtime("seqlen_copy", 4096, multiple_of=tile)
    copied = (AxisBound.of(0, same_number),)
    assert _required_checks(graph, inside, copied) == (
        BoundCheck("row", None, same_number, (CheckReason.TENSOR_BOUND,)),
    )


def test_identity_graph_over_a_runtime_length_checks_that_axis_only():
    seqlen = _runtime("seqlen", 8192)
    graph = identity_graph((seqlen, 128))
    assert tuple(graph.inputs.items()) == (("axis0", None), ("axis1", 128))
    assert graph.outputs == ("axis0", "axis1")
    bounds = (AxisBound.of(0, seqlen), AxisBound.of(0, 128))
    checks = _required_checks(graph, (Interval(0, None), Interval(0, 128)), bounds)
    # Per side: the lower side is proven (None), the upper side is reported.
    assert checks == (BoundCheck("axis0", None, seqlen, (CheckReason.TENSOR_BOUND,)),)


def _enumerated_sides(origins: range, tile: int, length: int) -> tuple[bool, bool]:
    positions = [o + t for o in origins for t in range(tile)]
    return min(positions) < 0, max(positions) >= length


@pytest.mark.parametrize("lengths, tile", [((10, 8), (4, 4)), ((256, 128), (64, 64))])
def test_identity_graph_decisions_equal_enumeration_over_every_origin(lengths, tile):
    graph = identity_graph(lengths)
    steps = [
        range(-4, 1),
        range(1),
        range(0, 9, 4),
        range(16, 17),
        range(4, 200, 32),
    ]
    for o0, o1 in itertools.product(steps, repeat=2):
        ranges = tuple((o.start, o[-1] + t) for o, t in zip((o0, o1), tile))
        checks = {
            c.coordinate: c for c in required_checks(graph, _tensor(*lengths), ranges)
        }
        for axis, (origins, t, length) in enumerate(zip((o0, o1), tile, lengths)):
            low, high = _enumerated_sides(origins, t, length)
            check = checks.get(f"axis{axis}")
            assert (check is not None and check.lo is not None) == low
            assert (check is not None and check.hi is not None) == high


def test_a_tile_that_overhangs_the_store_at_origin_16_0_reports_a_check():
    """The 256x128 store of a 256x128x64 GEMM at origin (16, 0): the tile overhangs axis 0."""
    checks = required_checks(
        identity_graph((256, 128)), _tensor(256, 128), ((16, 272), (0, 128))
    )
    assert checks == (BoundCheck("axis0", None, 256, (CheckReason.INPUT_DOMAIN,)),)


# --- graph_injective ---------------------------------------------------------------------------


def _injective_by_enumeration(graph, ranges) -> bool:
    seen = set()
    for coords in itertools.product(*(range(lo, hi) for lo, hi in ranges)):
        out = graph.evaluate(coords)
        if out in seen:
            return False
        seen.add(out)
    return True


@pytest.mark.parametrize(
    "graph, ranges, expected",
    [
        (_SWIZZLE, ((0, 8), (0, 8)), True),
        (_MERGE_SLICE, ((0, 3), (0, 4), (0, 6)), True),
        (_conv_graph(2, 5, 5, 3, 1, 1, 1, 0)[0], ((0, 50), (0, 3)), True),
        (
            make_transform_graph(
                inputs(ho=4, y=3),
                embed(("ho", "y"), into="h", coeffs=(1, 1)),
                outputs("h"),
            ),
            ((0, 4), (0, 3)),
            False,
        ),
        (
            make_transform_graph(
                inputs(ho=4, y=3),
                embed(("ho", "y"), into="h", coeffs=(3, 1)),
                outputs("h"),
            ),
            ((0, 4), (0, 3)),
            True,
        ),
        (
            make_transform_graph(inputs(m=4, b=3), tt.broadcast(("b",)), outputs("m")),
            ((0, 4), (0, 3)),
            False,
        ),
    ],
)
def test_graph_injective(graph, ranges, expected):
    assert graph_injective(graph, ranges) is expected
    assert _injective_by_enumeration(graph, ranges) is expected


def test_an_unbounded_read_is_injective_only_as_the_widest_coefficient():
    """A read with no upper end can carry the largest coefficient of an embed and nothing else."""
    graph = make_transform_graph(
        inputs(row=None, col=4),
        embed(("row", "col"), into="flat", coeffs=(4, 1)),
        outputs("flat"),
    )
    assert graph_injective(graph, ((0, None), (0, 4)))
    swapped = make_transform_graph(
        inputs(row=None, col=4),
        embed(("row", "col"), into="flat", coeffs=(1, 4)),
        outputs("flat"),
    )
    assert not graph_injective(swapped, ((0, None), (0, 4)))
    tied = make_transform_graph(
        inputs(row=None, col=None),
        embed(("row", "col"), into="flat", coeffs=(4, 4)),
        outputs("flat"),
    )
    assert not graph_injective(tied, ((0, None), (0, 2)))


def test_dilated_conv_taps_overlap_and_are_not_injective():
    graph, _ = _conv_graph(1, 6, 6, 1, 3, 3, 1, 1)
    assert not graph_injective(graph, ((0, 36), (0, 9)))


# --- public values carry public types only -----------------------------------------------------


_INTERNAL = (Interval, Congruence, AxisBound, CheckSide, VectorRun)


def test_public_values_carry_no_internal_type():
    ms = _runtime("Ms", 4096, multiple_of=8)
    graph = _tile_inside_graph(16)
    found = value_ranges(graph, ((0, ms), (0, 16)))
    for pair in found.values():
        assert isinstance(pair, tuple) and not any(
            isinstance(e, _INTERNAL) for e in pair
        )
    checks = required_checks(identity_graph((32,)), _tensor(32), ((None, None),))
    for check in checks:
        for f in fields(BoundCheck):
            assert not isinstance(getattr(check, f.name), _INTERNAL)
    assert [a for a in dir(ms) if not a.startswith("_")] == ["value"]
    for name in tt.__all__:
        assert getattr(tt, name) not in _INTERNAL


def test_runtime_value_reads_its_facts_in_plain_words():
    k = _runtime("K", 65536, multiple_of=8)
    assert str(k) == "K: 0 <= value <= 65536, a multiple of 8"


@pytest.mark.parametrize(
    "pair",
    [(0, 8), (None, 8), (-3, None), (None, None)],
)
def test_interval_pairs_round_trip(pair):
    assert Interval.from_pair(pair).as_pair() == pair


def test_interval_pairs_round_trip_runtime_ends():
    ms = _runtime("Ms", 4096)
    for pair in [(0, ms), (ms._shifted(-4), None), (ms, ms._shifted(3))]:
        assert Interval.from_pair(pair).as_pair() == pair


# --- errors ------------------------------------------------------------------------------------


def test_two_runtime_values_over_one_value_with_different_facts_raise():
    ms = _runtime("Ms", 4095)
    retyped = RuntimeValue(ms.value, -63, Interval(0, 64), ms._congruence)
    message = (
        "two RuntimeValues over the same value Ms state different facts (envelope [0, 4096) vs "
        "[0, 64), congruence 0 mod 1 vs 0 mod 1) -- write ends relative to a length as "
        "length.shifted(k), never restate its envelope"
    )
    with _raises(ValueError, message):
        Interval(0, ms).intersect(Interval(0, retyped))
    graph = _tile_inside_graph(16)
    with pytest.raises(ValueError, match="state different facts"):
        _required_checks(
            graph, (Interval(0, retyped), Interval(0, 16)), (AxisBound.of(0, ms),)
        )
    recongruent = RuntimeValue(ms.value, 0, ms._envelope, Congruence(8, 0))
    with pytest.raises(ValueError, match="state different facts"):
        value_ranges(graph, ((0, ms), (0, recongruent)))


@pytest.mark.parametrize("owner", [Interval(0, 4), Congruence(4, 1)])
@pytest.mark.parametrize("operation", ["floor_div", "floor_mod"])
def test_divisor_below_one_raises(owner, operation):
    name = type(owner).__name__
    with _raises(ValueError, f"{name}.{operation} needs a divisor >= 1, got 0"):
        getattr(owner, operation)(0)


def test_congruence_scaled_by_zero_raises():
    with _raises(
        ValueError,
        "Congruence.scaled(0): the product is the constant 0 -- fold it as the int 0 instead of "
        "a congruence",
    ):
        Congruence(4, 1).scaled(0)


def test_empty_interval_raises():
    with _raises(
        ValueError,
        "Interval [4, 4) is empty -- int ends need lo < hi; an empty range states no fact (to "
        "ask whether two ranges meet, use Interval.overlaps)",
    ):
        Interval(4, 4)


def test_input_ranges_count_raises():
    graph = identity_graph((4, 8))
    message = (
        "input_ranges has 1 entries but the graph has 2 inputs ('axis0', 'axis1') -- give one "
        "(lo, hi) pair per input, in inputs(...) order; use (None, None) when a range is not "
        "known"
    )
    with _raises(ValueError, message):
        value_ranges(graph, ((0, 4),))
    with _raises(ValueError, message):
        required_checks(graph, _tensor(4, 8), ((0, 4),))
    with _raises(ValueError, message):
        graph_injective(graph, ((0, 4),))
    with _raises(ValueError, message):
        _value_ranges(graph, (UNBOUNDED,))


def test_axis_bound_with_a_shifted_runtime_end_raises():
    ms = _runtime("Ms", 4095)
    with _raises(
        ValueError,
        "AxisBound end Ms - 1 is RuntimeValue(..., plus=-1) -- a bound is a length (plus 0); an "
        "end relative to a length belongs in an Interval: length.shifted(k)",
    ):
        AxisBound.of(0, ms._shifted(-1))


def test_empty_bound_raises():
    with _raises(
        ValueError,
        "bound of axis 1 is empty: [8, 8) -- an always-empty window cannot be read",
    ):
        _required_checks(
            identity_graph((4, 8)),
            (UNBOUNDED, UNBOUNDED),
            (AxisBound.of(0, 4), AxisBound.of(8, 8)),
        )


def test_bounds_count_raises():
    with _raises(
        ValueError,
        "bounds has 1 entries but the tensor has rank 2 -- give one AxisBound per tensor axis",
    ):
        _required_checks(
            identity_graph((4, 8)), (UNBOUNDED, UNBOUNDED), (AxisBound.of(0, 4),)
        )


def test_two_failing_ends_are_merged_into_the_one_that_is_always_tighter():
    graph = identity_graph((8,))
    length = _runtime("L", 99, at_least=8)
    checks = _required_checks(graph, (Interval(0, 20),), (AxisBound.of(0, length),))
    assert checks == (
        BoundCheck(
            "axis0", None, 8, (CheckReason.INPUT_DOMAIN, CheckReason.TENSOR_BOUND)
        ),
    )


def test_two_failing_ends_that_cannot_be_ordered_raise():
    graph = identity_graph((8,))
    length = _runtime("L", 99)
    with _raises(
        ValueError,
        "coordinate 'axis0' must be checked against both 8 and L on its upper side, and which "
        "of the two is tighter is not known at build time -- one check holds one end per side; "
        "bound the coordinate by one of them (an input over a runtime-length axis is declared "
        "inputs(name=None))",
    ):
        _required_checks(graph, (Interval(0, 20),), (AxisBound.of(0, length),))


def test_graph_outputs_must_match_the_tensor_rank():
    with _raises(
        ValueError, "graph has 2 outputs ('axis0', 'axis1') but the tensor has rank 1"
    ):
        required_checks(identity_graph((4, 8)), _tensor(4), ((0, 4), (0, 8)))


# --- import order ------------------------------------------------------------------------------


def test_descriptors_import_before_tensor_transforms():
    code = (
        "import rocke.helpers.tiling.descriptors\n"
        "import rocke.helpers.tiling.tensor_transforms.ranges\n"
        "import rocke.helpers.tiling.tensor_transforms\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
