# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the IR range walk (no GPU): the range and residue it derives for every row of its
rule table, how it splits a plain value into constant, per-wave and per-lane parts, how it
rebuilds a runtime length from its range marker, which leaves it finds that may differ between
the waves of a workgroup, and how it reads and records the workgroup size.

Values are built on small kernels with the real IR builder. Where a row's soundness is the
question, the derived facts are checked against every value the expression takes, computed with
i32 semantics.
"""

from __future__ import annotations

import itertools
import random
import re

import pytest

from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    UNBOUNDED,
    AffineCoord,
    AxisBound,
    Congruence,
    Interval,
    RuntimeValue,
    UniformPart,
    _required_checks,
)
from rocke.helpers.tiling.tensor_transforms.walk import (
    WORKGROUP_SIZE_RECORD,
    leaf_varying_within_workgroup,
    runtime_value_of,
    split_coordinate,
    value_facts,
)

I32_LO, I32_HI = -(2**31), 2**31
UNKNOWN = (UNBOUNDED, Congruence(1, 0))


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _builder(workgroup_size: int | None = 256) -> IRBuilder:
    b = IRBuilder("walk_test")
    if workgroup_size is not None:
        b.kernel.attrs["max_workgroup_size"] = workgroup_size
    return b


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _stated(b: IRBuilder, name: str, **facts) -> Value:
    """A kernel parameter with stated facts (a range marker over it)."""
    return b._assume_range(b.param(name, I32), **facts)


def _runtime_length(
    b: IRBuilder, name: str, at_most: int, *, multiple_of: int = 1
) -> RuntimeValue:
    """A runtime length the way its producer states it: an envelope marker, then rebuilt."""
    marked = _stated(b, name, lo=0, hi=at_most + 1, multiple_of=multiple_of)
    return runtime_value_of(marked)


def _op_count(b: IRBuilder) -> int:
    def count(ops) -> int:
        return sum(1 + sum(count(r.ops) for r in op.regions) for op in ops)

    return count(b.kernel.body.ops)


def _wrap32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


def _evaluate(value: Value | int, tid: int) -> int:
    """What ``value`` computes for one thread id, with i32 semantics (truncating div / mod)."""
    if isinstance(value, int):
        return value
    op = value.op
    if op.name == "arith.constant":
        return op.attrs["value"]
    if op.name == "gpu.thread_id":
        return tid
    a, *rest = (_evaluate(v, tid) for v in op.operands)
    c = rest[0] if rest else None
    if op.name == "arith.add":
        return a + c
    if op.name == "arith.sub":
        return a - c
    if op.name == "arith.mul":
        return a * c
    if op.name == "arith.div":
        return abs(a) // c * (1 if a >= 0 else -1)
    if op.name == "arith.mod":
        return a - c * (abs(a) // c * (1 if a >= 0 else -1))
    if op.name == "arith.shl":
        return _wrap32(a << c)
    if op.name == "arith.and":
        return a & c
    if op.name == "arith.smin":
        return min(a, c)
    if op.name == "arith.smax":
        return max(a, c)
    raise AssertionError(f"no evaluator for {op.name}")


def _holds(facts: tuple[Interval, Congruence], v: int) -> bool:
    r, congruence = facts
    lo_ok = r.lo is None or v >= r.lo
    hi_ok = r.hi is None or v < r.hi
    return lo_ok and hi_ok and (v - congruence.residue) % congruence.modulus == 0


def test_the_walk_is_internal_and_not_on_the_package_door():
    for name in ("split_coordinate", "value_facts", "runtime_value_of", "AffineCoord"):
        assert not hasattr(tt, name)
        assert name not in tt.__all__


# --- the range rules, one row at a time ---------------------------------------------------------


@pytest.mark.parametrize(
    "c, expected",
    [
        (5, (Interval(5, 6), Congruence(5, 0))),
        (-12, (Interval(-12, -11), Congruence(12, 0))),
        (0, (Interval(0, 1), Congruence(1, 0))),
    ],
)
def test_a_constant_is_exact_and_a_multiple_of_itself(c, expected):
    b = _builder()
    assert value_facts(b, _c(b, c)) == expected
    assert value_facts(b, c) == expected


def test_thread_and_block_ids_and_readfirstlane():
    b = _builder(128)
    tid, bid = b.thread_id_x(), b.block_id_x()
    assert value_facts(b, tid) == (Interval(0, 128), Congruence(1, 0))
    assert value_facts(b, bid) == (Interval(0, None), Congruence(1, 0))
    quarter = b.mul(b.div(tid, _c(b, 64)), _c(b, 32))
    assert value_facts(b, b.readfirstlane(quarter)) == value_facts(b, quarter)


def test_a_constant_added_or_subtracted_shifts_the_facts():
    b = _builder()
    x = b.mul(b.thread_id_x(), _c(b, 4))  # [0, 1021), 0 mod 4
    assert value_facts(b, b.add(x, _c(b, 3))) == (Interval(3, 1024), Congruence(4, 3))
    assert value_facts(b, b.add(_c(b, 3), x)) == (Interval(3, 1024), Congruence(4, 3))
    assert value_facts(b, b.sub(x, _c(b, 3))) == (Interval(-3, 1018), Congruence(4, 1))
    # c - x mirrors the range and the residue
    assert value_facts(b, b.sub(_c(b, 10), x)) == (
        Interval(-1010, 11),
        Congruence(4, 2),
    )


def test_sums_of_two_unknown_terms_add_ranges_and_take_the_common_residue():
    b = _builder()
    tid = b.thread_id_x()
    x = b.mul(tid, _c(b, 6))  # [0, 1531), 0 mod 6
    y = b.mul(tid, _c(b, 4))  # [0, 1021), 0 mod 4
    assert value_facts(b, b.add(x, y)) == (Interval(0, 2551), Congruence(2, 0))
    assert value_facts(b, b.sub(x, y)) == (Interval(-1020, 1531), Congruence(2, 0))


def test_a_product_by_a_constant_scales_and_by_zero_is_the_constant_zero():
    b = _builder()
    tid = b.thread_id_x()
    assert value_facts(b, b.mul(tid, _c(b, 8))) == (Interval(0, 2041), Congruence(8, 0))
    assert value_facts(b, b.mul(_c(b, -2), tid)) == (
        Interval(-510, 1),
        Congruence(2, 0),
    )
    assert value_facts(b, b.mul(tid, _c(b, 0))) == (Interval(0, 1), Congruence(1, 0))
    assert value_facts(b, b.mul(tid, tid)) == UNKNOWN  # two non-exact values


def test_div_and_mod_need_an_exact_positive_divisor_and_a_dividend_proven_non_negative():
    b = _builder()
    tid = b.thread_id_x()
    assert value_facts(b, b.div(tid, _c(b, 4))) == (Interval(0, 64), Congruence(1, 0))
    assert value_facts(b, b.mod(tid, _c(b, 4))) == (Interval(0, 4), Congruence(1, 0))
    by_8 = b.mul(tid, _c(b, 8))
    assert value_facts(b, b.div(by_8, _c(b, 4))) == (Interval(0, 511), Congruence(2, 0))
    below_zero = b.sub(tid, _c(b, 1))
    assert value_facts(b, b.div(below_zero, _c(b, 4))) == UNKNOWN
    assert value_facts(b, b.mod(below_zero, _c(b, 4))) == UNKNOWN
    assert value_facts(b, b.div(tid, b.thread_id_x())) == UNKNOWN


def test_smin_and_smax_take_the_range_rule_and_join_the_residues():
    b = _builder()
    tid = b.thread_id_x()
    x = b.mul(tid, _c(b, 8))  # [0, 2041), 0 mod 8
    assert value_facts(b, b.smin(x, _c(b, 100))) == (Interval(0, 101), Congruence(4, 0))
    assert value_facts(b, b.smax(_c(b, 100), x)) == (
        Interval(100, 2041),
        Congruence(4, 0),
    )
    y = b.add(b.mul(tid, _c(b, 12)), _c(b, 4))  # 4 mod 12
    assert value_facts(b, b.smin(x, y))[1] == Congruence(4, 0)
    assert value_facts(b, b.smin(_c(b, 3), _c(b, 7))) == (
        Interval(3, 4),
        Congruence(3, 0),
    )


def test_ops_outside_the_rule_set_are_unknown():
    b = _builder()
    tid = b.thread_id_x()
    p = b.param("p", I32)
    ptr = b.param("lens", PtrType(I32, "global"))
    unknown = [
        p,
        b.global_load(ptr, tid, I32),
        b.lane_id(),
        b.lor(tid, _c(b, 1)),
        b.xor(tid, _c(b, 1)),
        b.lshr(tid, _c(b, 3)),
        b.land(tid, _c(b, 6)),  # not a low mask
        b.shl(tid, tid),  # a shift amount that is not exact
        b.add_nuw(
            tid, tid
        ),  # may pass the signed maximum: no-signed-wrap rule does not hold
    ]
    for value in unknown:
        assert value_facts(b, value) == UNKNOWN, value
    with b.scf_for_iter(_c(b, 0), _c(b, 8), _c(b, 1), [("acc", tid)]) as (_, (acc,)):
        assert value_facts(b, acc) == UNKNOWN


def test_unknown_is_a_fact_so_a_partial_residue_survives():
    b = _builder()
    p = b.param("p", I32)
    assert value_facts(b, b.mul(p, _c(b, 128))) == (UNBOUNDED, Congruence(128, 0))


def test_every_arithmetic_row_holds_for_every_thread_id():
    b = _builder(256)
    tid = b.thread_id_x()
    c = lambda v: _c(b, v)
    expressions = [
        b.add(b.mul(tid, c(3)), c(-7)),
        b.sub(c(5), b.mul(tid, c(2))),
        b.sub(b.mul(tid, c(6)), b.mul(tid, c(4))),
        b.div(b.add(tid, c(9)), c(5)),
        b.mod(b.mul(tid, c(6)), c(9)),
        b.shl(b.mod(tid, c(4)), c(3)),
        b.land(b.sub(tid, c(300)), c(7)),
        b.smin(b.sub(tid, c(3)), c(10)),
        b.smax(b.mul(tid, c(4)), c(18)),
        b.add(b.div(tid, c(64)), b.mod(tid, c(16))),
    ]
    for value in expressions:
        facts = value_facts(b, value)
        reached = [_evaluate(value, t) for t in range(256)]
        assert all(_holds(facts, v) for v in reached), value


# --- the worked examples of the rule table ---------------------------------------------------------


def test_worked_example_a_masked_shifted_thread_id_is_one_per_lane_part():
    b = _builder(256)
    value = b.shl(b.mod(b.thread_id_x(), _c(b, 4)), _c(b, 3))
    assert value_facts(b, value) == (Interval(0, 25), Congruence(8, 0))
    split = split_coordinate(b, value)
    assert (split.const, split.uniform, split.lane) == (0, (), value)
    assert (split.lane_range, split.lane_congruence) == (
        Interval(0, 25),
        Congruence(8, 0),
    )


def test_worked_example_a_raw_block_origin_keeps_its_residue_but_shl_drops_the_range():
    b = _builder()
    by_mul = b.mul(b.block_id_x(), _c(b, 128))
    by_shl = b.shl(b.block_id_x(), _c(b, 7))
    assert value_facts(b, by_mul) == (Interval(0, None), Congruence(128, 0))
    assert value_facts(b, by_shl) == (UNBOUNDED, Congruence(128, 0))
    for origin in (by_mul, by_shl):
        split = split_coordinate(b, origin)
        assert split.lane is None and split.const == 0
        assert [part.value for part in split.uniform] == [origin]


def test_a_guard_marker_gives_a_multiplied_and_a_shifted_origin_the_same_facts():
    b = _builder()
    m = _runtime_length(b, "M", 1 << 20, multiple_of=128)
    for origin in (b.mul(b.block_id_x(), _c(b, 128)), b.shl(b.block_id_x(), _c(b, 7))):
        guarded = b._assume_range(origin, lo=0, hi_value=m.value)
        assert value_facts(b, guarded) == (
            Interval(0, m._shifted(-127)),
            Congruence(128, 0),
        )


def test_worked_example_and_with_a_low_mask():
    b = _builder()
    wide = _stated(b, "x", lo=-300, hi=300)
    narrow = _stated(b, "y", lo=16, hi=20)
    assert value_facts(b, b.land(wide, _c(b, 7)))[0] == Interval(0, 8)
    assert value_facts(b, b.land(_c(b, 31), narrow))[0] == Interval(16, 20)


def test_anti_example_a_right_shift_loses_the_range_a_division_keeps():
    b = _builder()
    x = b.mul(b.thread_id_x(), _c(b, 16))
    assert value_facts(b, b.lshr(x, _c(b, 3))) == UNKNOWN
    assert value_facts(b, b.div(x, _c(b, 8))) == (Interval(0, 511), Congruence(2, 0))


def test_a_grouped_one_dimensional_grid_remap_gives_origins_that_are_multiples_of_the_tile():
    """A 1-D grid remapped in-kernel into groups of row tiles (the order that shares column tiles
    in cache): only the residue survives, the range is unknown until a guard states it.
    """
    b = _builder()
    tile_m = tile_n = 128
    group_m = 8
    m_rt, n_rt = b.param("M", I32), b.param("N", I32)
    bid = b.block_id_x()
    n_m_blocks = b.div(b.add(m_rt, _c(b, tile_m - 1)), _c(b, tile_m))
    n_n_blocks = b.div(b.add(n_rt, _c(b, tile_n - 1)), _c(b, tile_n))
    per_group = b.mul(_c(b, group_m), n_n_blocks)
    group_id = b.div(bid, per_group)
    first_m = b.mul(group_id, _c(b, group_m))
    group_rows = b.smin(b.sub(n_m_blocks, first_m), _c(b, group_m))
    in_group = b.mod(bid, per_group)
    m_block = b.add(first_m, b.mod(in_group, group_rows))
    n_block = b.div(in_group, group_rows)
    for origin in (b.mul(m_block, _c(b, tile_m)), b.mul(n_block, _c(b, tile_n))):
        assert value_facts(b, origin) == (UNBOUNDED, Congruence(128, 0))


# --- shl and the low-mask and against i32 semantics ------------------------------------------------

_SHL_RANGES = [
    (None, None),
    (0, None),
    (None, 0),
    (0, 1),
    (0, 128),
    (-300, 300),
    (5, 6),
    (-7, -3),
    (0, 2**24),
    (0, 2**31 - 1),
    (I32_LO, I32_HI),
    (-(2**20), 2**20),
    (1000, 1001),
]
_SHL_CONGRUENCES = [(1, 0), (2, 0), (4, 1), (128, 0), (3, 2), (8, 5), (6, 0)]


def _samples(facts: tuple[Interval, Congruence], rng: random.Random) -> list[int]:
    r, _ = facts
    lo = I32_LO if r.lo is None else max(r.lo, I32_LO)
    hi = I32_HI if r.hi is None else min(r.hi, I32_HI)
    found = set(range(lo, min(hi, lo + 24))) | set(range(max(lo, hi - 24), hi))
    found |= {rng.randint(lo, hi - 1) for _ in range(24)}
    return [v for v in found if _holds(facts, v)]


def test_shl_and_low_mask_rows_hold_for_every_i32_value_and_their_precondition_matters():
    """Every operand range and residue sampled, every shift amount: the derived facts hold for
    the wrapped i32 result. The negative control keeps the operand's own scaled facts when the
    scaled range is not proven inside i32, and finds values that break them."""
    rng = random.Random(13)
    b = _builder()
    unproven_violations = 0
    for index, ((lo, hi), (m, r)) in enumerate(
        itertools.product(_SHL_RANGES, _SHL_CONGRUENCES)
    ):
        if (
            lo is not None
            and hi is not None
            and not any((v - r) % m == 0 for v in range(lo, min(hi, lo + m)))
        ):
            continue  # no value of this range has this residue
        x = _stated(b, f"x{index}", lo=lo, hi=hi, multiple_of=m, residue=r)
        x_facts = value_facts(b, x)
        xs = _samples(x_facts, rng)
        for c in range(32):
            shifted = value_facts(b, b.shl(x, _c(b, c)))
            masked = value_facts(b, b.land(x, _c(b, (1 << c) - 1)))
            naive = (x_facts[0].scaled(1 << c), x_facts[1].scaled(1 << c))
            for v in xs:
                assert _holds(shifted, _wrap32(v << c)), (lo, hi, m, r, c, v)
                assert _holds(masked, v & ((1 << c) - 1)), (lo, hi, m, r, c, v)
                if not _holds(naive, _wrap32(v << c)):
                    unproven_violations += 1
    assert unproven_violations > 0


# --- the loop-counter row ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (0, 256, 32, (Interval(0, 225), Congruence(32, 0))),
        (3, 100, 8, (Interval(3, 100), Congruence(8, 3))),
        (0, 16, 32, (Interval(0, 1), Congruence(32, 0))),  # one trip
        (16, 16, 8, (Interval(16, 17), Congruence(8, 0))),  # never runs
        (20, 4, 8, (Interval(20, 21), Congruence(8, 4))),  # never runs
    ],
)
def test_a_counter_with_int_bounds_has_the_tight_range_of_its_trips(
    start, stop, step, expected
):
    b = _builder()
    with b.scf_for(_c(b, start), _c(b, stop), _c(b, step)) as k:
        assert value_facts(b, k) == expected


def test_a_counter_up_to_a_runtime_length_minus_a_tile_proves_the_full_tile_by_identity():
    b = _builder()
    tk = 32
    ks = _runtime_length(b, "K", 65536, multiple_of=8)
    with b.scf_for(_c(b, 0), b.add(ks.value, _c(b, 1 - tk)), _c(b, tk)) as k:
        assert value_facts(b, k) == (
            Interval(0, ks._shifted(-(tk - 1))),
            Congruence(tk, 0),
        )
    with b.scf_for(_c(b, 0), ks.value, _c(b, tk)) as k:
        assert value_facts(b, k) == (Interval(0, ks), Congruence(tk, 0))
    two_steps = b.sub(b.add(ks.value, _c(b, 1)), _c(b, 8))
    with b.scf_for(_c(b, 0), two_steps, _c(b, tk)) as k:
        assert value_facts(b, k)[0] == Interval(0, ks._shifted(-7))


def test_any_other_stop_gives_its_own_upper_end_and_a_non_exact_step_is_unknown():
    b = _builder()
    p = b.param("p", I32)
    with b.scf_for(_c(b, 0), b.smin(p, _c(b, 100)), _c(b, 8)) as k:
        assert value_facts(b, k) == (Interval(0, 101), Congruence(8, 0))
    with b.scf_for(_c(b, 16), b.smin(p, _c(b, 10)), _c(b, 8)) as k:
        assert value_facts(b, k) == (Interval(16, 17), Congruence(8, 0))  # never runs
    with b.scf_for(_c(b, 0), p, _c(b, 8)) as k:
        assert value_facts(b, k) == (Interval(0, None), Congruence(8, 0))
    with b.scf_for(_c(b, 0), _c(b, 64), b.param("step", I32)) as k:
        assert value_facts(b, k) == UNKNOWN


def test_full_tile_proof_end_to_end_needs_no_check():
    """The counter of a loop over full tiles plus a per-lane offset inside the tile stays below
    the runtime length: no check, also when the walk rebuilt the length from its marker and the
    tensor bound holds the producer's object."""
    b = _builder()
    tk = 16
    producer_ks = _runtime_length(b, "K", 4096, multiple_of=8)
    with b.scf_for(_c(b, 0), b.add(producer_ks.value, _c(b, 1 - tk)), _c(b, tk)) as k:
        k_range, _ = value_facts(b, k)
    tile = k_range.plus(Interval(0, tk))
    graph = identity_graph((producer_ks,))
    assert _required_checks(graph, (tile,), (AxisBound.of(0, producer_ks),)) == ()
    rebuilt_ks = runtime_value_of(producer_ks.value)
    assert rebuilt_ks is not producer_ks and rebuilt_ks == producer_ks
    assert _required_checks(graph, (tile,), (AxisBound.of(0, rebuilt_ks),)) == ()


# --- the marker row and runtime ends ----------------------------------------------------------------


def test_a_marker_narrows_the_derived_facts_and_snaps_to_the_residue():
    b = _builder()
    length = _runtime_length(b, "L", 1 << 20, multiple_of=128)
    origin = b.mul(b.block_id_x(), _c(b, 128))
    guarded = b._assume_range(origin, lo=0, hi_value=length.value)
    assert value_facts(b, guarded) == (
        Interval(0, length._shifted(-127)),
        Congruence(128, 0),
    )
    stated_residue = _stated(b, "s", lo=0, hi=1000, multiple_of=8, residue=3)
    assert value_facts(b, stated_residue) == (Interval(3, 996), Congruence(8, 3))


def test_a_marker_that_contradicts_the_ir_raises_the_intersect_error():
    b = _builder(100)
    proven = b.add(b.thread_id_x(), _c(b, 100))  # [100, 200)
    marked = b._assume_range(proven, lo=0, hi=64)
    message = (
        "Interval.intersect: [100, 200) and [0, 64) have no value in common -- the two ranges "
        "state contradictory facts; check where each was derived"
    )
    with _raises(ValueError, message):
        value_facts(b, marked)


def test_a_marker_on_a_per_lane_value_keeps_it_per_lane():
    b = _builder()
    lane = b.mod(b.thread_id_x(), _c(b, 64))
    marked = b._assume_range(lane, lo=0, hi=64)
    split = split_coordinate(b, marked)
    assert split.uniform == () and split.lane is marked


def test_runtime_value_of_rebuilds_exactly_what_the_marker_states():
    b = _builder()
    p = b.param("p", I32)
    marked = b._assume_range(p, lo=0, hi=4097, multiple_of=8)
    assert runtime_value_of(marked) == RuntimeValue(
        marked, 0, Interval(0, 4097), Congruence(8, 0)
    )
    residue = b._assume_range(p, lo=0, hi=100, multiple_of=8, residue=3)
    assert runtime_value_of(residue)._congruence == Congruence(8, 3)
    assert runtime_value_of(p) is None
    assert runtime_value_of(b.thread_id_x()) is None
    assert runtime_value_of(7) is None
    with_runtime_end = b._assume_range(p, lo=0, hi_value=marked)
    assert runtime_value_of(with_runtime_end) is None


def test_runtime_value_of_takes_the_stated_envelope_not_the_derived_range():
    b = _builder()
    capped = b.smin(b.param("p", I32), _c(b, 100))  # derived [?, 101)
    marked = b._assume_range(capped, lo=0, hi=1001)
    assert value_facts(b, marked)[0] == Interval(0, 101)
    rebuilt = runtime_value_of(marked)
    assert rebuilt._envelope == Interval(0, 1001)
    producer = RuntimeValue(marked, 0, Interval(0, 1001), Congruence(1, 0))
    # an end of each kind in one call: no two-RuntimeValue error
    assert Interval(0, rebuilt).intersect(Interval(0, producer)) == Interval(0, rebuilt)


def test_a_marker_runtime_end_resolves_through_the_length_marker():
    b = _builder()
    ks = _runtime_length(b, "K", 4096)
    origin = b.block_id_x()
    guarded = b._assume_range(origin, lo=0, hi_value=ks.value, hi_plus=-3)
    assert value_facts(b, guarded)[0] == Interval(0, ks._shifted(-3))
    not_a_length = b.param("n", I32)
    bad = b._assume_range(origin, lo=0, hi_value=not_a_length)
    message = (
        f"IR range walk: the marker on {origin} states an end over {not_a_length}, which is not "
        "a runtime value's marker result -- a runtime end is the .value of a RuntimeValue (from "
        "runtime_value)"
    )
    with _raises(ValueError, message):
        value_facts(b, bad)


# --- the split ------------------------------------------------------------------------------------


def _parts(split: AffineCoord) -> tuple[int, list[Value], Value | None]:
    return split.const, [part.value for part in split.uniform], split.lane


def test_uniform_leaves_are_one_uniform_part():
    b = _builder()
    p = b.param("p", I32)
    bid = b.block_id_x()
    first_lane = b.readfirstlane(b.thread_id_x())
    for value in (p, bid, first_lane, b.sub(bid, p), b.mul(bid, _c(b, 128))):
        assert _parts(split_coordinate(b, value)) == (0, [value], None)
    with b.scf_for(_c(b, 0), p, _c(b, 16)) as k:
        split = split_coordinate(b, k)
        assert _parts(split) == (0, [k], None)
        assert (split.uniform[0].range, split.uniform[0].congruence) == (
            Interval(0, None),
            Congruence(16, 0),
        )


def test_thread_ids_loads_and_loop_carried_values_are_per_lane():
    b = _builder()
    tid = b.thread_id_x()
    ptr = b.param("p", PtrType(I32, "global"), readonly=True, noalias=True)
    loaded = b.global_load(ptr, b.block_id_x(), I32)
    first = b.readfirstlane(loaded)
    assert _parts(split_coordinate(b, tid)) == (0, [], tid)
    assert _parts(split_coordinate(b, loaded)) == (0, [], loaded)
    assert _parts(split_coordinate(b, first)) == (0, [first], None)
    with b.scf_for_iter(_c(b, 0), _c(b, 8), _c(b, 1), [("acc", b.block_id_x())]) as (
        _,
        (acc,),
    ):
        assert _parts(split_coordinate(b, acc)) == (0, [], acc)


def test_values_that_cannot_be_split_further_are_one_per_lane_part_with_their_derived_range():
    b = _builder(256)
    tid = b.thread_id_x()
    from_constant = b.sub(_c(b, 5), tid)
    capped = b.smin(b.sub(tid, _c(b, 3)), _c(b, 10))
    for value, derived in (
        (from_constant, Interval(-250, 6)),
        (capped, Interval(-3, 11)),
    ):
        split = split_coordinate(b, value)
        assert _parts(split) == (0, [], value)
        assert split.lane_range == derived


def test_an_origin_plus_a_lane_position_plus_a_constant_splits_into_three_parts():
    b = _builder(256)
    origin = b.mul(b.block_id_x(), _c(b, 64))
    lane = b.div(b.thread_id_x(), _c(b, 4))
    split = split_coordinate(b, b.add(b.add(origin, lane), _c(b, 3)))
    assert _parts(split) == (3, [origin], lane)
    assert split.uniform[0].congruence == Congruence(64, 0)
    assert split.lane_range == Interval(0, 64)


def test_adds_split_their_operands_unless_both_carry_a_lane_part():
    b = _builder()
    tid, bid, p = b.thread_id_x(), b.block_id_x(), b.param("p", I32)
    lane_a, lane_b = b.mod(tid, _c(b, 4)), b.div(tid, _c(b, 4))
    two_origins = b.add(b.add(bid, lane_a), p)
    assert _parts(split_coordinate(b, two_origins)) == (0, [bid, p], lane_a)
    both_lanes = b.add(b.add(bid, lane_a), lane_b)
    assert _parts(split_coordinate(b, both_lanes)) == (0, [], both_lanes)
    x = b.add(bid, lane_a)
    assert _parts(split_coordinate(b, b.sub(x, _c(b, 5)))) == (-5, [bid], lane_a)


def test_anti_example_a_product_of_a_sum_is_one_per_lane_part():
    b = _builder()
    mixed = b.mul(b.add(b.block_id_x(), b.thread_id_x()), _c(b, 2))
    assert _parts(split_coordinate(b, mixed)) == (0, [], mixed)


def test_a_constant_splits_into_the_constant_part_only():
    b = _builder()
    assert _parts(split_coordinate(b, 7)) == (7, [], None)
    assert _parts(split_coordinate(b, _c(b, -4))) == (-4, [], None)


def test_affine_coord_parts_need_ranges_and_one_lane_part():
    b = _builder()
    tid = b.thread_id_x()
    with _raises(
        ValueError,
        "AffineCoord lane part has no range -- every part needs an explicit Interval "
        "(UNBOUNDED if unknown)",
    ):
        AffineCoord(0, (), tid, None)
    with _raises(
        ValueError,
        "AffineCoord uniform part has no range -- every part needs an explicit Interval "
        "(UNBOUNDED if unknown)",
    ):
        AffineCoord(0, (UniformPart(b.block_id_x(), None, Congruence(1, 0)),))
    lane = AffineCoord(1, (), tid, Interval(0, 256))
    other = AffineCoord(2, (), b.lane_id(), UNBOUNDED)
    with _raises(
        ValueError,
        f"AffineCoord + AffineCoord: both operands have a lane part ({tid}, {other.lane}) -- a "
        "coordinate has one lane part; add the per-lane values first and split the sum with the "
        "walk",
    ):
        lane + other
    total = AffineCoord.of_int(4) + lane
    assert (total.const, total.lane, total.lane_range) == (5, tid, Interval(0, 256))


# --- values the same for the whole workgroup ------------------------------------------------------


def test_block_ids_parameters_and_their_arithmetic_are_the_same_for_the_workgroup():
    b = _builder()
    p = b.param("M", I32)
    same = [
        b.mul(b.block_id_y(), _c(b, 128)),
        b.add(b.mul(b.div(b.block_id_x(), _c(b, 8)), _c(b, 128)), _c(b, 64)),
        p,
        b._assume_range(p, lo=0, hi=4097),
        b.readfirstlane(b.block_id_x()),
        5,
    ]
    for value in same:
        assert leaf_varying_within_workgroup(b, value) is None


def test_a_wave_uniform_value_that_differs_between_waves_returns_its_thread_id_leaf():
    b = _builder()
    tid = b.thread_id_x()
    wave_origin = b.add(
        b.mul(b.readfirstlane(b.div(tid, _c(b, 64))), _c(b, 32)),
        b.mul(b.block_id_x(), _c(b, 128)),
    )
    assert leaf_varying_within_workgroup(b, wave_origin) is tid


@pytest.mark.parametrize(
    "attrs, same",
    [
        ({"readonly": True, "noalias": True}, True),
        ({"addr_space": "constant"}, True),
        ({"readonly": True}, False),
        ({}, False),
    ],
)
def test_a_length_loaded_per_workgroup_needs_memory_no_pointer_of_the_kernel_writes(
    attrs, same
):
    b = _builder()
    lens = b.param("lens", PtrType(I32, "global"), **attrs)
    loaded = b.global_load(lens, b.block_id_z(), I32)
    length = b.readfirstlane(loaded)
    assert leaf_varying_within_workgroup(b, length) is (None if same else loaded)


def test_a_load_at_an_index_that_differs_between_waves_is_the_leaf_returned():
    b = _builder()
    lens = b.param("lens", PtrType(I32, "global"), readonly=True, noalias=True)
    loaded = b.global_load(lens, b.div(b.thread_id_x(), _c(b, 64)), I32)
    assert leaf_varying_within_workgroup(b, b.readfirstlane(loaded)) is loaded


def test_an_unknown_op_and_a_loop_carried_value_are_their_own_leaf():
    b = _builder()
    lane = b.lane_id()
    assert leaf_varying_within_workgroup(b, b.add(b.block_id_x(), lane)) is lane
    with b.scf_for_iter(_c(b, 0), _c(b, 8), _c(b, 1), [("acc", b.block_id_x())]) as (
        k,
        (acc,),
    ):
        assert leaf_varying_within_workgroup(b, acc) is acc
        assert leaf_varying_within_workgroup(b, k) is None


# --- the workgroup size, types, purity --------------------------------------------------------------


def test_a_thread_id_needs_an_explicit_workgroup_size():
    b = _builder(None)
    value = b.add(b.thread_id_x(), _c(b, 1))
    message = (
        f"IR range walk: {value} depends on gpu.thread_id but the kernel states no "
        'max_workgroup_size -- set b.kernel.attrs["max_workgroup_size"] to the launch block size '
        "right after creating the builder"
    )
    with _raises(ValueError, message):
        value_facts(b, value)
    with _raises(ValueError, message):
        split_coordinate(b, value)


def test_the_workgroup_size_is_recorded_once_and_never_silently_changed():
    b = _builder(128)
    tid = b.thread_id_x()
    assert WORKGROUP_SIZE_RECORD not in b.kernel.attrs
    value_facts(b, b.block_id_x())
    assert WORKGROUP_SIZE_RECORD not in b.kernel.attrs  # no thread id reached yet
    value_facts(b, tid)
    assert b.kernel.attrs[WORKGROUP_SIZE_RECORD] == 128
    b.kernel.attrs["max_workgroup_size"] = 256
    message = (
        "IR range walk: max_workgroup_size is now 256 but thread-id ranges were already derived "
        "with 128 -- set it once, before positioning windows from thread ids"
    )
    with _raises(ValueError, message):
        value_facts(b, tid)
    with _raises(ValueError, message):
        split_coordinate(b, b.block_id_x())


@pytest.mark.parametrize(
    "make", [lambda b: b.const_i64(3), lambda b: b.param("f", F32)]
)
def test_a_value_that_is_not_i32_is_refused(make):
    b = _builder()
    value = make(b)
    message = (
        f"IR range walk: {value} has type {value.type} -- a coordinate is an i32 value"
    )
    for walk in (value_facts, split_coordinate, leaf_varying_within_workgroup):
        with _raises(TypeError, message):
            walk(b, value)
    assert value.type in (I64, F32)


def test_the_walk_emits_nothing_and_answers_the_same_twice():
    b = _builder()
    ks = _runtime_length(b, "K", 4096, multiple_of=8)
    with b.scf_for(_c(b, 0), b.add(ks.value, _c(b, -15)), _c(b, 16)) as k:
        value = b.add(b.add(k, b.div(b.thread_id_x(), _c(b, 4))), _c(b, 2))
    before = _op_count(b)
    facts = value_facts(b, value)
    split = split_coordinate(b, value)
    leaf = leaf_varying_within_workgroup(b, value)
    assert _op_count(b) == before
    assert value_facts(b, value) == facts
    again = split_coordinate(b, value)
    assert _parts(again) == _parts(split)
    assert (again.lane_range, again.lane_congruence) == (
        split.lane_range,
        split.lane_congruence,
    )
    assert [(p.range, p.congruence) for p in again.uniform] == [
        (p.range, p.congruence) for p in split.uniform
    ]
    assert leaf_varying_within_workgroup(b, value) is leaf
