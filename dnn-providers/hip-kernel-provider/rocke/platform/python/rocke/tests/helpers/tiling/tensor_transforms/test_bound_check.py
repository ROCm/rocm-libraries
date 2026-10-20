# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the one bound predicate of tensor_transforms, ``emit_bound_check``.

Each check is a fold of its per-wave side into two saturated ends and one unsigned compare. The
host tests run the emitted IR with each op's integer width (``test_emit``'s interpreter, which
fails on any no-signed-wrap op that overflows) and compare every answer with the Python
``lo <= value < hi``. The op-count tests pin which width each fold takes and where the work
lands; the ISA tests pin what the gfx90a compiler makes of it (64-bit compares only where a
per-wave value needs 64 bits). The GPU tests run the same checks on whatever GPU is present.
"""

from __future__ import annotations

import itertools
import random

import pytest

from rocke.core.ir import F16, I32, I64, IRBuilder, Value
from rocke.helpers.tiling.tensor_transforms import emit_accesses, emit_bound_check
from rocke.helpers.tiling.tensor_transforms.emit import (
    _emit_bound_check,
    _fold_bound_check,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    UNBOUNDED,
    AffineCoord,
    AxisBound,
    Congruence,
    Interval,
    UniformPart,
    _required_checks,
)
from rocke.helpers.tiling.tensor_transforms.walk import (
    runtime_value_of,
    split_coordinate,
)
from rocke.tests.gpu import current_device_arch

from ._harness import (
    WAVE,
    RuntimeTensor,
    builder,
    count,
    gfx90a_isa,
    isa_count,
    ops,
    raises,
    run_lanes,
    runtime_length,
    store_values,
)
from .test_emit import _run

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
EDGE_VALUES = (I32_MIN, -1, 0, 1, 1023, 1024, 1025, I32_MAX)


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _host(valid: Value, *leaves: tuple[Value, int]) -> bool:
    """The validity the emitted IR computes for these ``(leaf, value)`` pairs, run on the host."""
    return bool(_run(valid, {id(leaf): v for leaf, v in leaves}, {}))


def _readers(b: IRBuilder, value: Value) -> list:
    return [op for op in ops(b) if any(v is value for v in op.operands)]


# --- one compare, or the fold -------------------------------------------------------------------


def test_a_proven_access_emits_no_compare_and_a_required_check_exactly_one():
    """An access whose ranges prove it in bounds emits nothing; one that can leave the tensor
    on one axis emits exactly one unsigned compare."""
    b = builder()
    graph, tensor = identity_graph((100, 64)), RuntimeTensor((100, 64), (64, 1), F16)
    tid = b.thread_id_x()
    origin = b._assume_range(b.mul(b.block_id_x(), _c(b, 16)), lo=0, hi=100)
    row = b.add(origin, b.div(tid, _c(b, 4)))
    col = b.mul(b.mod(tid, _c(b, 4)), _c(b, 16))
    edge, inner = emit_accesses(
        b, graph, tensor, [(row, col), (b.div(tid, _c(b, 4)), col)]
    )
    assert inner.valid is None and edge.valid is not None
    assert count(b, "arith.cmp", pred="ult") == 1
    block = next(op for op in ops(b) if op.name == "gpu.block_id").results[0]
    for blk, t in itertools.product(range(7), range(WAVE)):
        assert _host(edge.valid, (tid, t), (block, blk)) == (blk * 16 + t // 4 < 100)


def test_an_unbounded_uniform_value_folds_in_64_bits_and_never_subtracts_it_in_32():
    """A kernel parameter (wave-uniform, range unknown) against [1, 1025): the fold sign-extends
    it once and every subtraction that reads it is a 64-bit one."""
    b = builder()
    u = b.param("u", I32)
    valid = emit_bound_check(b, u, 1, 1025)
    (widen,) = _readers(b, u)
    assert widen.name == "arith.sext"
    subs = _readers(b, widen.results[0])
    assert subs and all(
        op.name == "arith.sub" and op.results[0].type == I64 for op in subs
    )
    for v in EDGE_VALUES:
        assert _host(valid, (u, v)) == (1 <= v < 1025), v


def test_a_lone_uniform_value_against_zero_and_a_length_is_one_unsigned_compare():
    """``0 <= u < 1024`` and ``0 <= u < M`` (``M`` a runtime length below 2^31) are one
    ``icmp ult`` on the value itself, whatever its range: no extension, no subtraction.
    """
    for runtime in (False, True):
        b = builder()
        u = b.param("u", I32)
        hi = runtime_length(b, "M") if runtime else 1024
        before = len(ops(b))
        valid = emit_bound_check(b, u, 0, hi)
        new = ops(b)[before:]
        assert [op.name for op in new if op.name != "arith.constant"] == ["arith.cmp"]
        assert valid.op.attrs["pred"] == "ult" and valid.op.operands[0] is u
        assert count(b, "arith.sext") == count(b, "arith.sub") == 0
        if not runtime:
            for v in (I32_MIN, -1, 0, 1023, 1024, I32_MAX):
                assert _host(valid, (u, v)) == (0 <= v < 1024), v
    # A constant on the value, or a lower end other than 0, takes the fold.
    b = builder()
    u = b.param("u", I32)
    emit_bound_check(b, b.add(u, _c(b, 1)), 0, 1024)
    emit_bound_check(b, u, 2, 1024)
    assert count(b, "arith.sext") == 2


def test_the_single_compare_and_the_fold_agree_on_the_host():
    """The fold, forced, answers what the single compare answers on every edge value."""
    for hi in (1, 2, 1024, I32_MAX):
        b = builder()
        u = b.param("u", I32)
        single = emit_bound_check(b, u, 0, hi)
        folded = _fold_bound_check(b, split_coordinate(b, u), 0, hi)
        assert single.op.operands[0] is u and folded is not single
        for v in (*EDGE_VALUES, hi - 1, hi, hi + 1 if hi < I32_MAX else hi):
            assert _host(single, (u, v)) == _host(folded, (u, v)) == (0 <= v < hi), (
                hi,
                v,
            )


def test_a_tile_origin_proven_by_the_guard_marker_emits_no_compare():
    """An origin marked ``0 <= origin < M`` (what the tile guard returns), a multiple of the
    tile, with ``M`` stated a multiple of the tile: the whole tile is proven inside ``[0, M)``.
    """
    b = builder()
    tile = 16
    m = runtime_length(b, "M", multiple_of=tile)
    origin = b._assume_range(b.mul(b.block_id_x(), _c(b, tile)), lo=0, hi_value=m.value)
    row = b.add(origin, b.mod(b.thread_id_x(), _c(b, tile)))
    tensor = RuntimeTensor((m, 64), (64, 1), F16)
    (access,) = emit_accesses(b, identity_graph((m, 64)), tensor, [(row, 0)])
    assert access.valid is None
    assert (
        count(b, "arith.cmp") == count(b, "arith.smin") == count(b, "arith.smax") == 0
    )


def test_a_value_is_checked_before_its_marker_and_proven_after_it():
    """Checking the raw value emits the compare; the same call on the marker's result reads the
    stated fact back and proves the check away (so a caller checks first, then marks).
    """
    b = builder()
    origin = b.mul(b.block_id_x(), _c(b, 16))
    assert emit_bound_check(b, origin, 0, 1024) is not None
    compares = count(b, "arith.cmp")
    assert compares == 1
    marked = b._assume_range(origin, lo=0, hi=1024)
    assert emit_bound_check(b, marked, 0, 1024) is None
    assert count(b, "arith.cmp") == compares


def _origin_with_lane(b: IRBuilder, m, upper):
    """A hand-split coordinate ``origin + lane``: the origin's range is ``[0, upper)``."""
    origin = b._assume_range(b.mul(b.block_id_x(), _c(b, 16)), lo=0, hi_value=m.value)
    lane = b.mod(b.thread_id_x(), _c(b, 16))
    return AffineCoord(
        0,
        (UniformPart(origin, Interval(0, upper), Congruence(16, 0)),),
        lane,
        Interval(0, 16),
        Congruence(1, 0),
    )


def test_a_runtime_bounded_origin_keeps_its_upper_check_in_32_bits_and_unbounded_takes_64():
    """An origin in ``[0, M)`` plus a lane in ``[0, 16)`` against ``[0, M)`` keeps the upper
    check, folded with 32-bit scalar ops only; the same origin known only to be ``>= 0`` takes
    the 64-bit pre-clamp."""
    for wide in (False, True):
        b = builder()
        m = runtime_length(b, "M")
        coord = _origin_with_lane(b, m, None if wide else m)
        (check,) = _required_checks(
            identity_graph((None,)), (coord.range,), (AxisBound.of(0, m),)
        )
        assert check.lo is None and check.hi is m
        valid = _emit_bound_check(b, coord, check.lo, check.hi)
        store_values(b, [valid], per_block=False)
        isa = gfx90a_isa(b)
        assert (count(b, "arith.sext") > 0) == wide
        assert (isa_count(isa, r"v_cmp\w*_[iu]64\w*") > 0) == wide, isa


def test_an_unbounded_uniform_origin_against_a_runtime_bound_compares_lanes_in_32_bits():
    """The per-lane compare is 32-bit even when the per-wave side needs 64 bits: no 64-bit
    unsigned compare and no 64-bit extension of a per-lane value."""
    b = builder()
    m = runtime_length(b, "M")
    origin = b.param("origin", I32)
    lane = b.mod(b.thread_id_x(), _c(b, 16))
    valid = emit_bound_check(b, b.add(origin, lane), 0, m)
    assert valid.op.operands[0].type == I32
    assert all(op.operands[0] is not lane for op in ops(b) if op.name == "arith.sext")
    store_values(b, [valid], per_block=False)
    isa = gfx90a_isa(b)
    assert isa_count(isa, r"v_cmp\w*_u64\w*") == 0
    assert isa_count(isa, r"v_cmp\w*_u32\w*") >= 1


def test_two_uniform_parts_are_summed_in_32_bits_when_proven_and_in_64_otherwise():
    """``begin + length - 1`` held as two per-wave parts: summed with one 32-bit add when their
    ranges fit, else each part sign-extended and added in 64 bits; once per call."""
    for stated in (True, False):
        b = builder()
        begin, length = b.param("begin", I32), b.param("length", I32)
        r = Interval(0, 1 << 20) if stated else UNBOUNDED
        coord = AffineCoord(
            -1,
            (
                UniformPart(begin, r, Congruence(1, 0)),
                UniformPart(length, r, Congruence(1, 0)),
            ),
        )
        valid = _emit_bound_check(b, coord, 0, 4096)
        adds = [
            op for op in ops(b) if op.name == "arith.add" and op.operands[0] is begin
        ]
        if stated:
            assert len(adds) == 1 and count(b, "arith.sext") == 0
        else:
            assert not adds
            wide = [
                op
                for op in ops(b)
                if op.name == "arith.add" and op.results[0].type == I64
            ]
            assert len(wide) == 1 and count(b, "arith.sext") == 2
        values = (
            (I32_MIN, -1, 0, 1, I32_MAX) if not stated else (0, 1, 4095, (1 << 20) - 1)
        )
        for x, y in itertools.product(values, values):
            assert _host(valid, (begin, x), (length, y)) == (0 <= x + y - 1 < 4096), (
                x,
                y,
            )


def test_a_lower_only_check_is_one_subtract_and_one_compare_never_the_single_compare():
    """With no upper end: a per-lane coordinate is one 32-bit subtract and one compare against
    ``L - a`` (no ``smax``); a per-wave value folds in 64 bits when unbounded and in 32 when its
    stated range fits."""
    b = builder()
    halo = b.sub(b.div(b.thread_id_x(), _c(b, 4)), _c(b, 1))
    before = len(ops(b))
    valid = emit_bound_check(b, halo, 0, None)
    new = [op.name for op in ops(b)[before:] if op.name != "arith.constant"]
    assert new == ["arith.sub", "arith.cmp"]
    assert valid.op.operands[1].op.attrs["value"] == 15
    for unbounded in (True, False):
        b = builder()
        u = b.param("u", I32)
        if not unbounded:
            u = b._assume_range(u, lo=-(1 << 20), hi=1 << 20)
        valid = emit_bound_check(b, u, 0, None)
        assert valid.op.operands[0] is not u
        assert (count(b, "arith.sext") > 0) == unbounded
        for v in (-(1 << 20), -1, 0, 1, (1 << 20) - 1):
            assert _host(valid, (b.get_param("u"), v)) == (v >= 0)


def test_checking_nothing_or_an_empty_range_or_a_non_value_raises_before_emitting():
    b = builder()
    u = b.param("u", I32)
    before = len(ops(b))
    with raises(
        ValueError,
        "emit_bound_check of '%u' got lo=None and hi=None -- that checks nothing; pass the "
        "end(s) to check, or emit no check",
    ):
        emit_bound_check(b, u, None, None)
    with raises(
        ValueError,
        "bound of '%u' is empty: [4, 4) -- an always-empty window cannot be read",
    ):
        emit_bound_check(b, u, 4, 4)
    with raises(
        TypeError,
        "emit_bound_check takes a plain i32 IR value or an int, got Value of type i64",
    ):
        emit_bound_check(b, b.sext(u, I64), 0, 4)
    with raises(
        TypeError, "emit_bound_check takes a plain i32 IR value or an int, got float"
    ):
        emit_bound_check(b, 1.0, 0, 4)
    assert len(ops(b)) == before + 1  # only the test's own sext


def _rebase_cases(b: IRBuilder, lane: Value):
    """``c - lane`` (``c = 40``) and ``smin(lane - 3, 10)``: per-lane values whose walked range
    starts below 0, with their Python value per lane."""
    return (
        (b.sub(_c(b, 40), lane), lambda t: 40 - t),
        (b.smin(b.sub(lane, _c(b, 3)), _c(b, 10)), lambda t: min(t - 3, 10)),
    )


def test_a_per_lane_value_starting_below_zero_is_checked_exactly_after_its_rebase():
    for hi_runtime in (False, True):
        b = builder()
        lane = b.thread_id_x()
        hi = runtime_length(b, "M", at_most=4096) if hi_runtime else 12
        for value, truth in _rebase_cases(b, lane):
            for lo in (-2, 0, 5):
                valid = emit_bound_check(b, value, lo, hi)
                leaves = [(hi.value.op.operands[0], 12)] if hi_runtime else []
                for t in range(WAVE):
                    expected = lo <= truth(t) < 12
                    assert _host(valid, (lane, t), *leaves) == expected, (lo, t)


def test_a_hand_split_coordinate_whose_lane_starts_below_zero_is_refused():
    b = builder()
    lane = b.thread_id_x()
    coord = AffineCoord(0, (), lane, Interval(-3, 61), Congruence(1, 0))
    with raises(
        ValueError,
        f"_emit_bound_check: lane part {lane} has range [-3, 61), starting below 0 -- pass the "
        "coordinate through _emit_coord_table or emit_bound_check, which rebase it to start at 0",
    ):
        _emit_bound_check(b, coord, 0, 32)


def test_a_lane_range_plus_constant_spread_reaching_two_to_the_31_raises():
    """Two accesses share an unbounded per-wave value: the 64-bit pre-clamp covers the lane
    range plus the spread of their constants, which must stay below 2^31."""
    b = builder()
    u = b.param("u", I32)
    spread = b.mul(b.thread_id_x(), _c(b, 1 << 25))
    x = b.add(u, spread)
    tensor = RuntimeTensor((I32_MAX,), (1,), F16)
    with raises(
        ValueError,
        "bound check of 'axis0' needs lane range + constant spread below 2^31, got 2113929217 "
        "+ 1073741824 -- narrow the lane range or split the accesses into separate calls",
    ):
        emit_accesses(
            b, identity_graph((I32_MAX,)), tensor, [(x,), (b.add(x, _c(b, 1 << 30)),)]
        )


# --- 64-bit work and vector compares in the gfx90a ISA --------------------------------------------


@pytest.mark.parametrize("wide", [False, True])
def test_eight_accesses_against_a_runtime_bound_compare_once_per_folded_bound(wide):
    """Eight accesses (rows ``origin + lane + 8 i``, each an 8-wide vector along the columns)
    against a runtime row count: at most one vector compare per access, none of them 64-bit;
    with an unbounded per-wave origin the 64-bit compares are one pair per (origin, end), not
    one per access."""
    b = builder()
    m = runtime_length(b, "M")
    tid = b.thread_id_x()
    if wide:
        origin = b.param("origin", I32)
    else:
        origin = b._assume_range(
            b.mul(b.block_id_x(), _c(b, 64)), lo=0, hi_value=m.value
        )
    row = b.add(origin, b.div(tid, _c(b, 8)))
    col = b.mul(b.mod(tid, _c(b, 8)), _c(b, 8))
    accesses = [(b.add(row, _c(b, 8 * i)) if i else row, col) for i in range(8)]
    tensor = RuntimeTensor((m, 64), (64, 1), F16)
    results = emit_accesses(
        b, identity_graph((m, 64)), tensor, accesses, vector_axis=1, vector_width=8
    )
    assert all(a.valid_span is None for a in results)
    store_values(b, [a.valid for a in results], per_block=False)
    isa = gfx90a_isa(b)
    wide_compares = isa_count(isa, r"v_cmp\w*_[iu]64\w*")
    narrow_compares = isa_count(isa, r"v_cmp\w*_[iu]32\w*")
    assert narrow_compares <= 8, isa
    if wide:
        assert 0 < wide_compares <= 4, isa
    else:
        assert wide_compares == 0, isa


# --- on the GPU -------------------------------------------------------------------------------


def _paths_agree_values(hi: int) -> list[int]:
    rng = random.Random(1234)
    values = set(range(-(1 << 16), 1 << 16))
    for centre in (0, hi, I32_MIN, I32_MAX):
        values.update(
            v for v in range(centre - 1024, centre + 1025) if I32_MIN <= v <= I32_MAX
        )
    values.update(rng.randint(I32_MIN, I32_MAX) for _ in range(1 << 20))
    return sorted(values)


def test_gpu_the_single_compare_and_the_fold_agree_on_every_value():
    """One kernel computes, for one wave-uniform value per workgroup, both the single compare
    and the forced fold, for int upper ends 1, 2, 1024 and 2^31 - 1 and a runtime length; the
    two answers are equal and equal ``0 <= u < hi`` on every value."""
    current_device_arch()
    import numpy as np

    int_ends = (1, 2, 1024, I32_MAX)
    values = sorted(set().union(*(_paths_agree_values(h) for h in (*int_ends, 1024))))

    def build(b):
        m = runtime_length(b, "M")
        u = b.readfirstlane(b.global_load(b.get_param("values"), b.block_id_x(), I32))
        out = []
        for hi in (*int_ends, m):
            out.append(emit_bound_check(b, u, 0, hi))
            out.append(_fold_bound_check(b, split_coordinate(b, u), 0, hi))
        return out

    array = np.array(values, dtype=np.int64)
    for m in (1, 1024, I32_MAX):
        stored = run_lanes(
            build,
            scalars={"M": m},
            buffers={"values": array},
            n_blocks=len(values),
            per_block=True,
        )
        for j, hi in enumerate((*int_ends, m)):
            expected = (array >= 0) & (array < hi)
            assert (stored[:, 2 * j] == expected).all(), (m, hi)
            assert (stored[:, 2 * j + 1] == expected).all(), (m, hi)


def test_gpu_unbounded_and_single_compare_checks_are_exact_on_edge_values():
    current_device_arch()
    import numpy as np

    values = np.array(EDGE_VALUES, dtype=np.int64)

    def build(b):
        m = runtime_length(b, "M")
        u = b.readfirstlane(b.global_load(b.get_param("values"), b.block_id_x(), I32))
        return [
            emit_bound_check(b, u, 1, 1025),
            emit_bound_check(b, u, 0, 1024),
            emit_bound_check(b, u, 0, m),
        ]

    for m in (1023, 1024):
        stored = run_lanes(
            build,
            scalars={"M": m},
            buffers={"values": values},
            n_blocks=len(values),
            per_block=True,
        )
        assert (stored[:, 0] == ((values >= 1) & (values < 1025))).all()
        assert (stored[:, 1] == ((values >= 0) & (values < 1024))).all()
        assert (stored[:, 2] == ((values >= 0) & (values < m))).all()


SIDE_VALUES = (I32_MIN, -1, 0, 1, I32_MAX)


@pytest.mark.parametrize("extent", [1, 4, 64])
@pytest.mark.parametrize("wide", [True, False])
def test_gpu_a_lower_only_check_is_exact_for_every_lane_at_both_widths(extent, wide):
    """``lo <= s + lane`` with no upper end, ``s`` and ``lo`` at the extremes of their stated
    ranges (the full i32 range for the 64-bit fold, a narrower one for the 32-bit fold), every
    lane of ``[0, extent)``."""
    current_device_arch()
    import numpy as np

    limit = 1 << 31 if wide else 1 << 29
    side_values = (-limit, -1, 0, 1, limit - 1)
    pairs = list(itertools.product(side_values, side_values))
    s_values = np.array([s for s, _ in pairs], dtype=np.int64)
    lo_values = np.array([lo for _, lo in pairs], dtype=np.int64)

    def build(b):
        blk = b.block_id_x()
        s = b.readfirstlane(b.global_load(b.get_param("s"), blk, I32))
        lo_raw = b.readfirstlane(b.global_load(b.get_param("lo"), blk, I32))
        if not wide:
            s = b._assume_range(s, lo=-limit, hi=limit)
        lo_end = runtime_value_of(b._assume_range(lo_raw, lo=-limit, hi=limit))
        lane = b.land(b.thread_id_x(), _c(b, extent - 1))
        return [emit_bound_check(b, b.add(s, lane), lo_end, None)]

    stored = run_lanes(
        build, buffers={"s": s_values, "lo": lo_values}, n_blocks=len(pairs)
    )
    lanes = np.arange(WAVE) & (extent - 1)
    expected = (s_values[:, None] + lanes[None, :]) >= lo_values[:, None]
    assert (stored[:, :, 0] == expected).all()


def _uniform(b: IRBuilder, name: str) -> Value:
    """This workgroup's entry of the read-only buffer ``name``, as a wave-uniform value."""
    return b.readfirstlane(b.global_load(b.get_param(name), b.block_id_x(), I32))


def _stated(b: IRBuilder, value: Value, lo: int, hi: int):
    """``value`` behind a marker stating ``lo <= value < hi``, as its ``RuntimeValue``."""
    return runtime_value_of(b._assume_range(value, lo=lo, hi=hi))


def test_gpu_runtime_ends_including_empty_and_inverted_windows_are_exact():
    """``lo <= s + lane < hi`` with both ends runtime values, through ``emit_bound_check`` and
    through ``_required_checks`` + ``_emit_bound_check`` with a two-sided ``AxisBound``: exact for
    every lane, and an empty (``hi == lo``) or inverted (``hi < lo``) window accepts nothing,
    including a window wider than 2^31."""
    current_device_arch()
    import numpy as np

    cases = []
    for s in (I32_MIN, -100, -1, 0, 5, 1000, I32_MAX - 63):
        for lo, hi in (
            (0, 64),
            (3, 40),
            (-50, 7),
            (10, 10),
            (20, 3),
            (I32_MIN, I32_MAX),
            (-5, -5),
            (s + 7, s + 9) if I32_MIN <= s + 7 and s + 9 <= I32_MAX else (0, 1),
        ):
            cases.append((s, lo, hi))
    s_values, lo_values, hi_values = (np.array(c, dtype=np.int64) for c in zip(*cases))

    def build(b):
        s = _uniform(b, "s")
        lo = _stated(b, _uniform(b, "lo"), I32_MIN, 1 << 31)
        hi = _stated(b, _uniform(b, "hi"), I32_MIN, 1 << 31)
        x = b.add(s, b.thread_id_x())
        coord = split_coordinate(b, x)
        (check,) = _required_checks(
            identity_graph((None,)), (coord.range,), (AxisBound.of(lo, hi),)
        )
        return [
            emit_bound_check(b, x, lo, hi),
            _emit_bound_check(b, coord, check.lo, check.hi),
        ]

    stored = run_lanes(
        build,
        buffers={"s": s_values, "lo": lo_values, "hi": hi_values},
        n_blocks=len(cases),
    )
    x = s_values[:, None] + np.arange(WAVE)[None, :]
    expected = (lo_values[:, None] <= x) & (x < hi_values[:, None])
    assert (stored[:, :, 0] == expected).all()
    assert (stored[:, :, 1] == expected).all()
    assert not stored[lo_values >= hi_values].any()


@pytest.mark.parametrize("wide", [False, True])
def test_gpu_the_saturated_fold_is_exact_at_its_clamp_edges(wide):
    """Accesses ``s + lane + c`` for several constants ``c`` against ``[0, L)`` (``L`` a runtime
    length): with ``s`` unbounded the ends are pre-clamped in 64 bits once and finished per
    constant in 32 bits, and ``lo - s`` / ``L - s`` reach -2^31, 2^31 - 1, 2^31 and beyond; with
    ``s`` and ``L`` stated inside 2^29 the fold is 32-bit, exact at the edges of those ranges.
    """
    current_device_arch()
    import numpy as np

    from rocke.helpers.tiling.tensor_transforms import emit_access_validity

    limit = 1 << 31 if wide else 1 << 29
    constants = (0, 5, 37)
    s_choices = (-limit, -limit + 1, -64, -1, 0, 1, 63, limit - 64, limit - 1)
    l_choices = (1, 64, 100, limit - 1)
    cases = [(s, ln) for s in s_choices for ln in l_choices]
    s_values, l_values = (np.array(c, dtype=np.int64) for c in zip(*cases))

    def build(b):
        s = _uniform(b, "s")
        if not wide:
            s = b._assume_range(s, lo=-limit, hi=limit)
        length = _stated(b, _uniform(b, "L"), 1, limit)
        x = b.add(s, b.land(b.thread_id_x(), _c(b, 31)))
        tensor = RuntimeTensor((length,), (1,), F16)
        accesses = [(b.add(x, _c(b, c)) if c else x,) for c in constants]
        valid = emit_access_validity(b, identity_graph((length,)), tensor, accesses)
        assert (count(b, "arith.sext") > 0) == wide
        return list(valid)

    stored = run_lanes(
        build, buffers={"s": s_values, "L": l_values}, n_blocks=len(cases)
    )
    lanes = np.arange(WAVE) & 31
    for j, c in enumerate(constants):
        x = s_values[:, None] + lanes[None, :] + c
        expected = (x >= 0) & (x < l_values[:, None])
        assert (stored[:, :, j] == expected).all(), c


def test_gpu_two_uniform_parts_are_summed_exactly_at_the_i32_extremes():
    current_device_arch()
    import numpy as np

    extremes = (I32_MIN, I32_MIN + 1, -1, 0, 1, 4096, I32_MAX)
    cases = list(itertools.product(extremes, extremes))
    begin_values, length_values = (np.array(c, dtype=np.int64) for c in zip(*cases))

    def build(b):
        begin, length = _uniform(b, "begin"), _uniform(b, "length")
        coord = AffineCoord(
            -1,
            (
                UniformPart(begin, UNBOUNDED, Congruence(1, 0)),
                UniformPart(length, UNBOUNDED, Congruence(1, 0)),
            ),
        )
        return [_emit_bound_check(b, coord, 0, 4096)]

    stored = run_lanes(
        build,
        buffers={"begin": begin_values, "length": length_values},
        n_blocks=len(cases),
        per_block=True,
    )
    total = begin_values + length_values - 1
    assert (stored[:, 0] == ((total >= 0) & (total < 4096))).all()


def test_gpu_a_per_lane_value_starting_below_zero_is_checked_exactly_on_every_lane():
    current_device_arch()
    import numpy as np

    ends = ((-2, 12), (0, 12), (5, 40), (0, None))

    def build(b):
        m = runtime_length(b, "M", at_most=4096)
        lane = b.thread_id_x()
        out = []
        for value, _ in _rebase_cases(b, lane):
            for lo, hi in ends:
                out.append(emit_bound_check(b, value, lo, hi))
            out.append(emit_bound_check(b, value, 0, m))
        return out

    stored = run_lanes(build, scalars={"M": 12})
    t = np.arange(WAVE)
    slot = 0
    for truth in (40 - t, np.minimum(t - 3, 10)):
        for lo, hi in (*ends, (0, 12)):
            expected = (truth >= lo) & (True if hi is None else truth < hi)
            assert (stored[0, :, slot] == expected).all(), (lo, hi)
            slot += 1
