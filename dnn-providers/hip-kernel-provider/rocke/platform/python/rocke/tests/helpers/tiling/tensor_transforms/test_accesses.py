# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the access entries of tensor_transforms: ``emit_accesses``, its buffer-checked
sibling and ``emit_access_validity``.

They pin what an entry accepts (one tuple of plain coordinates per access, in ``inputs(...)``
order; a vector along one graph input), where its bounds come from (the tensor's own lengths,
never a parameter), which sides its ``valid`` covers, what the package door exports, and that
the range marker a check proves away is stated only by the tiling layer's checked callers. The
GPU test runs a convolution input tile and compares every access's validity with the graph
evaluated in Python.
"""

from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path

from rocke.core.ir import F16, F32, I32, I64, IRBuilder, Value
from rocke.helpers.tiling import tensor_transforms as tt
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    emit_access_validity,
    emit_accesses,
    emit_addresses,
    emit_coords,
)
from rocke.helpers.tiling.tensor_transforms.emit import (
    _plan_validity,
    _split_access,
    emit_accesses_except_buffer_checked,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    AxisBound,
    CheckSide,
    Side,
    _required_checks,
)
from rocke.tests.gpu import current_device_arch

from ._harness import (
    WAVE,
    RuntimeTensor,
    builder,
    count,
    ops,
    raises,
    run_lanes,
    runtime_length,
)
from .test_emit import _conv, _ConvAccess, _guarded, _halo_xor_case, _run

ROCKE_DIR = Path(__file__).resolve().parents[4]
TILING_DIR = ROCKE_DIR / "helpers" / "tiling"
LIBRARY_DIR = ROCKE_DIR.parents[2] / "library"


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _host(valid: Value, *leaves: tuple[Value, int]) -> bool:
    return bool(_run(valid, {id(leaf): v for leaf, v in leaves}, {}))


# --- which sides valid covers ---------------------------------------------------------------------


def _mixed_constants(b: IRBuilder):
    """An origin in ``[0, 65)`` plus a lane in ``[0, 4)`` against a length of 70, at constants
    0 and 12: the upper side is proven at 0 and fails at 12."""
    origin = b._assume_range(b.block_id_x(), lo=0, hi=65)
    x = b.add(origin, b.land(b.thread_id_x(), _c(b, 3)))
    return x, [(x,), (b.add(x, _c(b, 12)),)]


def test_a_buffer_checked_side_proven_for_one_constant_and_failing_for_another_emits_nothing():
    b = builder()
    _, accesses = _mixed_constants(b)
    graph, tensor = identity_graph((70,)), TensorDesc((70,), (1,), F16)
    upper = frozenset({CheckSide("axis0", Side.UPPER)})
    results = emit_accesses_except_buffer_checked(
        b, graph, tensor, accesses, buffer_checked_sides=upper
    )
    assert [a.valid for a in results] == [None, None]
    assert count(b, "arith.cmp") == 0


def test_buffer_checked_sides_drop_exactly_the_named_sides_and_refuse_any_other():
    """``valid`` keeps every side ``emit_accesses`` keeps except the named ones; a side no
    access reports, or an empty set, is refused."""
    graph, tensor = identity_graph((64,)), TensorDesc((64,), (1,), F32)
    for dropped, truth in (
        (None, lambda v: 0 <= v < 64),
        (Side.UPPER, lambda v: v >= 0),
        (Side.LOWER, lambda v: v < 64),
    ):
        b = builder()
        s, lane = b.param("s", I32), b.thread_id_x()
        x = b.add(s, lane)
        if dropped is None:
            (access,) = emit_accesses(b, graph, tensor, [(x,)])
        else:
            (access,) = emit_accesses_except_buffer_checked(
                b,
                graph,
                tensor,
                [(x,)],
                buffer_checked_sides=frozenset({CheckSide("axis0", dropped)}),
            )
        for sv, t in itertools.product((-70, -3, 0, 10, 60, 70), (0, 5, 63)):
            assert _host(access.valid, (s, sv), (lane, t)) == truth(sv + t), (
                dropped,
                sv,
                t,
            )

    b = builder()
    _, accesses = _mixed_constants(b)
    tensor = TensorDesc((70,), (1,), F16)
    before = len(ops(b))
    with raises(
        ValueError,
        "buffer_checked_sides names the lower side of 'axis0', which required_checks did not "
        "report as failing for any access of this call -- only a side that can fail can be "
        "left to the buffer",
    ):
        emit_accesses_except_buffer_checked(
            b,
            identity_graph((70,)),
            tensor,
            accesses,
            buffer_checked_sides=frozenset({CheckSide("axis0", Side.LOWER)}),
        )
    with raises(ValueError, "buffer_checked_sides is empty -- call emit_accesses"):
        emit_accesses_except_buffer_checked(
            b, identity_graph((70,)), tensor, accesses, buffer_checked_sides=frozenset()
        )
    assert len(ops(b)) == before


def _imports_of(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def test_the_buffer_checked_entry_is_off_the_door_and_reached_only_from_safe_access():
    assert "emit_accesses_except_buffer_checked" not in tt.__all__
    assert not hasattr(tt, "emit_accesses_except_buffer_checked")
    name = "emit_accesses_except_buffer_checked"
    users = sorted(
        str(path.relative_to(TILING_DIR))
        for path in TILING_DIR.rglob("*.py")
        if "tmp" not in path.relative_to(TILING_DIR).parts and _imports_of(path, name)
    )
    assert set(users) <= {"safe_access.py"}, users


# --- what an entry accepts ------------------------------------------------------------------------


def test_every_malformed_access_is_refused_by_every_entry_before_anything_is_emitted():
    b = builder()
    graph = identity_graph((64, 16))
    tensor = TensorDesc((64, 16), (16, 1), F32)
    tid = b.thread_id_x()
    entries = {
        "emit_accesses": lambda acc: emit_accesses(b, graph, tensor, acc),
        "sibling": lambda acc: emit_accesses_except_buffer_checked(
            b,
            graph,
            tensor,
            acc,
            buffer_checked_sides=frozenset({CheckSide("axis0", Side.UPPER)}),
        ),
        "emit_coords": lambda acc: emit_coords(b, graph, acc[0]),
        "emit_addresses": lambda acc: emit_addresses(b, graph, tensor, acc),
        "emit_access_validity": lambda acc: emit_access_validity(b, graph, tensor, acc),
    }
    before = len(ops(b))
    for name, entry in entries.items():
        with raises(
            TypeError,
            "access 0 is a list, not a tuple -- give each access as a tuple of coordinates in "
            "inputs(...) order",
        ):
            entry([[tid, 0]])
        with raises(
            TypeError,
            "access 0 is a dict, not a tuple -- give each access as a tuple of coordinates in "
            "inputs(...) order",
        ):
            entry([{"axis0": tid, "axis1": 0}])
        with raises(
            TypeError,
            "coordinate 1 ('axis1') of access 0 is a float -- a coordinate is a plain i32 IR "
            "value or an int",
        ):
            entry([(tid, 1.0)])
        with raises(
            ValueError,
            "access 0 gives 1 coordinates, the graph has 2 inputs ('axis0', 'axis1') -- give "
            "one coordinate per input, in inputs(...) order",
        ):
            entry([(tid,)])
        assert len(ops(b)) == before, name
    for entry in (emit_accesses, emit_addresses, emit_access_validity):
        with raises(
            ValueError,
            "graph has 2 outputs ('axis0', 'axis1') but the tensor has rank 1",
        ):
            entry(b, graph, TensorDesc((64,), (1,), F32), [(tid, 0)])
    assert len(ops(b)) == before


def test_the_vector_keywords_are_checked_and_address_the_first_element():
    b = builder()
    graph = identity_graph((64, 64))
    tensor = TensorDesc((64, 64), (64, 1), F16)
    tid = b.thread_id_x()
    row = b.div(tid, _c(b, 8))
    col = b.mul(b.mod(tid, _c(b, 8)), _c(b, 8))
    with raises(ValueError, "vector_width must be >= 1, got 0"):
        emit_accesses(b, graph, tensor, [(row, col)], vector_width=0)
    with raises(
        ValueError,
        "vector_width 8 needs vector_axis -- give the index, in inputs(...) order, of the input "
        "the vector runs along",
    ):
        emit_accesses(b, graph, tensor, [(row, col)], vector_width=8)
    with raises(
        ValueError,
        "vector_axis 2 is not an input index -- the graph has 2 inputs ('axis0', 'axis1')",
    ):
        emit_accesses(b, graph, tensor, [(row, col)], vector_axis=2, vector_width=8)
    (vector,) = emit_accesses(
        b, graph, tensor, [(row, col)], vector_axis=1, vector_width=8
    )
    (single,) = emit_addresses(b, graph, tensor, [(row, col)])
    assert vector.valid is None and vector.valid_span is None
    for t in range(WAVE):
        first = _run(vector.address.lane, {id(tid): t}, {})
        assert (
            first == _run(single.lane, {id(tid): t}, {}) == (t // 8) * 64 + (t % 8) * 8
        )


def test_vector_axis_is_a_graph_input_index_not_a_tile_axis_index():
    """On a batched ``(G, M, K)`` tensor (K contiguous) a rank-2 tile's K axis is graph input 2:
    ``vector_axis=2`` is one contiguous vector along K; ``vector_axis=1`` runs along M.
    """
    b = builder()
    graph = identity_graph((4, 64, 64))
    tensor = TensorDesc((4, 64, 64), (4096, 64, 1), F16)
    tid = b.thread_id_x()
    access = (
        b.block_id_x(),
        b.div(tid, _c(b, 8)),
        b.mul(b.mod(tid, _c(b, 8)), _c(b, 8)),
    )
    (vector,) = emit_accesses(b, graph, tensor, [access], vector_axis=2, vector_width=8)
    assert vector.address.lane is not None
    with raises(
        ValueError,
        "vector along input 1 ('axis1') of width 8 is not contiguous at coordinate 'axis1' "
        "(its elements are 64 apart in memory) -- use a narrower vector, or start it at a "
        "multiple of 8 (the walk reads that fact from how the coordinate was computed; a "
        "runtime value the IR cannot show it for is stated with "
        "runtime_value(..., multiple_of=...))",
    ):
        emit_accesses(b, graph, tensor, [access], vector_axis=1, vector_width=8)


def test_the_bounds_are_the_tensors_own_lengths_and_no_parameter():
    """The same accesses against a tensor with an int length and with a runtime length give the
    checks ``AxisBound.of(0, length)`` gives; no entry takes a bounds parameter."""
    b = builder()
    m = runtime_length(b, "M", at_most=4096)
    tid = b.thread_id_x()
    row = b.add(b.param("r", I32), b.div(tid, _c(b, 8)))
    accesses = [(row, b.mod(tid, _c(b, 8))), (b.add(row, _c(b, 3)), 4)]
    split = [_split_access(b, a) for a in accesses]
    for length in (100, m):
        tensor = RuntimeTensor((length, 8), (8, 1), F16)
        graph = identity_graph((length, 8))
        plans = _plan_validity(graph, tensor, split, None, None)
        bounds = (AxisBound.of(0, length), AxisBound.of(0, 8))
        for coords, plan in zip(split, plans):
            expected = _required_checks(graph, tuple(c.range for c in coords), bounds)
            assert tuple(pc.check for pc in plan) == expected
            assert expected and expected[0].hi == length
    for entry in (
        emit_accesses,
        emit_accesses_except_buffer_checked,
        emit_access_validity,
    ):
        assert "bounds" not in inspect.signature(entry).parameters


def test_the_package_door_exports_the_access_entry_and_no_internal_value():
    exported = set(tt.__all__)
    assert {
        "emit_accesses",
        "Access",
        "RuntimeValue",
        "BoundCheck",
        "CheckReason",
    } <= exported
    assert {"emit_bound_check", "emit_access_validity"} <= exported
    internal = {
        "emit_accesses_except_buffer_checked",
        "RecomputedCoords",
        "CoordinateSource",
        "RunSpec",
        "Interval",
        "Congruence",
        "AxisBound",
        "CheckSide",
        "identity_graph",
        "AffineCoord",
        "address_contiguous_run",
        "validity_uniform_over_run",
        "checks_monotone_over_run",
    }
    assert not internal & exported
    assert not any(hasattr(tt, name) for name in internal)
    assert "_stepped" not in inspect.signature(emit_accesses).parameters


# The range marker states a fact the IR range walk trusts; only callers that checked (or proved)
# it may state it. Those callers are the tiling layer's runtime_value, exit_if_tile_outside and
# slice_tensor, plus the tests of core and of this package.
_MARKER_CALLERS = {"runtime_value", "exit_if_tile_outside", "slice_tensor"}


def _unchecked_marker_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_assume_range"
            and function not in _MARKER_CALLERS
        ):
            found.append(f"{path}:{node.lineno} in {function}")
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def test_the_range_marker_is_stated_only_by_its_checked_callers():
    allowed_tests = (
        ROCKE_DIR / "tests" / "core",
        ROCKE_DIR.parent.parent / "tests" / "core",
        Path(__file__).parent,
    )
    roots = [ROCKE_DIR, LIBRARY_DIR]
    unchecked = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if any(path.is_relative_to(t) for t in allowed_tests):
                continue
            if "tmp" in path.relative_to(root).parts:
                continue
            unchecked.extend(_unchecked_marker_calls(path))
    assert not unchecked, unchecked


def test_emit_accesses_emits_the_shared_coordinate_chain_once():
    """The validity reads the same coordinate table as the address: the conv graph's unmerge
    divisions appear exactly as often as for the addresses alone."""

    def build(entry):
        b = builder()
        graph, tensor = _conv(2, 6, 6, 3)
        conv = _ConvAccess(b, graph)
        accesses = [
            (conv.m, b.add(conv.k_lane, _c(b, k)) if k else conv.k_lane)
            for k in (0, 1, 30)
        ]
        entry(b, graph, tensor, accesses)
        return b

    alone = build(emit_addresses)
    both = build(emit_accesses)
    for name in ("arith.umul_hi_i32", "arith.lshr", "arith.div", "arith.mod"):
        assert count(both, name) == count(alone, name), name
    assert count(both, "arith.cmp", pred="ult") > 0


def test_a_halo_column_fed_into_an_xor_is_valid_exactly_on_the_lanes_inside_the_tensor():
    """The swizzle's halo lanes (``c < 0``) are masked by the column's lower check, as any halo
    is; the other lanes are valid."""
    b = builder()
    graph, tensor, access = _halo_xor_case(b)
    (entry,) = emit_accesses(b, graph, tensor, [access])
    tid = next(op.results[0] for op in ops(b) if op.name == "gpu.thread_id")
    for t in range(WAVE):
        assert _host(entry.valid, (tid, t)) == (t % 16 >= 4), t


# --- on the GPU -------------------------------------------------------------------------------


def test_gpu_a_conv_input_tile_with_a_halo_and_ragged_edges_is_valid_exactly_where_enumerated():
    """A 2 x 5 x 7 x 3 input, a 3 x 3 filter with pad 1: rows past ``N * Ho * Wo`` (the ragged M
    tile), columns past ``Y * X * C`` (the K tail) and halo taps outside ``H`` / ``W`` on both
    sides are invalid; every other access is valid."""
    current_device_arch()
    n, h, w, c = 2, 5, 7, 3
    graph, tensor = _conv(n, h, w, c)
    m_length, k_length = graph.inputs["m"], graph.inputs["k"]
    constants = tuple(range(8))
    n_blocks = -(-m_length // 16)

    def build(b):
        conv = _ConvAccess(b, graph)
        accesses = [
            (conv.m, b.add(conv.k_lane, _c(b, k)) if k else conv.k_lane)
            for k in constants
        ]
        results = emit_accesses(b, graph, tensor, accesses)
        assert all(r.valid is not None for r in results)
        return [r.valid for r in results]

    stored = run_lanes(build, n_blocks=n_blocks)
    for blk, t in itertools.product(range(n_blocks), range(WAVE)):
        for j, k in enumerate(constants):
            m, kk = blk * 16 + t // 4, (t % 4) * 8 + k
            coords = graph.resolve((m, kk))
            expected = (
                m < m_length
                and kk < k_length
                and 0 <= coords["hi"] < h
                and 0 <= coords["wi"] < w
            )
            assert stored[blk, t, j] == expected, (blk, t, k)


def test_an_unguarded_origin_is_refused_and_a_guarded_one_is_accepted():
    """The validity reads the same table as the address, so an access the address refuses (an
    unbounded origin into an unmerge) is refused by every entry."""
    graph, tensor = _conv(2, 6, 6, 3)
    b = builder()
    m = b.add(b.mul(b.block_id_x(), _c(b, 16)), b.div(b.thread_id_x(), _c(b, 4)))
    with raises(
        ValueError,
        "unmerge of 'm' needs its dividend in [0, 2^31) for magic division, but its range is "
        "[0, +inf) -- bound the origin with exit_if_tile_outside(b, origin, length), or state a "
        "trusted range with runtime_value(b, v, at_most=...)",
    ):
        emit_accesses(b, graph, tensor, [(m, 0)])
    b = builder()
    origin = _guarded(b, b.mul(b.block_id_x(), _c(b, 16)), graph.inputs["m"])
    (access,) = emit_accesses(b, graph, tensor, [(b.add(origin, b.thread_id_x()), 0)])
    assert access.valid is not None and access.address.uniform[0].value.type in (
        I32,
        I64,
    )
