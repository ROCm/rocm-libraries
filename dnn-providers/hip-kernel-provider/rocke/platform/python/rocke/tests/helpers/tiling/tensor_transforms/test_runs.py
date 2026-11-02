# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the run queries of tensor_transforms and the vector accesses they keep wide.

A run is one access standing for ``width`` consecutive values of one graph input. Each query
is checked against exhaustive enumeration at every start residue modulo ``width``: the query
says yes exactly when every start of that residue gives a contiguous address (or one validity
answer for the whole run), across affine, halo, slice, unmerge and swizzle graphs. The access
tests pin when a vector keeps one compare, when it takes its first and last element plus the
span of its valid elements, and that the span is exact on a GPU.
"""

from __future__ import annotations

import itertools

import pytest

from rocke.core.ir import F16, I32, IRBuilder, Value
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    embed,
    emit_accesses,
    emit_addresses,
    freeze,
    inputs,
    make_transform_graph,
    offset,
    outputs,
    slice_window,
    transform_address_of,
    unmerge,
    xor,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    AffineCoord,
    AxisBound,
    Congruence,
    Interval,
    UniformPart,
    VectorRun,
    _required_checks,
)
from rocke.helpers.tiling.tensor_transforms.runs import (
    address_contiguous_run,
    checks_monotone_over_run,
    validity_uniform_over_run,
)
from rocke.tests.gpu import current_device_arch

from ._harness import (
    WAVE,
    RuntimeTensor,
    builder,
    count,
    raises,
    run_lanes,
    runtime_length,
)
from .test_emit import _conv, _ConvAccess

WIDTH = 8


def _c(b: IRBuilder, value: int) -> Value:
    return b.const_i32(value)


def _start(residue: int, upper: int) -> AffineCoord:
    """A run start known only to be ``residue mod WIDTH`` and in ``[residue, upper)``."""
    part = UniformPart(
        Value("origin", I32), Interval(0, upper - residue), Congruence(WIDTH, 0)
    )
    return AffineCoord(residue, (part,))


def _starts(residue: int, upper: int) -> range:
    return range(residue, upper, WIDTH)


def _contiguous_everywhere(graph, tensor, residue: int, upper: int) -> bool:
    for x0 in _starts(residue, upper):
        addresses = [
            transform_address_of(graph, tensor, (x0 + e,)) for e in range(WIDTH)
        ]
        if any(b - a != 1 for a, b in itertools.pairwise(addresses)):
            return False
    return True


def _swizzle(xor_read):
    """``k -> (row, col)``, ``col`` swizzled by ``xor_read`` (a coordinate built from ``row``),
    on a ``(4, 64)`` tensor with rows 64 apart."""
    return make_transform_graph(
        inputs(k=256),
        unmerge("k", into=("row", "col"), lengths=(4, 64)),
        *xor_read,
        xor(("sw", "col"), into=("sw_out", "col_out")),
        embed(("row", "sw_out"), into="row_out", coeffs=(1, 0)),
        outputs("row_out", "col_out"),
    )


CONTIGUITY_GRAPHS = {
    "affine": (identity_graph((256,)), TensorDesc((256,), (1,), F16)),
    "halo": (
        make_transform_graph(
            inputs(k=256), offset("k", into="h", shift=-3), outputs("h")
        ),
        TensorDesc((256,), (1,), F16),
    ),
    "slice": (
        make_transform_graph(
            inputs(k=200), slice_window("k", into="s", begin=5, end=205), outputs("s")
        ),
        TensorDesc((256,), (1,), F16),
    ),
    "unmerge dividing": (
        make_transform_graph(
            inputs(k=256),
            unmerge("k", into=("a", "b"), lengths=(16, 16)),
            outputs("a", "b"),
        ),
        TensorDesc((16, 16), (32, 1), F16),
    ),
    "unmerge not dividing": (
        make_transform_graph(
            inputs(k=240),
            unmerge("k", into=("a", "b"), lengths=(20, 12)),
            outputs("a", "b"),
        ),
        TensorDesc((20, 12), (16, 1), F16),
    ),
    "swizzle": (
        _swizzle((embed(("row",), into="sw", coeffs=(8,)),)),
        TensorDesc((4, 64), (64, 1), F16),
    ),
    "xor of a constant 1": (
        _swizzle((freeze(into="sw", value=1),)),
        TensorDesc((4, 64), (64, 1), F16),
    ),
}


@pytest.mark.parametrize("case", sorted(CONTIGUITY_GRAPHS))
def test_contiguity_matches_exhaustive_enumeration_at_every_start_residue(case):
    graph, tensor = CONTIGUITY_GRAPHS[case]
    upper = next(iter(graph.inputs.values())) - WIDTH + 1
    run = VectorRun(0, WIDTH)
    answers = []
    for residue in range(WIDTH):
        coords = (_start(residue, upper),)
        answer = address_contiguous_run(graph, tensor, run, coords)
        assert answer == _contiguous_everywhere(graph, tensor, residue, upper), residue
        answers.append(answer)
    expected_aligned = case not in ("unmerge not dividing", "xor of a constant 1")
    assert answers[0] == expected_aligned
    if case.startswith(("unmerge", "swizzle", "xor")):
        assert not any(answers[1:])


def _uniform_everywhere(graph, length: int, residue: int, upper: int) -> bool:
    """Every run of that residue has one validity answer (all elements valid, or none)."""
    for x0 in _starts(residue, upper):
        valid = []
        for e in range(WIDTH):
            coords = graph.resolve((x0 + e,))
            ok = all(0 <= coords[o] < length for o in graph.outputs)
            for name, domain in graph.inputs.items():
                ok = ok and (domain is None or 0 <= coords[name] < domain)
            valid.append(ok)
        if len(set(valid)) > 1:
            return False
    return True


@pytest.mark.parametrize("shift", [0, -3, 2])
@pytest.mark.parametrize("length", [64, 60, 61])
def test_uniform_validity_matches_exhaustive_enumeration_at_every_start_residue(
    shift, length
):
    graph = make_transform_graph(
        inputs(k=None), offset("k", into="h", shift=shift), outputs("h")
    )
    upper = 80
    run = VectorRun(0, WIDTH)
    for residue in range(WIDTH):
        coords = (_start(residue, upper),)
        ranges = (coords[0].range.plus(Interval(0, WIDTH)),)
        checks = _required_checks(graph, ranges, (AxisBound.of(0, length),))
        answer = validity_uniform_over_run(graph, checks, run, coords)
        assert answer == _uniform_everywhere(graph, length, residue, upper), residue
        assert checks_monotone_over_run(graph, checks, run, coords)


def test_a_k_tail_check_is_uniform_when_the_unmerge_length_is_a_multiple_of_the_width():
    graph = make_transform_graph(
        inputs(k=None),
        unmerge("k", into=("a", "b"), lengths=(10, 16)),
        outputs("a", "b"),
    )
    run = VectorRun(0, WIDTH)
    coords = (_start(0, 200),)
    (check,) = _required_checks(
        graph,
        (coords[0].range.plus(Interval(0, WIDTH)),),
        (AxisBound.of(0, 10), AxisBound.of(0, 16)),
    )
    assert check.coordinate == "k" and check.hi == 160
    assert validity_uniform_over_run(graph, (check,), run, coords)
    assert not validity_uniform_over_run(graph, (check,), run, (_start(4, 200),))


# --- vector accesses ------------------------------------------------------------------------------


def _vector_case(b: IRBuilder, length, lane_extra: int = 0):
    """8-wide vectors along a runtime-length axis: each lane's vector starts at ``8 * lane``
    (plus ``lane_extra``)."""
    tid = b.thread_id_x()
    x = b.mul(tid, _c(b, WIDTH))
    if lane_extra:
        x = b.add(x, _c(b, lane_extra))
    tensor = RuntimeTensor((length,), (1,), F16)
    return tid, x, identity_graph((length,)), tensor


def test_a_vector_against_a_length_stated_a_multiple_of_its_width_is_one_compare():
    for multiple, uniform in ((WIDTH, True), (1, False)):
        b = builder()
        length = runtime_length(b, "L", at_most=4096, multiple_of=multiple)
        _, x, graph, tensor = _vector_case(b, length)
        (access,) = emit_accesses(
            b, graph, tensor, [(x,)], vector_axis=0, vector_width=WIDTH
        )
        compares = count(b, "arith.cmp", pred="ult")
        if uniform:
            assert compares == 1 and access.valid_span is None
        else:
            assert compares == 2 and access.valid_span is not None


def test_the_conv_k_vector_is_accepted_at_a_multiple_of_its_width_and_refused_one_past_it():
    graph, tensor = _conv(2, 6, 6, 64)
    b = builder()
    conv = _ConvAccess(b, graph)
    (access,) = emit_accesses(
        b, graph, tensor, [(conv.m, conv.k_lane)], vector_axis=1, vector_width=WIDTH
    )
    assert access.address.lane is not None
    b = builder()
    conv = _ConvAccess(b, graph)
    shifted = b.add(conv.k_lane, _c(b, 1))
    with raises(
        ValueError,
        "vector along input 1 ('k') of width 8 is not contiguous at coordinate 'k' (it may "
        "start at 1 mod 8, so the unmerge into ('y', 'x', 'c') can split the vector) -- use a "
        "narrower vector, or start it at a multiple of 8 (the walk reads that fact from how the "
        "coordinate was computed; a runtime value the IR cannot show it for is stated with "
        "runtime_value(..., multiple_of=...))",
    ):
        emit_accesses(
            b, graph, tensor, [(conv.m, shifted)], vector_axis=1, vector_width=WIDTH
        )
    assert count(b, "arith.cmp", pred="ult") == 0


def test_an_odd_f16_tail_takes_its_span_from_the_coordinate_table_without_reemitting_it():
    """The span of a vector straddling the length reuses the coordinates the address already
    emitted: the validity adds no multiply or division."""

    def build(entry):
        b = builder()
        length = runtime_length(b, "L", at_most=4096)
        _, x, graph, tensor = _vector_case(b, length)
        if entry is emit_accesses:
            return b, entry(b, graph, tensor, [(x,)], vector_axis=0, vector_width=WIDTH)
        return b, entry(b, graph, tensor, [(x,)])

    alone, _ = build(emit_addresses)
    b, (access,) = build(emit_accesses)
    assert access.valid_span is not None
    for name in ("arith.mul", "arith.div", "arith.mod", "arith.umul_hi_i32"):
        assert count(b, name) == count(alone, name), name


# --- on the GPU -------------------------------------------------------------------------------


@pytest.mark.parametrize("lane_extra", [0, -3])
def test_gpu_a_vector_straddling_a_runtime_length_has_an_exact_valid_span(lane_extra):
    """8-wide f16 vectors at ``8 * lane + lane_extra`` against a runtime length, odd lengths
    included (the tail ends mid-dword): a vector wholly inside is valid with the full span, a
    straddling one is invalid with exactly its valid elements as its span, and one wholly
    outside has an empty span."""
    current_device_arch()

    def build(b):
        length = runtime_length(b, "L", at_most=4096)
        _, x, graph, tensor = _vector_case(b, length, lane_extra)
        (access,) = emit_accesses(
            b, graph, tensor, [(x,)], vector_axis=0, vector_width=WIDTH
        )
        first, end = access.valid_span
        return [access.valid, first, end]

    for length in (1, 7, 8, 101, 260, 4096):
        stored = run_lanes(build, scalars={"L": length})[0]
        for t in range(WAVE):
            start = WIDTH * t + lane_extra
            valid = [0 <= start + e < length for e in range(WIDTH)]
            assert stored[t, 0] == all(valid), (length, t)
            first, end = stored[t, 1], stored[t, 2]
            if any(valid):
                inside = [e for e in range(WIDTH) if valid[e]]
                assert (first, end) == (inside[0], inside[-1] + 1), (length, t)
            else:
                assert end <= first, (length, t)


def test_gpu_the_conv_k_vector_span_is_exact_against_the_k_tail():
    """An 8-wide vector along the conv K axis (C = 64, a multiple of the width): only the K tail
    and the halo decide it, and every element agrees with the graph evaluated in Python.
    """
    current_device_arch()
    graph, tensor = _conv(1, 3, 3, 64, y=1, x=1, pad=0)
    m_length, k_length = graph.inputs["m"], graph.inputs["k"]

    def build(b):
        conv = _ConvAccess(b, graph, tile_m=16)
        k = b.add(conv.k_lane, _c(b, 32))
        (access,) = emit_accesses(
            b, graph, tensor, [(conv.m, k)], vector_axis=1, vector_width=WIDTH
        )
        assert access.valid_span is None
        return [access.valid]

    stored = run_lanes(build, n_blocks=1)
    for t in range(WAVE):
        m, k = t // 4, (t % 4) * 8 + 32
        expected = m < m_length and k + WIDTH <= k_length
        assert stored[0, t, 0] == expected
