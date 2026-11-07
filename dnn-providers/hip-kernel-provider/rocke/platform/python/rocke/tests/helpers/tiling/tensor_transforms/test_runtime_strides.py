# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for runtime strides and slice begins in the structured address.

A runtime stride is an i32 ``Value`` with stated facts, here a kernel parameter: a packed
``A[M, K]`` whose row stride is the runtime ``K``, or a batch stride stated at least a runtime
sequence length times the row size. A
slice's begin is the tensor's start in its parent, one int or per-wave value per axis. Both
reach the address through the stride fold, and the tests pin where each part lands:

- a uniform part times a runtime stride is one i64 multiply in its input set's term; a per-lane
  part times it is one i32 multiply per call, its range following the stride's stated range;
- a constant times a runtime stride cannot be the int constant: the call's least constant joins
  the uniform term, and what each access has above it is its ``uniform_delta``;
- a slice's begin is folded like a per-wave origin part: its constant into the int constant on an
  int-stride axis, every other product into one uniform term keyed by no graph input, i32 or
  i64 by its range, so a slice may start past ``2^31`` elements; a per-lane begin is refused.

Every emitted address is run on the host, part by part with each op's integer width, and
compared with ``transform_address_of`` given the launch values. The GPU tests at the end run
the same coordinates on the GPU present and compare every lane's parts with the golden.
"""

from __future__ import annotations

import itertools
import re

import pytest

from rocke.core.intexpr import product_of
from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType, Value
from rocke.core.ir_print import print_ir
from rocke.helpers.tiling.descriptors import TensorDesc
from rocke.helpers.tiling.tensor_transforms import (
    address_of,
    emit_accesses,
    emit_addresses,
    transform_address_of,
)
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.tests.gpu import current_device_arch

from ._harness import RuntimeTensor, run_lanes
from .test_emit import _address, _builder, _guarded, _leaf, _ops, _varies_per_lane

WAVE = 64
I32_END = 1 << 31


def _raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def _new_ops(b: IRBuilder, before: int) -> list:
    """The ops emitted since ``before`` ops existed (the call under test)."""
    return _ops(b)[before:]


def _muls(ops: list, ty, *, per_lane: bool) -> list:
    return [
        op
        for op in ops
        if op.name == "arith.mul"
        and op.results[0].type == ty
        and _varies_per_lane(op.results[0]) == per_lane
    ]


def _part_values(address) -> list[Value]:
    """Every emitted part of an address, in a fixed order: uniform terms, delta, lane."""
    parts = [term.value for term in address.uniform]
    if address.uniform_delta is not None:
        parts.append(address.uniform_delta)
    if address.lane is not None:
        parts.append(address.lane)
    return parts


# --- runtime strides ----------------------------------------------------------------------------

# Register constants along M: the rows a lane reads in one call.
ROW_CONSTANTS = (0, 16, 48)


def _packed_rows(b: IRBuilder, *, k_max: int, dtype=F32):
    """A packed ``A[256, K]`` (row stride the runtime ``K``) and its row / column coordinates:
    row ``origin + tid / 4`` (origin a guarded per-wave value), column ``(tid % 4) * 8``.
    """
    k = b.param("K", I32, ge=1, le=k_max)
    tensor = TensorDesc((256, k), (k, 1), dtype)
    tid = b.thread_id_x()
    origin = _guarded(b, b.mul(b.block_id_x(), b.const_i32(16)), 256)
    row = b.add(origin, b.div(tid, b.const_i32(4)))
    col = b.mul(b.mod(tid, b.const_i32(4)), b.const_i32(8))
    return k, tensor, row, col


def _row_accesses(b: IRBuilder, row: Value, col: Value) -> list[tuple]:
    return [(b.add(row, b.const_i32(c)) if c else row, col) for c in ROW_CONSTANTS]


@pytest.mark.parametrize("k_max", [4096, 1 << 20])
def test_a_runtime_stride_costs_one_i64_uniform_multiply_one_lane_multiply_and_a_delta(
    k_max,
):
    b = _builder()
    k, tensor, row, col = _packed_rows(b, k_max=k_max)
    graph = identity_graph((256, k))
    accesses = _row_accesses(b, row, col)
    before = len(_ops(b))
    addresses = emit_addresses(b, graph, tensor, accesses)
    ops = _new_ops(b, before)

    # The uniform part times K: one i64 multiply for the call, in the term of input 0.
    (wide,) = _muls(ops, I64, per_lane=False)
    for address in addresses:
        (term,) = address.uniform
        assert term.inputs == frozenset({0}) and term.value is wide.results[0]
    # The per-lane row times K: one i32 multiply shared by every access.
    (lane_mul,) = _muls(ops, I32, per_lane=True)
    assert any(v is k for v in lane_mul.operands)
    # Each constant above the least one is its access's own wave-uniform delta, on the SALU.
    deltas = [address.uniform_delta for address in addresses]
    assert deltas[0] is None and addresses[0].uniform_delta_range is None
    for address, c in zip(addresses[1:], ROW_CONSTANTS[1:]):
        assert address.uniform_delta.type == I32
        assert not _varies_per_lane(address.uniform_delta)
        assert address.uniform_delta_range == (c, c * k_max + 1)
    assert len(_muls(ops, I32, per_lane=False)) == len(ROW_CONSTANTS) - 1
    # The constant along the runtime-stride axis is not in the int constant.
    assert all(address.const == 0 for address in addresses)
    # The lane range follows the stride's stated range: row lane 15 times K, plus column 24.
    for address in addresses:
        assert address.lane_range == (0, 15 * k_max + 24 + 1)

    tid, bx = _leaf(b, "thread_id"), _leaf(b, "bx")
    k_param = b.get_param("K")
    for k_int, blk, t in itertools.product((1, 37, 64, 65), range(3), range(WAVE)):
        leaves = {id(tid): t, id(bx): blk, id(k_param): k_int}
        for address, c in zip(addresses, ROW_CONSTANTS):
            position = (blk * 16 + t // 4 + c, (t % 4) * 8)
            expected = transform_address_of(graph, tensor, position, [(k, k_int)])
            assert _address(address, leaves) == expected, (k_int, blk, t, c)


def test_the_least_constant_on_a_runtime_stride_axis_joins_the_uniform_term():
    """Constants (-1, 2, 5): -1 * K joins the uniform term once, the other accesses carry 3 * K
    and 6 * K as their deltas."""
    b = _builder()
    k, tensor, row, col = _packed_rows(b, k_max=4096)
    graph = identity_graph((256, k))
    constants = (-1, 2, 5)
    accesses = [(b.add(row, b.const_i32(c)), col) for c in constants]
    addresses = emit_addresses(b, graph, tensor, accesses)
    least_terms = {id(a.uniform[0].value) for a in addresses}
    assert len(least_terms) == 1
    assert [a.uniform_delta_range for a in addresses] == [
        None,
        (3, 3 * 4096 + 1),
        (6, 6 * 4096 + 1),
    ]
    tid, bx = _leaf(b, "thread_id"), _leaf(b, "bx")
    for k_int, blk, t in itertools.product((1, 65), range(2), range(WAVE)):
        leaves = {id(tid): t, id(bx): blk, id(b.get_param("K")): k_int}
        for address, c in zip(addresses, constants):
            position = (blk * 16 + t // 4 + c, (t % 4) * 8)
            expected = transform_address_of(graph, tensor, position, [(k, k_int)])
            assert _address(address, leaves) == expected


def test_a_delta_past_32_bits_is_i64():
    """Rows with no per-lane part and K up to 2^31 - 1: 2 * K leaves i32."""
    b = _builder()
    k = b.param("K", I32, ge=1, le=I32_END - 1)
    tensor = TensorDesc((4, k), (k, 1), F32)
    origin = _guarded(b, b.block_id_x(), 2)
    graph = identity_graph((4, k))
    addresses = emit_addresses(
        b, graph, tensor, [(origin, 0), (b.add(origin, b.const_i32(2)), 0)]
    )
    assert addresses[0].uniform_delta is None
    assert addresses[1].uniform_delta.type == I64
    assert addresses[1].uniform_delta_range == (2, 2 * (I32_END - 1) + 1)


def test_the_lane_multiply_of_a_runtime_stride_does_not_depend_on_the_k_loop():
    """Inside a K loop, the per-lane row times K is one multiply for the call, and it reads only
    values defined before the loop: it is the same on every trip."""
    b = _builder()
    k, tensor, row, col = _packed_rows(b, k_max=4096)
    graph = identity_graph((256, k))
    loop_values = set()
    with b.scf_for(b.const_i32(0), k, b.const_i32(32)) as k0:
        before = len(_ops(b))
        accesses = [
            (b.add(row, b.const_i32(c)) if c else row, b.add(k0, col))
            for c in ROW_CONSTANTS
        ]
        emit_addresses(b, graph, tensor, accesses)
        loop_ops = _new_ops(b, before)
        loop_values = {id(r) for op in loop_ops for r in op.results} | {id(k0)}
    lane_muls = _muls(loop_ops, I32, per_lane=True)
    assert len(lane_muls) == 1
    assert not any(id(v) in loop_values for v in lane_muls[0].operands)


def test_a_vector_along_a_runtime_stride_axis_is_not_contiguous():
    b = _builder()
    k, tensor, row, col = _packed_rows(b, k_max=4096)
    graph = identity_graph((256, k))
    with _raises(
        ValueError,
        "vector along input 0 ('axis0') of width 4 is not contiguous at coordinate 'axis0' "
        "(runtime stride) -- use a narrower vector, or start it at a multiple of 4 (the walk "
        "reads that fact from how the coordinate was computed; for a value the IR cannot show "
        "it for, state multiple_of on its b.param declaration or check it with "
        "b.gpu_checked_value(v, multiple_of=...))",
    ):
        emit_accesses(b, graph, tensor, [(row, col)], vector_axis=0, vector_width=4)
    # Along the stride-1 axis the vector stays one access.
    (access,) = emit_accesses(
        b, graph, tensor, [(row, col)], vector_axis=1, vector_width=4
    )
    assert access.address.lane is not None


def test_a_runtime_stride_that_can_be_0_is_refused_before_anything_is_emitted():
    """A tensor description built without TensorDesc's own checks reaches the entries' check
    (TensorDesc refuses it first, with its own message)."""
    b = _builder()
    s = b.param("s", I32, ge=0, le=100)
    tensor = RuntimeTensor((8, 16), (s, 1), F32)
    graph = identity_graph((8, 16))
    before = len(_ops(b))
    message = (
        "tensor axis 0 has a runtime stride with stated range [0, 101) -- a runtime stride "
        "must be >= 1 (state ge=1 where s is stated)"
    )
    with _raises(ValueError, message):
        emit_addresses(b, graph, tensor, [(0, 0)])
    with _raises(ValueError, message):
        emit_accesses(b, graph, tensor, [(0, 0)])
    assert len(_ops(b)) == before


# --- a slice's begin ----------------------------------------------------------------------------


def _tile(b: IRBuilder):
    """Row ``origin + tid / 16``, column ``tid % 16`` (origin a guarded per-wave value)."""
    tid = b.thread_id_x()
    origin = _guarded(b, b.mul(b.block_id_x(), b.const_i32(4)), 64)
    return b.add(origin, b.div(tid, b.const_i32(16))), b.mod(tid, b.const_i32(16))


def _addresses_with_begin(begin):
    b = _builder()
    row, col = _tile(b)
    tensor = TensorDesc((64, 16), (40, 1), F16, begin=begin)
    graph = identity_graph((64, 16))
    accesses = [(row, col), (b.add(row, b.const_i32(1)), col)]
    return b, emit_addresses(b, graph, tensor, accesses)


def test_an_int_begin_on_an_int_stride_axis_lands_in_the_constant_only():
    b_plain, plain = _addresses_with_begin((0, 0))
    b_slice, sliced = _addresses_with_begin((3, 5))
    assert print_ir(b_plain.kernel) == print_ir(b_slice.kernel)
    for p, s in zip(plain, sliced):
        assert s.const == p.const + 3 * 40 + 5
        assert [t.range for t in s.uniform] == [t.range for t in p.uniform]
        assert s.lane_range == p.lane_range and s.uniform_delta is None


def test_a_plain_tensor_emits_the_same_ir_as_before_slices_existed():
    """Every begin 0, written or defaulted: the IR is the same, nothing is added."""
    b_default, defaulted = _addresses_with_begin(None)
    b_zero, _ = _addresses_with_begin((0, 0))
    assert print_ir(b_default.kernel) == print_ir(b_zero.kernel)
    assert all(not any(t.inputs == frozenset() for t in a.uniform) for a in defaulted)


def test_per_wave_begins_on_two_axes_make_one_term_keyed_by_no_input():
    b = _builder()
    row, col = _tile(b)
    cu = b.param("cu", PtrType(I32, "global"), readonly=True)
    row_begin = b.param("row_begin", I32)
    col_begin = b.readfirstlane(b.global_load(cu, b.const_i32(0), I32))
    tensor = TensorDesc((64, 16), (40, 1), F16, begin=(row_begin, col_begin))
    graph = identity_graph((64, 16))
    (address,) = emit_addresses(b, graph, tensor, [(row, col)])
    begin_terms = [t for t in address.uniform if t.inputs == frozenset()]
    assert len(begin_terms) == 1
    # Summed in tensor-axis order: (row_begin * 40) + col_begin, the col begin unbounded (i64).
    (term,) = begin_terms
    add = term.value.op
    assert add.name == "arith.add" and term.value.type == I64
    first, second = add.operands
    assert (
        first.op.name == "arith.mul"
        and first.op.operands[0].op.operands[0] is row_begin
    )
    assert second.op.name == "arith.sext" and second.op.operands[0] is col_begin
    assert not _varies_per_lane(term.value)

    tid, bx = _leaf(b, "thread_id"), _leaf(b, "bx")
    for rb, cb, blk, t in itertools.product(
        (0, 7), (0, 3, 1 << 20), range(2), range(WAVE)
    ):
        leaves = {id(tid): t, id(bx): blk, id(row_begin): rb, id(col_begin): cb}
        position = (blk * 4 + t // 16, t % 16)
        golden = address_of(tensor, position, [(row_begin, rb), (col_begin, cb)])
        parent = TensorDesc((64 + 7, 16 + (1 << 20)), (40, 1), F16)
        assert golden == address_of(parent, (position[0] + rb, position[1] + cb))
        assert _address(address, leaves) == golden


def test_an_int_begin_on_a_runtime_stride_axis_joins_the_begin_term():
    b = _builder()
    k, _, row, col = _packed_rows(b, k_max=4096)
    tensor = TensorDesc((256, k), (k, 1), F32, begin=(5, 0))
    graph = identity_graph((256, k))
    before = len(_ops(b))
    (address,) = emit_addresses(b, graph, tensor, [(row, col)])
    ops = _new_ops(b, before)
    (term,) = [t for t in address.uniform if t.inputs == frozenset()]
    assert term.value.op.name == "arith.mul"
    assert any(v is k for v in term.value.op.operands)
    assert term.range == (5, 5 * 4096 + 1)
    assert address.const == 0 and address.uniform_delta is None
    # One SALU multiply for the begin, one for the uniform row part: no more.
    assert (
        len(_muls(ops, I32, per_lane=False)) + len(_muls(ops, I64, per_lane=False)) == 2
    )


def test_a_begin_computed_from_the_thread_id_is_refused_with_its_axis():
    b = _builder()
    row, col = _tile(b)
    tid = _leaf(b, "thread_id")
    tensor = TensorDesc((64, 16), (40, 1), F16, begin=(0, tid))
    graph = identity_graph((64, 16))
    message = (
        f"the begin of tensor (64, 16) on axis 1 has a per-lane part ({tid.name}) -- a slice "
        "starts at the same element for the whole wave; compute its begin from per-wave values "
        "(pass a value loaded from memory through readfirstlane)"
    )
    before = len(_ops(b))
    with _raises(ValueError, message):
        emit_addresses(b, graph, tensor, [(row, col)])
    with _raises(ValueError, message):
        emit_accesses(b, graph, tensor, [(row, col)])
    assert len(_ops(b)) == before


def test_the_checks_of_a_slice_are_those_of_its_own_lengths():
    def validity_ir(begin):
        b = _builder()
        row, col = _tile(b)
        tensor = TensorDesc((8, 12), (40, 1), F16, begin=begin)
        (access,) = emit_accesses(b, identity_graph((8, 12)), tensor, [(row, col)])
        compares = [op for op in _ops(b) if op.name == "arith.cmp"]
        return access, [
            (op.attrs, [str(v.type) for v in op.operands]) for op in compares
        ]

    plain, plain_compares = validity_ir((0, 0))
    sliced, sliced_compares = validity_ir((3, 5))
    assert plain.valid is not None and plain_compares == sliced_compares
    assert sliced.address.const == plain.address.const + 3 * 40 + 5


@pytest.mark.parametrize(
    "max_t, wide", [((I32_END // (8 * 64)) + 1, True), (I32_END // (8 * 64) - 1, False)]
)
def test_a_thd_begin_term_is_i64_exactly_when_its_product_can_reach_2_to_the_31(
    max_t, wide
):
    """A (T, H, D) slice of one sequence: begin ``start`` on the token axis, ``start * H * D``."""
    h, d = 8, 64
    b = _builder()
    start = b.param("start", I32, ge=0, le=max_t)
    seqlen = b.param("seqlen", I32, ge=1, le=4096)
    tensor = TensorDesc((seqlen, h, d), (h * d, d, 1), F16, begin=(start, 0, 0))
    tid = b.thread_id_x()
    accesses = [
        (
            b.div(tid, b.const_i32(8)),
            0,
            b.mul(b.mod(tid, b.const_i32(8)), b.const_i32(8)),
        )
    ]
    before = len(_ops(b))
    (address,) = emit_addresses(b, identity_graph((seqlen, h, d)), tensor, accesses)
    ops = _new_ops(b, before)
    (term,) = [t for t in address.uniform if t.inputs == frozenset()]
    assert term.value.type == (I64 if wide else I32)
    assert term.value.op.name == "arith.mul"
    uniform_muls = [
        op
        for op in ops
        if op.name == "arith.mul" and not _varies_per_lane(op.results[0])
    ]
    assert [op.results[0].type for op in uniform_muls] == [I64 if wide else I32]


def test_an_unstated_begin_loaded_from_memory_is_accepted_as_an_i64_term():
    b = _builder()
    cu = b.param("cu", PtrType(I32, "global"), readonly=True)
    start = b.readfirstlane(b.global_load(cu, b.const_i32(0), I32))
    tensor = TensorDesc((16, 8, 64), (8 * 64, 64, 1), F16, begin=(start, 0, 0))
    (address,) = emit_addresses(b, identity_graph((16, 8, 64)), tensor, [(0, 0, 0)])
    (term,) = address.uniform
    assert term.inputs == frozenset() and term.value.type == I64
    assert term.range == (None, None)


# --- the golden ---------------------------------------------------------------------------------


def test_address_of_a_slice_is_the_parent_address_at_position_plus_begin():
    b = _builder()
    k = b.param("K", I32, ge=1, le=4096)
    start = b.param("start", I32)
    parent = TensorDesc((64, k), (k, 1), F32)
    sliced = TensorDesc((8, k), (k, 1), F32, begin=(start, 3))
    for k_int, s, position in itertools.product((37, 65), (0, 9), ((0, 0), (5, 7))):
        values = [(k, k_int), (start, s)]
        assert address_of(sliced, position, values) == address_of(
            parent, (position[0] + s, position[1] + 3), values
        )
    assert (
        transform_address_of(
            identity_graph((8, k)), sliced, (2, 1), [(k, 10), (start, 4)]
        )
        == (2 + 4) * 10 + 1 + 3
    )


def test_address_of_names_the_missing_launch_value():
    b = _builder()
    k = b.param("K", I32, ge=1, le=4096)
    start = b.param("start", I32)
    tensor = TensorDesc((8, k), (k, 1), F32, begin=(start, 0))
    with _raises(
        ValueError,
        "address_of needs the value of the runtime stride of axis 0 -- pass it in "
        "launch_values",
    ):
        address_of(tensor, (0, 0), [(start, 1)])
    with _raises(
        ValueError,
        "address_of needs the value of the runtime begin of axis 0 -- pass it in "
        "launch_values",
    ):
        address_of(tensor, (0, 0), [(k, 8)])


# --- GPU: every lane's parts against the golden -------------------------------------------------


def _lane_sums(stored, addresses) -> list[list[list[int]]]:
    """``sums[block][lane][i]``: access ``i``'s parts as that lane stored them, summed in Python
    integers, plus the constant."""
    sums = []
    for block in stored.tolist():
        rows = []
        for lane in block:
            values, at = [], 0
            for address in addresses:
                n = len(_part_values(address))
                values.append(sum(lane[at : at + n]) + address.const)
                at += n
            rows.append(values)
        sums.append(rows)
    return sums


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("k_int", [37, 64, 65])
def test_gpu_packed_rows_with_a_runtime_k(dtype, k_int):
    current_device_arch()
    held = {}

    def build(b):
        k, tensor, row, col = _packed_rows(b, k_max=4096, dtype=dtype)
        addresses = emit_addresses(
            b, identity_graph((256, k)), tensor, _row_accesses(b, row, col)
        )
        held.update(k=k, tensor=tensor, addresses=addresses)
        return [v for address in addresses for v in _part_values(address)]

    stored = run_lanes(build, scalars={"K": k_int}, n_blocks=3)
    sums = _lane_sums(stored, held["addresses"])
    graph = identity_graph((256, held["k"]))
    for blk, t in itertools.product(range(3), range(WAVE)):
        for i, c in enumerate(ROW_CONSTANTS):
            position = (blk * 16 + t // 4 + c, (t % 4) * 8)
            expected = transform_address_of(
                graph, held["tensor"], position, [(held["k"], k_int)]
            )
            assert sums[blk][t][i] == expected


def test_gpu_a_bshd_batch_through_its_runtime_batch_stride():
    """(B, S, H, D) with a runtime S: the batch stride is a kernel parameter stated at least
    S * H * D."""
    current_device_arch()
    h, d, s_int, batch = 4, 32, 37, 3
    held = {}

    def build(b):
        s = b.param("S", I32, ge=1, le=4096)
        batch_stride = b.param(
            "batch_stride", I32, ge=product_of(s, h * d), le=4096 * h * d
        )
        tensor = TensorDesc((4, s, h, d), (batch_stride, h * d, d, 1), F16)
        tid = b.thread_id_x()
        seq = b.add(b.mul(b.block_id_x(), b.const_i32(16)), b.div(tid, b.const_i32(4)))
        col = b.mul(b.mod(tid, b.const_i32(4)), b.const_i32(8))
        index = b.param("batch", I32)
        accesses = [(index, seq, b.const_i32(hh), col) for hh in (0, 3)]
        addresses = emit_addresses(b, identity_graph((4, s, h, d)), tensor, accesses)
        held.update(stride=batch_stride, tensor=tensor, addresses=addresses, s=s)
        return [v for address in addresses for v in _part_values(address)]

    stored = run_lanes(
        build,
        scalars={"S": s_int, "batch_stride": s_int * h * d, "batch": batch},
        n_blocks=2,
    )
    sums = _lane_sums(stored, held["addresses"])
    values = [(held["stride"], s_int * h * d)]
    for blk, t in itertools.product(range(2), range(WAVE)):
        for i, hh in enumerate((0, 3)):
            position = (batch, blk * 16 + t // 4, hh, (t % 4) * 8)
            expected = address_of(held["tensor"], position, values)
            assert (
                expected == ((batch * s_int + position[1]) * h + hh) * d + position[3]
            )
            assert sums[blk][t][i] == expected


@pytest.mark.parametrize("start_int", [5, (1 << 26) + 3])
def test_gpu_a_thd_slice_from_a_loaded_begin(start_int):
    """One sequence of an (H, T, D) tensor, its begin read from a cu_seqlens-style buffer and
    made per-wave with readfirstlane; a begin of 2^26 + 3 puts start * H * D past 2^31
    (coordinates only: nothing is allocated at that size)."""
    current_device_arch()
    import numpy as np

    h, d = 8, 64
    held = {}

    def build(b):
        cu = b.get_param("cu")
        start = b.readfirstlane(b.global_load(cu, b.const_i32(1), I32))
        seqlen = b.param("seqlen", I32, ge=1, le=4096)
        tensor = TensorDesc((h, seqlen, d), (d, h * d, 1), F16, begin=(0, start, 0))
        tid = b.thread_id_x()
        tok = b.add(b.mul(b.block_id_x(), b.const_i32(16)), b.div(tid, b.const_i32(4)))
        col = b.mul(b.mod(tid, b.const_i32(4)), b.const_i32(8))
        accesses = [(b.const_i32(hh), tok, col) for hh in (0, 5)]
        addresses = emit_addresses(b, identity_graph((h, seqlen, d)), tensor, accesses)
        held.update(start=start, tensor=tensor, addresses=addresses)
        return [v for address in addresses for v in _part_values(address)]

    stored = run_lanes(
        build,
        scalars={"seqlen": 40},
        buffers={"cu": np.array([0, start_int, start_int + 40], dtype=np.int32)},
        n_blocks=2,
    )
    sums = _lane_sums(stored, held["addresses"])
    parent = TensorDesc((h, start_int + 40, d), (d, h * d, 1), F16)
    for blk, t in itertools.product(range(2), range(WAVE)):
        for i, hh in enumerate((0, 5)):
            tok, col = blk * 16 + t // 4, (t % 4) * 8
            expected = address_of(parent, (hh, start_int + tok, col))
            assert expected == address_of(
                held["tensor"], (hh, tok, col), [(held["start"], start_int)]
            )
            assert sums[blk][t][i] == expected
    if start_int > 1 << 25:
        assert expected >= I32_END
