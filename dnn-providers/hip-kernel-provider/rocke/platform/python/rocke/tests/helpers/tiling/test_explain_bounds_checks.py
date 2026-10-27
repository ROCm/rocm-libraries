# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``explain_bounds_checks``: which bound checks a global access needs, decided by the shared rule.

For a window, a tile and a global pointer, the plan says per tensor axis whether each side of
``0 <= position < length`` is proven at build time or checked (by one scalar compare when the
position is the same for every lane, else per lane), how a vector access's validity is decided and
which memory instructions are used. These tests pin:

- that the plan is the tensor_transforms ``required_checks`` answer for the window's positions --
  each axis's walked origin range plus the tile's ``[0, extent)`` -- with no rule of its own;
- the facts the IR range walk reads through guarded origins, loop counters and the tail-tile origin,
  seen through the plan (the guard's own tests pin the walk);
- leading axes reported on both sides, scalar checks, the vector axis of a rank-3 window;
- that the plan emits nothing and is what the verbs apply;
- its plain-words print -- with the fix it names for a side only a loop stop keeps checked -- and
  its refusals, the constant-space pointer refusal shared with the verbs among them.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType, Value
from rocke.helpers.tiling import (
    AccessPath,
    AxisChecks,
    BoundsCheckPlan,
    SideCheck,
    TensorWindow,
    VectorValidity,
    exit_if_tile_outside,
    explain_bounds_checks,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.clipping import GlobalForm, global_store_width
from rocke.helpers.tiling.tensor_transforms.graph import identity_graph
from rocke.helpers.tiling.tensor_transforms.ranges import (
    AxisBound,
    Interval,
    required_checks_within,
)
from rocke.helpers.tiling.tensor_transforms.walk import value_facts

NOT_NEEDED = SideCheck.NOT_NEEDED
SCALAR = SideCheck.SCALAR_CHECK
PER_LANE = SideCheck.PER_LANE_CHECK

TM = TK = 16

# A (M, K) operand tile, K-contiguous: each lane holds 4 consecutive K elements (one vector of 4),
# 16 lanes along M and 4 along K -- the manual demo's A layout.
A_TILE = make_tile_desc(
    shape=[TM, TK],
    thread_tile=[1, 4],
    thread_dist=[16, 4],
    thread_order=[1, 0],
    wave_size=64,
)


def _builder(
    name: str = "plan_test", addr_space: str | None = None
) -> tuple[IRBuilder, Value]:
    """A builder and its pointer parameter ``A`` (declared with ``addr_space``, if given)."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    return b, b.param("A", PtrType(F16, "global"), align=16, addr_space=addr_space)


def _length(b: IRBuilder, name: str, **facts) -> Value:
    """A runtime length: the kernel parameter ``name`` declared ``0 <= value <= 2^20`` and
    ``facts``."""
    return b.param(name, I32, **{"ge": 0, "le": 1 << 20, **facts})


def _sides(plan: BoundsCheckPlan) -> list[tuple[SideCheck, SideCheck]]:
    return [(axis.lower, axis.upper) for axis in plan.axes]


def _block_origin(b: IRBuilder, tile: int, axis: str = "y") -> Value:
    block = {"x": b.block_id_x, "y": b.block_id_y, "z": b.block_id_z}[axis]()
    return b.mul(block, b.const_i32(tile))


def _all_ops(b: IRBuilder) -> list:
    def walk(ops):
        for op in ops:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def _rule(b, tensor, origin, tile_shape):
    """The shared rule asked directly: each axis's walked origin range plus the tile's extent
    against the tensor's bounds, as ``{axis: (lower checked, upper checked)}``."""
    lead = tensor.rank - len(tile_shape)
    ranges = []
    for axis, value in enumerate(origin):
        r = value_facts(b, value)[0]
        if axis >= lead:
            r = r.plus(Interval(0, tile_shape[axis - lead]))
        ranges.append(r)
    graph = identity_graph(tensor.lengths)
    bounds = tuple(AxisBound.of(0, n) for n in tensor.lengths)
    checks = required_checks_within(graph, tuple(ranges), bounds)
    found = {c.coordinate: (c.lo is not None, c.hi is not None) for c in checks}
    return [found.get(name, (False, False)) for name in graph.outputs]


# --- the plan is the shared rule -------------------------------------------------------------------


def _cases():
    """(name, builder) pairs: each builder returns ``(b, ptr, window, tile)``."""

    def aligned():
        b, ptr = _builder()
        td = make_tensor_desc((256, 64), (64, 1), F16)
        m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
        return b, ptr, make_window(td, (m0, 16)), A_TILE

    def ragged_runtime_m():
        b, ptr = _builder()
        ms = _length(b, "M")
        td = make_tensor_desc((ms, 64), (64, 1), F16)
        m0 = exit_if_tile_outside(b, _block_origin(b, TM), ms)
        return b, ptr, make_window(td, (m0, 0)), A_TILE

    def unguarded():
        b, ptr = _builder()
        td = make_tensor_desc((256, 64), (64, 1), F16)
        return b, ptr, make_window(td, (_block_origin(b, TM), 0)), A_TILE

    def halo():
        b, ptr = _builder()
        td = make_tensor_desc((256, 64), (64, 1), F16)
        return b, ptr, make_window(td, (-4, -2)), A_TILE

    def parameter_origin():
        b, ptr = _builder()
        td = make_tensor_desc((256, 64), (64, 1), F16)
        return b, ptr, make_window(td, (b.param("M0", I32), 32)), A_TILE

    return [
        ("aligned", aligned),
        ("ragged_runtime_m", ragged_runtime_m),
        ("unguarded", unguarded),
        ("halo", halo),
        ("parameter_origin", parameter_origin),
    ]


@pytest.mark.parametrize("name,build", _cases(), ids=[c[0] for c in _cases()])
def test_the_plan_is_the_shared_rule_on_the_walked_positions(name, build):
    b, ptr, window, tile = build()
    plan = explain_bounds_checks(b, ptr, window, tile)
    expected = _rule(b, window.tensor, window.origin, tile.shape)
    assert [
        (lower is not NOT_NEEDED, upper is not NOT_NEEDED)
        for lower, upper in _sides(plan)
    ] == expected


def test_a_plain_window_has_no_graph_inputs_and_no_read_domains():
    b, ptr, window, tile = _cases()[1][1]()
    plan = explain_bounds_checks(b, ptr, window, tile)
    assert plan.inputs == () and plan.read_domains == ()
    assert plan.path is AccessPath.GLOBAL
    assert plan.edge_vectors_load_whole is False


def test_explaining_emits_nothing():
    b, ptr, window, tile = _cases()[1][1]()
    before = len(_all_ops(b))
    explain_bounds_checks(b, ptr, window, tile)
    assert len(_all_ops(b)) == before


# --- the walk's facts through the plan -------------------------------------------------------------


def _k_window_plan(b, ptr, ks, m0, k_origin):
    td = make_tensor_desc((256, ks), (ks, 1), F16)
    return explain_bounds_checks(b, ptr, make_window(td, (m0, k_origin)), A_TILE)


def test_a_guarded_shift_gives_the_plan_of_the_guarded_product():
    plans = []
    for spell in ("mul", "shl"):
        b, ptr = _builder()
        ms = _length(b, "M")
        block = b.block_id_y()
        raw = (
            b.mul(block, b.const_i32(128))
            if spell == "mul"
            else b.shl(block, b.const_i32(7))
        )
        m0 = exit_if_tile_outside(b, raw, ms)
        td = make_tensor_desc((ms, 64), (64, 1), F16)
        plans.append(explain_bounds_checks(b, ptr, make_window(td, (m0, 0)), A_TILE))
    assert plans[0] == plans[1]
    assert _sides(plans[0])[0] == (NOT_NEEDED, PER_LANE)


@pytest.mark.parametrize("multiple_of", [1, TK])
def test_a_full_tile_loop_counter_proves_its_k_window(multiple_of):
    # The loop stop Ks + (1 - TK) gives the counter the end Ks - (TK - 1): k + TK <= Ks by the
    # identity of the one length Value, with or without a stated multiple.
    b, ptr = _builder()
    ks = _length(b, "K", ge=TK, multiple_of=multiple_of)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    stop = b.add(ks, b.const_i32(1 - TK))
    with b.scf_for(b.const_i32(0), stop, b.const_i32(TK)) as k:
        plan = _k_window_plan(b, ptr, ks, m0, k)
    assert _sides(plan) == [(NOT_NEEDED, NOT_NEEDED), (NOT_NEEDED, NOT_NEEDED)]
    assert plan.vector_validity is VectorValidity.NOT_CHECKED


@pytest.mark.parametrize(
    "multiple_of,upper", [(TK, NOT_NEEDED), (2 * TK, NOT_NEEDED), (1, PER_LANE)]
)
def test_the_loop_up_to_k_proves_its_k_window_when_k_is_a_tile_multiple(
    multiple_of, upper
):
    # The aligned kernel's natural loop [0, K) step TK: with K stated a multiple of TK the last
    # trip is K - TK, so the K side is proven with no change to the loop stop. Without that fact
    # the last trip may be partial and the upper K side stays checked (the ragged kernel writes
    # the stop K - (TK - 1) and a tail tile instead).
    b, ptr = _builder()
    ks = _length(b, "K", ge=multiple_of, multiple_of=multiple_of)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    with b.scf_for(b.const_i32(0), ks, b.const_i32(TK)) as k:
        plan = _k_window_plan(b, ptr, ks, m0, k)
    assert _sides(plan) == [(NOT_NEEDED, NOT_NEEDED), (NOT_NEEDED, upper)]


def test_the_tail_tile_origin_checks_only_its_upper_k_side():
    b, ptr = _builder()
    ks = _length(b, "K", ge=TK)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    tail = b.mul(b.div(ks, b.const_i32(TK)), b.const_i32(TK))
    plan = _k_window_plan(b, ptr, ks, m0, tail)
    assert _sides(plan) == [(NOT_NEEDED, NOT_NEEDED), (NOT_NEEDED, PER_LANE)]


@pytest.mark.parametrize(
    "length,upper", [(256, NOT_NEEDED), (1024, NOT_NEEDED), (250, PER_LANE)]
)
def test_an_int_length_proves_the_whole_tile_only_when_a_tile_multiple(length, upper):
    b, ptr = _builder()
    td = make_tensor_desc((length, 64), (64, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), length)
    plan = explain_bounds_checks(b, ptr, make_window(td, (m0, 0)), A_TILE)
    assert _sides(plan)[0] == (NOT_NEEDED, upper)


def test_an_unguarded_block_origin_keeps_its_upper_side_checked():
    b, ptr, window, tile = _cases()[2][1]()
    assert _sides(explain_bounds_checks(b, ptr, window, tile))[0] == (
        NOT_NEEDED,
        PER_LANE,
    )


# --- leading axes, scalar checks and the vector axis -----------------------------------------------


def _batched(b, g_origin, batches=2):
    td = make_tensor_desc((batches, 256, 64), (256 * 64, 64, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    return make_window(td, (g_origin, m0, 0))


def test_every_leading_axis_reports_both_sides():
    b, ptr = _builder()
    index = b.param("G", I32)  # a runtime batch index nothing bounds
    plan = explain_bounds_checks(b, ptr, _batched(b, index), A_TILE)
    assert len(plan.axes) == 3
    assert _sides(plan)[0] == (SCALAR, SCALAR)
    b, ptr = _builder()
    guarded = exit_if_tile_outside(b, b.block_id_z(), 2)
    plan = explain_bounds_checks(b, ptr, _batched(b, guarded), A_TILE)
    assert _sides(plan)[0] == (NOT_NEEDED, NOT_NEEDED)


def test_a_halo_axis_of_tile_extent_one_is_a_scalar_check():
    # A (rows, cols) image read one row at a time: the row is the same for every lane, so a halo
    # row before the image start is one scalar compare; no side is left to a buffer.
    b, ptr = _builder()
    row_tile = make_tile_desc(
        shape=[1, 64], thread_tile=[1, 1], thread_dist=[1, 64], wave_size=64
    )
    td = make_tensor_desc((32, 64), (64, 1), F16)
    row = exit_if_tile_outside(b, b.block_id_y(), 32)
    window = make_window(td, (b.add(row, b.const_i32(-1)), 0))
    plan = explain_bounds_checks(b, ptr, window, row_tile)
    assert _sides(plan) == [(SCALAR, NOT_NEEDED), (NOT_NEEDED, NOT_NEEDED)]
    assert plan._buffer_checked_sides == frozenset()
    assert all(
        SideCheck.BY_BUFFER not in (axis.lower, axis.upper) for axis in plan.axes
    )
    load_fragment(b, ptr, window, row_tile, b.thread_id_x())  # builds


def test_a_runtime_leading_index_is_a_scalar_check_and_builds():
    b, ptr = _builder()
    window = _batched(b, b.param("G", I32))
    plan = explain_bounds_checks(b, ptr, window, A_TILE)
    assert _sides(plan)[0] == (SCALAR, SCALAR)
    load_fragment(b, ptr, window, A_TILE, b.thread_id_x())  # builds


def test_the_vector_axis_of_a_rank_3_window_is_a_window_axis():
    # (G, M, K) read with a rank-2 (M, K) tile whose vector runs along K: the vector axis is window
    # axis 2, and the masked loads are 4 halves wide.
    b, ptr = _builder()
    window = _batched(b, b.param("G", I32))
    plan = explain_bounds_checks(b, ptr, window, A_TILE)
    assert (plan.vector_axis, plan.vector_width) == (2, 4)
    assert plan.vector_validity is VectorValidity.FIRST_ELEMENT
    load_fragment(b, ptr, window, A_TILE, b.thread_id_x())
    loads = [op for op in _all_ops(b) if op.name == "memref.global_load_vN"]
    assert loads and all(op.attrs["vec"] == 4 for op in loads)


def test_a_vector_that_can_cross_a_bound_is_decided_by_its_first_and_last_element():
    # K = 66 stated with no multiple: a vector of 4 starting at a multiple of 4 can end past 66.
    b, ptr = _builder()
    td = make_tensor_desc((256, 66), (68, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    plan = explain_bounds_checks(b, ptr, make_window(td, (m0, 64)), A_TILE)
    assert _sides(plan)[1] == (NOT_NEEDED, PER_LANE)
    assert plan.vector_validity is VectorValidity.FIRST_AND_LAST_ELEMENT
    # A K of 68 ends on a vector boundary: every vector is wholly in or out.
    b, ptr = _builder()
    td = make_tensor_desc((256, 68), (68, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    plan = explain_bounds_checks(b, ptr, make_window(td, (m0, 64)), A_TILE)
    assert plan.vector_validity is VectorValidity.FIRST_ELEMENT


# --- the verb applies the plan it explains ----------------------------------------------------------


def _vector_loads(ops):
    return [op for op in ops if op.name == "memref.global_load_vN"]


def test_the_verb_takes_the_form_and_width_the_plan_states():
    # Proven, int strides, offsets inside i32: the one-index form, no pointer arithmetic.
    b, ptr, window, tile = _cases()[0][1]()
    plan = explain_bounds_checks(b, ptr, window, tile)
    assert plan.form is GlobalForm.INDEX32
    start = len(_all_ops(b))
    load_fragment(b, ptr, window, tile, b.thread_id_x())
    emitted = _all_ops(b)[start:]
    assert not any(op.name.startswith("tile.global_ptr") for op in emitted)
    assert not any(op.name in ("scf.if", "scf.if_else") for op in emitted)
    assert len(_vector_loads(emitted)) == tile.register_count // plan.vector_width
    # Checked: every access is a vector of the stated width under its own validity branch.
    b, ptr, window, tile = _cases()[1][1]()
    plan = explain_bounds_checks(b, ptr, window, tile)
    assert plan.form is GlobalForm.MASKED
    assert plan.vector_validity is VectorValidity.FIRST_ELEMENT
    start = len(_all_ops(b))
    load_fragment(b, ptr, window, tile, b.thread_id_x())
    emitted = _all_ops(b)[start:]
    branches = [op for op in emitted if op.name == "scf.if_else"]
    accesses = tile.register_count // plan.vector_width
    assert len(branches) == accesses
    for branch in branches:
        (load,) = _vector_loads(branch.regions[0].ops)
        assert load.attrs["vec"] == plan.vector_width
    assert explain_bounds_checks(b, ptr, window, tile) == plan


# --- the print and the refusals --------------------------------------------------------------------


def test_the_print_names_every_side_in_plain_words():
    b, ptr, window, tile = _cases()[1][1]()
    assert str(explain_bounds_checks(b, ptr, window, tile)).splitlines() == [
        "bounds checks (global memory instructions):",
        "  tensor axis 0, lower side: not needed (proven at build time)",
        "  tensor axis 0, upper side: one compare per lane",
        "  tensor axis 1, lower side: not needed (proven at build time)",
        "  tensor axis 1, upper side: not needed (proven at build time)",
        (
            "  each access is 4 elements along window axis 1, validity decided by its "
            "first element"
        ),
    ]


def _k_loop_plan(
    b: IRBuilder,
    ptr: Value,
    stop: Value,
    length: Value,
    *,
    row_stride: int | None = None,
) -> BoundsCheckPlan:
    """The plan of a K window positioned by the loop [0, stop) step TK, over a (256, length)
    tensor (row stride ``length`` itself unless ``row_stride`` is given)."""
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    stride = length if row_stride is None else row_stride
    with b.scf_for(b.const_i32(0), stop, b.const_i32(TK)) as k:
        window = make_window(make_tensor_desc((256, length), (stride, 1), F16), (m0, k))
        return explain_bounds_checks(b, ptr, window, A_TILE)


def _k_upper_line(plan: BoundsCheckPlan) -> str:
    """The printed line of the K window's upper side."""
    (line,) = [
        line
        for line in str(plan).splitlines()
        if line.startswith("  tensor axis 1, upper side:")
    ]
    return line


_K_CHECKED = "  tensor axis 1, upper side: one compare per lane"


@pytest.mark.parametrize(
    "stated_multiple,fix_multiple", [(1, TK), (TK // 2, TK), (24, 48)]
)
def test_the_print_names_the_fix_for_a_side_only_a_loop_stop_keeps_checked(
    stated_multiple, fix_multiple
):
    # K not stated a multiple of the step: the last trip may be partial, so the K side is checked
    # on every trip, and stating the fact is all it takes to prove it. The multiple to state keeps
    # what K already states (24 and the step 16 give 48).
    b, ptr = _builder()
    ks = _length(b, "K", ge=stated_multiple, multiple_of=stated_multiple)
    plan = _k_loop_plan(b, ptr, ks, ks)
    assert _k_upper_line(plan) == (
        f"{_K_CHECKED} -- state multiple_of={fix_multiple} on K's b.param declaration"
    )
    assert str(plan).count("state multiple_of") == 1


def test_the_print_names_the_gpu_checked_value_call_for_a_checked_loop_length():
    b, ptr = _builder()
    lengths = b.param("lengths", PtrType(I32, "global"), readonly=True, noalias=True)
    loaded = b.readfirstlane(b.global_load(lengths, b.const_i32(0), I32))
    ks = b.gpu_checked_value(loaded, ge=1, le=1 << 20)
    plan = _k_loop_plan(b, ptr, ks, ks)
    assert _k_upper_line(plan) == (
        f"{_K_CHECKED} -- state multiple_of={TK} in the b.gpu_checked_value call that returned "
        f"{ks.name}"
    )


def test_a_plan_with_a_loop_stop_fix_equals_the_same_window_explained_again():
    # The loop stops a plan keeps take part in its equality: explaining one window twice gives
    # equal plans, as for every other window.
    b, ptr = _builder()
    ks = _length(b, "K", ge=1)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    with b.scf_for(b.const_i32(0), ks, b.const_i32(TK)) as k:
        window = make_window(make_tensor_desc((256, ks), (ks, 1), F16), (m0, k))
        first = explain_bounds_checks(b, ptr, window, A_TILE)
        assert explain_bounds_checks(b, ptr, window, A_TILE) == first


def test_the_print_names_no_fix_when_k_is_stated_a_tile_multiple():
    b, ptr = _builder()
    ks = _length(b, "K", ge=TK, multiple_of=TK)
    plan = _k_loop_plan(b, ptr, ks, ks)
    assert _k_upper_line(plan) == (
        "  tensor axis 1, upper side: not needed (proven at build time)"
    )
    assert "state multiple_of" not in str(plan)


def test_the_print_names_no_fix_for_a_side_checked_for_another_reason():
    # The loop runs to K but the tensor's length is another parameter, N: stating K a tile
    # multiple would not prove the N bound, so no fix is suggested.
    b, ptr = _builder()
    ks, ns = _length(b, "K", ge=1), _length(b, "N", ge=1)
    assert _k_upper_line(_k_loop_plan(b, ptr, ks, ns)) == _K_CHECKED
    # A runtime M with a guarded origin: checked because M is ragged, not because of a loop.
    b, ptr, window, tile = _cases()[1][1]()
    assert "state multiple_of" not in str(explain_bounds_checks(b, ptr, window, tile))


def test_the_print_names_no_fix_for_a_loop_length_the_author_does_not_state():
    # The loop stops at a guarded origin, whose facts exit_if_tile_outside states, not the author:
    # there is nowhere to state a multiple, so no fix is printed though the side is checked.
    b, ptr = _builder()
    ms = _length(b, "M", ge=1, le=2048)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), ms)
    assert _k_upper_line(_k_loop_plan(b, ptr, m0, m0, row_stride=2048)) == _K_CHECKED


def test_the_public_values_are_plain():
    plan = explain_bounds_checks(*_cases()[0][1]())
    assert plan.axes == (AxisChecks(NOT_NEEDED, NOT_NEEDED),) * 2
    assert (plan.vector_axis, plan.vector_width) == (1, 4)
    assert plan.vector_validity is VectorValidity.NOT_CHECKED


def test_an_lds_buffer_is_refused():
    b, _ = _builder()
    td = make_tensor_desc((16, 16), (16, 1), F16)
    lds = b.smem_alloc(F16, [td.element_span])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "explain_bounds_checks: ptr is an LDS buffer -- LDS accesses are not bound-checked "
            "(the buffer holds the descriptor's whole span); explain an access through a global "
            "pointer"
        ),
    ):
        explain_bounds_checks(b, lds, make_window(td, (0, 0)), A_TILE)


def test_a_window_of_another_kind_is_refused():
    b, ptr = _builder()
    with pytest.raises(
        TypeError,
        match=re.escape(
            "explain_bounds_checks takes a TensorWindow -- got a tuple; build one with "
            "make_window(tensor, origin)"
        ),
    ):
        explain_bounds_checks(b, ptr, (0, 0), A_TILE)


def test_a_tile_of_higher_rank_than_the_window_is_refused():
    b, ptr = _builder()
    td = make_tensor_desc((64,), (1,), F32)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "explain_bounds_checks: the tile has rank 2 but the window has rank 1 -- a rank-t "
            "tile covers the window's last t axes, so t <= r"
        ),
    ):
        explain_bounds_checks(b, ptr, make_window(td, (0,)), A_TILE)


# --- a constant-space pointer ----------------------------------------------------------------------


def _checked_window(addr_space: str | None) -> tuple[IRBuilder, Value, TensorWindow]:
    """A runtime M: the M side is checked, so every access takes the masked global form."""
    b, ptr = _builder(addr_space=addr_space)
    ms = _length(b, "M")
    td = make_tensor_desc((ms, 64), (64, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), ms)
    return b, ptr, make_window(td, (m0, 0))


def _proven_runtime_stride_window(
    addr_space: str | None,
) -> tuple[IRBuilder, Value, TensorWindow]:
    """Every side proven, but a runtime row stride: the address goes through the 64-bit pointer."""
    b, ptr = _builder(addr_space=addr_space)
    lda = b.param("lda", I32, ge=64, le=1 << 16, multiple_of=8)
    td = make_tensor_desc((256, 64), (lda, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    return b, ptr, make_window(td, (m0, 16))


def _proven_int_strides_window(
    addr_space: str | None,
) -> tuple[IRBuilder, Value, TensorWindow]:
    """Every side proven, int strides, offsets inside i32: the one-i32-index form."""
    b, ptr = _builder(addr_space=addr_space)
    td = make_tensor_desc((256, 64), (64, 1), F16)
    m0 = exit_if_tile_outside(b, _block_origin(b, TM), 256)
    return b, ptr, make_window(td, (m0, 16))


# Each window builder with the form its window takes through a global pointer.
_WINDOW_FORMS = {
    "checked": (_checked_window, GlobalForm.MASKED),
    "proven_runtime_stride": (_proven_runtime_stride_window, GlobalForm.PLACED),
    "proven_int_strides": (_proven_int_strides_window, GlobalForm.INDEX32),
}
_REFUSED = [
    name for name, (_, form) in _WINDOW_FORMS.items() if form is not GlobalForm.INDEX32
]


def _constant_space_error(verb: str) -> str:
    return re.escape(
        f'{verb}: ptr is the parameter A, declared addr_space="constant"; a bound-checked or '
        "64-bit-addressed access needs a global pointer -- declare A without addr_space"
    )


@pytest.mark.parametrize("name", list(_WINDOW_FORMS))
def test_each_window_takes_its_form_through_a_global_pointer(name):
    build, form = _WINDOW_FORMS[name]
    b, ptr, window = build(None)
    assert explain_bounds_checks(b, ptr, window, A_TILE).form is form


@pytest.mark.parametrize("name", list(_WINDOW_FORMS))
def test_a_global_store_writes_the_width_global_store_width_states(name):
    # One store instruction per global_store_width elements on every form: a whole vector on the
    # placed and masked forms, one element per register on the 32-bit-index form.
    build, form = _WINDOW_FORMS[name]
    b, ptr, window = build(None)
    plan = explain_bounds_checks(b, ptr, window, A_TILE)
    width = global_store_width(plan)
    if form is GlobalForm.INDEX32:
        assert width == 1 < plan.vector_width
    else:
        assert width == plan.vector_width
    fragment = load_fragment(b, ptr, window, A_TILE, b.thread_id_x())
    start = len(_all_ops(b))
    store_fragment(b, ptr, window, fragment, b.thread_id_x())
    stores = [
        op
        for op in _all_ops(b)[start:]
        if op.name in ("memref.global_store_typed", "memref.global_store_vN")
    ]
    assert len(stores) == A_TILE.register_count // width


@pytest.mark.parametrize("name", _REFUSED)
def test_a_constant_space_pointer_is_refused_off_the_32_bit_index_form(name):
    build, _ = _WINDOW_FORMS[name]
    b, ptr, window = build("constant")
    thread = b.thread_id_x()
    fragment = make_fragment(A_TILE, F16, b.zero_vec(F16, A_TILE.register_count))
    before = len(_all_ops(b))
    with pytest.raises(TypeError, match=_constant_space_error("explain_bounds_checks")):
        explain_bounds_checks(b, ptr, window, A_TILE)
    with pytest.raises(TypeError, match=_constant_space_error("load_fragment")):
        load_fragment(b, ptr, window, A_TILE, thread)
    with pytest.raises(TypeError, match=_constant_space_error("store_fragment")):
        store_fragment(b, ptr, window, fragment, thread)
    assert len(_all_ops(b)) == before  # refused at build, before any IR


def test_a_constant_space_pointer_is_not_refused_on_the_32_bit_index_form():
    # The one-index form stays exempt from the refusal: it hands the constant-space pointer itself
    # to core's global loads and stores and relies on their lowering to handle it (read through it
    # in the constant address space, refuse a store through it). The tiling layer refuses nothing
    # here and passes the pointer on unchanged; what the lowering makes of it is core's to test.
    build, form = _WINDOW_FORMS["proven_int_strides"]
    b, ptr, window = build("constant")
    assert explain_bounds_checks(b, ptr, window, A_TILE).form is form
    fragment = load_fragment(b, ptr, window, A_TILE, b.thread_id_x())
    store_fragment(b, ptr, window, fragment, b.thread_id_x())
    accesses = [
        op
        for op in _all_ops(b)
        if op.name in ("memref.global_load_vN", "memref.global_store_typed")
    ]
    assert {op.name for op in accesses} == {
        "memref.global_load_vN",
        "memref.global_store_typed",
    }
    assert all(op.operands[0] is ptr for op in accesses)
