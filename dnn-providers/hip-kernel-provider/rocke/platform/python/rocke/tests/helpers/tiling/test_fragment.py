# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The `Fragment` register invariant: a fragment's `value` is always an IR vector of exactly
``tile_desc.register_count`` elements of type `dtype`, checked on every write (construction
included); `tile_desc` and `dtype` are fixed; an unfilled fragment refuses to be read.

The headline case: registers re-bound under a descriptor with more registers than they hold would
otherwise be accepted silently, and any verb slicing past the value's end would emit out-of-range
``extractelement`` -- LLVM poison, a silent wrong answer. The bad assignment itself fails instead.

Offline (no GPU): values come from a real ``IRBuilder``; descriptors are real MMA operand layouts
plus one hand-authored single-register layout.

Not covered here (by design -- this is a TYPE check): a value of the right type whose registers are
in the wrong order, or an SSA value used outside the scope that defines it. Register order is owned
by the MMA layout checks and the on-GPU parity tests.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType, SmemType, VectorType
from rocke.helpers.tiling import (
    Fragment,
    TileMma,
    Tiling,
    classify_transform,
    fill_fragment,
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
    transform_fragment,
)
from rocke.helpers.tiling import tiling_recorder

# 32x32x16 f16 -> f32 wave tile from 16x16x16 atoms: A is a 32x16 tile and C a 32x32 tile over one
# 64-lane wave.
_MMA = TileMma(
    (32, 32, 16),
    a="f16",
    b="f16",
    c="f32",
    target="gfx90a",
    tiling=Tiling(atom_shape=(16, 16, 16)),
)

# An 8x8 tile spread one element per lane over a 64-lane wave: the single-register boundary case.
_ONE_REGISTER = make_tile_desc(
    shape=[8, 8],
    thread_tile=[1, 1],
    thread_dist=[8, 8],
    wave_size=64,
)

_DESCS = {
    "one-register": _ONE_REGISTER,
    "mma-A": _MMA.a_desc,
    "mma-C": _MMA.c_desc,
}

# Hand-derived from the tile sizes, independent of the layout code: elements / 64 lanes.
_EXPECTED_REGISTERS = {
    "one-register": 8 * 8 // 64,
    "mma-A": 32 * 16 // 64,
    "mma-C": 32 * 32 // 64,
}

_DTYPES = [F16, BF16, F32, I32]


def _builder() -> IRBuilder:
    return IRBuilder("fragment_invariant")


@pytest.mark.parametrize("name", sorted(_DESCS))
def test_fixture_register_counts_match_hand_derived_values(name):
    # Self-check: every case below leans on these counts, so they must be what the tile sizes say.
    assert _DESCS[name].register_count == _EXPECTED_REGISTERS[name]


# ---- accepted writes ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("name", sorted(_DESCS))
def test_matching_vector_is_accepted_and_kept_as_the_same_object(name, dtype):
    b = _builder()
    registers = b.zero_vec(dtype, _EXPECTED_REGISTERS[name])
    fragment = make_fragment(_DESCS[name], dtype, registers)
    # The very same SSA value comes back: the tiling recorder tracks dataflow by its identity.
    assert fragment.value is registers
    assert fragment.value.type == VectorType(dtype, _EXPECTED_REGISTERS[name])


def test_raw_mma_update_through_the_setter_is_accepted():
    # The lowest-layer authoring path: one atom, the accumulator re-bound in place every K step.
    mma = TileMma(target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    b = _builder()
    a = make_fragment(mma.a_desc, F16, b.zero_vec(F16, mma.a_desc.register_count))
    b_ = make_fragment(mma.b_desc, F16, b.zero_vec(F16, mma.b_desc.register_count))
    accumulator = make_fragment(
        mma.c_desc, F32, b.zero_vec(F32, mma.c_desc.register_count)
    )
    before = accumulator.value
    accumulator.value = b.mma(mma.emit_op(), a.value, b_.value, accumulator.value)
    # The setter stored the MMA's result, and that MMA consumed the very previous accumulator
    # object (identity, not equality: IR values compare equal field-by-field).
    assert accumulator.value.op.name == "tile.mma"
    consumed = accumulator.value.op.operands
    assert len(consumed) == 3
    assert all(got is want for got, want in zip(consumed, [a.value, b_.value, before]))
    assert accumulator.value.type == VectorType(F32, mma.c_desc.register_count)


def test_tile_mma_result_has_the_accumulator_register_type():
    b = _builder()
    a = make_fragment(_MMA.a_desc, F16, b.zero_vec(F16, _EXPECTED_REGISTERS["mma-A"]))
    b_ = make_fragment(_MMA.b_desc, F16, b.zero_vec(F16, _MMA.b_desc.register_count))
    accumulator = make_fragment(
        _MMA.c_desc, F32, b.zero_vec(F32, _EXPECTED_REGISTERS["mma-C"])
    )
    result = _MMA(b, a, b_, accumulator)
    assert result.value.type == VectorType(F32, _EXPECTED_REGISTERS["mma-C"])


# ---- rejected writes ----------------------------------------------------------------------------


_OFF_BY_ONE = [
    pytest.param(name, delta, id=f"{name}{delta:+d}")
    for name in sorted(_DESCS)
    for delta in (-1, +1)
    if _EXPECTED_REGISTERS[name] + delta > 0
]


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("name,delta", _OFF_BY_ONE)
def test_register_count_off_by_one_is_rejected(name, delta, dtype):
    want = _EXPECTED_REGISTERS[name]
    b = _builder()
    wrong = b.zero_vec(dtype, want + delta)
    with pytest.raises(
        ValueError,
        match=(
            rf"expected vec<{dtype.name}x{want}> .* got vec<{dtype.name}x{want + delta}> -- "
            rf"the value holds {want + delta} registers per lane but this TileDesc lays out "
            rf"{want}; wrap it in a new Fragment using the TileDesc that produced these "
            r"registers, or, if they are one MMA atom's result, drive the whole tile through "
            r"TileMma instead of raw b\.mma$"
        ),
    ):
        make_fragment(_DESCS[name], dtype, wrong)


@pytest.mark.parametrize("name", sorted(_DESCS))
def test_right_count_wrong_element_type_is_rejected(name):
    want = _EXPECTED_REGISTERS[name]
    b = _builder()
    with pytest.raises(
        ValueError,
        match=(
            rf"expected vec<f32x{want}> .* got vec<f16x{want}> -- the registers are f16, not "
            r"f32; build the Fragment at f16$"
        ),
    ):
        make_fragment(_DESCS[name], F32, b.zero_vec(F16, want))


def test_wrong_count_and_wrong_element_type_name_both_fixes():
    b = _builder()
    with pytest.raises(
        ValueError,
        match=(
            r"got vec<f16x8> -- the value holds 8 registers per lane but this TileDesc lays out "
            r"16; .*; and the registers are f16, not f32; build the Fragment at f16$"
        ),
    ):
        make_fragment(_MMA.c_desc, F32, b.zero_vec(F16, 8))


def test_scalar_value_is_rejected_even_for_one_register():
    b = _builder()
    with pytest.raises(
        ValueError, match=r"got scalar f32 -- wrap the single register as a vec<f32x1>$"
    ):
        make_fragment(_ONE_REGISTER, F32, b.const_f32(0.0))


def test_scalar_of_the_wrong_element_type_names_both_fixes():
    # Wrapping alone would still fail here: the register is i32, the fragment is f32.
    b = _builder()
    with pytest.raises(
        ValueError,
        match=(
            r"got scalar i32 -- wrap the single register as a vec<f32x1>; and the register is "
            r"i32, not f32; build the Fragment at i32$"
        ),
    ):
        make_fragment(_ONE_REGISTER, F32, b.const_i32(0))


def test_scalar_value_for_a_multi_register_tile_names_the_register_count():
    b = _builder()
    with pytest.raises(
        ValueError,
        match=(
            r"got scalar f32 -- this TileDesc lays out 16 registers per lane; pass them as one "
            r"vector$"
        ),
    ):
        make_fragment(_MMA.c_desc, F32, b.const_f32(0.0))


def test_pointer_value_is_rejected_as_non_vector():
    b = _builder()
    pointer = b.param("P", PtrType(F16, "global"))
    with pytest.raises(
        ValueError,
        match=(
            r"expected vec<f16x1> .*, got non-vector ptr.* read memory into one with "
            r"load_fragment$"
        ),
    ):
        make_fragment(_ONE_REGISTER, F16, pointer)


def test_non_ir_value_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"must be an IR Value of type vec<f16x1> .* -- got tuple \('zeros', 'f16', 1\)$",
    ):
        make_fragment(_ONE_REGISTER, F16, ("zeros", "f16", 1))


def test_rebinding_registers_under_a_mismatched_descriptor_is_rejected():
    # Registers handed to the wrong descriptor: an 8-register A fragment wrapped as the
    # 16-register C tile.
    b = _builder()
    a = make_fragment(_MMA.a_desc, F16, b.zero_vec(F16, _EXPECTED_REGISTERS["mma-A"]))
    with pytest.raises(ValueError, match=r"expected vec<f16x16> .* got vec<f16x8>"):
        make_fragment(_MMA.c_desc, F16, a.value)


@pytest.mark.parametrize(
    "role,dtype,short",
    [("A", F16, 4), ("B", F16, 4), ("C", F32, 8)],
)
def test_short_operands_are_rejected_when_wrapped(role, dtype, short):
    # Short operands: 4/4/8 registers where this plan needs 8/8/16 are rejected at construction.
    desc = {"A": _MMA.a_desc, "B": _MMA.b_desc, "C": _MMA.c_desc}[role]
    assert desc.register_count == 2 * short  # self-check: the value really is short
    b = _builder()
    with pytest.raises(
        ValueError,
        match=rf"got vec<{dtype.name}x{short}> -- the value holds {short} registers",
    ):
        make_fragment(desc, dtype, b.zero_vec(dtype, short))


def test_setter_rejects_a_bad_value_and_keeps_the_old_one():
    # A short vector assigned to a valid accumulator through the setter (the raw-MMA idiom).
    b = _builder()
    good = b.zero_vec(F32, _EXPECTED_REGISTERS["mma-C"])
    accumulator = make_fragment(_MMA.c_desc, F32, good)
    with pytest.raises(
        ValueError,
        match=(
            r"expected vec<f32x16> .* got vec<f32x8> -- the value holds 8 registers per lane but "
            r"this TileDesc lays out 16; wrap it in a new Fragment using the TileDesc that "
            r"produced these registers, or, if they are one MMA atom's result, drive the whole "
            r"tile through TileMma instead of raw b\.mma$"
        ),
    ):
        accumulator.value = b.zero_vec(F32, 8)
    assert accumulator.value is good


def test_setter_rejects_none():
    b = _builder()
    accumulator = make_fragment(_MMA.c_desc, F32, b.zero_vec(F32, 16))
    with pytest.raises(
        ValueError,
        match=(
            r"cannot be None -- assign a vec<f32x16>, or build a new unfilled Fragment with "
            r"make_fragment$"
        ),
    ):
        accumulator.value = None


@pytest.mark.parametrize("attribute", ["tile_desc", "dtype"])
def test_layout_and_dtype_are_read_only(attribute):
    fragment = make_fragment(_MMA.c_desc, F32)
    # Python 3.12+ says "has no setter"; 3.10/3.11 say "can't set attribute".
    with pytest.raises(AttributeError, match=r"has no setter|can't set attribute"):
        setattr(fragment, attribute, getattr(fragment, attribute))


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("f16", id="string-token"),
        pytest.param(VectorType(F16, 4), id="vector-type"),
        pytest.param(PtrType(F16, "global"), id="pointer-type"),
        pytest.param(SmemType(F16, (16, 16)), id="lds-type"),
    ],
)
def test_non_scalar_ir_dtype_is_rejected_at_construction(dtype):
    with pytest.raises(ValueError, match=r"dtype must be a scalar IR Type"):
        make_fragment(_MMA.c_desc, dtype)


@pytest.mark.parametrize(
    "tile_desc",
    [
        pytest.param(None, id="none"),
        pytest.param((32, 32), id="bare-shape"),
    ],
)
def test_non_tile_desc_layout_is_rejected_at_construction(tile_desc):
    with pytest.raises(ValueError, match=r"tile_desc must be a TileDesc .* -- got "):
        make_fragment(tile_desc, F32)


# ---- unfilled fragments -------------------------------------------------------------------------


def test_reading_an_unfilled_fragment_names_the_fix():
    fragment = make_fragment(_MMA.c_desc, F32)
    with pytest.raises(
        ValueError,
        match=r"Fragment\(shape=\(32, 32\), dtype=f32\) not filled -- call fill_fragment",
    ):
        _ = fragment.value


def _consume_with_mma(b, unfilled):
    a = make_fragment(_MMA.a_desc, F16, b.zero_vec(F16, _EXPECTED_REGISTERS["mma-A"]))
    b_ = make_fragment(_MMA.b_desc, F16, b.zero_vec(F16, _MMA.b_desc.register_count))
    _MMA(b, a, b_, unfilled)


def _consume_with_store(b, unfilled):
    out = b.param("OUT", PtrType(F32, "global"))
    c_td = make_tensor_desc((32, 32), (32, 1), F32)
    zero = b.const_i32(0)
    store_fragment(b, out, make_window(c_td, (zero, zero)), unfilled, b.thread_id_x())


def _consume_with_transform(b, unfilled):
    transform_fragment(b, unfilled, unfilled.tile_desc)


@pytest.mark.parametrize(
    "consume",
    [_consume_with_mma, _consume_with_store, _consume_with_transform],
    ids=["mma", "store_fragment", "transform_fragment"],
)
def test_unfilled_fragment_reaching_a_consumer_fails_with_the_fill_message(consume):
    b = _builder()
    with pytest.raises(ValueError, match=r"not filled -- call fill_fragment"):
        consume(b, make_fragment(_MMA.c_desc, F32))


def test_repr_of_an_unfilled_fragment_does_not_raise():
    assert repr(make_fragment(_MMA.c_desc, F32)) == (
        "Fragment(shape=(32, 32), dtype=f32, value=<unfilled>)"
    )


def test_repr_of_a_filled_fragment_shows_the_register_type():
    b = _builder()
    fragment = Fragment(_MMA.c_desc, F32, b.zero_vec(F32, 16))
    assert repr(fragment).endswith(": vec<f32x16>)")


# ---- recorder dataflow --------------------------------------------------------------------------


# A's registers in the reverse bucket order: a genuine (non-identity) reorder back to the MMA
# layout.
_A_REVERSED = _MMA.a_desc.reorder_registers(
    tuple(reversed(range(len(_MMA.a_desc.layout.register_to_rh_major))))
)


def _row_major(shape, dtype):
    return make_tensor_desc(shape, (shape[1], 1), dtype)


def _build_chain():
    """load A (reversed), load B, fill C -> transform A -> TileMma -> store C, through the public
    verbs, with the two hand-offs a real kernel makes: B and the result re-wrapped in a NEW
    Fragment."""
    b = IRBuilder("fragment_recorder_chain")
    a_ptr = b.param("A", PtrType(F16, "global"))
    b_ptr = b.param("B", PtrType(F16, "global"))
    c_ptr = b.param("C", PtrType(F32, "global"))
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a_win = make_window(_row_major(_A_REVERSED.shape, F16), (zero, zero))
    b_win = make_window(_row_major(_MMA.b_desc.shape, F16), (zero, zero))
    a = load_fragment(b, a_ptr, a_win, _A_REVERSED, lane)
    b_ = load_fragment(b, b_ptr, b_win, _MMA.b_desc, lane)
    b_ = make_fragment(_MMA.b_desc, F16, b_.value)
    accumulator = make_fragment(_MMA.c_desc, F32)
    fill_fragment(b, accumulator, 0)
    a = transform_fragment(b, a, _MMA.a_desc)
    accumulator = _MMA(b, a, b_, accumulator)
    accumulator = make_fragment(_MMA.c_desc, F32, accumulator.value)
    c_win = make_window(_row_major(_MMA.c_desc.shape, F32), (zero, zero))
    store_fragment(b, c_ptr, c_win, accumulator, lane)
    return b.kernel


def test_chain_fixture_reorders_for_real():
    # Self-check: the transform in the chain must move registers, or "distinct values" below would
    # lean on an identity transform still emitting a shuffle.
    plan = classify_transform(_A_REVERSED.layout, _MMA.a_desc.layout)
    assert plan.tier == "reorder"
    assert list(plan.permutation) != sorted(plan.permutation)


def test_recorder_links_every_hop_through_the_same_register_values():
    # The recorder joins producers to consumers by the identity of `Fragment.value`, so a value
    # handed into a NEW Fragment (the re-wraps in the chain) must be stored as the very object
    # given. A broken link here means the setter copied or re-wrapped the value.
    _kernel, pipeline = tiling_recorder.record_build(_build_chain)
    load_a, load_b, fill_c = pipeline.transactions[:3]
    (store_c,) = [t for t in pipeline.transactions if t.kind == "store"]
    (transform_a,) = [op for op in pipeline.ops if op.kind == "reorder"]
    (mma_op,) = [op for op in pipeline.ops if op.kind == "mma"]
    assert [load_a.kind, load_b.kind, fill_c.kind] == ["load", "load", "fill"]
    assert transform_a.consumes == (load_a.produces,)
    assert mma_op.consumes == (transform_a.produces, load_b.produces, fill_c.produces)
    assert store_c.consumes == (mma_op.produces,)
    produced = [load_a.produces, load_b.produces, fill_c.produces, transform_a.produces]
    assert len(set(produced + [mma_op.produces])) == 5  # five distinct register values
