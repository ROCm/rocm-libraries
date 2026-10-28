# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the public Mma object (target-aware resolution, IR-free)."""

from __future__ import annotations

import pytest

from rocke.core import ir
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    U8,
    Type,
    VectorType,
)
from rocke.helpers.tiling.mma import TileMma, TileMmaPlan, Tiling


def test_mma_resolves_gfx90a_16x16x16_f16() -> None:
    mma_op = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.wave_size == 64
    # Layouts are the calculator's encodings (IR-free).
    assert mma_op.c_layout.hierarchical_lengths == ((1, 4, 4), (16,))
    assert mma_op.a_layout.hierarchical_lengths == ((16,), (1, 4, 4))


# ---- dtypes are IR types --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,fix",
    [
        ("f16", "-- pass rocke.core.ir.F16"),
        ("f128", "-- rocke.core.ir has no type for 'f128'"),
    ],
)
def test_string_dtype_is_refused_with_the_constant_to_pass(token, fix) -> None:
    # The string goes through a variable so that a search for string dtypes in TileMma calls finds
    # none in this package.
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a=token, b=F16, c=F32, target="gfx90a")
    assert str(excinfo.value) == (
        f"a= must be a scalar IR type such as rocke.core.ir.F16, not the string {token!r} {fix}"
    )


def test_every_dtype_keyword_is_checked() -> None:
    token = "bf16"
    with pytest.raises(
        ValueError, match=r"^d= must be a scalar IR type .* rocke.core.ir.BF16$"
    ):
        TileMma((16, 16, 16), a=F16, b=F16, c=F32, d=token, target="gfx90a")


def test_composite_ir_type_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a=VectorType(F16, 4), b=F16, c=F32, target="gfx90a")
    assert str(excinfo.value) == (
        "a= must be a scalar IR type such as rocke.core.ir.F16 -- got vec<f16x4> (VectorType)"
    )


_SCALAR_IR_TYPES = sorted(
    (value for value in vars(ir).values() if type(value) is Type),
    key=lambda t: t.name,
)


@pytest.mark.parametrize("dtype", _SCALAR_IR_TYPES, ids=lambda t: t.name)
def test_refusal_names_the_constant_of_every_scalar_type(dtype) -> None:
    # The refusal spells the constant to pass as rocke.core.ir.<NAME UPPER CASE>; that constant must
    # exist and be the very type the string names.
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a=dtype.name, b=F16, c=F32, target="gfx90a")
    constant = dtype.name.upper()
    assert str(excinfo.value).endswith(f"-- pass rocke.core.ir.{constant}")
    assert getattr(ir, constant) is dtype


def test_dtypes_are_the_ir_types_given() -> None:
    mma_op = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    assert mma_op.a_dtype is F16 and mma_op.b_dtype is F16
    assert mma_op.plan.c_dtype is F32
    # An omitted result dtype is the accumulator's.
    assert mma_op.d_dtype is F32


def test_atom_override_derives_ir_dtypes() -> None:
    plan = TileMmaPlan(target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert (plan.a_dtype, plan.b_dtype, plan.c_dtype, plan.d_dtype) == (
        F16,
        F16,
        F32,
        F32,
    )


def test_tf32_has_no_instruction_on_gfx950() -> None:
    # gfx950 has no tf32 MMA instruction; the request fails like any other dtype with no row.
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 8), a=ir.TF32, b=ir.TF32, c=F32, target="gfx950")
    message = str(excinfo.value)
    assert message.startswith(
        "no MMA instruction found for a=tf32, b=tf32, c=f32, shape=(16, 16, 8), "
        "family='dense', target='gfx950' -- available on 'gfx950' (a x b -> c, MxNxK): ["
    )


def test_mma_is_target_agnostic_authorship_swap_target() -> None:
    # Same authored intent, different bound target -> distinct valid resolution.
    on_gfx90a = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    on_gfx942 = TileMma((16, 16, 16), a=F16, b=F16, c=F32, target="gfx942")
    assert on_gfx90a.op_id == on_gfx942.op_id == "mfma_f32_16x16x16f16"
    # (both CDNA MFMA here; the point is authorship carries no gfx branch)
    assert on_gfx90a.wave_size == 64


def test_default_atom_picks_native_32x32_single_atom() -> None:
    # With no atom knob, a 32x32x8 wave tile resolves the NATIVE 32x32 atom -> single atom,
    # NOT a grid of 16x16 atoms. This is the behaviour we override below.
    mma_op = TileMma((32, 32, 8), a=F16, b=F16, c=F32, target="gfx90a")
    assert mma_op.op_id == "mfma_f32_32x32x8f16"
    assert mma_op.atom_shape == (32, 32, 8)
    assert mma_op.subtiles == (1, 1, 1)


def test_atom_override_derives_shape_and_dtypes() -> None:
    # Name the exact intrinsic; TileMma derives M/N/K + dtypes from its traits (no shape/dtypes given).
    mma_op = TileMma(target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.shape == (16, 16, 16)
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (1, 1, 1)
    assert mma_op.wave_size == 64


def test_atom_override_with_wave_shape_is_a_grid() -> None:
    # atom_override still derives dtypes; an explicit shape is the wave tile (a multiple of the atom).
    mma_op = TileMma(
        (32, 32, 16), target="gfx90a", atom_override="mfma_f32_16x16x16f16"
    )
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)


def test_atom_override_conflicts_with_tiling_atom_shape() -> None:
    with pytest.raises(ValueError, match="atom_override OR tiling.atom_shape"):
        TileMma(
            target="gfx90a",
            atom_override="mfma_f32_16x16x16f16",
            tiling=Tiling(atom_shape=(16, 16, 16)),
        )


def test_forcing_16x16_atom_on_32x32_tile() -> None:
    # Force the small atom via the tiling knob: a 32x32x16 wave tile over a 16x16x16 atom is
    # a 2x2x1 subtile grid, NOT the native 32x32 atom.
    mma_op = TileMma(
        (32, 32, 16),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.shape == (32, 32, 16)
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)
    # The wave layouts carry the subtile counts: M gets an outer m_iter=2 level, N an
    # n_iter=2 level (K stays single: k_iter=1).
    assert mma_op.a_layout.hierarchical_lengths == ((2, 16), (1, 4, 4))
    assert mma_op.c_layout.hierarchical_lengths == ((2, 1, 4, 4), (2, 16))


def test_atom_selected_by_explicit_intrinsic_name() -> None:
    # Escape hatch: name the exact backend intrinsic; it resolves to the same atom as the
    # shape tuple, and the wave tile subtiles over it.
    mma_op = TileMma(
        (32, 32, 16),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        atom_override="mfma_f32_16x16x16f16",
    )
    assert mma_op.op_id == "mfma_f32_16x16x16f16"
    assert mma_op.atom_shape == (16, 16, 16)
    assert mma_op.subtiles == (2, 2, 1)


def test_atom_override_with_matching_dtypes_is_the_same_plan() -> None:
    # Dtypes given next to a pinned instruction are checked, not used to choose: when they are the
    # ones it takes, the plan is the one built without them.
    given = TileMma(
        (32, 32, 16),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        atom_override="mfma_f32_16x16x16f16",
    )
    derived = TileMma(
        (32, 32, 16), target="gfx90a", atom_override="mfma_f32_16x16x16f16"
    )
    assert repr(given.plan) == repr(derived.plan)
    assert given.c_layout == derived.c_layout


def test_atom_override_with_some_dtypes_takes_the_rest_from_the_instruction() -> None:
    mma_op = TileMma(c=F32, target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert (mma_op.a_dtype, mma_op.b_dtype, mma_op.d_dtype) == (F16, F16, F32)


def test_atom_override_dtype_mismatch_names_each_dtype_and_the_fix() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(a=BF16, c=F16, target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert str(excinfo.value) == (
        "MMA instruction dtype mismatch for 'mfma_f32_16x16x16f16' -- a=bf16 but it takes "
        "A=f16; c=f16 but it takes C=f32 -- pass a=rocke.core.ir.F16, c=rocke.core.ir.F32, "
        "or omit a/b/c to take them from the instruction"
    )


def test_atom_override_result_dtype_mismatch_names_the_result() -> None:
    # The result dtype is checked like the others; an omitted d is the instruction's, not c.
    with pytest.raises(ValueError) as excinfo:
        TileMma(d=BF16, target="gfx90a", atom_override="mfma_f32_16x16x16f16")
    assert str(excinfo.value) == (
        "MMA instruction dtype mismatch for 'mfma_f32_16x16x16f16' -- d=bf16 but it returns "
        "D=f32 -- pass d=rocke.core.ir.F32, or omit a/b/c/d to take them from the instruction"
    )


def test_atom_override_keeps_an_unsigned_operand_the_instruction_takes() -> None:
    # An integer instruction whose operand signedness is selectable takes u8 as well as i8; the
    # author's u8 is kept (it decides the signedness the instruction is issued with) and the
    # omitted B is the instruction's i8.
    plan = TileMmaPlan(a=U8, target="gfx1250", atom_override="wmma_i32_16x16x64_iu8")
    assert (plan.a_dtype, plan.b_dtype, plan.c_dtype) == (U8, I8, I32)


def test_atom_override_unknown_fails_fast() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(target="gfx90a", atom_override="mfma_f32_16x16x16_does_not_exist")
    assert "mfma_f32_16x16x16_does_not_exist" in str(excinfo.value)


def test_atom_override_wrong_target_fails_fast() -> None:
    # A real wide atom of another target named while bound to gfx90a must fail fast.
    with pytest.raises(ValueError) as excinfo:
        TileMma(target="gfx90a", atom_override="mfma_f32_16x16x32_f16")
    message = str(excinfo.value)
    assert message.startswith(
        "MMA instruction not available on target -- op_id='mfma_f32_16x16x32_f16', "
        "target='gfx90a', supported=["
    )


def test_atom_override_operand_dtype_mismatch_fails_fast() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(
            (16, 16, 16),
            a=F16,
            b=F16,
            c=F32,
            target="gfx90a",
            atom_override="mfma_i32_16x16x16i8",
        )
    message = str(excinfo.value)
    assert message.startswith(
        "MMA instruction dtype mismatch for 'mfma_i32_16x16x16i8' -- a=f16 but it takes A=i8"
    )


def test_tiling_refuses_an_instruction_name_as_atom_shape() -> None:
    with pytest.raises(ValueError) as excinfo:
        Tiling(atom_shape="mfma_f32_16x16x16f16")
    assert str(excinfo.value) == (
        "atom_shape is the instruction's (M, N, K) -- got the string 'mfma_f32_16x16x16f16'; "
        "pin an instruction with TileMma(atom_override='mfma_f32_16x16x16f16')"
    )


def test_dtypes_are_required_without_a_pinned_instruction() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a=F16, b=F16, target="gfx90a")
    assert str(excinfo.value) == (
        "a/b/c dtypes are required unless the instruction is pinned with atom_override=..."
    )


def test_mma_exposes_per_operand_tiledescs() -> None:
    # The MMA object hands back ready-to-use TileDescs (wave shape + resolved layout).
    mma_op = TileMma(
        (32, 32, 16),
        a=F16,
        b=F16,
        c=F32,
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    assert mma_op.a_desc.shape == (32, 16)  # (wave M, wave K)
    assert mma_op.b_desc.shape == (32, 16)  # (wave N, wave K)
    assert mma_op.c_desc.shape == (32, 32)  # (wave M, wave N)
    assert mma_op.a_desc.layout == mma_op.a_layout  # desc bundles the resolved layout
    assert mma_op.c_desc.layout == mma_op.c_layout


def test_mma_mixed_ab_without_instruction_fails_fast() -> None:
    # A and B may differ, but gfx90a has no f16 x bf16 instruction.
    with pytest.raises(ValueError) as excinfo:
        TileMma((16, 16, 16), a=F16, b=BF16, c=F32, target="gfx90a")
    assert str(excinfo.value).startswith(
        "no MMA instruction found for a=f16, b=bf16, c=f32, shape=(16, 16, 16)"
    )


def test_mma_mixed_fp8_bf8_selects_the_mixed_instruction() -> None:
    mma_op = TileMma((16, 16, 32), a=FP8E4M3, b=BF8E5M2, c=F32, target="gfx942")
    assert mma_op.op_id == "mfma_f32_16x16x32_fp8_bf8"
    assert (mma_op.a_dtype, mma_op.b_dtype) == (FP8E4M3, BF8E5M2)


def test_mma_selects_an_integer_instruction() -> None:
    mma_op = TileMma((16, 16, 32), a=I8, b=I8, c=I32, target="gfx942")
    assert mma_op.op_id == "mfma_i32_16x16x32_i8"


def test_mma_unsupported_shape_fails_fast_with_candidates() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma((13, 16, 16), a=F16, b=F16, c=F32, target="gfx90a")
    assert "no MMA instruction found" in str(excinfo.value)
    assert "(13, 16, 16)" in str(excinfo.value)
    assert "available on 'gfx90a'" in str(excinfo.value)


# The gfx1250 'wmma_bf16f32_16x16x32_bf16' instruction accumulates in fp32 but returns bf16. The
# TileMma driver keeps the result in the accumulator dtype, so it must never pick that instruction,
# and naming it must fail rather than silently issue the fp32-result instruction of the same shape.
_RESULT_DTYPE_CHANGE_MESSAGE = (
    "MMA instruction 'wmma_bf16f32_16x16x32_bf16' accumulates in fp32 but returns bf16 -- the "
    "TileMma driver does not drive an instruction whose result dtype differs from its accumulator "
    "dtype; choose one whose result dtype is fp32"
)


def test_selection_by_intent_never_picks_a_result_dtype_change() -> None:
    mma_op = TileMma((16, 16, 32), a=BF16, b=BF16, c=F32, target="gfx1250")
    assert mma_op.op_id == "wmma_f32_16x16x32_bf16"
    assert mma_op.traits.d_dtype == mma_op.traits.c_dtype


def test_atom_override_refuses_a_result_dtype_change() -> None:
    with pytest.raises(ValueError) as excinfo:
        TileMma(target="gfx1250", atom_override="wmma_bf16f32_16x16x32_bf16")
    assert str(excinfo.value) == _RESULT_DTYPE_CHANGE_MESSAGE
