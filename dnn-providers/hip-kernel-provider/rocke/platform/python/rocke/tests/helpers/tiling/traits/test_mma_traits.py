# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocke.helpers.tiling.traits.mma_traits.

Anchors on the gfx90a M1 atom (mfma_f32_16x16x16f16) whose field values come from the catalog.
Also checks the per-operand dtypes of the rows whose instruction names carry them, selection by
intent (result dtype, scale block size, integer signedness), multi-block rows, and construction
validation.
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.helpers.tiling.mma import TileMmaPlan
from rocke.helpers.tiling.traits import MmaLayoutParams, load_mma_traits


def _m1_atom():
    catalog = load_mma_traits()
    return catalog, catalog.select(
        target="gfx90a", a="fp16", b="fp16", c="fp32", m=16, n=16, k=16
    )


def test_catalog_loads() -> None:
    catalog = load_mma_traits()
    assert catalog.by_op_id, "catalog should contain instructions"


def test_gfx90a_16x16x16_f16_matches_the_catalog() -> None:
    _catalog, traits = _m1_atom()
    # Field values come from the catalog (the gfx90a 16x16x16 f16 MFMA row).
    assert traits.op_id == "mfma_f32_16x16x16f16"
    assert traits.wave_size == 64
    assert (traits.a_dtype, traits.b_dtype, traits.c_dtype, traits.d_dtype) == (
        "fp16",
        "fp16",
        "fp32",
        "fp32",
    )
    assert (traits.m, traits.n, traits.k, traits.b, traits.r, traits.s) == (
        16,
        16,
        16,
        1,
        1,
        1,
    )
    assert traits.layout_params == MmaLayoutParams(
        k_ab_per_lane=4,  # ABK
        a_k_num_access=1,  # AKN
        a_repeat=1,  # AR
        b_k_num_access=1,  # BKN
        b_repeat=1,  # BR
        c_m_per_lane=4,  # CM
        c_m_num_access=1,  # CMN
        a_m_register_blocks=1,  # AMB
    )
    assert traits.a_layout == "K{4} L{K1M} V{K0}"
    assert traits.c_d_layout == "M{4} L{M1N} V{M0}"
    assert traits.scale is None and traits.index_layout is None
    assert traits.supports("gfx90a")


def test_select_unknown_combo_raises_valueerror_with_query() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError) as excinfo:
        catalog.select(target="gfx90a", a="fp16", b="fp16", c="fp32", m=13, n=16, k=16)
    message = str(excinfo.value)
    assert message.startswith(
        "no MMA instruction for target='gfx90a' a='fp16' b='fp16' c='fp32' d='fp32' "
        "shape=(13,16,16) family='dense' -- selectable on 'gfx90a': ["
    )
    assert "'mfma_f32_16x16x16f16'" in message  # lists the candidates


@pytest.mark.parametrize(
    "target,op_id",
    [
        ("gfx1100", "wmma_f32_16x16x16_f16_w32"),
        ("gfx1154", "wmma_f32_16x16x16_f16_w32"),
        ("gfx11-generic", "wmma_f32_16x16x16_f16_w32"),
        ("gfx1200", "wmma_f32_16x16x16_f16_w32_gfx12"),
        ("gfx942", "mfma_f32_16x16x16f16"),
    ],
)
def test_select_on_real_target_ids(target, op_id) -> None:
    chosen = load_mma_traits().select(
        target=target, a="fp16", b="fp16", c="fp32", m=16, n=16, k=16
    )
    assert chosen.op_id == op_id


def test_family_name_is_not_a_target() -> None:
    # Rows list real target ids; a family name such as 'gfx11' selects nothing.
    with pytest.raises(ValueError) as excinfo:
        load_mma_traits().select(
            target="gfx11", a="fp16", b="fp16", c="fp32", m=16, n=16, k=16
        )
    assert str(excinfo.value) == (
        "no MMA instruction for target='gfx11' a='fp16' b='fp16' c='fp32' d='fp32' "
        "shape=(16,16,16) family='dense' -- selectable on 'gfx11': []"
    )


# ---- per-operand dtypes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_id,dtypes",
    [
        # Mixed fp8 / bf8 operands: the A and B formats are the ones the instruction name states.
        ("mfma_f32_16x16x32_fp8_bf8", ("fp8e4m3", "bf8e5m2", "fp32", "fp32")),
        ("mfma_f32_16x16x32_bf8_fp8", ("bf8e5m2", "fp8e4m3", "fp32", "fp32")),
        ("wmma_f32_16x16x64_fp8_bf8", ("fp8e4m3", "bf8e5m2", "fp32", "fp32")),
        ("swmmac_f16_16x16x128_bf8_fp8", ("bf8e5m2", "fp8e4m3", "fp16", "fp16")),
        # One row per (A format, B format) of an f8f6f4 instruction.
        (
            "mfma_scale_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1",
            ("fp8e4m3", "fp4e2m1", "fp32", "fp32"),
        ),
        (
            "wmma_f32_16x16x128_f8f6f4_fp6e3m2_bf8e5m2",
            ("fp6e3m2", "bf8e5m2", "fp32", "fp32"),
        ),
        # Accumulates in f32 and returns bf16.
        ("wmma_bf16f32_16x16x32_bf16", ("bf16", "bf16", "fp32", "bf16")),
        # Its sparse namesake returns f32, as the compiler's signature for it says.
        ("swmmac_bf16f32_16x16x64_bf16", ("bf16", "bf16", "fp32", "fp32")),
        ("mfma_f32_16x16x8_xf32", ("tf32", "tf32", "fp32", "fp32")),
        ("wmma_i32_16x16x16_iu4_w32", ("i4", "i4", "i32", "i32")),
    ],
)
def test_rows_carry_per_operand_canonical_dtypes(op_id, dtypes) -> None:
    traits = load_mma_traits().get(op_id)
    assert (traits.a_dtype, traits.b_dtype, traits.c_dtype, traits.d_dtype) == dtypes


def test_scaled_rows_carry_their_scale_block_size() -> None:
    catalog = load_mma_traits()
    stem = "wmma_scale{}_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3"
    assert catalog.get(stem.format("")).scale.block_k == 32
    assert catalog.get(stem.format("16")).scale.block_k == 16
    assert catalog.get(stem.format("16")).family == "scaled"


# ---- selection ------------------------------------------------------------------------------------


def test_result_dtype_is_part_of_the_selection() -> None:
    # gfx1250 has two dense bf16 x bf16, f32-accumulator 16x16x32 instructions that differ only in
    # the result: d defaults to c, and naming d="bf16" picks the bf16-result instruction.
    catalog = load_mma_traits()
    query = {
        "target": "gfx1250",
        "a": "bf16",
        "b": "bf16",
        "c": "fp32",
        "m": 16,
        "n": 16,
        "k": 32,
    }
    assert catalog.select(**query).op_id == "wmma_f32_16x16x32_bf16"
    assert catalog.select(**query, d="bf16").op_id == "wmma_bf16f32_16x16x32_bf16"


def test_scaled_selection_names_the_scale_block_size() -> None:
    catalog = load_mma_traits()
    query = {
        "target": "gfx1250",
        "a": "fp8e4m3",
        "b": "fp4e2m1",
        "c": "fp32",
        "m": 16,
        "n": 16,
        "k": 128,
        "family": "scaled",
    }
    assert (
        catalog.select(**query, scale_block_k=32).op_id
        == "wmma_scale_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1"
    )
    assert (
        catalog.select(**query, scale_block_k=16).op_id
        == "wmma_scale16_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1"
    )
    with pytest.raises(ValueError, match=r"needs scale_block_k"):
        catalog.select(**query)


def test_scale_block_size_is_refused_outside_family_scaled() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError) as excinfo:
        catalog.select(
            target="gfx90a",
            a="fp16",
            b="fp16",
            c="fp32",
            m=16,
            n=16,
            k=16,
            scale_block_k=32,
        )
    assert str(excinfo.value) == (
        "scale_block_k=32 applies to family 'scaled' only -- family='dense'"
    )


def test_pin_only_rows_are_never_selected() -> None:
    # The gfx1250 f8f6f4 8-bit x 8-bit variant answers the same intent as the dedicated fp8
    # instruction; the dedicated one is selected, the f8f6f4 variant is reachable only by name.
    catalog = load_mma_traits()
    pinned = catalog.get("wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3")
    assert not pinned.auto_select
    chosen = catalog.select(
        target="gfx1250", a="fp8e4m3", b="fp8e4m3", c="fp32", m=16, n=16, k=128
    )
    assert chosen.op_id == "wmma_f32_16x16x128_fp8_fp8"


def test_selectable_signedness_also_takes_unsigned_operands() -> None:
    # Integer WMMA has a signedness flag per operand, so u8 matches its i8 row, independently for A
    # and B; integer MFMA is signed only.
    catalog = load_mma_traits()
    for a, b in (("u8", "i8"), ("i8", "u8"), ("u8", "u8")):
        chosen = catalog.select(target="gfx1250", a=a, b=b, c="i32", m=16, n=16, k=64)
        assert chosen.op_id == "wmma_i32_16x16x64_iu8"
        assert chosen.operand_signedness == "selectable"
    signed = catalog.select(target="gfx942", a="i8", b="i8", c="i32", m=16, n=16, k=32)
    assert signed.operand_signedness == "signed"
    with pytest.raises(ValueError, match="no MMA instruction"):
        catalog.select(target="gfx942", a="u8", b="i8", c="i32", m=16, n=16, k=32)


# ---- multi-block rows -----------------------------------------------------------------------------


def test_multi_block_row_loads_and_plan_build_refuses_it() -> None:
    # mfma_f32_16x16x1f32 computes 4 independent blocks per instruction: the catalog describes it,
    # and building a plan on it fails by name because the encoder does not model the block axis.
    catalog = load_mma_traits()
    assert catalog.get("mfma_f32_16x16x1f32").b == 4
    with pytest.raises(NotImplementedError) as excinfo:
        TileMmaPlan(target="gfx90a", atom_override="mfma_f32_16x16x1f32")
    assert str(excinfo.value) == (
        "MMA instruction 'mfma_f32_16x16x1f32' cannot be laid out by the tiling encoder: it "
        "computes 4 independent blocks per instruction and the block axis is not modelled "
        "-- TODO: not implemented yet"
    )


def test_multi_block_rows_are_never_selected() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError, match="no MMA instruction"):
        catalog.select(target="gfx90a", a="fp32", b="fp32", c="fp32", m=16, n=16, k=1)


def test_get_unknown_op_raises_valueerror() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError) as excinfo:
        catalog.get("mfma_does_not_exist")
    assert "unknown op_id='mfma_does_not_exist'" in str(excinfo.value)


# ---- construction validation ----------------------------------------------------------------------


def test_non_positive_dim_fails_construction() -> None:
    _catalog, traits = _m1_atom()
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(traits, m=0)
    assert "non-positive fragment dim" in str(excinfo.value)
    assert "m=0" in str(excinfo.value)


def test_invalid_wave_size_fails_construction() -> None:
    _catalog, traits = _m1_atom()
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(traits, wave_size=17)
    assert "invalid wave_size" in str(excinfo.value)


def test_unknown_family_fails_construction() -> None:
    _catalog, traits = _m1_atom()
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(traits, family="quantum")
    assert str(excinfo.value) == (
        "unknown MMA family -- op_id='mfma_f32_16x16x16f16', family='quantum', "
        "expected one of ['dense', 'scaled', 'sparse']"
    )


def test_non_canonical_dtype_fails_construction() -> None:
    _catalog, traits = _m1_atom()
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(traits, a_dtype="f16")
    assert str(excinfo.value) == (
        "mma_traits.json row 'mfma_f32_16x16x16f16': a_dtype 'f16' is not a canonical rocke "
        "dtype name -- use the name core.dtypes.dtype_info returns, e.g. 'fp8e4m3'"
    )


def test_integer_rows_state_operand_signedness() -> None:
    catalog = load_mma_traits()
    integer_row = catalog.get("mfma_i32_16x16x32_i8")
    with pytest.raises(ValueError, match="integer operands need operand_signedness"):
        dataclasses.replace(integer_row, operand_signedness=None)
    with pytest.raises(ValueError, match="is not one of"):
        dataclasses.replace(integer_row, operand_signedness="unsigned")
    _catalog, float_row = _m1_atom()
    with pytest.raises(ValueError, match="operand_signedness is for integer rows only"):
        dataclasses.replace(float_row, operand_signedness="signed")
