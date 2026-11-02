# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the MMA warp-encoding calculators (dense no-block, M1).

Structural correctness (bijection holds; field values match the derivation) for the
gfx90a dense MFMA atoms. The cross-check against rocke's make_c_warp_dstr_encoding lives
in the oracle test (tests/mma/test_layout_oracle.py) which imports the rocke substrate.
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.helpers.tiling.encoding import WarpDistributionEncoding
from rocke.helpers.tiling.mma.warp_encoding import (
    a_warp_encoding,
    b_warp_encoding,
    c_warp_encoding,
    encoder_supports_layout,
)
from rocke.helpers.tiling.traits import load_mma_traits


def _traits(op_id: str):
    return load_mma_traits().get(op_id)


def test_c_encoding_gfx90a_16x16x16_f16_fields() -> None:
    encoding = c_warp_encoding(_traits("mfma_f32_16x16x16f16"))
    # Matches rocke make_c_warp_dstr_encoding for this atom (CM=4, CMN=1, M=N=16).
    assert encoding.replication_lengths == ()
    assert encoding.hierarchical_lengths == ((1, 4, 4), (16,))
    assert encoding.lane_to_rh_major == ((1, 2),)
    assert encoding.lane_to_rh_minor == ((1, 0),)
    assert encoding.register_to_rh_major == (1, 1)
    assert encoding.register_to_rh_minor == (0, 2)


def test_a_encoding_gfx90a_16x16x16_f16_fields() -> None:
    encoding = a_warp_encoding(_traits("mfma_f32_16x16x16f16"))
    # ABK=4, AKN=1, AR=1, K=16, M=16 -> Hs = ((16,), (1, 4, 4)).
    assert encoding.replication_lengths == (1,)
    assert encoding.hierarchical_lengths == ((16,), (1, 4, 4))
    assert encoding.lane_to_rh_major == ((2, 0, 1),)
    assert encoding.lane_to_rh_minor == ((1, 0, 0),)
    assert encoding.register_to_rh_major == (2, 2)
    assert encoding.register_to_rh_minor == (0, 2)


def test_b_encoding_uses_n_as_major_dim() -> None:
    encoding = b_warp_encoding(_traits("mfma_f32_16x16x16f16"))
    assert encoding.hierarchical_lengths[0] == (16,)  # major dim = N = 16


@pytest.mark.parametrize(
    "op_id",
    [
        "mfma_f32_16x16x16f16",
        "mfma_f32_32x32x8f16",
        "mfma_i32_16x16x16i8",
        "mfma_f32_16x16x8bf16",
    ],
)
def test_bijection_holds_for_gfx90a_dense_atoms(op_id: str) -> None:
    # Construction runs __post_init__ bijection validation; must not raise.
    traits = _traits(op_id)
    for encoding in (
        a_warp_encoding(traits),
        b_warp_encoding(traits),
        c_warp_encoding(traits),
    ):
        assert isinstance(encoding, WarpDistributionEncoding)


def test_duplicate_bucket_reference_is_rejected() -> None:
    # Two register dims claiming the same (major, minor) bucket -> not a bijection.
    with pytest.raises(ValueError) as excinfo:
        WarpDistributionEncoding(
            replication_lengths=(),
            hierarchical_lengths=((4, 4),),
            lane_to_rh_major=((1,),),
            lane_to_rh_minor=((0,),),
            register_to_rh_major=(1, 1),
            register_to_rh_minor=(1, 1),  # both claim X-dim0 level1
        )
    assert "not a bijection" in str(excinfo.value)


def test_uncovered_h_bucket_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        WarpDistributionEncoding(
            replication_lengths=(),
            hierarchical_lengths=((4, 4),),  # two levels
            lane_to_rh_major=((1,),),
            lane_to_rh_minor=((0,),),  # only level 0 claimed
            register_to_rh_major=(),
            register_to_rh_minor=(),
        )
    assert "no contributor" in str(excinfo.value)


def test_non_divisible_c_shape_fails_fast() -> None:
    good = _traits("mfma_f32_16x16x16f16")
    bad = dataclasses.replace(good, m=15)  # 15 % c_m_per_lane(4) != 0
    with pytest.raises(ValueError) as excinfo:
        c_warp_encoding(bad)
    assert "not divisible" in str(excinfo.value)


def test_encoder_supports_a_single_block_instruction() -> None:
    assert encoder_supports_layout(_traits("mfma_f32_16x16x16f16")) == (True, "")


@pytest.mark.parametrize(
    "encoding_fn", [a_warp_encoding, b_warp_encoding, c_warp_encoding]
)
def test_multi_block_instruction_is_a_named_encoder_boundary(encoding_fn) -> None:
    # mfma_f32_16x16x1f32 computes 4 independent 16x16 blocks per instruction. The encoder does not
    # model the block axis, so every operand encoding refuses it by name instead of laying out one
    # block as if it were the whole instruction.
    traits = _traits("mfma_f32_16x16x1f32")
    supported, reason = encoder_supports_layout(traits)
    assert not supported
    with pytest.raises(NotImplementedError) as excinfo:
        encoding_fn(traits)
    assert str(excinfo.value) == (
        "MMA instruction 'mfma_f32_16x16x1f32' cannot be laid out by the tiling encoder: it "
        "computes 4 independent blocks per instruction and the block axis is not modelled "
        "-- TODO: not implemented yet"
    )
    assert reason in str(excinfo.value)


def test_a_operand_with_two_m_register_blocks_is_two_stacked_row_blocks() -> None:
    # wmma_f32_32x16x128_f4 holds A's 32 rows as two blocks of 16: the lanes span 16 rows and the
    # second block sits in the atom's most significant registers. That is exactly a 16-row A
    # operand stacked twice along M.
    traits = _traits("wmma_f32_32x16x128_f4")
    assert traits.layout_params.a_m_register_blocks == 2
    one_block = dataclasses.replace(
        traits,
        m=16,
        layout_params=dataclasses.replace(traits.layout_params, a_m_register_blocks=1),
    )
    assert a_warp_encoding(traits) == a_warp_encoding(one_block, m_iter=2)


def test_one_m_register_block_adds_no_encoding_level() -> None:
    # Every other row has one M block of A per lane, and its A encoding is the CK encoding: the
    # lanes span all M rows, with no extra register level.
    for traits in load_mma_traits().by_op_id.values():
        if traits.b == 1 and traits.layout_params.a_m_register_blocks == 1:
            assert a_warp_encoding(traits).hierarchical_lengths[0] == (
                traits.m,
            ), traits.op_id
