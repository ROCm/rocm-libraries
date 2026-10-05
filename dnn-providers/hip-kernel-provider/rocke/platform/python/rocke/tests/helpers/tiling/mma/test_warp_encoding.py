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
    for encoding in (a_warp_encoding(traits), b_warp_encoding(traits), c_warp_encoding(traits)):
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
