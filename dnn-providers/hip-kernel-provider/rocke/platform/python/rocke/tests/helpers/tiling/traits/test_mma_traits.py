# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocke.helpers.tiling.traits.mma_traits.

Anchors on the gfx90a M1 atom (mfma_f32_16x16x16f16) whose field values come from the traits
table. Also checks fail-fast behavior (reserved block ops, bad input, construction validation).
"""

from __future__ import annotations

import dataclasses

import pytest

from rocke.helpers.tiling.traits import MmaTraits, load_mma_traits


def _m1_atom():
    catalog = load_mma_traits()
    return catalog, catalog.select(
        target="gfx90a",
        input_dtype="f16",
        output_dtype="f32",
        m=16,
        n=16,
        k=16,
    )


def test_catalog_loads_and_reports_source_of_truth() -> None:
    catalog = load_mma_traits()
    assert catalog.by_op_id, "catalog should contain usable intrinsics"
    assert catalog.source_of_truth == "intrinsic support matrix (MFMA/WMMA)"


def test_gfx90a_16x16x16_f16_matches_sot() -> None:
    _catalog, traits = _m1_atom()
    # Field values come from the traits table (the gfx90a 16x16x16 f16 MFMA row).
    assert traits.op_id == "mfma_f32_16x16x16f16"
    assert traits.wave_size == 64
    assert (traits.m, traits.n, traits.k, traits.b, traits.r, traits.s) == (16, 16, 16, 1, 1, 1)
    assert traits.k_ab_per_lane == 4  # ABK
    assert traits.a_k_num_access == 1  # AKN
    assert traits.c_m_per_lane == 4  # CM
    assert traits.c_m_num_access == 1  # CMN
    assert traits.a_layout == "K{4} L{K1M} V{K0}"
    assert traits.c_d_layout == "M{4} L{M1N} V{M0}"
    assert traits.supports("gfx90a")


def test_select_unknown_combo_raises_valueerror_with_query() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError) as excinfo:
        catalog.select(
            target="gfx90a", input_dtype="f16", output_dtype="f32", m=13, n=16, k=16
        )
    message = str(excinfo.value)
    assert "no MMA intrinsic" in message
    assert "(13,16,16)" in message  # names the bad shape
    assert "available on 'gfx90a'" in message  # lists candidates


def test_block_hiding_op_is_reserved_not_missing() -> None:
    catalog = load_mma_traits()
    # mfma_f32_16x16x1f32 carries block markers (M="16X"); it must be reserved.
    assert "mfma_f32_16x16x1f32" in catalog.reserved
    with pytest.raises(NotImplementedError) as excinfo:
        catalog.get("mfma_f32_16x16x1f32")
    assert "block-hiding" in str(excinfo.value)


def test_get_truly_unknown_op_raises_valueerror() -> None:
    catalog = load_mma_traits()
    with pytest.raises(ValueError) as excinfo:
        catalog.get("mfma_does_not_exist")
    assert "unknown op_id" in str(excinfo.value)


def _valid_traits_kwargs() -> dict:
    _catalog, traits = _m1_atom()
    return dataclasses.asdict(traits)


def test_non_positive_dim_fails_construction() -> None:
    kwargs = _valid_traits_kwargs()
    kwargs["m"] = 0
    with pytest.raises(ValueError) as excinfo:
        MmaTraits(**kwargs)
    assert "non-positive fragment dim" in str(excinfo.value)
    assert "m=0" in str(excinfo.value)


def test_invalid_wave_size_fails_construction() -> None:
    kwargs = _valid_traits_kwargs()
    kwargs["wave_size"] = 17
    with pytest.raises(ValueError) as excinfo:
        MmaTraits(**kwargs)
    assert "invalid wave_size" in str(excinfo.value)


def test_unknown_family_fails_construction() -> None:
    kwargs = _valid_traits_kwargs()
    kwargs["family"] = "quantum"
    with pytest.raises(ValueError) as excinfo:
        MmaTraits(**kwargs)
    assert "unknown MMA family" in str(excinfo.value)
