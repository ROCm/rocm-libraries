# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocke.helpers.tiling.traits.mma_traits.

Anchors on the gfx90a M1 atom (mfma_f32_16x16x16f16) whose field values come from the traits
table. Also checks fail-fast behavior (reserved block ops, bad input, construction validation).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from rocke.helpers.tiling.traits import (
    DEFAULT_TRAITS_PATH,
    MmaTraits,
    load_mma_traits,
)


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
    assert (traits.m, traits.n, traits.k, traits.b, traits.r, traits.s) == (
        16,
        16,
        16,
        1,
        1,
        1,
    )
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


# ---- element widths -------------------------------------------------------------------------------


def _all_records():
    """Every operation record of the committed table, as written (the catalog keeps one record per
    op_id, and the f8f6f4 instructions have one record per format family)."""
    return json.loads(DEFAULT_TRAITS_PATH.read_text())["operations"]


def test_every_record_names_known_dtype_tags() -> None:
    # The table names operand types by instruction-name tags; each must map to dtypes of one width.
    # The loader turns a construction error into a reserved entry, so check the records directly.
    for record in _all_records():
        try:
            traits = MmaTraits.from_record(record)
        except NotImplementedError:
            continue  # block-hiding rows are reserved by design
        except ValueError as reason:
            assert "dtype tag" not in str(reason), reason
            continue  # rows without CK layout params are reserved by design
        assert all(bits > 0 for bits in traits.element_bits()), traits.op_id


def _record_bits(op_id, input_tag):
    (record,) = [
        r
        for r in _all_records()
        if r["op_id"] == op_id and r["input_dtype"] == input_tag
    ]
    return MmaTraits.from_record(record).element_bits()


@pytest.mark.parametrize(
    "op_id,input_tag,bits",
    [
        ("mfma_f32_16x16x16f16", "f16", (16, 16, 32, 32)),
        # Mixed fp8 / bf8 operands: both 8 bits (not the 88 a digit parse of the tag gives).
        ("mfma_f32_16x16x32_fp8_bf8", "fp8bf8", (8, 8, 32, 32)),
        ("mfma_f32_16x16x32_bf8_fp8", "bf8f8", (8, 8, 32, 32)),
        # One record per format family of the f8f6f4 instructions.
        ("mfma_scale_f32_16x16x128_f8f6f4", "f8", (8, 8, 32, 32)),
        ("mfma_scale_f32_16x16x128_f8f6f4", "f6", (6, 6, 32, 32)),
        ("mfma_scale_f32_16x16x128_f8f6f4", "f4", (4, 4, 32, 32)),
        # Accumulates in f32, returns bf16.
        ("wmma_bf16f32_16x16x32_bf16", "bf16", (16, 16, 32, 16)),
    ],
)
def test_element_bits(op_id, input_tag, bits) -> None:
    assert _record_bits(op_id, input_tag) == bits


def test_unknown_dtype_tag_fails_construction() -> None:
    kwargs = _valid_traits_kwargs()
    kwargs["input_dtype"] = "f12"
    with pytest.raises(ValueError) as excinfo:
        MmaTraits(**kwargs)
    assert "unknown MMA input dtype tag" in str(excinfo.value)
    assert "input_dtype='f12'" in str(excinfo.value)
