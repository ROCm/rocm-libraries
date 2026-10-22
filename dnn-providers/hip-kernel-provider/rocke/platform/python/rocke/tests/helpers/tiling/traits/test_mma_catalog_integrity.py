# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integrity of the MMA catalog (``traits/data/mma_traits.json``) and its strict loader.

The catalog is hand-maintained, so these tests hold the invariants a hand edit could break: one
row per instruction variant, canonical dtype names, op_ids spelled from the builtin, replication
stated consistently, and a unique answer for every selection. The loader tests edit a copy of the
committed catalog in a temporary directory and check that each kind of data error fails the load
with its exact message.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections import Counter

import pytest

from rocke.core.dtypes import normalize_dtype
from rocke.helpers.tiling.traits import (
    DEFAULT_TRAITS_PATH,
    load_mma_traits,
)
from rocke.helpers.tiling.traits.mma_traits import _selection_key

_BUILTIN_PREFIX = "__builtin_amdgcn_"


def _document() -> dict:
    return json.loads(DEFAULT_TRAITS_PATH.read_text())


def _row(document: dict, op_id: str) -> dict:
    (row,) = [r for r in document["operations"] if r["op_id"] == op_id]
    return row


def _load_edited(tmp_path, document: dict):
    path = tmp_path / "mma_traits.json"
    path.write_text(json.dumps(document))
    return load_mma_traits(path)


def _load_error(tmp_path, document: dict) -> str:
    with pytest.raises(ValueError) as excinfo:
        _load_edited(tmp_path, document)
    return str(excinfo.value)


# ---- the committed catalog ------------------------------------------------------------------------


def test_op_ids_are_unique() -> None:
    counts = Counter(row["op_id"] for row in _document()["operations"])
    assert [op_id for op_id, n in counts.items() if n > 1] == []


def test_dtypes_are_canonical_names() -> None:
    for row in _document()["operations"]:
        for field in ("a_dtype", "b_dtype", "c_dtype", "d_dtype"):
            assert normalize_dtype(row[field]) == row[field], (row["op_id"], field)


def test_op_id_is_the_builtin_name_plus_the_formats_of_a_multi_format_builtin() -> None:
    # An op_id is its builtin without the __builtin_amdgcn_ prefix. A builtin whose A and B formats
    # are picked by an operand (the f8f6f4 instructions) has one row per format pair, named
    # '<stem>_<a_dtype>_<b_dtype>'.
    rows = _document()["operations"]
    rows_per_builtin = Counter(row["llvm_builtin"] for row in rows)
    for row in rows:
        assert row["llvm_builtin"].startswith(_BUILTIN_PREFIX), row["op_id"]
        stem = row["llvm_builtin"][len(_BUILTIN_PREFIX) :]
        if rows_per_builtin[row["llvm_builtin"]] == 1:
            assert row["op_id"] == stem
        else:
            assert row["op_id"] == f"{stem}_{row['a_dtype']}_{row['b_dtype']}"


def test_replication_is_stated_once() -> None:
    # dims R is the operand replication; the A and B repeats must both equal it.
    for traits in load_mma_traits().by_op_id.values():
        params = traits.layout_params
        assert traits.r == params.a_repeat == params.b_repeat, traits.op_id


def test_every_selection_has_one_answer_and_every_pin_only_row_one_twin() -> None:
    # Selection keys are unique among selectable rows (the loader enforces it); a pin-only row exists
    # only because it answers the same selection as exactly one selectable row on each target.
    rows = list(load_mma_traits().by_op_id.values())
    answers = Counter(
        _selection_key(t, target)
        for t in rows
        if t.b == 1 and t.auto_select
        for target in t.supported_targets
    )
    assert [key for key, n in answers.items() if n > 1] == []
    pin_only = [t for t in rows if not t.auto_select]
    assert pin_only, "the catalog has pin-only rows"
    for traits in pin_only:
        for target in traits.supported_targets:
            assert answers[_selection_key(traits, target)] == 1, (traits.op_id, target)


# ---- loader errors --------------------------------------------------------------------------------


def test_committed_catalog_loads_through_an_edited_copy(tmp_path) -> None:
    assert _load_edited(tmp_path, _document()).by_op_id


def test_missing_table_fails(tmp_path) -> None:
    missing = tmp_path / "absent.json"
    with pytest.raises(ValueError) as excinfo:
        load_mma_traits(missing)
    assert str(excinfo.value) == f"mma_traits table not found -- path={missing}"


def test_old_schema_version_fails(tmp_path) -> None:
    document = _document()
    document["_meta"]["schema_version"] = 1
    assert _load_error(tmp_path, document) == (
        "mma_traits.json schema_version=1, expected 2 -- see helpers/tiling/traits/README.md"
    )


def test_row_with_missing_and_unexpected_fields_fails(tmp_path) -> None:
    document = _document()
    row = _row(document, "mfma_f32_32x32x8f16")
    del row["d_dtype"]
    row["note"] = "free text"
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_32x32x8f16': missing fields ['d_dtype']; unexpected "
        "fields ['note'] -- a row carries exactly the fields listed in "
        "helpers/tiling/traits/README.md"
    )
    del row["note"]
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_32x32x8f16': missing fields ['d_dtype'] -- a row "
        "carries exactly the fields listed in helpers/tiling/traits/README.md"
    )


def test_repeated_op_id_fails(tmp_path) -> None:
    document = _document()
    template = _row(document, "wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3")
    for _ in range(3):
        repeated = copy.deepcopy(template)
        repeated["op_id"] = "wmma_f32_16x16x128_f8f6f4"
        document["operations"].append(repeated)
    assert _load_error(tmp_path, document) == (
        "mma_traits.json: op_id 'wmma_f32_16x16x128_f8f6f4' appears 3 times -- one row per "
        "instruction variant; name multi-format variants '<stem>_<a_dtype>_<b_dtype>'"
    )


def test_non_integer_dim_or_layout_parameter_fails(tmp_path) -> None:
    document = _document()
    _row(document, "mfma_f32_32x32x1f32")["dims"]["M"] = "32X"
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_32x32x1f32': dims['M']='32X' is not a positive integer "
        "-- put the block count in dims['B']"
    )
    document = _document()
    _row(document, "mfma_f32_32x32x1f32")["layout_params"]["AR"] = "Y"
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_32x32x1f32': layout_params['AR']='Y' is not a positive "
        "integer -- put the block count in dims['B']"
    )


def test_non_canonical_dtype_fails(tmp_path) -> None:
    document = _document()
    _row(document, "mfma_f32_16x16x32_fp8_fp8")["a_dtype"] = "fp8"
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_16x16x32_fp8_fp8': a_dtype 'fp8' is not a canonical "
        "rocke dtype name -- use the name core.dtypes.dtype_info returns, e.g. 'fp8e4m3'"
    )


def test_two_selectable_rows_with_one_selection_fail(tmp_path) -> None:
    document = _document()
    _row(document, "wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3")["auto_select"] = True
    message = _load_error(tmp_path, document)
    assert message.startswith("mma_traits.json: rows [")
    assert "'wmma_f32_16x16x128_fp8_fp8'" in message
    assert "'wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp8e4m3'" in message
    assert message.endswith(
        "answer the same selection (gfx1250, dense, fp8e4m3 x fp8e4m3, c=fp32, d=fp32, "
        "16x16x128) -- set auto_select: false on all but one (README: Selection)"
    )


def test_rows_that_differ_only_in_result_dtype_are_distinct_selections(
    tmp_path,
) -> None:
    # The result dtype is part of a selection: the gfx1250 f32-result and bf16-result bf16
    # instructions load side by side, and become a conflict once their results agree.
    document = _document()
    pair = ["wmma_f32_16x16x32_bf16", "wmma_bf16f32_16x16x32_bf16"]
    document["operations"] = [_row(document, op_id) for op_id in pair]
    assert sorted(_load_edited(tmp_path, document).by_op_id) == sorted(pair)
    _row(document, "wmma_bf16f32_16x16x32_bf16")["d_dtype"] = "fp32"
    assert _load_error(tmp_path, document) == (
        "mma_traits.json: rows ['wmma_f32_16x16x32_bf16', 'wmma_bf16f32_16x16x32_bf16'] answer "
        "the same selection (gfx1250, dense, bf16 x bf16, c=fp32, d=fp32, 16x16x32) -- set "
        "auto_select: false on all but one (README: Selection)"
    )


def test_scale_block_size_is_part_of_a_scaled_selection(tmp_path) -> None:
    # The block-32 and block-16 scaled twins answer different selections until their block sizes
    # agree.
    document = _document()
    pair = [
        "wmma_scale_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1",
        "wmma_scale16_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1",
    ]
    document["operations"] = [_row(document, op_id) for op_id in pair]
    assert sorted(_load_edited(tmp_path, document).by_op_id) == sorted(pair)
    _row(document, pair[1])["scale"]["block_k"] = 32
    assert _load_error(tmp_path, document).endswith(
        "answer the same selection (gfx1250, scaled, fp4e2m1 x fp4e2m1, c=fp32, d=fp32, "
        "16x16x128, scale_block_k=32) -- set auto_select: false on all but one "
        "(README: Selection)"
    )


def test_scale_contract_follows_the_family(tmp_path) -> None:
    document = _document()
    _row(document, "mfma_f32_16x16x16f16")["scale"] = {
        "block_k": 32,
        "a_dtypes": ["e8m0"],
        "b_dtypes": ["e8m0"],
    }
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'mfma_f32_16x16x16f16': family 'dense' takes no scale contract"
    )
    document = _document()
    _row(document, "wmma_scale_f32_32x16x128_f4")["scale"] = None
    assert _load_error(tmp_path, document) == (
        "mma_traits.json row 'wmma_scale_f32_32x16x128_f4': family 'scaled' needs a scale "
        "contract"
    )


def test_layout_parameters_are_positive_integers() -> None:
    params = load_mma_traits().get("mfma_f32_16x16x16f16").layout_params
    with pytest.raises(ValueError) as excinfo:
        dataclasses.replace(params, a_m_register_blocks=0)
    assert str(excinfo.value) == (
        "layout parameter a_m_register_blocks=0 must be a positive integer"
    )
