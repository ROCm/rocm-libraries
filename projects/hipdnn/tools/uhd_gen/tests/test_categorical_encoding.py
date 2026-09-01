#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The categorical encoding is one table with two copies (RFC 0019 6.5).

The authoritative copy is CATEGORICAL_ENCODING_TABLE in
plugin_sdk/include/hipdnn_plugin_sdk/ingestor/uhd/CategoricalEncoding.hpp, which the
runtime reads; features.py holds the copy training encodes with. Nothing about two
hand-maintained tables is safe on its own -- the last time this repo kept a Python
writer and a C++ reader in step by hand, the Python FlatBuffer writer emitted a vtable
two slots short of what C++ expected and shipped green for months, because each side
only ever tested against itself.

So these tests do not check that features.py agrees with features.py. They parse the
C++ header and compare it entry for entry. Editing either side alone fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from uhd_gen.features import (
    CATEGORICAL_ENCODING,
    CATEGORICAL_ENCODING_FROZEN_DIGEST,
    CATEGORICAL_ENCODING_FROZEN_ENTRIES,
    CATEGORICAL_ENCODING_VERSION,
    categorical_encoding_digest,
    categorical_encoding_entries,
    category_of_reference,
    encode_categorical,
    encode_feature_value,
)

#: The C++ table this file is the mirror of. A relative path on purpose: a search would
#: turn a moved or deleted header into a skip, and a skipped pin is not a pin.
HEADER_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk"
    / "include"
    / "hipdnn_plugin_sdk"
    / "ingestor"
    / "uhd"
    / "CategoricalEncoding.hpp"
)

_ENTRY_RE = re.compile(r'\{"([^"]+)",\s*"([^"]+)",\s*(-?\d+)\}')


def _header_text() -> str:
    assert HEADER_PATH.is_file(), (
        f"{HEADER_PATH} is missing. It is the authoritative categorical encoding; "
        "without it this file is a table agreeing with itself."
    )
    return HEADER_PATH.read_text(encoding="utf-8")


def _cpp_entries() -> list[tuple[str, str, int]]:
    """Entries in CATEGORICAL_ENCODING_TABLE, in declaration order.

    Order carries the ordering rule (dtype by byte width) and the digest depends on it,
    so a reordering has to show up here as a difference.
    """
    text = _header_text()
    _, _, after = text.partition("CATEGORICAL_ENCODING_TABLE = {{")
    block, _, _ = after.partition("}};")
    assert block, "could not locate CATEGORICAL_ENCODING_TABLE in the header"
    return [(category, value, int(code)) for category, value, code in _ENTRY_RE.findall(block)]


def _cpp_constant(name: str) -> str:
    match = re.search(rf"{name}\s*(?:=|\n\s*=)\s*\"?([^\";\n]+)\"?\s*;", _header_text())
    assert match is not None, f"{name} is not declared in {HEADER_PATH.name}"
    return match.group(1).strip()


# ---- The two tables are the same table -----------------------------------------


def test_python_table_matches_the_cpp_header_entry_for_entry():
    """Fails if either side is edited alone -- which is the entire point of this file."""
    assert categorical_encoding_entries() == _cpp_entries()


def test_version_matches_the_cpp_header():
    assert str(CATEGORICAL_ENCODING_VERSION) == _cpp_constant("CATEGORICAL_ENCODING_VERSION")


def test_frozen_entry_count_matches_the_cpp_header():
    assert str(CATEGORICAL_ENCODING_FROZEN_ENTRIES) == _cpp_constant(
        "CATEGORICAL_ENCODING_FROZEN_ENTRIES"
    )


def test_frozen_digest_matches_the_cpp_header_and_the_python_table():
    """Three-way: the literal here, the literal in the header, and the digest actually
    computed from the Python table. A copy edited without its digest fails; a digest
    edited without its copy fails."""
    assert CATEGORICAL_ENCODING_FROZEN_DIGEST == _cpp_constant(
        "CATEGORICAL_ENCODING_FROZEN_DIGEST"
    )
    assert categorical_encoding_digest() == CATEGORICAL_ENCODING_FROZEN_DIGEST


def test_frozen_prefix_is_not_shortened():
    """The digest covers a prefix, so a deletion past its end would slip by it."""
    assert len(categorical_encoding_entries()) >= CATEGORICAL_ENCODING_FROZEN_ENTRIES


def test_appending_leaves_the_frozen_digest_alone():
    """The pin has to distinguish growth from mutation, or it gets edited away.

    Simulated rather than asserted about the real table: an append past the frozen
    prefix must not move the digest, while an edit inside it must.
    """
    from uhd_gen import features

    original = features.CATEGORICAL_ENCODING
    grown = {**{k: dict(v) for k, v in original.items()}, "pipeline": {"intrawave": 0}}
    try:
        features.CATEGORICAL_ENCODING = grown
        assert features.categorical_encoding_digest() == CATEGORICAL_ENCODING_FROZEN_DIGEST

        mutated = {k: dict(v) for k, v in original.items()}
        mutated["dtype"]["fp16"] = 99
        features.CATEGORICAL_ENCODING = mutated
        assert features.categorical_encoding_digest() != CATEGORICAL_ENCODING_FROZEN_DIGEST
    finally:
        features.CATEGORICAL_ENCODING = original


# ---- What the table means -------------------------------------------------------


def test_dtype_codes_ascend_with_byte_width():
    """RFC 0019 6.5 "Ordering": a tree splits on these numbers, so `dtype < 12` has to
    separate narrow precision from wide rather than nothing at all."""
    dtype = CATEGORICAL_ENCODING["dtype"]
    assert dtype["fp4_e2m1"] < dtype["fp8_e4m3"] < dtype["fp16"] < dtype["fp32"] < dtype["fp64"]


def test_dtype_members_are_the_spellings_the_library_produces():
    """to_string(DataType) in hipdnn_frontend/Types.hpp is the only place this library
    turns a data type into a string, so it is the only vocabulary a binding can hold.
    "float16" is the plausible near-miss that must not resolve."""
    assert encode_categorical("dtype", "fp16") == 13
    assert encode_categorical("dtype", "float16") is None
    assert encode_categorical("dtype", "unknown") is None


@pytest.mark.parametrize(
    ("reference", "category"),
    [
        ("$kernel.dtype", "dtype"),
        ("$q.dtype", "dtype"),
        ("$device.dtype", "dtype"),
        ("fp16", ""),
        ("$kernel", ""),
    ],
)
def test_category_is_the_field_not_the_namespace(reference, category):
    """`$kernel.dtype` and `$q.dtype` encode identically: the category belongs to the
    value, not to whoever holds it. That is what makes two engines' feature vectors
    comparable (RFC 0019 11.3). A bare literal is not a reference and never encodes."""
    assert category_of_reference(reference) == category


def test_same_value_encodes_identically_through_any_namespace():
    assert encode_feature_value("$kernel.dtype", "bf16") == encode_feature_value(
        "$q.dtype", "bf16"
    )


# ---- Encoding a logged value ----------------------------------------------------


def test_numbers_and_bools_pass_through():
    assert encode_feature_value("$kernel.tile_m", 64) == 64.0
    assert encode_feature_value("$kernel.split_k", True) == 1.0


def test_unencodable_value_in_a_known_category_raises():
    """Not NaN. A GBDT reads NaN as missing, routes it down default_left and returns an
    ordinary leaf, so the row would train as data with nothing in the log."""
    with pytest.raises(ValueError) as excinfo:
        encode_feature_value("$kernel.dtype", "float16")
    assert "float16" in str(excinfo.value)
    assert "dtype" in str(excinfo.value)


def test_string_in_a_category_with_no_table_raises():
    """`pipeline` has no encoding, so this string means nothing numerically -- a
    different failure from a known category's unknown value, and it reads differently."""
    with pytest.raises(ValueError) as excinfo:
        encode_feature_value("$kernel.pipeline", "intrawave")
    assert "no categorical encoding" in str(excinfo.value)
