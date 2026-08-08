#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the features_signature / features_hash contract (RFC 0019 7.2, 7.3).

The hash is a cross-language contract: this tool writes it into the descriptor and
the model artifact, and the C++ runtime recomputes it from the descriptor it loads
(FeatureExtractor::computeHash in backend/src/heuristics/uhd/). If the two
canonicalizations drift, every descriptor this tool emits stops validating at load.
"""
from __future__ import annotations

import json

import pytest

from uhd_gen.features import (
    build_features_signature,
    canonicalize_signature,
    compute_features_hash,
    parse_signature_entry,
)


def test_signature_uses_bare_field_references():
    """RFC 0019 7.2 canonical spelling is a bare `$ref`, not a quoted JSON string."""
    assert build_features_signature(["q.batch", "kernel.tile_m"]) == [
        "$q.batch",
        "$kernel.tile_m",
    ]


def test_hash_matches_runtime_canonical_form():
    """Pinned against FeatureExtractor::computeHash.

    The mirror of this assertion lives in
    backend/tests/heuristics/TestFeatureExtractor.cpp
    (HashMatchesGeneratorCanonicalForm). Both must be updated together, and only
    with a deliberate format change -- editing one alone silently breaks loading.
    """
    signature = ["$q.batch", "$kernel.tile_m", "$device.cu_count"]
    assert compute_features_hash(signature) == "sha256:fe9d0487031089e0"


def test_hash_is_order_sensitive():
    """RFC 0019 7.2 requires the signature to match training exactly.

    Hashing a sorted copy would make a permuted signature -- a real feature-contract
    break -- produce a matching hash.
    """
    signature = ["$q.batch", "$kernel.tile_m", "$device.cu_count"]
    permuted = list(reversed(signature))

    assert compute_features_hash(signature) != compute_features_hash(permuted)


def test_hash_is_stable_for_same_signature():
    signature = ["$q.batch", "$kernel.tile_m"]
    assert compute_features_hash(signature) == compute_features_hash(list(signature))


def test_hash_distinguishes_different_signatures():
    assert compute_features_hash(["$q.batch"]) != compute_features_hash(["$q.seqlen_q"])


def test_hash_has_expected_format():
    """23 chars: "sha256:" plus a 16-char truncated digest, matching the C++ side."""
    digest = compute_features_hash(["$q.batch"])
    assert digest.startswith("sha256:")
    assert len(digest) == 23


def test_raw_column_names_are_rejected():
    """The signature is what gets hashed -- raw column names are not a signature.

    ``q.batch`` is neither a bare ``$ref`` nor valid JSON, so it cannot be
    canonicalized. Failing here beats hashing it as if it were a valid signature: the
    resulting digest would embed in a descriptor and never match the runtime.
    """
    columns = ["q.batch", "kernel.tile_m"]

    with pytest.raises(json.JSONDecodeError):
        compute_features_hash(columns)

    # The signature built from those columns is well-formed.
    assert compute_features_hash(build_features_signature(columns))


# ---- Canonicalization is structural, not textual -------------------------------
#
# Hashing raw entry strings would make the two legal spellings of a reference hash
# differently, and would hash a derived expression as an opaque string here while the
# runtime hashes it as a parsed node. Every case below is pinned against the C++ value
# produced by FeatureExtractor::computeHash for the same input.


def test_parse_lifts_bare_reference_without_parsing():
    assert parse_signature_entry("$q.batch") == "$q.batch"


def test_parse_reads_derived_expression_as_structure():
    assert parse_signature_entry('{"log2": ["$q.batch"]}') == {"log2": ["$q.batch"]}


def test_bare_and_prequoted_reference_agree():
    """`$q.batch` and `"$q.batch"` are the same reference and must hash the same."""
    assert compute_features_hash(["$q.batch"]) == compute_features_hash(['"$q.batch"'])


@pytest.mark.parametrize(
    ("signature", "canonical", "digest"),
    [
        (
            ["$q.batch", "$kernel.tile_m", "$device.cu_count"],
            '["$q.batch","$kernel.tile_m","$device.cu_count"]',
            "sha256:fe9d0487031089e0",
        ),
        (['"$q.batch"'], '["$q.batch"]', "sha256:611513da8e8614b2"),
        (
            ["$q.batch", '{"*": ["$q.batch", "$q.num_heads"]}'],
            '["$q.batch",{"*":["$q.batch","$q.num_heads"]}]',
            "sha256:d5ae6976facefe74",
        ),
        (
            ['{"log2": [{"*": ["$q.batch", "$q.num_heads"]}]}'],
            '[{"log2":[{"*":["$q.batch","$q.num_heads"]}]}]',
            "sha256:8f014cf81bab5f8c",
        ),
        ([], "[]", "sha256:4f53cda18c2baa0c"),
    ],
    ids=["bare", "prequoted", "derived", "nested", "empty"],
)
def test_canonical_form_matches_runtime(signature, canonical, digest):
    """Mirrored in backend/tests/heuristics/TestFeatureExtractor.cpp.

    Both sides must be updated together. Editing one alone silently breaks loading for
    every descriptor this tool emits.
    """
    assert canonicalize_signature(signature) == canonical
    assert compute_features_hash(signature) == digest


def test_canonical_form_has_no_whitespace():
    """nlohmann::json::dump() emits none; a stray space changes the digest."""
    canonical = canonicalize_signature(["$q.batch", '{"log2": ["$q.batch"]}'])
    assert " " not in canonical


# ---- Numeric literals the two languages render differently ----------------------
#
# Strings, keys, escaping and unicode canonicalize identically. Numbers are the one
# axis where Python and nlohmann disagree, so literals near or past the divergence are
# rejected rather than hashed into a digest the runtime cannot reproduce. Mirrored by
# backend/tests/heuristics/TestFeatureExtractor.cpp.


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ("1e15", "nlohmann renders 1e+15, Python 1000000000000000.0"),
        ("1000000000000000.0", "same value, same divergence"),
        ("1234567890123456.0", "mid-decade"),
        ('{">": ["$q.batch", 1e15]}', "reachable through a derived expression"),
        ('{"log2": [{"+": ["$q.batch", 1e16]}]}', "nested inside an operator tree"),
        ("18446744073709551616", "beyond uint64; nlohmann degrades to double"),
        ("-9223372036854775809", "beyond int64 min"),
        ("123456789012345678901234567890", "bignum"),
        ("NaN", "Python json extension, nlohmann rejects"),
        ("Infinity", "Python json extension, nlohmann rejects"),
        ("-Infinity", "Python json extension, nlohmann rejects"),
        ("1e400", "overflows to inf in Python, out_of_range in nlohmann"),
    ],
)
def test_rejects_literals_the_runtime_renders_differently(entry, why):
    with pytest.raises(ValueError):
        compute_features_hash([entry])
    assert why  # documents the divergence for the reader


def test_integer_overflow_would_have_collided():
    """nlohmann maps these two distinct values to one double, so it would accept a
    hash computed over a different signature. Both must be rejected outright."""
    for entry in ("18446744073709551616", "18446744073709551617"):
        with pytest.raises(ValueError):
            compute_features_hash([entry])


def test_accepts_literal_just_below_threshold():
    """Boundary, and a cross-language pin against the C++ digest."""
    assert compute_features_hash(["999999999999999.0"]) == "sha256:1449061ef40ea91e"


@pytest.mark.parametrize(
    "entry",
    [
        '{">": ["$q.seqlen_q", 4096]}',
        '{"*": ["$kernel.tile_m", 0.5]}',
        "1e14",
        "-1e14",
        "0",
        "-0.0",
        "5e-324",
        "true",
        "false",
        "null",
    ],
)
def test_accepts_realistic_feature_literals(entry):
    # Tile sizes, dimensions and thresholds are orders of magnitude clear of the bound.
    assert compute_features_hash([entry]).startswith("sha256:")


def test_booleans_are_not_treated_as_numbers():
    """bool subclasses int in Python; the validator must not fall into the numeric
    branch for True/False."""
    assert compute_features_hash(["true"]) != compute_features_hash(["false"])
