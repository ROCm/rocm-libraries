# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-language feature identity and canonical inline authoring boundaries."""
import pytest

from uhd_gen.features import (
    build_features_signature, canonicalize_signature, compute_features_hash,
    parse_signature_entry, signature_references,
)


def test_published_names_are_not_rewritten_into_q_namespace():
    assert build_features_signature(["attention.query.dims[2]", "M", "kernel.tile_m"]) == [
        "$attention.query.dims[2]", "$M", "$kernel.tile_m",
    ]


@pytest.mark.parametrize("entry", ['"$q.batch"', '{"log2":["$q.batch"]}', "q.batch", "", 1, None, []])
def test_legacy_stringified_or_noncanonical_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        parse_signature_entry(entry)


@pytest.mark.parametrize(("signature", "canonical", "digest"), [
    (["$q.batch", "$kernel.tile_m", "$device.cu_count"],
     '["$q.batch","$kernel.tile_m","$device.cu_count"]', "sha256:fe9d0487031089e0"),
    (["$q.batch", {"*": ["$q.batch", "$q.num_heads"]}],
     '["$q.batch",{"*":["$q.batch","$q.num_heads"]}]', "sha256:d5ae6976facefe74"),
    ([{"log2": [{"*": ["$q.batch", "$q.num_heads"]}]}],
     '[{"log2":[{"*":["$q.batch","$q.num_heads"]}]}]', "sha256:8f014cf81bab5f8c"),
])
def test_hash_matches_runtime_canonical_ast(signature, canonical, digest):
    assert canonicalize_signature(signature) == canonical
    assert compute_features_hash(signature) == digest


def test_signature_order_changes_model_identity():
    signature = ["$q.batch", "$kernel.tile_m"]
    assert compute_features_hash(signature) != compute_features_hash(list(reversed(signature)))


def test_categorical_codes_not_mapping_insertion_order_define_identity():
    signature = ["$kernel.dtype"]
    a = {"$kernel.dtype": {"fp32": 1, "fp16": 0}}
    b = {"$kernel.dtype": {"fp16": 0, "fp32": 1}}
    swapped = {"$kernel.dtype": {"fp16": 1, "fp32": 0}}
    assert compute_features_hash(signature, a) == compute_features_hash(signature, b)
    assert compute_features_hash(signature, a) != compute_features_hash(signature, swapped)
    assert compute_features_hash(signature, a) != compute_features_hash(signature)


def test_empty_encoding_preserves_existing_raw_reference_hash():
    signature = ["$q.batch", "$kernel.tile_m", "$device.cu_count"]
    assert compute_features_hash(signature, {}) == "sha256:fe9d0487031089e0"


@pytest.mark.parametrize("literal", [1e15, -1e15, 18446744073709551616, float("nan"), float("inf")])
def test_nonportable_literals_are_rejected_inside_nested_ast(literal):
    with pytest.raises(ValueError):
        compute_features_hash([{"log2": [{"+": ["$attention.dims[2]", literal]}]}])


def test_reference_collection_preserves_generic_and_categorical_dependencies():
    signature = ["$kernel.tile", {"if": [{"==": ["$attention.dtype", "fp16"]},
                                        {"/": ["$attention.dims[2]", "$device.cu_count"]}, 0]}]
    assert signature_references(signature) == ["$kernel.tile", "$attention.dtype", "$attention.dims[2]", "$device.cu_count"]


def test_changing_only_an_expression_changes_the_fingerprint():
    """Why an expression has to be *in* the signature rather than named beside it.

    Two signatures that compute reciprocal quantities from the same two fields read
    alike to anything that inspects references only, so a model would consume something
    other than what it was trained on. Here the expression is the entry, so the change
    is in the canonical form and the hash moves with it -- the same hole §6.5 describes
    for the categorical encoding, closed by the same mechanism.
    """
    intensity = [{"/": ["$q.flops", "$q.bytes"]}]
    flipped = [{"/": ["$q.bytes", "$q.flops"]}]
    assert compute_features_hash(intensity) != compute_features_hash(flipped)
