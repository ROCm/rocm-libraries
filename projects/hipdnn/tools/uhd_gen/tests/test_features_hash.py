# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-language feature identity and canonical inline authoring boundaries.

RFC 0019 §6.3 gives `features_hash` one definition, shared by the tool that stamps it
and the loader that verifies it. This file used to assert agreement between two
implementations by pinning the same literal digest on both sides of the language
boundary, which held only until someone changed one canonicalisation; now there is a
single definition (FeatureExtractor::computeHash, reached through the shared evaluator)
and these tests check the properties that definition has to have.
"""
import os
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from uhd_gen import features
from uhd_gen.features import (
    build_features_signature, compute_features_hash, evaluate_feature_rows,
    parse_signature_entry, signature_references,
)

#: The digest TestFeatureExtractor.cpp pins for this signature. Retained deliberately:
#: it no longer guards the algorithm -- that is C++'s and is pinned there -- but it does
#: guard what uhd_gen SENDS. A request that reordered entries, stringified an inline AST
#: or dropped the categorical map would still produce a well-formed digest from the one
#: true implementation, and nothing else in either language would notice.
RAW_SIGNATURE = ["$q.batch", "$kernel.tile_m", "$device.cu_count"]
RAW_DIGEST = "sha256:fe9d0487031089e0"


def test_published_names_are_not_rewritten_into_q_namespace():
    assert build_features_signature(["attention.query.dims[2]", "M", "kernel.tile_m"]) == [
        "$attention.query.dims[2]", "$M", "$kernel.tile_m",
    ]


@pytest.mark.parametrize("entry", ['"$q.batch"', '{"log2":["$q.batch"]}', "q.batch", "", 1, None, []])
def test_legacy_stringified_or_noncanonical_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        parse_signature_entry(entry)


def test_a_supplied_evaluator_that_cannot_run_is_refused_not_rehashed_in_python(monkeypatch, tmp_path):
    """The behaviour that replaced the second implementation.

    Raw-reference signatures used to take a pure-Python digest, so generation succeeded
    on a machine with nothing built and the disagreement -- if the two canonicalisations
    had ever drifted -- surfaced as a descriptor the runtime refuses to load. There is
    now nothing to fall back to, and a name that was asked for and cannot be run is
    refused by name rather than quietly replaced by whatever else is around.
    """
    monkeypatch.setenv(features.EVALUATOR_ENV_VAR, "hipdnn_uhd_features_not_installed")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="hipdnn_uhd_features_not_installed"):
        compute_features_hash(RAW_SIGNATURE)


def test_an_installed_evaluator_is_found_without_path_or_environment(monkeypatch, tmp_path):
    """Discovery is relative, so a tree mounted at a different root still resolves.

    A committed batch script that named the executable absolutely ran on the login node
    and failed inside the container, where the same tree is mounted elsewhere. Nothing
    committed should have to know the root: `<prefix>/bin` is where CMAKE_INSTALL_BINDIR
    puts it, and finding it there means no script needs a path at all.
    """
    monkeypatch.delenv(features.EVALUATOR_ENV_VAR, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    stub = tmp_path / "bin" / (features.EVALUATOR_NAME + (".exe" if os.name == "nt" else ""))
    stub.parent.mkdir()
    stub.write_text("")
    stub.chmod(0o755)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert Path(features.resolve_feature_evaluator()).samefile(stub)


def test_no_evaluator_anywhere_names_the_variable_that_would_supply_one(monkeypatch, tmp_path):
    monkeypatch.delenv(features.EVALUATOR_ENV_VAR, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    # The search roots come from this file's location, and this file lives in a checkout
    # that has a build tree in it; overriding them is the only way to ask what a machine
    # without one is told.
    monkeypatch.setattr(features, "_evaluator_search_roots", lambda: [tmp_path])
    with pytest.raises(ValueError, match=features.EVALUATOR_ENV_VAR):
        compute_features_hash(RAW_SIGNATURE)


def test_the_request_uhd_gen_builds_reaches_the_runtimes_hash_unaltered(evaluator):
    """A raw-reference signature is hashed by the same routine the loader uses."""
    assert compute_features_hash(RAW_SIGNATURE, executable=evaluator) == RAW_DIGEST


@pytest.mark.parametrize("signature", [
    RAW_SIGNATURE,
    ["$q.batch", {"*": ["$q.batch", "$q.num_heads"]}],
    [{"log2": [{"*": ["$q.batch", "$q.num_heads"]}]}],
])
def test_both_entry_points_report_one_digest_for_a_signature(signature, evaluator):
    """Whether a corpus is being extracted or not cannot change model identity.

    `evaluate_feature_rows` stamps the descriptor during training and
    `compute_features_hash` restamps it after constant pruning and recomputes it during
    evaluation. A fork between them -- which is exactly what existed, keyed off whether
    the signature contained an inline expression -- publishes a model whose declared
    identity depends on which code path happened to produce it.
    """
    references = [reference[1:] for reference in signature_references(signature)]
    corpus = pd.DataFrame({reference: [4.0, 8.0] for reference in references})
    digest, values = evaluate_feature_rows(corpus, signature, executable=evaluator)
    assert len(values) == 2
    assert digest == compute_features_hash(signature, executable=evaluator)


def test_signature_order_changes_model_identity(evaluator):
    signature = ["$q.batch", "$kernel.tile_m"]
    assert (compute_features_hash(signature, executable=evaluator)
            != compute_features_hash(list(reversed(signature)), executable=evaluator))


def test_changing_only_an_expression_changes_the_fingerprint(evaluator):
    """Why an expression has to be *in* the signature rather than named beside it.

    Two signatures that compute reciprocal quantities from the same two fields read
    alike to anything that inspects references only, so a model would consume something
    other than what it was trained on. Here the expression is the entry, so the change
    is in the canonical form and the hash moves with it -- the same hole §6.5 describes
    for the categorical encoding, closed by the same mechanism.
    """
    intensity = [{"/": ["$q.flops", "$q.bytes"]}]
    flipped = [{"/": ["$q.bytes", "$q.flops"]}]
    assert (compute_features_hash(intensity, executable=evaluator)
            != compute_features_hash(flipped, executable=evaluator))


def test_categorical_codes_not_mapping_insertion_order_define_identity(evaluator):
    signature = ["$kernel.dtype"]
    a = {"$kernel.dtype": {"fp32": 1, "fp16": 0}}
    b = {"$kernel.dtype": {"fp16": 0, "fp32": 1}}
    swapped = {"$kernel.dtype": {"fp16": 1, "fp32": 0}}
    digest = compute_features_hash(signature, a, evaluator)
    assert digest == compute_features_hash(signature, b, evaluator)
    assert digest != compute_features_hash(signature, swapped, evaluator)
    assert digest != compute_features_hash(signature, executable=evaluator)


def test_empty_encoding_preserves_existing_raw_reference_hash(evaluator):
    """An encoding nobody declared is not a contract change.

    Descriptors generated before §6.5 existed carry no categorical map, and they must
    keep loading against models trained since; `{}` therefore has to hash as absent
    rather than as an empty vocabulary.
    """
    assert compute_features_hash(RAW_SIGNATURE, {}, evaluator) == RAW_DIGEST


@pytest.mark.parametrize("literal", [1e15, -1e15, 18446744073709551616, float("nan"), float("inf")])
def test_nonportable_literals_are_rejected_inside_nested_ast(literal):
    # Rejected while parsing the entry, before any request is built, so this holds on a
    # machine with no evaluator: a literal C++ cannot round-trip never reaches a digest.
    with pytest.raises(ValueError):
        compute_features_hash([{"log2": [{"+": ["$attention.dims[2]", literal]}]}])


def test_reference_collection_preserves_generic_and_categorical_dependencies():
    signature = ["$kernel.tile", {"if": [{"==": ["$attention.dtype", "fp16"]},
                                        {"/": ["$attention.dims[2]", "$device.cu_count"]}, 0]}]
    assert signature_references(signature) == ["$kernel.tile", "$attention.dtype", "$attention.dims[2]", "$device.cu_count"]
