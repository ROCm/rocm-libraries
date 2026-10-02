# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for features_hash identity and canonical signature entries.

The hash has one definition, FeatureExtractor::computeHash, reached through the shared
evaluator (RFC 0019 §6.3).
"""
import os
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from uhd_gen import features
from uhd_gen.features import (
    build_features_signature,
    compute_features_hash,
    evaluate_feature_rows,
    parse_signature_entry,
    signature_references,
)

#: Same digest TestFeatureExtractor.cpp pins; here it guards the request uhd_gen sends
#: (entry order, inline ASTs, categorical map), not the hash algorithm.
RAW_SIGNATURE = ["$q.batch", "$kernel.tile_m", "$device.cu_count"]
RAW_DIGEST = "sha256:fe9d0487031089e0"


def test_published_names_are_not_rewritten_into_q_namespace():
    assert build_features_signature(
        ["attention.query.dims[2]", "M", "kernel.tile_m"]
    ) == [
        "$attention.query.dims[2]",
        "$M",
        "$kernel.tile_m",
    ]


@pytest.mark.parametrize(
    "entry", ['"$q.batch"', '{"log2":["$q.batch"]}', "q.batch", "", 1, None, []]
)
def test_legacy_stringified_or_noncanonical_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        parse_signature_entry(entry)


def test_a_supplied_evaluator_that_cannot_run_is_refused_not_rehashed_in_python(
    monkeypatch, tmp_path
):
    """There is no Python fallback hash; an unrunnable evaluator is an error."""
    monkeypatch.setenv(features.EVALUATOR_ENV_VAR, "hipdnn_uhd_features_not_installed")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match="hipdnn_uhd_features_not_installed"):
        compute_features_hash(RAW_SIGNATURE)


def test_an_installed_evaluator_is_found_without_path_or_environment(
    monkeypatch, tmp_path
):
    """Discovery via `<prefix>/bin` keeps working when the tree is mounted elsewhere."""
    monkeypatch.delenv(features.EVALUATOR_ENV_VAR, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    stub = (
        tmp_path
        / "bin"
        / (features.EVALUATOR_NAME + (".exe" if os.name == "nt" else ""))
    )
    stub.parent.mkdir()
    stub.write_text("")
    stub.chmod(0o755)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert Path(features.resolve_feature_evaluator()).samefile(stub)


def test_no_evaluator_anywhere_names_the_variable_that_would_supply_one(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(features.EVALUATOR_ENV_VAR, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    # The real search roots are relative to this checkout, which may have a build tree.
    monkeypatch.setattr(features, "_evaluator_search_roots", lambda: [tmp_path])
    with pytest.raises(ValueError, match=features.EVALUATOR_ENV_VAR):
        compute_features_hash(RAW_SIGNATURE)


def test_the_request_uhd_gen_builds_reaches_the_runtimes_hash_unaltered(evaluator):
    """A raw-reference signature is hashed by the same routine the loader uses."""
    assert compute_features_hash(RAW_SIGNATURE, executable=evaluator) == RAW_DIGEST


def test_the_feature_semantics_revision_is_the_evaluators_and_absent_means_1(
    evaluator_reporting,
):
    """The revision comes from the evaluator; reporting none means revision 1."""
    assert features.evaluator_feature_semantics_revision(evaluator_reporting(7)) == 7
    assert features.evaluator_feature_semantics_revision(evaluator_reporting(None)) == 1


@pytest.mark.parametrize("reported", [0, -1, True, 1.5, "1"])
def test_a_malformed_feature_semantics_revision_is_refused(
    evaluator_reporting, reported
):
    """Only positive ints pass; the loader rejects spellings like True or "1"."""
    with pytest.raises(ValueError, match="feature_semantics_revision"):
        features.evaluator_feature_semantics_revision(evaluator_reporting(reported))


@pytest.mark.parametrize(
    "signature",
    [
        RAW_SIGNATURE,
        ["$q.batch", {"*": ["$q.batch", "$q.num_heads"]}],
        [{"log2": [{"*": ["$q.batch", "$q.num_heads"]}]}],
    ],
)
def test_both_entry_points_report_one_digest_for_a_signature(signature, evaluator):
    """Training and evaluation stamp the hash via different entry points."""
    references = [reference[1:] for reference in signature_references(signature)]
    corpus = pd.DataFrame({reference: [4.0, 8.0] for reference in references})
    digest, values = evaluate_feature_rows(corpus, signature, executable=evaluator)
    assert len(values) == 2
    assert digest == compute_features_hash(signature, executable=evaluator)


def test_signature_order_changes_model_identity(evaluator):
    signature = ["$q.batch", "$kernel.tile_m"]
    assert compute_features_hash(
        signature, executable=evaluator
    ) != compute_features_hash(list(reversed(signature)), executable=evaluator)


def test_changing_only_an_expression_changes_the_fingerprint(evaluator):
    """Signatures with the same references but different expressions must differ."""
    intensity = [{"/": ["$q.flops", "$q.bytes"]}]
    flipped = [{"/": ["$q.bytes", "$q.flops"]}]
    assert compute_features_hash(
        intensity, executable=evaluator
    ) != compute_features_hash(flipped, executable=evaluator)


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
    """`{}` hashes as absent so descriptors without a categorical map still load."""
    assert compute_features_hash(RAW_SIGNATURE, {}, evaluator) == RAW_DIGEST


@pytest.mark.parametrize(
    "literal", [1e15, -1e15, 18446744073709551616, float("nan"), float("inf")]
)
def test_nonportable_literals_are_rejected_inside_nested_ast(literal):
    # Rejected at parse time, so no evaluator is needed.
    with pytest.raises(ValueError):
        compute_features_hash([{"log2": [{"+": ["$attention.dims[2]", literal]}]}])


def test_reference_collection_preserves_generic_and_categorical_dependencies():
    signature = [
        "$kernel.tile",
        {
            "if": [
                {"==": ["$attention.dtype", "fp16"]},
                {"/": ["$attention.dims[2]", "$device.cu_count"]},
                0,
            ]
        },
    ]
    assert signature_references(signature) == [
        "$kernel.tile",
        "$attention.dtype",
        "$attention.dims[2]",
        "$device.cu_count",
    ]
