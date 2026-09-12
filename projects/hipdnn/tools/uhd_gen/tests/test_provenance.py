# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor snapshots name real owned dependencies, never inferred versions."""
import copy
import json

import pytest

from uhd_gen.provenance import ProvenanceError, compare_provenance, snapshot_provenance, validate_provenance

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
MATCH_A = "11be5fe7-02a7-4ec2-b79c-e849951f8c24"
MATCH_B = "22be5fe7-02a7-4ec2-b79c-e849951f8c24"
PACK_A = "33be5fe7-02a7-4ec2-b79c-e849951f8c24"
PACK_B = "44be5fe7-02a7-4ec2-b79c-e849951f8c24"


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "1.0", **doc}), encoding="utf-8")


def _tree(root):
    _write(root / "engine.ued.json", {"id": UED, "name": "test:engine", "metadata": KMD})
    _write(root / "metadata.kmd.json", {"id": KMD, "revision": "3.12"})
    _write(root / "a.umd.json", {"id": MATCH_A, "revision": "2.10"})
    _write(root / "b.umd.json", {"id": MATCH_B})
    _write(root / "a.kdp.json", {"id": PACK_A, "engine": UED, "arch": ["gfx942"], "matchers": [MATCH_B, MATCH_A]})
    _write(root / "b.kdp.json", {"id": PACK_B, "engine": UED, "arch": ["gfx950"], "matchers": [MATCH_A]})
    return root


def test_snapshot_resolves_actual_semantic_revisions_and_sorted_matchers(tmp_path):
    tree = _tree(tmp_path)
    assert snapshot_provenance(tree, "test:engine", "gfx942") == {
        "ued": {"id": UED, "revision": "1.0"},
        "kmd": {"id": KMD, "revision": "3.12"},
        "umd": [{"id": MATCH_A, "revision": "2.10"}, {"id": MATCH_B, "revision": "1.0"}],
    }
    assert snapshot_provenance(tree, arch="gfx950")["umd"] == [{"id": MATCH_A, "revision": "2.10"}]


@pytest.mark.parametrize("missing", ["metadata.kmd.json", "a.umd.json"])
def test_missing_dependency_never_receives_an_invented_version(tmp_path, missing):
    tree = _tree(tmp_path)
    (tree / missing).unlink()
    with pytest.raises(ProvenanceError):
        snapshot_provenance(tree, arch="gfx942")


def test_missing_non_target_matcher_does_not_poison_target_snapshot(tmp_path):
    tree = _tree(tmp_path)
    (tree / "b.umd.json").unlink()
    assert snapshot_provenance(tree, arch="gfx950")["umd"] == [{"id": MATCH_A, "revision": "2.10"}]


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_definitions_are_ambiguous_even_if_identical(tmp_path, conflicting):
    tree = _tree(tmp_path)
    _write(tree / "duplicate.kmd.json", {"id": KMD, "revision": "4.0" if conflicting else "3.12"})
    with pytest.raises(ProvenanceError):
        snapshot_provenance(tree)


def test_metadata_cannot_resolve_to_a_matcher_of_the_same_identity(tmp_path):
    tree = _tree(tmp_path)
    (tree / "metadata.kmd.json").unlink()
    _write(tree / "impostor.umd.json", {"id": KMD})
    with pytest.raises(ProvenanceError):
        snapshot_provenance(tree)


def test_compatible_minor_revisions_and_new_matchers_preserve_model_contract(tmp_path):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx950")
    _write(tree / "a.umd.json", {"id": MATCH_A, "revision": "2.11"})
    actual = snapshot_provenance(tree, arch="gfx942")
    compare_provenance(trained, actual)
    assert trained["umd"] == [{"id": MATCH_A, "revision": "2.10"}]


@pytest.mark.parametrize("kind", ["ued", "kmd", "umd"])
@pytest.mark.parametrize("change", ["identity", "major", "minor_regression"])
def test_incompatible_dependency_identity_or_revision_is_rejected(tmp_path, kind, change):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree)
    actual = copy.deepcopy(trained)
    entry = actual[kind][0] if kind == "umd" else actual[kind]
    if change == "identity":
        entry["id"] = PACK_A
    elif change == "major":
        entry["revision"] = "99.0"
    else:
        original = trained[kind][0] if kind == "umd" else trained[kind]
        major, minor = map(int, original["revision"].split("."))
        original["revision"] = f"{major}.{minor + 1}"
    with pytest.raises(ProvenanceError):
        compare_provenance(trained, actual)


def test_removed_matcher_is_a_contract_break_not_a_coverage_addition(tmp_path):
    tree = _tree(tmp_path)
    trained = snapshot_provenance(tree, arch="gfx942")
    with pytest.raises(ProvenanceError):
        compare_provenance(trained, snapshot_provenance(tree, arch="gfx950"))


@pytest.mark.parametrize("revision", [None, "1", "1.2.3", 1.0, "a.b"])
def test_malformed_explicit_revision_does_not_default(tmp_path, revision):
    tree = _tree(tmp_path)
    _write(tree / "metadata.kmd.json", {"id": KMD, "revision": revision})
    with pytest.raises(ProvenanceError):
        snapshot_provenance(tree)


def test_snapshot_rejects_legacy_version_only_provenance():
    with pytest.raises(ProvenanceError):
        validate_provenance({"ued": "1.0", "kmd": "1.0", "umd": "1.0"})


def test_an_engine_with_no_descriptors_is_trained_against_its_selector_revision():
    """RFC 0019 §4.1, Open Question 7: an engine with no UED binds its model by declared
    UUID, so there is no ued/kmd/umd to name -- only the provider build whose behaviour was
    measured. The loader has accepted this since opaque engines gained L1 (`UhdParser.hpp`
    :146-161); this validator rejecting it stopped every AITER and MIOpen L1 collection
    after the whole corpus had been measured (runs 67929293, 67929294)."""
    recorded = validate_provenance({"selector_revision": "miopen-provider/0.2.0/aiter-fwd-1"})
    assert recorded == {"selector_revision": "miopen-provider/0.2.0/aiter-fwd-1"}


@pytest.mark.parametrize("snapshot", [
    {},
    {"selector_revision": ""},
    {"selector_revision": 3},
    {"ued": {"id": UED, "revision": "1.0"}},
    {"selector_revision": "rev", "unexpected": "value"},
])
def test_provenance_that_names_neither_form_completely_is_refused(snapshot):
    """Two thirds of a descriptor set is not a weaker claim, it is an unverifiable one --
    and an empty or absent revision names nothing at all."""
    with pytest.raises(ProvenanceError):
        validate_provenance(snapshot)


def test_a_stale_selector_revision_is_refused_rather_than_scored():
    """L1 is the one score compared ACROSS engines, so a model measured against another
    provider build does not merely misreport a number -- it changes which engine wins."""
    trained = {"selector_revision": "aiter-fwd-1"}
    compare_provenance(trained, {"selector_revision": "aiter-fwd-1"})
    with pytest.raises(ProvenanceError, match="selector_revision"):
        compare_provenance(trained, {"selector_revision": "aiter-fwd-2"})
