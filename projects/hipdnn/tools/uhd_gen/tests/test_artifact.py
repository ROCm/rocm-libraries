# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Offline tools refuse exactly the `tree_data` artifacts TreeDataAdapter refuses (R8)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

flatbuffers = pytest.importorskip("flatbuffers")
pytest.importorskip("lightgbm")  # the shipping converter builds every artifact here
pytest.importorskip("numpy")
pytest.importorskip("pandas")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtGroup import GbdtGroupT  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT  # noqa: E402

from uhd_gen.artifact import (
    artifact_digest,
    is_grouped_tree,
    verify_tree_artifact,
)  # noqa: E402
from uhd_gen.evaluate import load_model  # noqa: E402


def _stump(left=(1, -1, -1), right=(2, -1, -1)) -> GbdtTreeT:
    tree = GbdtTreeT()
    tree.featureIndices = [0, 0, 0]
    tree.thresholds = [10.0, 0.0, 0.0]
    tree.leftChildren = list(left)
    tree.rightChildren = list(right)
    tree.leafValues = [0.0, 1.0, 9.0]
    tree.defaultLeft = [True, True, True]
    tree.decisionLte = [True, True, True]
    return tree


def _artifact(path: Path, *, trees=None, grouped=False, identifier=b"HGBM") -> Path:
    model = GbdtModelT()
    model.trees = trees or [_stump()]
    model.baseScore = 0.0
    model.numFeatures = 2
    model.featuresHash = "sha256:0123456789abcdef"
    model.groupByFeatureIndex = 1 if grouped else -1
    if grouped:
        group = GbdtGroupT()
        group.value = 0.0
        group.trees = [_stump()]
        model.groups = [group]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    data = bytearray(builder.Output())
    data[4:8] = identifier
    path.write_bytes(bytes(data))
    return path


def test_a_buffer_the_runtime_does_not_identify_as_a_gbdt_model_is_refused(tmp_path):
    """`BufferHasIdentifier(buffer, "HGBM")` is the loader's first structural check; the
    Python object API decodes a `NOPE` buffer happily, which is how evaluate reported
    regret for bytes the engine would never load."""
    nope = _artifact(tmp_path / "model.bin", identifier=b"NOPE")
    with pytest.raises(ValueError, match="identifier"):
        verify_tree_artifact(nope, None)
    (tmp_path / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "tree_data": {"artifact": "model.bin"},
                "features_signature": ["$q.size", "$kernel.group"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identifier"):
        load_model(tmp_path)


def test_the_declared_digest_is_bare_hex_over_the_file_and_is_enforced(tmp_path):
    path = _artifact(tmp_path / "model.bin")
    digest = artifact_digest(path)
    # TreeDataAdapter compares `sha256(buffer, size)`: 64 lowercase hex, no `sha256:` prefix.
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_tree_artifact(path, digest) == path.read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_tree_artifact(path, "sha256:" + digest)


def test_a_declared_digest_the_evaluated_artifact_does_not_match_is_refused(tmp_path):
    """load_model holds the descriptor's `tree_data.hash` to the file it scores."""
    _artifact(tmp_path / "model.bin")
    (tmp_path / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "tree_data": {"artifact": "model.bin", "hash": "0" * 64},
                "features_signature": ["$q.size", "$kernel.group"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(tmp_path)


def test_a_tree_the_runtime_cannot_prepare_is_refused(tmp_path):
    """`prepareTrees` rejects a child cycle at load, reached or not by any row."""
    cyclic = _artifact(
        tmp_path / "model.bin", trees=[_stump(left=(1, 0, -1), right=(2, 2, -1))]
    )
    with pytest.raises(ValueError, match="cycle"):
        verify_tree_artifact(cyclic, None)


def test_a_buffer_the_flatbuffers_verifier_refuses_is_refused(tmp_path):
    """`VerifyGbdtModelBuffer` runs before any accessor: a string whose NUL terminator is
    gone still decodes through the generated Python accessors, and the runtime refuses it.
    """
    path = _artifact(tmp_path / "model.bin")
    data = bytearray(path.read_bytes())
    end = data.index(b"sha256:0123456789abcdef") + len("sha256:0123456789abcdef")
    data[end] = ord("!")
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="NUL-terminated"):
        verify_tree_artifact(path, None)
    (tmp_path / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "tree_data": {"artifact": "model.bin"},
                "features_signature": ["$q.size", "$kernel.group"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verifier"):
        load_model(tmp_path)


def test_an_artifact_whose_arity_is_not_its_signatures_is_refused(tmp_path):
    """EnginePredictor and UhdKernelHeuristic refuse a model whose `num_features` differs
    from its signature's slot count, whatever the features_hash says."""
    path = _artifact(tmp_path / "model.bin")
    assert verify_tree_artifact(path, None, feature_count=2) == path.read_bytes()
    with pytest.raises(ValueError, match="num_features 2"):
        verify_tree_artifact(path, None, feature_count=1)
    (tmp_path / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "tree_data": {"artifact": "model.bin"},
                "features_signature": ["$q.size"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_features 2"):
        load_model(tmp_path)


def test_grouping_is_read_from_the_artifact(tmp_path):
    grouped = verify_tree_artifact(
        _artifact(tmp_path / "grouped.bin", grouped=True), None
    )
    flat = verify_tree_artifact(_artifact(tmp_path / "flat.bin"), None)
    assert is_grouped_tree(grouped)
    assert not is_grouped_tree(flat)
