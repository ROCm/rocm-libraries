# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Offline tools refuse exactly the `tree_data` artifacts TreeDataAdapter refuses."""
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


def _artifact(
    path: Path, *, trees=None, grouped=False, identifier=b"HGBM", group_values=(0.0,)
) -> Path:
    model = GbdtModelT()
    model.trees = trees or [_stump()]
    model.baseScore = 0.0
    model.numFeatures = 2
    model.featuresHash = "sha256:0123456789abcdef"
    model.groupByFeatureIndex = 1 if grouped else -1
    if grouped:
        model.groups = []
        for value in group_values:
            group = GbdtGroupT()
            group.value = value
            group.trees = [_stump()]
            model.groups.append(group)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    data = bytearray(builder.Output())
    data[4:8] = identifier
    path.write_bytes(bytes(data))
    return path


def test_a_buffer_the_runtime_does_not_identify_as_a_gbdt_model_is_refused(tmp_path):
    """The runtime checks the `HGBM` identifier; the Python object API does not."""
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


def test_trees_sharing_one_table_past_what_the_buffer_holds_are_refused(tmp_path):
    """`trees` entries may point at one table, so the node count is not bounded by the
    entries; the runtime refuses more nodes than the buffer has int32s for, and so does this.
    """
    from hipdnn_flatbuffers_sdk.data_objects import GbdtModel

    builder = flatbuffers.Builder(1024)
    shared = _stump().Pack(builder)
    entries = 4000  # 3 nodes each: far more than a buffer of a few KiB can hold
    GbdtModel.GbdtModelStartTreesVector(builder, entries)
    for _ in range(entries):
        builder.PrependUOffsetTRelative(shared)
    trees = builder.EndVector()
    digest = builder.CreateString("sha256:0123456789abcdef")
    GbdtModel.GbdtModelStart(builder)
    GbdtModel.GbdtModelAddTrees(builder, trees)
    GbdtModel.GbdtModelAddNumFeatures(builder, 2)
    GbdtModel.GbdtModelAddFeaturesHash(builder, digest)
    GbdtModel.GbdtModelAddGroupByFeatureIndex(builder, -1)
    builder.Finish(GbdtModel.GbdtModelEnd(builder), file_identifier=b"HGBM")
    path = tmp_path / "model.bin"
    path.write_bytes(bytes(builder.Output()))
    assert 3 * entries > len(path.read_bytes()) // 4

    with pytest.raises(ValueError, match="more nodes than the model buffer can hold"):
        verify_tree_artifact(path, None)


def test_a_buffer_the_flatbuffers_verifier_refuses_is_refused(tmp_path):
    """`VerifyGbdtModelBuffer` refuses a string missing its NUL; Python does not."""
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
    """The runtime refuses `num_features` other than the signature's slot count."""
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


@pytest.mark.parametrize(
    "values", [(float("inf"),), (float("nan"),), (0.0, -0.0)], ids=str
)
def test_a_group_value_the_runtime_cannot_key_is_refused(tmp_path, values):
    """TreeDataAdapter refuses a nonfinite group value or two groups of one value."""
    path = _artifact(tmp_path / "model.bin", grouped=True, group_values=values)
    with pytest.raises(ValueError, match="must be finite and unique"):
        verify_tree_artifact(path, None)
    verify_tree_artifact(
        _artifact(tmp_path / "distinct.bin", grouped=True, group_values=(0.0, 1.0)),
        None,
    )


def test_an_engine_prediction_model_cannot_be_grouped(tmp_path):
    """EnginePredictor refuses to bind a grouped artifact to predict_engine."""
    _artifact(tmp_path / "model.bin", grouped=True)
    (tmp_path / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "score": {"metric": "tflops", "calibrated": True},
                "trained_against": {"selector_revision": "provider-1"},
                "features_signature": ["$graph.size", "$graph.group"],
                "tree_data": {"artifact": "model.bin"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "train_manifest.json").write_text(
        json.dumps({"role": "predict_engine"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="predict_engine"):
        load_model(tmp_path)


@pytest.mark.parametrize(
    ("payload", "admitted"),
    [
        ("./model.bin", True),
        ("nested/../model.bin", False),
        ("../model.bin", False),
        ("nested\\..\\model.bin", False),
        ("absolute", False),
        ("C:model.bin", False),
        ("nested/.", False),
    ],
)
def test_evaluation_opens_only_an_artifact_the_loader_would(
    tmp_path, payload, admitted
):
    """UhdParser reads `tree_data.artifact` only inside the descriptor's directory."""
    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    _artifact(model / "model.bin")
    _artifact(tmp_path / "model.bin")
    if payload == "absolute":
        payload = str((tmp_path / "model.bin").resolve())
    (model / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "max",
                "tree_data": {"artifact": payload},
                "features_signature": ["$q.size", "$kernel.group"],
            }
        ),
        encoding="utf-8",
    )
    if admitted:
        assert load_model(model).source == str(model / payload)
    else:
        with pytest.raises(ValueError, match="relative path inside"):
            load_model(model)


def _write(path: Path, model: GbdtModelT) -> Path:
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    path.write_bytes(bytes(builder.Output()))
    return path


def _model(**changes) -> GbdtModelT:
    model = GbdtModelT()
    model.trees = [_stump()]
    model.baseScore = 0.0
    model.numFeatures = 2
    model.featuresHash = "sha256:0123456789abcdef"
    model.groupByFeatureIndex = -1
    for name, value in changes.items():
        setattr(model, name, value)
    return model


def _tree(**changes) -> GbdtTreeT:
    tree = _stump()
    for name, value in changes.items():
        setattr(tree, name, value)
    return tree


def _grouped(index: int) -> GbdtModelT:
    group = GbdtGroupT()
    group.value = 0.0
    group.trees = [_stump()]
    return _model(groups=[group], groupByFeatureIndex=index)


@pytest.mark.parametrize(
    ("model", "reason"),
    [
        (_model(numFeatures=-1), "negative feature count"),
        (_model(baseScore=float("inf")), "base score is not finite"),
        (_model(baseScore=float("nan")), "base score is not finite"),
        (_model(trees=[_tree(thresholds=None)]), "missing nodes"),
        (_model(trees=[_tree(leftChildren=[])]), "missing nodes"),
        (_model(trees=[_tree(thresholds=[10.0, 0.0])]), "different lengths"),
        (_model(trees=[_tree(leafValues=[0.0, 1.0])]), "leaf has no prediction"),
        (
            _model(trees=[_tree(leafValues=[0.0, float("nan"), 9.0])]),
            "leaf prediction is not finite",
        ),
        (
            _model(trees=[_tree(thresholds=[float("inf"), 0.0, 0.0])]),
            "split threshold is not finite",
        ),
        (_model(trees=[_tree(rightChildren=[3, -1, -1])]), "child index outside"),
        (_model(trees=[_tree(leftChildren=[-2, -1, -1])]), "child index outside"),
        (_model(trees=[_tree(featureIndices=[2, 0, 0])]), "split feature outside"),
        (_model(trees=[_tree(featureIndices=[-1, 0, 0])]), "split feature outside"),
        (_grouped(2), "group_by_feature_index 2"),
        (_grouped(-1), "group_by_feature_index -1"),
    ],
    ids=lambda case: case if isinstance(case, str) else "",
)
def test_an_artifact_the_runtime_would_not_load_is_refused_with_its_reason(
    tmp_path, model, reason
):
    """Each `TreeDataAdapter::loadFromBuffer` refusal, one field off a loadable model."""
    assert verify_tree_artifact(_write(tmp_path / "base.bin", _model()), None)
    with pytest.raises(ValueError, match=reason):
        verify_tree_artifact(_write(tmp_path / "model.bin", model), None)


def test_an_artifact_is_admitted_up_to_the_runtimes_size_limit_and_no_further(
    tmp_path, monkeypatch
):
    """The runtime refuses an empty buffer, one under eight bytes, and one over 256 MiB."""
    from uhd_gen import artifact

    assert artifact.MAX_ARTIFACT_BYTES == 256 * 1024 * 1024
    path = _write(tmp_path / "model.bin", _model())
    size = path.stat().st_size
    monkeypatch.setattr(artifact, "MAX_ARTIFACT_BYTES", size)
    assert verify_tree_artifact(path, None) == path.read_bytes()
    monkeypatch.setattr(artifact, "MAX_ARTIFACT_BYTES", size - 1)
    with pytest.raises(ValueError, match="outside"):
        verify_tree_artifact(path, None)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="outside"):
        verify_tree_artifact(empty, None)
    short = tmp_path / "short.bin"
    short.write_bytes(path.read_bytes()[:7])
    with pytest.raises(ValueError, match="too short"):
        verify_tree_artifact(short, None)
