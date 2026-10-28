# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The split/leaf `model.bin` encoding: what the converter writes, and what the offline
verifier refuses, reason for reason, as TreeDataAdapter refuses it."""
from __future__ import annotations

from pathlib import Path

import pytest

flatbuffers = pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.Int32Array import Int32ArrayT  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.IntegerArray import IntegerArray  # noqa: E402
from hipdnn_flatbuffers_sdk.data_objects.RealArray import RealArray  # noqa: E402
from uhd_gen.artifact import verify_tree_artifact  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model  # noqa: E402


def _booster(trees: int = 60, leaves: int = 31, features: int = 6) -> lgb.Booster:
    rng = np.random.default_rng(5)
    X = rng.random((3000, features))
    y = X @ rng.random(features) + rng.normal(0, 0.05, len(X))
    return lgb.train(
        {"objective": "regression", "num_leaves": leaves, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=trees,
    )


def _model(buffer: bytes) -> GbdtModelT:
    return GbdtModelT.InitFromPackedBuf(bytearray(buffer), 0)


def _pack(model: GbdtModelT, path: Path) -> Path:
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    path.write_bytes(bytes(builder.Output()))
    return path


def _leaf(value: float) -> dict:
    return {"leaf_value": value}


def _split(feature: int, threshold: float, left: dict, right: dict) -> dict:
    return {
        "split_feature": feature,
        "threshold": threshold,
        "default_left": True,
        "decision_type": "<=",
        "left_child": left,
        "right_child": right,
    }


def _dump(*roots: dict, features: int = 2) -> dict:
    return {
        "tree_info": [{"tree_structure": root} for root in roots],
        "max_feature_idx": features - 1,
        "objective": "regression",
    }


def test_split_leaf_is_the_default_and_a_fraction_of_the_node_array_size():
    dump = _booster().dump_model()
    node_arrays = build_gbdt_model(dump, "sha256:x", encoding="node_arrays")
    split_leaf = build_gbdt_model(dump, "sha256:x")
    assert all(tree.leavesType != 0 for tree in _model(split_leaf).trees)
    assert len(split_leaf) < 0.3 * len(node_arrays), (len(split_leaf), len(node_arrays))


def test_each_array_takes_the_narrowest_type_that_holds_it():
    """Six features and 31-leaf trees: features and children fit a byte, signed for the
    children (a leaf is ~index); the threshold ids are as wide as the table needs."""
    model = _model(build_gbdt_model(_booster().dump_model(), "sha256:x"))
    tree = model.trees[0]
    assert tree.splitFeaturesType == IntegerArray.UInt8Array
    assert tree.splitLeftType == tree.splitRightType == IntegerArray.Int8Array
    expected_ids = (
        IntegerArray.UInt8Array
        if len(model.thresholdTable.values) <= 256
        else IntegerArray.UInt16Array
    )
    assert tree.splitThresholdsType == expected_ids
    assert tree.leavesType == RealArray.Float32Array


@pytest.mark.parametrize(
    ("thresholds", "table_type"),
    [
        pytest.param((0.5, 2.25), RealArray.Float32Array, id="every-value-a-float32"),
        pytest.param((0.1, 2.25), RealArray.Float64Array, id="one-value-not-a-float32"),
    ],
)
def test_thresholds_are_stored_exactly(thresholds, table_type):
    """0.1 is not a float32: the whole table stays float64 rather than move a split."""
    low, high = thresholds
    dump = _dump(
        _split(0, low, _leaf(1.0), _split(1, high, _leaf(2.0), _leaf(3.0))),
    )
    model = _model(build_gbdt_model(dump, "sha256:x"))
    assert model.thresholdTableType == table_type
    assert [float(v) for v in model.thresholdTable.values] == sorted(thresholds)


def test_routing_flags_are_stored_only_when_they_vary():
    dump = _dump(_split(0, 0.5, _leaf(1.0), _leaf(2.0)))
    uniform = _model(build_gbdt_model(dump, "sha256:x")).trees[0]
    assert uniform.splitDefaultLeft is None and uniform.defaultLeftFill
    assert uniform.splitDecisionLte is None

    mixed = _dump(
        _split(
            0,
            0.5,
            {**_split(1, 0.25, _leaf(1.0), _leaf(2.0)), "default_left": False},
            {**_split(1, 0.75, _leaf(3.0), _leaf(4.0)), "decision_type": "<"},
        )
    )
    tree = _model(build_gbdt_model(mixed, "sha256:x")).trees[0]
    assert [bool(v) for v in tree.splitDefaultLeft] == [True, False, True]
    assert [bool(v) for v in tree.splitDecisionLte] == [True, True, False]


def test_a_single_leaf_tree_is_its_leaf_alone():
    model = _model(build_gbdt_model(_dump(_leaf(4.5)), "sha256:x"))
    tree = model.trees[0]
    assert tree.splitLeftType == IntegerArray.NONE
    assert [float(v) for v in tree.leaves.values] == [4.5]
    assert model.thresholdTableType == RealArray.NONE


@pytest.mark.parametrize(
    "keyword", [{"encoding": "packed"}, {"leaf_precision": "float16"}]
)
def test_an_unknown_encoding_choice_is_refused(keyword):
    with pytest.raises(ValueError, match="must be one of"):
        build_gbdt_model(_dump(_leaf(1.0)), "sha256:x", **keyword)


# ---- The offline verifier refuses what TreeDataAdapter refuses ---------------------


def _two_splits() -> GbdtModelT:
    dump = _dump(_split(0, 0.5, _split(1, 1.5, _leaf(1.0), _leaf(2.0)), _leaf(3.0)))
    return _model(build_gbdt_model(dump, "sha256:x"))


def test_the_unbroken_model_verifies(tmp_path):
    verify_tree_artifact(_pack(_two_splits(), tmp_path / "model.bin"), None)


def _with(change) -> GbdtModelT:
    model = _two_splits()
    change(model, model.trees[0])
    return model


def _int32(values) -> Int32ArrayT:
    array = Int32ArrayT()
    array.values = list(values)
    return array


def _set(tree, field: str, values) -> None:
    setattr(tree, f"{field}Type", IntegerArray.Int32Array)
    setattr(tree, field, _int32(values))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param(
            lambda m, t: _set(t, "splitLeft", [2, ~0]),
            "child index outside the tree",
            id="split-index-past-the-splits",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitRight", [~0, ~3]),
            "child index outside the tree",
            id="leaf-index-past-the-leaves",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitLeft", [0, ~0]),
            "cycle in child indices",
            id="cycle",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitThresholds", [0, 2]),
            "split threshold index outside the threshold table",
            id="threshold-id",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitFeatures", [0, 2]),
            "split feature outside the declared feature count",
            id="feature",
        ),
        pytest.param(
            lambda m, t: _set(t, "splitFeatures", [0]),
            "split arrays do not hold one entry per split",
            id="short-array",
        ),
        pytest.param(
            lambda m, t: setattr(t, "splitDefaultLeft", [True]),
            "split arrays do not hold one entry per split",
            id="short-flags",
        ),
        pytest.param(
            lambda m, t: setattr(t, "splitFeaturesType", 9),
            "split_features is stored as integer array type 9, which this runtime "
            "cannot read",
            id="unknown-integer-tag",
        ),
        pytest.param(
            lambda m, t: (
                setattr(t, "splitLeftType", 9),
                setattr(t, "splitRightType", 9),
            ),
            "split children are stored in a type this runtime cannot read",
            id="unknown-children-tag",
        ),
        pytest.param(
            lambda m, t: setattr(t, "leavesType", 7),
            "leaves are stored in a type this runtime cannot read",
            id="unknown-leaves-tag",
        ),
        pytest.param(
            lambda m, t: setattr(m, "thresholdTableType", 7),
            "threshold_table is stored as real array type 7, which this runtime "
            "cannot read",
            id="unknown-table-tag",
        ),
        pytest.param(
            lambda m, t: (
                setattr(m, "thresholdTableType", 0),
                setattr(m, "thresholdTable", None),
            ),
            "split threshold index outside the threshold table",
            id="no-table",
        ),
    ],
)
def test_a_malformed_split_leaf_tree_is_refused_with_the_runtimes_reason(
    tmp_path, change, reason
):
    path = _pack(_with(change), tmp_path / "model.bin")
    with pytest.raises(ValueError, match=reason.replace("(", r"\(")):
        verify_tree_artifact(path, None)


def test_a_non_finite_threshold_or_leaf_is_refused(tmp_path):
    model = _two_splits()
    model.thresholdTable.values = np.asarray([0.5, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="split threshold is not finite"):
        verify_tree_artifact(_pack(model, tmp_path / "a.bin"), None)

    model = _two_splits()
    model.trees[0].leaves.values = np.asarray([1.0, np.nan, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="leaf prediction is not finite"):
        verify_tree_artifact(_pack(model, tmp_path / "b.bin"), None)
