# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""evaluate's own walk of a shipped `model.bin` scores what LightGBM scores."""
from __future__ import annotations

import pytest

flatbuffers = pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT  # noqa: E402

from uhd_gen import train_uhd  # noqa: E402
from uhd_gen.evaluate import _flatbuffer_scorer  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model  # noqa: E402

FEATURES = ["f0", "f1", "f2", "f3"]


def _missing_data(seed: int = 3):
    """Rows where a missing f0 means a high label, so LightGBM learns both directions."""
    rng = np.random.default_rng(seed)
    X = rng.random((1200, len(FEATURES)))
    y = X @ np.array([2.0, -1.0, 0.5, 0.25]) + rng.normal(0, 0.05, len(X))
    missing = rng.random(len(X)) < 0.3
    X[missing, 0] = np.nan
    y[missing] += 3.0
    sometimes = rng.random(len(X)) < 0.2
    X[sometimes, 1] = np.nan
    y[sometimes] -= 2.0
    return X, y


def _shipped(path, booster, base_score):
    model = GbdtModelT.InitFromPackedBuf(
        bytearray(build_gbdt_model(booster.dump_model(), "sha256:test")), 0
    )
    model.baseScore = base_score
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    path.write_bytes(bytes(builder.Output()))
    return model


@pytest.mark.parametrize("base_score", [0.0, 0.75])
def test_the_shipped_artifact_scores_missing_values_as_lightgbm_does(
    tmp_path, monkeypatch, base_score
):
    """A NaN feature takes its node's `default_left`, either way, and every row starts
    from the artifact's base score, as `TreeDataAdapter::score` sums it."""
    X, y = _missing_data()
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 15, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=20,
    )
    model = _shipped(tmp_path / "model.bin", booster, base_score)
    directions = {
        bool(flag)
        for tree in model.trees
        for flag, left in zip(tree.defaultLeft, tree.leftChildren)
        if left >= 0
    }
    assert directions == {True, False}, "the model must route missing values both ways"

    probes = np.vstack([X[:200], np.full((1, len(FEATURES)), np.nan)])
    monkeypatch.setattr(train_uhd, "build_feature_matrix", lambda *_a, **_k: probes)
    score = _flatbuffer_scorer(
        tmp_path / "model.bin", FEATURES, score_transform="identity", physical=False
    )
    expected = booster.predict(probes, raw_score=True) + base_score
    assert np.isnan(probes).any(axis=1).sum() > 50
    np.testing.assert_allclose(
        score(pd.DataFrame(index=range(len(probes)))), expected, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize(
    ("default_left", "decision_lte", "expected"),
    [
        # Missing `default_left` is false: NaN goes right. Absent `decision_lte` is `<=`.
        ([], None, [1.0, 9.0, 5.0]),
        ([True], None, [1.0, 5.0, 5.0]),
        # An empty `decision_lte` is absent too.
        ([], [], [1.0, 9.0, 5.0]),
        # A nonempty `decision_lte` shorter than the tree is `<` past its end.
        ([], [True], [5.0, 9.0, 5.0]),
        ([False], [False], [5.0, 9.0, 9.0]),
    ],
)
def test_short_routing_arrays_are_read_as_the_runtime_reads_them(
    tmp_path, monkeypatch, default_left, decision_lte, expected
):
    """`prepareTrees` pads `default_left` with false and a short `decision_lte` with `<`."""
    from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

    tree = GbdtTreeT()
    # x <= 10 ? (x <= 5 ? 1 : 5) : 9, with the second split past a one-entry array.
    tree.featureIndices = [0, 0, 0, 0, 0]
    tree.thresholds = [10.0, 5.0, 0.0, 0.0, 0.0]
    tree.leftChildren = [1, 3, -1, -1, -1]
    tree.rightChildren = [2, 4, -1, -1, -1]
    tree.leafValues = [0.0, 0.0, 9.0, 1.0, 5.0]
    tree.defaultLeft = default_left
    tree.decisionLte = decision_lte
    model = GbdtModelT()
    model.trees = [tree]
    model.baseScore = 0.0
    model.numFeatures = 1
    model.featuresHash = "sha256:test"
    model.groupByFeatureIndex = -1
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")
    (tmp_path / "model.bin").write_bytes(bytes(builder.Output()))

    probes = np.array([[5.0], [np.nan], [10.0]])
    monkeypatch.setattr(train_uhd, "build_feature_matrix", lambda *_a, **_k: probes)
    score = _flatbuffer_scorer(
        tmp_path / "model.bin", ["f0"], score_transform="identity", physical=False
    )
    assert score(pd.DataFrame(index=range(3))).tolist() == expected
