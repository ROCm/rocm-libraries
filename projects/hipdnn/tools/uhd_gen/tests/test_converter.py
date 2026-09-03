#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for LightGBM to FlatBuffer converter.

These tests verify that:
1. The FlatBuffer output has correct structure
2. Tree traversal produces same results as LightGBM predict()
3. Features hash is correctly embedded
"""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

# Must precede the imports below: these are optional, heavyweight training
# dependencies, and importing them directly would turn a missing dep into a
# collection error instead of a skip.
flatbuffers = pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pytest.importorskip("pandas")  # imported transitively by uhd_gen.__main__

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel  # noqa: E402
from uhd_gen.__main__ import _looks_like_cost_metric  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import (  # noqa: E402
    GBDT_MODEL_FILE_IDENTIFIER,
    _objective_name,
    build_gbdt_model,
    convert,
)
from uhd_gen.train_uhd import train_model  # noqa: E402


def _create_synthetic_data(n_samples: int = 1000, n_features: int = 5, seed: int = 42):
    """Create synthetic regression data."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features))
    y = np.sum(X * rng.random(n_features), axis=1) + rng.normal(0, 0.1, n_samples)
    return X, y


def _train_simple_model(X: np.ndarray, y: np.ndarray, num_trees: int = 5) -> lgb.Booster:
    """Train a simple LightGBM model for testing."""
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "verbose": -1,
    }
    return lgb.train(params, train_data, num_boost_round=num_trees)


def _score_flatbuffer_model(buffer: bytes, features) -> float:
    """Mirror of `TreeDataAdapter::score()` over a GbdtModel FlatBuffer.

    Deliberately a re-implementation rather than a call into the tool: the point is
    to walk the emitted bytes exactly the way the C++ runtime walks them, so a
    writer that produces a buffer the runtime misreads fails here.

    Semantics copied from
    plugin_sdk/include/hipdnn_plugin_sdk/ingestor/uhd/adapters/TreeDataAdapter.hpp:
      - score() is `base_score + sum(tree)`; `learning_rate` is metadata and is NOT
        applied, because LightGBM folds shrinkage into the dumped leaf values (:294-299)
      - a node is a leaf when `left_children[node] < 0` (:342-350)
      - `decision_lte` absent or true means `<=`, false means `<` (:356-362)
      - a NaN feature, or a feature index outside the row, takes `default_left` (:369-385)
    """
    model = GbdtModel.GetRootAs(buffer, 0)
    total = model.BaseScore()
    for tree_idx in range(model.TreesLength()):
        total += _score_flatbuffer_tree(model.Trees(tree_idx), features)
    return total


def _score_flatbuffer_tree(tree, features) -> float:
    has_default_left = tree.DefaultLeftLength() > 0
    has_decision_lte = tree.DecisionLteLength() > 0

    node = 0
    steps = 0
    max_steps = tree.LeftChildrenLength()
    while node < tree.LeftChildrenLength():
        steps += 1
        if steps > max_steps:
            raise AssertionError("tree descent exceeded node count: cyclic child indices")

        left = tree.LeftChildren(node)
        right = tree.RightChildren(node)
        if left < 0:
            return tree.LeafValues(node) if node < tree.LeafValuesLength() else 0.0

        feature_idx = tree.FeatureIndices(node)
        threshold = tree.Thresholds(node)
        use_lte = (not has_decision_lte) or bool(tree.DecisionLte(node))
        default_left = has_default_left and bool(tree.DefaultLeft(node))

        if 0 <= feature_idx < len(features):
            value = features[feature_idx]
            if value != value:  # NaN
                go_left = default_left
            else:
                go_left = value <= threshold if use_lte else value < threshold
        else:
            go_left = default_left

        node = left if go_left else right

    return 0.0


class TestTrainModelCvPaths:
    """Both cross-validation paths must actually run.

    Regression: the no-group-columns path (the default) passed an int as
    `folds=`, which lgb.cv rejects, and then hit `stratified=True` routing a
    continuous target through StratifiedKFold. The tool could not train at all
    without --group-by.
    """

    def test_trains_without_group_columns(self):
        import pandas as pd

        X, y = _create_synthetic_data(n_samples=200, n_features=3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = y

        model = train_model(df, ["a", "b", "c"], "target", None, num_boost_round=10)
        assert model.num_trees() > 0

    def test_trains_with_group_columns(self):
        import pandas as pd

        X, y = _create_synthetic_data(n_samples=200, n_features=3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        df["target"] = y
        df["grp"] = np.arange(len(df)) % 10

        model = train_model(df, ["a", "b", "c"], "target", ["grp"], num_boost_round=10)
        assert model.num_trees() > 0


class TestCostMetricDetection:
    """Guard on the objective/target mismatch.

    `--target` is free-form while `objective` used to be hardcoded to "max", so
    training on a latency column emitted a descriptor telling the runtime to
    maximize latency. This heuristic only warns; it never overrides the caller.
    """

    @pytest.mark.parametrize(
        "target",
        [
            "latency_ms",
            "kernel_time_us",
            "duration",
            "elapsed_ns",
            "total_sec",
            "cost",
            "rmse_error",
            "val_loss",
            "LATENCY_MS",
        ],
    )
    def test_cost_metrics_are_flagged(self, target):
        assert _looks_like_cost_metric(target)

    @pytest.mark.parametrize(
        "target", ["tflops", "throughput", "gflops", "bandwidth_gbps", "samples"]
    )
    def test_throughput_metrics_are_not_flagged(self, target):
        assert not _looks_like_cost_metric(target)


class TestObjectiveName:
    """Objective extraction across LightGBM dump_model shapes."""

    @pytest.mark.parametrize(
        ("dump", "expected"),
        [
            # LightGBM 4.x: a bare string. Reading this as a mapping raised
            # AttributeError and broke every conversion.
            ({"objective": "regression"}, "regression"),
            # 4.x with trailing objective parameters; only the leading token names it.
            ({"objective": "binary sigmoid:1"}, "binary"),
            ({"objective": "multiclass num_class:3"}, "multiclass"),
            # 2.x/3.x mapping form.
            ({"objective": {"name": "regression_l1"}}, "regression_l1"),
            # Degenerate inputs fall back rather than raising.
            ({}, "regression"),
            ({"objective": None}, "regression"),
            ({"objective": ""}, "regression"),
            ({"objective": {}}, "regression"),
        ],
    )
    def test_objective_name(self, dump, expected):
        assert _objective_name(dump) == expected

    def test_real_lightgbm_dump_is_supported(self):
        """The installed LightGBM's actual dump shape must convert, not just fixtures."""
        X, y = _create_synthetic_data(n_samples=100, n_features=3)
        model = _train_simple_model(X, y, num_trees=2)
        assert _objective_name(model.dump_model()) == "regression"


class TestConverter:
    """Test FlatBuffer conversion."""

    def test_convert_simple_model(self, tmp_path: Path):
        """Test converting a simple 2-tree model."""
        X, y = _create_synthetic_data(n_samples=100, n_features=3)
        model = _train_simple_model(X, y, num_trees=2)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:test_hash_1234", fb_path)

        assert fb_path.exists()
        with open(fb_path, "rb") as f:
            buffer = f.read()

        # Verify file identifier
        assert buffer[4:8] == GBDT_MODEL_FILE_IDENTIFIER
        assert len(buffer) > 100  # Should have some content

    def test_features_hash_embedded(self, tmp_path: Path):
        """Test that features hash is embedded in output."""
        X, y = _create_synthetic_data(n_samples=50, n_features=2)
        model = _train_simple_model(X, y, num_trees=1)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        test_hash = "sha256:abcdef0123456789"
        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, test_hash, fb_path)

        with open(fb_path, "rb") as f:
            buffer = f.read()

        # The hash should appear somewhere in the buffer
        assert test_hash.encode() in buffer

    def test_build_gbdt_model_structure(self):
        """Test that build_gbdt_model produces valid FlatBuffer."""
        X, y = _create_synthetic_data(n_samples=100, n_features=4)
        model = _train_simple_model(X, y, num_trees=3)
        model_json = model.dump_model()

        buffer = build_gbdt_model(model_json, "sha256:test", num_training_samples=100)

        # Should have file identifier
        assert buffer[4:8] == GBDT_MODEL_FILE_IDENTIFIER

        # Should be valid FlatBuffer (root offset in first 4 bytes)
        root_offset = struct.unpack_from("<I", buffer, 0)[0]
        assert root_offset > 0
        assert root_offset < len(buffer)

    def test_num_features_correct(self, tmp_path: Path):
        """Test that num_features matches training data."""
        n_features = 7
        X, y = _create_synthetic_data(n_samples=50, n_features=n_features)
        model = _train_simple_model(X, y, num_trees=2)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        model_json = model.dump_model()
        assert model_json["max_feature_idx"] == n_features - 1


class TestRoundTrip:
    """The converted model must predict what LightGBM predicts.

    This is the training<->runtime parity check RFC 0019 §15 Phase 4 asks for, and
    the only test in either language that would catch a converter emitting bytes the
    runtime walks differently. Structural assertions cannot: a buffer with the right
    tree count and the wrong split semantics passes every one of them.
    """

    def test_predictions_match_lightgbm(self):
        """Every row must score the same through both evaluators."""
        X, y = _create_synthetic_data(n_samples=800, n_features=6)
        model = _train_simple_model(X, y, num_trees=12)
        buffer = build_gbdt_model(model.dump_model(), "sha256:test")

        expected = model.predict(X, raw_score=True)
        for row in range(X.shape[0]):
            actual = _score_flatbuffer_model(buffer, X[row])
            assert actual == pytest.approx(expected[row], rel=1e-9, abs=1e-9), (
                f"row {row}: flatbuffer {actual} != lightgbm {expected[row]}"
            )

    def test_exact_threshold_values_match(self):
        """Feature values sitting exactly on a split threshold.

        This is where `decision_lte` is observable: `<=` sends the row left, `<`
        sends it right, and every other input in the corpus agrees either way. A
        converter that dropped the field, or inverted it, passes
        test_predictions_match_lightgbm on random data and fails here.
        """
        X, y = _create_synthetic_data(n_samples=400, n_features=4)
        model = _train_simple_model(X, y, num_trees=6)
        model_json = model.dump_model()
        buffer = build_gbdt_model(model_json, "sha256:test")

        thresholds = _collect_thresholds(model_json)
        assert thresholds, "model produced no internal splits to probe"

        probes = []
        for feature_idx, threshold in thresholds[:40]:
            row = np.full(X.shape[1], 0.5)
            row[feature_idx] = threshold
            probes.append(row)

        probe_matrix = np.array(probes)
        expected = model.predict(probe_matrix, raw_score=True)
        for i, row in enumerate(probes):
            actual = _score_flatbuffer_model(buffer, row)
            assert actual == pytest.approx(expected[i], rel=1e-9, abs=1e-9), (
                f"probe {i} on threshold: flatbuffer {actual} != lightgbm {expected[i]}"
            )

    def test_missing_values_match(self):
        """NaN features must take the direction `default_left` records.

        LightGBM decides missing-value routing per node at training time. If the
        converter drops `default_left`, the mirror falls back to `False` (go right)
        and diverges on exactly the rows that carry a NaN.
        """
        X, y = _create_synthetic_data(n_samples=600, n_features=5)
        model = _train_simple_model(X, y, num_trees=8)
        buffer = build_gbdt_model(model.dump_model(), "sha256:test")

        rng = np.random.default_rng(7)
        probes = X[:60].copy()
        for row in probes:
            row[rng.integers(0, probes.shape[1])] = np.nan

        expected = model.predict(probes, raw_score=True)
        for i, row in enumerate(probes):
            actual = _score_flatbuffer_model(buffer, row)
            assert actual == pytest.approx(expected[i], rel=1e-9, abs=1e-9), (
                f"NaN probe {i}: flatbuffer {actual} != lightgbm {expected[i]}"
            )

    def test_learning_rate_is_not_applied_twice(self):
        """A non-default shrinkage must not scale the ensemble a second time.

        LightGBM folds `learning_rate` into the dumped leaf values, so the emitted
        model carries 1.0 as provenance and `TreeDataAdapter::score()` deliberately
        ignores it. Training at 0.3 rather than the 0.1 the other cases use makes a
        double application a 3x error rather than a rounding difference.
        """
        X, y = _create_synthetic_data(n_samples=400, n_features=4)
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(
            {
                "objective": "regression",
                "num_leaves": 7,
                "learning_rate": 0.3,
                "verbose": -1,
                "min_data_in_leaf": 5,
            },
            train_data,
            num_boost_round=5,
        )
        buffer = build_gbdt_model(model.dump_model(), "sha256:test")

        assert GbdtModel.GetRootAs(buffer, 0).LearningRate() == pytest.approx(1.0)

        expected = model.predict(X[:50], raw_score=True)
        for i in range(50):
            assert _score_flatbuffer_model(buffer, X[i]) == pytest.approx(
                expected[i], rel=1e-9, abs=1e-9
            )


def _collect_thresholds(model_json) -> list[tuple[int, float]]:
    """Every (split_feature, threshold) pair in the ensemble, in DFS order."""
    found: list[tuple[int, float]] = []

    def walk(node):
        if "leaf_value" in node:
            return
        found.append((node["split_feature"], node["threshold"]))
        walk(node["left_child"])
        walk(node["right_child"])

    for tree_info in model_json["tree_info"]:
        walk(tree_info["tree_structure"])
    return found


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_tree_model(self, tmp_path: Path):
        """Test conversion with single tree."""
        X, y = _create_synthetic_data(n_samples=50, n_features=2)
        model = _train_simple_model(X, y, num_trees=1)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:single_tree", fb_path)

        assert fb_path.exists()
        with open(fb_path, "rb") as f:
            buffer = f.read()
        assert len(buffer) > 0

    def test_many_trees_model(self, tmp_path: Path):
        """Test conversion with many trees."""
        X, y = _create_synthetic_data(n_samples=200, n_features=3)
        model = _train_simple_model(X, y, num_trees=50)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:many_trees", fb_path)

        assert fb_path.exists()
        with open(fb_path, "rb") as f:
            buffer = f.read()

        # 50 trees should produce substantial output
        assert len(buffer) > 5000

    def test_deep_tree(self, tmp_path: Path):
        """Test conversion with deeper trees (more leaves)."""
        X, y = _create_synthetic_data(n_samples=500, n_features=5)

        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "num_leaves": 63,  # Deeper tree
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=5)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:deep_tree", fb_path)

        assert fb_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
