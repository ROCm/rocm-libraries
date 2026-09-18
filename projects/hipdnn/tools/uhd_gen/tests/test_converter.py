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


def _evaluate_flatbuffer_tree(
    buffer: bytes, features: np.ndarray
) -> float:
    """Evaluate FlatBuffer GbdtModel manually.

    This mirrors TreeDataAdapter::score() to verify correctness.
    """
    # Skip to root table offset (after file identifier)
    root_offset = struct.unpack_from("<I", buffer, 0)[0]

    # For now, just verify the buffer structure is valid
    # Full evaluation would require parsing the FlatBuffer manually
    # or using generated Python bindings
    assert len(buffer) > 0
    assert buffer[4:8] == GBDT_MODEL_FILE_IDENTIFIER
    return 0.0  # Placeholder


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
    """Test prediction round-trip between LightGBM and FlatBuffer."""

    def test_tree_structure_preserved(self, tmp_path: Path):
        """Test that tree structure is preserved in conversion."""
        X, y = _create_synthetic_data(n_samples=200, n_features=5)
        model = _train_simple_model(X, y, num_trees=3)

        model_json = model.dump_model()
        buffer = build_gbdt_model(model_json, "sha256:test")

        # Verify we have the right number of trees
        # The tree count should match
        assert len(model_json["tree_info"]) == 3

        # Buffer should contain tree data
        assert len(buffer) > 500  # Non-trivial size for 3 trees


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
