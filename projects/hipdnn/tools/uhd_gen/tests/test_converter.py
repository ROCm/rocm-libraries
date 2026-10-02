#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the LightGBM to FlatBuffer converter."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path

import pytest

# Before the imports below, so a missing optional dependency skips instead of
# failing collection.
flatbuffers = pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel  # noqa: E402
from uhd_gen.__main__ import _looks_like_cost_metric, main  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import (  # noqa: E402
    GBDT_MODEL_FILE_IDENTIFIER,
    _objective_name,
    build_gbdt_model,
    convert,
)
from uhd_gen.train_uhd import train_model  # noqa: E402


def _read_model(path: Path) -> GbdtModel:
    return GbdtModel.GetRootAs(path.read_bytes(), 0)


def _create_synthetic_data(n_samples: int = 1000, n_features: int = 5, seed: int = 42):
    """Create synthetic regression data."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features))
    y = np.sum(X * rng.random(n_features), axis=1) + rng.normal(0, 0.1, n_samples)
    return X, y


def _train_simple_model(
    X: np.ndarray, y: np.ndarray, num_trees: int = 5
) -> lgb.Booster:
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
    """Independent mirror of C++ `TreeDataAdapter::score()` over a GbdtModel buffer.

    `learning_rate` is not applied (LightGBM folds it into leaf values); absent
    `decision_lte` means `<=`; a NaN or out-of-range feature takes `default_left`.
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
            raise AssertionError(
                "tree descent exceeded node count: cyclic child indices"
            )

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


class TestCostMetricDetection:
    """Flags cost-like `--target` names; it only warns, never overrides."""

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
            # LightGBM 4.x: a bare string.
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
        """The installed LightGBM's real dump shape, not just fixtures."""
        X, y = _create_synthetic_data(n_samples=100, n_features=3)
        model = _train_simple_model(X, y, num_trees=2)
        assert _objective_name(model.dump_model()) == "regression"


class TestConverter:
    def test_convert_simple_model(self, tmp_path: Path):
        X, y = _create_synthetic_data(n_samples=100, n_features=3)
        model = _train_simple_model(X, y, num_trees=2)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:test_hash_1234", fb_path)

        assert fb_path.exists()
        with open(fb_path, "rb") as f:
            buffer = f.read()

        assert buffer[4:8] == GBDT_MODEL_FILE_IDENTIFIER
        assert len(buffer) > 100

    def test_features_hash_embedded(self, tmp_path: Path):
        X, y = _create_synthetic_data(n_samples=50, n_features=2)
        model = _train_simple_model(X, y, num_trees=1)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        test_hash = "sha256:abcdef0123456789"
        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, test_hash, fb_path)

        with open(fb_path, "rb") as f:
            buffer = f.read()

        assert test_hash.encode() in buffer

    def test_build_gbdt_model_structure(self):
        X, y = _create_synthetic_data(n_samples=100, n_features=4)
        model = _train_simple_model(X, y, num_trees=3)
        model_json = model.dump_model()

        buffer = build_gbdt_model(model_json, "sha256:test", num_training_samples=100)

        assert buffer[4:8] == GBDT_MODEL_FILE_IDENTIFIER

        # The root offset (first 4 bytes) must point inside the buffer.
        root_offset = struct.unpack_from("<I", buffer, 0)[0]
        assert root_offset > 0
        assert root_offset < len(buffer)

    def test_num_features_correct(self, tmp_path: Path):
        n_features = 7
        X, y = _create_synthetic_data(n_samples=50, n_features=n_features)
        model = _train_simple_model(X, y, num_trees=2)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        model_json = model.dump_model()
        assert model_json["max_feature_idx"] == n_features - 1


class TestRoundTrip:
    """Converted models must predict what LightGBM predicts (RFC 0019 §15 Phase 4)."""

    def test_predictions_match_lightgbm(self):
        X, y = _create_synthetic_data(n_samples=800, n_features=6)
        model = _train_simple_model(X, y, num_trees=12)
        buffer = build_gbdt_model(model.dump_model(), "sha256:test")

        expected = model.predict(X, raw_score=True)
        for row in range(X.shape[0]):
            actual = _score_flatbuffer_model(buffer, X[row])
            assert actual == pytest.approx(
                expected[row], rel=1e-9, abs=1e-9
            ), f"row {row}: flatbuffer {actual} != lightgbm {expected[row]}"

    def test_exact_threshold_values_match(self):
        """Only values exactly on a threshold observe `decision_lte` (`<=` vs `<`)."""
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
            assert actual == pytest.approx(
                expected[i], rel=1e-9, abs=1e-9
            ), f"probe {i} on threshold: flatbuffer {actual} != lightgbm {expected[i]}"

    def test_missing_values_match(self):
        """NaN features must take the direction `default_left` records."""
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
            assert actual == pytest.approx(
                expected[i], rel=1e-9, abs=1e-9
            ), f"NaN probe {i}: flatbuffer {actual} != lightgbm {expected[i]}"

    def test_learning_rate_is_not_applied_twice(self):
        """Leaf values already include `learning_rate`; it must not be applied again.

        0.3 (not the usual 0.1) makes a double application a 3x error, not rounding.
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
    def test_single_tree_model(self, tmp_path: Path):
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
        X, y = _create_synthetic_data(n_samples=200, n_features=3)
        model = _train_simple_model(X, y, num_trees=50)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:many_trees", fb_path)

        assert fb_path.exists()
        with open(fb_path, "rb") as f:
            buffer = f.read()

        assert len(buffer) > 5000

    def test_deep_tree(self, tmp_path: Path):
        X, y = _create_synthetic_data(n_samples=500, n_features=5)

        train_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "num_leaves": 63,
            "verbose": -1,
        }
        model = lgb.train(params, train_data, num_boost_round=5)

        lgbm_path = tmp_path / "model.lgbm"
        model.save_model(str(lgbm_path))

        fb_path = tmp_path / "model.bin"
        convert(lgbm_path, "sha256:deep_tree", fb_path)

        assert fb_path.exists()


class TestReproducibility:
    """Artifacts must rebuild byte for byte: RFC 0019 §10.5 records a hash over them."""

    @staticmethod
    def _saved(tmp_path: Path) -> Path:
        X, y = _create_synthetic_data(n_samples=200, n_features=4)
        lgbm_path = tmp_path / "model.lgbm"
        _train_simple_model(X, y, num_trees=8).save_model(str(lgbm_path))
        return lgbm_path

    def test_two_conversions_of_one_model_produce_identical_bytes(self, tmp_path: Path):
        lgbm_path = self._saved(tmp_path)
        first, second = tmp_path / "first.bin", tmp_path / "second.bin"
        first_digest = convert(
            lgbm_path,
            "sha256:0123456789abcdef",
            first,
            num_training_samples=200,
            training_arches=["gfx942"],
            model_version="1.0.0",
        )
        second_digest = convert(
            lgbm_path,
            "sha256:0123456789abcdef",
            second,
            num_training_samples=200,
            training_arches=["gfx942"],
            model_version="1.0.0",
        )
        assert first.read_bytes() == second.read_bytes()
        # The descriptor records this digest and TreeDataAdapter rechecks it, so it
        # must cover the bytes written.
        assert (
            first_digest
            == second_digest
            == hashlib.sha256(first.read_bytes()).hexdigest()
        )

    def test_an_unstamped_conversion_omits_the_date_rather_than_inventing_one(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
        convert(
            self._saved(tmp_path), "sha256:0123456789abcdef", tmp_path / "model.bin"
        )
        model = _read_model(tmp_path / "model.bin")
        assert model.TrainingDate() is None
        # Non-clock provenance is still written.
        assert model.Framework() == b"lightgbm"

    def test_source_date_epoch_supplies_the_stamp_a_reproducible_build_can_reproduce(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
        convert(
            self._saved(tmp_path), "sha256:0123456789abcdef", tmp_path / "model.bin"
        )
        assert (
            _read_model(tmp_path / "model.bin").TrainingDate()
            == b"2023-11-14T22:13:20+00:00"
        )

    def test_an_explicit_stamp_outranks_the_environment(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
        convert(
            self._saved(tmp_path),
            "sha256:0123456789abcdef",
            tmp_path / "model.bin",
            training_date="2024-02-29T00:00:00+00:00",
        )
        assert (
            _read_model(tmp_path / "model.bin").TrainingDate()
            == b"2024-02-29T00:00:00+00:00"
        )

    def test_an_unparseable_source_date_epoch_is_refused_not_ignored(
        self, tmp_path: Path, monkeypatch
    ):
        # The reproducible-builds specification requires the error.
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "yesterday")
        with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
            convert(
                self._saved(tmp_path), "sha256:0123456789abcdef", tmp_path / "model.bin"
            )

    def test_training_records_the_artifact_digest_in_the_descriptor_and_the_manifest(
        self, tmp_path: Path, evaluator
    ):
        block = [64 if row % 2 else 256 for row in range(80)]
        corpus = tmp_path / "corpus.csv"
        pd.DataFrame(
            {
                "kernel.block_size": block,
                "tflops": [120 - 0.2 * value for value in block],
            }
        ).to_csv(corpus, index=False)
        snapshot = tmp_path / "provenance.json"
        snapshot.write_text(
            json.dumps(
                {
                    "ued": {
                        "id": "13ab344f-4818-4772-bb8e-8e1441fec82c",
                        "revision": "1.0",
                    },
                    "kmd": {
                        "id": "46d64d06-18eb-483d-9bb4-94472d32b78d",
                        "revision": "1.0",
                    },
                    "umd": [],
                }
            ),
            encoding="utf-8",
        )
        output = tmp_path / "model"
        assert (
            main(
                [
                    "train",
                    "--input",
                    str(corpus),
                    "--provenance",
                    str(snapshot),
                    "--features",
                    "kernel.block_size",
                    "--target",
                    "tflops",
                    "--output-dir",
                    str(output),
                    "--num-boost-round",
                    "10",
                    "--early-stopping",
                    "5",
                ]
            )
            == 0
        )
        document = (output / "heuristic.uhd.json").read_text(encoding="utf-8")
        descriptor = json.loads(document)
        manifest = json.loads(
            (output / "train_manifest.json").read_text(encoding="utf-8")
        )
        artifact = (output / descriptor["tree_data"]["artifact"]).read_bytes()
        # Bare hex: TreeDataAdapter compares against `sha256(buffer, size)`, which has
        # no `sha256:` prefix.
        assert descriptor["tree_data"]["hash"] == hashlib.sha256(artifact).hexdigest()
        # RFC 0019.13 §10.5 wants the pair: the document and the artifact it names.
        assert manifest["model_sha256"] == descriptor["tree_data"]["hash"]
        assert (
            manifest["uhd_sha256"]
            == hashlib.sha256(document.encode("utf-8")).hexdigest()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
