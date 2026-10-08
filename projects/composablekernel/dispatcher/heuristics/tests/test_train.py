#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for train.py.

Covers: group key computation, TFLOPS efficiency calculation, edge cases
(single group, all-invalid data, tied predictions), and warm-start
incremental training (feature compat, lineage, quality).
"""

import json
import sys
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import GemmUniversalFeatureEngine
from train import (
    compute_group_keys,
    compute_tflops_efficiency,
    check_feature_compatibility,
    load_warm_start_model,
    train_final_model,
    DEFAULT_PARAMS,
)


class TestComputeGroupKeys:
    def test_basic(self):
        df = pd.DataFrame(
            {"m": [16, 16, 32], "n": [1536, 1536, 1536], "k": [7168, 7168, 7168]}
        )
        keys = compute_group_keys(df, "gemm_universal")
        assert keys[0] == keys[1]
        assert keys[0] != keys[2]

    def test_unique_shapes(self):
        df = pd.DataFrame({"m": [1, 2, 3], "n": [4, 5, 6], "k": [7, 8, 9]})
        keys = compute_group_keys(df, "gemm_universal")
        assert len(set(keys)) == 3


class TestComputeTflopsEfficiency:
    def test_perfect_prediction(self):
        """Model predicts highest TFLOPS kernel => efficiency = 1.0."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [50, 300, 100],  # correctly ranks kernel 1 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_worst_prediction(self):
        """Model picks the worst kernel."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 150],
                "pred_tflops": [999, 1, 1],  # incorrectly ranks kernel 0 highest
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert eff["efficiency"].iloc[0] == pytest.approx(100 / 200)

    def test_multiple_shapes(self):
        df = pd.DataFrame(
            {
                "m": [16, 16, 32, 32],
                "n": [1536, 1536, 1536, 1536],
                "k": [7168, 7168, 7168, 7168],
                "measured_tflops": [10, 20, 100, 200],
                "pred_tflops": [5, 25, 150, 190],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 2
        assert eff.iloc[0]["efficiency"] == pytest.approx(1.0)
        assert eff.iloc[1]["efficiency"] == pytest.approx(1.0)

    def test_zero_tflops_shape_skipped(self):
        df = pd.DataFrame(
            {
                "m": [16, 16],
                "n": [16, 16],
                "k": [16, 16],
                "measured_tflops": [0, 0],
                "pred_tflops": [1, 2],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 0

    def test_single_kernel_per_shape(self):
        df = pd.DataFrame(
            {
                "m": [1024],
                "n": [1024],
                "k": [1024],
                "measured_tflops": [150],
                "pred_tflops": [100],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] == pytest.approx(1.0)

    def test_tied_predictions(self):
        """When multiple kernels have the same predicted TFLOPS, pandas idxmax picks the first."""
        df = pd.DataFrame(
            {
                "m": [1024, 1024, 1024],
                "n": [1024, 1024, 1024],
                "k": [1024, 1024, 1024],
                "measured_tflops": [100, 200, 200],
                "pred_tflops": [50, 50, 50],
            }
        )
        eff = compute_tflops_efficiency(df, "gemm_universal", "pred_tflops")
        assert len(eff) == 1
        assert eff["efficiency"].iloc[0] >= 0.5


# ---------------------------------------------------------------------------
# Helpers for warm-start tests
# ---------------------------------------------------------------------------


def _make_dummy_data(n_rows=200, n_shapes=5):
    """Create a small synthetic benchmark DataFrame for testing training."""
    rng = np.random.RandomState(42)
    rows = []
    for _ in range(n_rows):
        m = rng.choice([64, 128, 256, 512, 1024])
        n = rng.choice([64, 128, 256, 512, 1024])
        k = rng.choice([64, 128, 256, 512, 1024])
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "split_k": 1,
                "dtype": "fp8",
                "layout": "rcr",
                "op_type": "gemm_universal",
                "tile_m": rng.choice([64, 128, 256]),
                "tile_n": rng.choice([64, 128, 256]),
                "tile_k": rng.choice([32, 64, 128]),
                "warp_m": rng.choice([1, 2, 4]),
                "warp_n": rng.choice([1, 2, 4]),
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 16,
                "pipeline": rng.choice(["compv3", "compv4", "mem"]),
                "scheduler": rng.choice(["intrawave", "interwave"]),
                "epilogue": "cshuffle",
                "pad_m": False,
                "pad_n": False,
                "pad_k": False,
                "persistent": False,
                "measured_tflops": float(rng.uniform(10, 500)),
                "latency_ms": float(rng.uniform(0.01, 1.0)),
                "bandwidth_gb_s": float(rng.uniform(50, 1500)),
                "is_valid": True,
                "kernel_name": f"test_kernel_{rng.randint(0, 100)}",
            }
        )
    return pd.DataFrame(rows)


def _save_feature_spec(model_dir, fe):
    """Save a feature_spec.json matching the given feature engine."""
    spec = {
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
    }
    with open(model_dir / "feature_spec.json", "w") as f:
        json.dump(spec, f)


def _train_and_save_base_model(model_dir, df, fe, target="tflops"):
    """Train a small base model and save it to model_dir."""
    params = dict(DEFAULT_PARAMS)
    params["n_estimators"] = 20
    params["n_jobs"] = 1
    model = train_final_model(df, fe, target, params, "gemm_universal")
    model.booster_.save_model(str(model_dir / f"model_{target}.lgbm"))
    _save_feature_spec(model_dir, fe)
    return model


# ---------------------------------------------------------------------------
# Warm-start tests
# ---------------------------------------------------------------------------


class TestCheckFeatureCompatibility:
    def test_compatible_passes(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        _save_feature_spec(tmp_path, fe)
        check_feature_compatibility(tmp_path, fe)

    def test_missing_spec_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        with pytest.raises(FileNotFoundError, match="feature_spec.json"):
            check_feature_compatibility(tmp_path, fe)

    def test_added_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names()[:-1],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_removed_feature_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names() + ["extra_feature"],
            "categorical_features": fe.get_categorical_features(),
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Feature schema mismatch"):
            check_feature_compatibility(tmp_path, fe)

    def test_categorical_mismatch_raises(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        spec = {
            "feature_names": fe.get_feature_names(),
            "categorical_features": ["layout", "pipeline"],
        }
        with open(tmp_path / "feature_spec.json", "w") as f:
            json.dump(spec, f)
        with pytest.raises(ValueError, match="Categorical feature mismatch"):
            check_feature_compatibility(tmp_path, fe)


class TestLoadWarmStartModel:
    def test_loads_existing_model(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe)
        path = load_warm_start_model(tmp_path, "tflops")
        assert path is not None
        assert Path(path).exists()

    def test_returns_none_for_missing_target(self, tmp_path):
        assert load_warm_start_model(tmp_path, "tflops") is None

    def test_returns_none_for_wrong_target(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data()
        _train_and_save_base_model(tmp_path, df, fe, target="tflops")
        assert load_warm_start_model(tmp_path, "bandwidth") is None


class TestWarmStartTraining:
    def test_warm_start_produces_more_trees(self, tmp_path):
        """A warm-started model should have more trees than the base."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)
        base_n_trees = base_model.booster_.num_trees()

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_n_trees = warm_model.booster_.num_trees()

        assert warm_n_trees > base_n_trees

    def test_warm_start_does_not_degrade(self, tmp_path):
        """Warm-started model on the same data should not be significantly worse."""
        fe = GemmUniversalFeatureEngine()
        df = _make_dummy_data(n_rows=300)

        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_model = _train_and_save_base_model(base_dir, df, fe)

        X = fe.extract_batch(df[df["is_valid"]].reset_index(drop=True))
        y = df[df["is_valid"]]["measured_tflops"].values
        base_rmse = np.sqrt(np.mean((base_model.predict(X) - y) ** 2))

        init_model_path = load_warm_start_model(base_dir, "tflops")
        params = dict(DEFAULT_PARAMS)
        params["n_estimators"] = 15
        params["n_jobs"] = 1
        warm_model = train_final_model(
            df, fe, "tflops", params, "gemm_universal", init_model=init_model_path
        )
        warm_rmse = np.sqrt(np.mean((warm_model.predict(X) - y) ** 2))

        assert warm_rmse <= base_rmse * 1.1

    def test_warm_start_from_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            check_feature_compatibility(
                Path("/nonexistent/model/dir"), GemmUniversalFeatureEngine()
            )


class TestEngineVariants:
    """gemm_universal_vec is gemm_universal DATA read through a wider engine.
    Only the engine lookup may branch on the variant; everything data-side must
    normalise, or the variant filters to an empty dataset and KeyErrors on
    TARGET_COLUMNS."""

    def test_base_operation_maps_the_variant(self):
        from train import base_operation

        assert base_operation("gemm_universal_vec") == "gemm_universal"

    def test_base_operation_is_identity_for_real_operations(self):
        from train import base_operation

        for op in ("gemm_universal", "grouped_conv", "fmha"):
            assert base_operation(op) == op

    def test_factory_returns_the_wider_engine(self):
        from train import get_feature_engine

        base = get_feature_engine("gemm_universal")
        vec = get_feature_engine("gemm_universal_vec")
        assert type(vec).__name__ == "GemmUniversalVecFeatureEngine"
        assert len(vec.get_feature_names()) == len(base.get_feature_names()) + 6

    def test_target_columns_resolve_for_the_variant(self):
        from train import TARGET_COLUMNS, base_operation

        assert "tflops" in TARGET_COLUMNS[base_operation("gemm_universal_vec")]

    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                "m": [128, 128, 256],
                "n": [256, 256, 512],
                "k": [512, 512, 1024],
                "measured_tflops": [10.0, 5.0, 8.0],
                "pred_tflops": [9.0, 6.0, 7.0],
            }
        )

    def test_group_keys_normalise_the_variant(self):
        """The normalisation lives INSIDE compute_group_keys, so asserting
        base_operation() in isolation does not reach it. Without it the variant
        falls through to ValueError and the variant cannot be trained at all."""
        from train import compute_group_keys

        df = self._frame()
        np.testing.assert_array_equal(
            compute_group_keys(df, "gemm_universal_vec"),
            compute_group_keys(df, "gemm_universal"),
        )

    def test_efficiency_normalises_the_variant(self):
        """Same normalisation, second call site."""
        from train import compute_tflops_efficiency

        df = self._frame()
        pd.testing.assert_frame_equal(
            compute_tflops_efficiency(df, "gemm_universal_vec", "pred_tflops"),
            compute_tflops_efficiency(df, "gemm_universal", "pred_tflops"),
        )

    def test_parser_accepts_the_variant_and_rejects_junk(self):
        """Build the real parser rather than grepping the source: a literal can
        sit in the help text while the choices list rejects the value."""
        import train

        argv = [
            "--data_dir",
            "d",
            "--out_dir",
            "o",
            "--operation",
            "gemm_universal_vec",
        ]
        with mock.patch.object(sys, "argv", ["train.py", *argv]):
            parser = train.build_arg_parser()
            assert parser.parse_args(argv).operation == "gemm_universal_vec"
            with pytest.raises(SystemExit):
                parser.parse_args(
                    [
                        "--data_dir",
                        "d",
                        "--out_dir",
                        "o",
                        "--operation",
                        "definitely_not_an_operation",
                    ]
                )


class TestFeatureSpecRecordsTheEngine:
    """The read side (predict._engine_for_spec) is only useful if the write side
    records the engine. Dropping the key leaves older specs loadable, so nothing
    else in the suite notices -- the failure surfaces later, as a vec model
    loaded with the base engine."""

    @staticmethod
    def _spec(operation):
        from train import build_feature_spec, get_feature_engine

        fe = get_feature_engine(operation)
        return fe, build_feature_spec(
            operation, fe, "bf16", "gfx950", ["tflops"], ["tflops"], {}
        )

    @pytest.mark.parametrize("operation", ["gemm_universal", "gemm_universal_vec"])
    def test_round_trip_through_predict(self, tmp_path, operation):
        """Build the spec the way train.py does, serialise it, and load it the
        way Predictor does. Constructing the dict in the test instead would pass
        even with the key removed from train.py."""
        import json

        from predict import _engine_for_spec

        fe, spec = self._spec(operation)
        path = tmp_path / "feature_spec.json"
        path.write_text(json.dumps(spec))
        assert type(_engine_for_spec(json.loads(path.read_text()))) is type(fe)

    def test_the_spec_records_the_two_engines_distinctly(self):
        """Guards the whole point: a spec hardcoded to the base engine would
        still round-trip for gemm_universal."""
        assert (
            self._spec("gemm_universal")[1]["feature_engine"]
            != self._spec("gemm_universal_vec")[1]["feature_engine"]
        )

    def test_the_spec_feature_names_match_the_engine(self):
        """A spec that names one engine while carrying another's feature list
        makes Predictor raise on load, so pin them together."""
        for operation in ("gemm_universal", "gemm_universal_vec"):
            fe, spec = self._spec(operation)
            assert spec["feature_names"] == fe.get_feature_names(), operation

    def test_the_two_operations_give_different_engines(self):
        """Guards the whole point: if both resolved to the base engine the
        round-trip above would still pass."""
        from train import get_feature_engine

        assert type(get_feature_engine("gemm_universal")) is not type(
            get_feature_engine("gemm_universal_vec")
        )


class TestVariantReachesTheRunners:
    """The three call sites that actually run a training job -- run_cv,
    train_final_model and the build_training_dataset call in main() -- each
    normalise the variant through base_operation(). Asserting
    TARGET_COLUMNS[base_operation(...)] in a test re-applies the helper itself,
    so it pins the helper and not the caller: deleting the normalisation inside
    run_cv left the whole suite green. That is the same caller-vs-helper gap the
    evaluate.py arity fix in this branch was written to close.
    """

    @staticmethod
    def _frame(n=24):
        rs = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "m": rs.choice([128, 256, 512], n),
                "n": rs.choice([256, 512], n),
                "k": rs.choice([512, 1024], n),
                "measured_tflops": rs.uniform(1.0, 100.0, n),
                "is_valid": True,
            }
        )

    class _Engine:
        """Minimal engine: run_cv only needs these three methods."""

        def get_feature_names(self):
            return ["m", "n", "k"]

        def get_categorical_features(self):
            return []

        def extract_batch(self, df):
            return df[["m", "n", "k"]].to_numpy(dtype=float)

    def test_run_cv_accepts_the_variant(self):
        """Without the normalisation this raises KeyError on TARGET_COLUMNS."""
        from train import run_cv

        out = run_cv(
            self._frame(),
            self._Engine(),
            target="tflops",
            params={"n_estimators": 2, "verbose": -1},
            operation="gemm_universal_vec",
            n_splits=2,
        )
        assert out is not None

    def test_train_final_model_accepts_the_variant(self):
        from train import train_final_model

        out = train_final_model(
            self._frame(),
            self._Engine(),
            target="tflops",
            params={"n_estimators": 2, "verbose": -1},
            operation="gemm_universal_vec",
        )
        assert out is not None

    def test_the_dataset_is_loaded_under_the_base_operation(self):
        """main() must ask build_training_dataset for gemm_universal data; the
        variant is an engine choice, not a different op_type in the parquet."""
        import train

        seen = {}

        def fake(data_dir, op_type=None, dtype=None, **kw):
            seen["op_type"] = op_type
            raise SystemExit  # stop before training; we only need the kwarg

        with mock.patch.object(train, "build_training_dataset", fake):
            argv = [
                "train.py",
                "--data_dir",
                "d",
                "--out_dir",
                "o",
                "--operation",
                "gemm_universal_vec",
                "--targets",
                "tflops",
            ]
            with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
                train.main()
        assert seen["op_type"] == "gemm_universal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
