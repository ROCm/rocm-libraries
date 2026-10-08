#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for predict.py.

Covers: Predictor initialization, single prediction, ranking, select_best,
missing model handling, and edge cases (single kernel, empty list).
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_engine import GemmUniversalFeatureEngine
from predict import Predictor


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal trained model for testing."""
    fe = GemmUniversalFeatureEngine()
    n_features = len(fe.get_feature_names())

    np.random.seed(42)
    X = np.random.rand(200, n_features)
    y = np.random.rand(200) * 100

    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))

    y_lat = np.random.rand(200) * 0.1
    model_lat = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model_lat.fit(X, y_lat)
    model_lat.booster_.save_model(str(tmp_path / "model_latency.lgbm"))

    spec = {
        "feature_names": fe.get_feature_names(),
        "categorical_features": fe.get_categorical_features(),
    }
    with open(tmp_path / "feature_spec.json", "w") as f:
        json.dump(spec, f)

    return tmp_path


@pytest.fixture
def predictor(model_dir):
    return Predictor(model_dir)


def _problem():
    return {
        "m": 1024,
        "n": 1024,
        "k": 1024,
        "dtype": "fp8",
        "layout": "rcr",
        "split_k": 1,
    }


def _kernel(tile_m=128, pipeline="compv3"):
    return {
        "kernel_name": f"test_kernel_{tile_m}_{pipeline}",
        "tile_m": tile_m,
        "tile_n": 128,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": pipeline,
        "scheduler": "intrawave",
        "epilogue": "cshuffle",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "persistent": False,
    }


class TestPredictor:
    def test_predict_tflops_returns_float(self, predictor):
        result = predictor.predict_tflops(_problem(), _kernel())
        assert isinstance(result, float)

    def test_predict_latency_returns_float(self, predictor):
        result = predictor.predict_latency(_problem(), _kernel())
        assert isinstance(result, float)

    def test_predict_all_returns_dict(self, predictor):
        result = predictor.predict_all(_problem(), _kernel())
        assert "tflops" in result
        assert "latency_ms" in result

    def test_rank_kernels_sorted_descending(self, predictor):
        kernels = [_kernel(64, "compv3"), _kernel(128, "compv4"), _kernel(256, "mem")]
        ranked = predictor.rank_kernels(_problem(), kernels)
        assert len(ranked) == 3
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_select_best_returns_name(self, predictor):
        kernels = [_kernel(64), _kernel(128)]
        best = predictor.select_best(_problem(), kernels)
        assert isinstance(best, str)
        assert best in [k["kernel_name"] for k in kernels]

    def test_single_kernel(self, predictor):
        kernels = [_kernel(128)]
        ranked = predictor.rank_kernels(_problem(), kernels)
        assert len(ranked) == 1

    def test_missing_bandwidth_model(self, model_dir):
        pred = Predictor(model_dir)
        with pytest.raises(FileNotFoundError):
            pred.predict_bandwidth(_problem(), _kernel())

    def test_empty_kernel_list(self, predictor):
        with pytest.raises(ValueError):
            predictor.select_best(_problem(), [])

    def test_corner_case_m1(self, predictor):
        prob = {
            "m": 1,
            "n": 4096,
            "k": 4096,
            "dtype": "fp8",
            "layout": "rcr",
            "split_k": 1,
        }
        result = predictor.predict_tflops(prob, _kernel())
        assert np.isfinite(result)

    def test_different_shapes_give_different_results(self, predictor):
        k = _kernel()
        r1 = predictor.predict_tflops(
            {
                "m": 16,
                "n": 1536,
                "k": 7168,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            k,
        )
        r2 = predictor.predict_tflops(
            {
                "m": 20480,
                "n": 7168,
                "k": 256,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            k,
        )
        assert r1 != r2


class TestPredictorEdgeCases:
    def test_nonexistent_model_dir(self):
        with pytest.raises(Exception):
            pred = Predictor("/nonexistent/path")
            pred.predict_tflops(_problem(), _kernel())


class TestEngineForSpec:
    """A vec-trained model must load without the caller naming the engine;
    otherwise Predictor raises on the six feature names the base engine cannot
    supply, and evaluate/predict/ml_heuristic_sweep can never read it."""

    def test_vec_spec_rebuilds_the_vec_engine(self):
        from predict import _engine_for_spec

        fe = _engine_for_spec({"feature_engine": "GemmUniversalVecFeatureEngine"})
        assert type(fe).__name__ == "GemmUniversalVecFeatureEngine"

    def test_absent_key_falls_back_to_the_base_engine(self):
        """Model directories written before the key existed must still load."""
        from predict import _engine_for_spec

        assert type(_engine_for_spec({})).__name__ == "GemmUniversalFeatureEngine"

    def test_unknown_name_raises_rather_than_guessing(self):
        """Falling back to the base engine for a name we do not recognise
        extracts the wrong features whenever the spec omits feature_names, which
        is the only other thing that would catch the mismatch."""
        from predict import _engine_for_spec

        with pytest.raises(ValueError, match="SomeFutureEngine"):
            _engine_for_spec({"feature_engine": "SomeFutureEngine"})

    def test_absent_key_resolves_from_op_type(self):
        """Every shipped model dir predates the feature_engine key but carries
        op_type. Defaulting straight to the GEMM engine hands a grouped_conv
        model an engine that cannot supply its features, and the error then
        blames the engine rather than the missing key."""
        from predict import _engine_for_spec

        cases = {
            "grouped_conv": "GroupedConvFeatureEngine",
            "gemm_universal": "GemmUniversalFeatureEngine",
            "gemm_universal_vec": "GemmUniversalVecFeatureEngine",
        }
        for op, expected in cases.items():
            assert type(_engine_for_spec({"op_type": op})).__name__ == expected, op

    def test_absent_key_and_unknown_op_type_falls_back(self):
        """Neither key present: the base engine is the only sane default."""
        from predict import _engine_for_spec

        assert type(_engine_for_spec({})).__name__ == "GemmUniversalFeatureEngine"
        assert (
            type(_engine_for_spec({"op_type": "something_else"})).__name__
            == "GemmUniversalFeatureEngine"
        )


class TestPredictorEngineWiring:
    """_engine_for_spec in isolation does not prove Predictor calls it. Replacing
    the call with a hardcoded base engine previously left the suite green."""

    @staticmethod
    def _model_dir(tmp_path, engine_name, feature_names):
        import lightgbm as lgb
        import numpy as np

        X = np.random.RandomState(0).rand(40, len(feature_names))
        y = np.random.RandomState(1).rand(40)
        lgb.LGBMRegressor(n_estimators=3, verbose=-1).fit(X, y).booster_.save_model(
            str(tmp_path / "model_tflops.lgbm")
        )
        spec = {"feature_engine": engine_name, "feature_names": feature_names}
        (tmp_path / "feature_spec.json").write_text(json.dumps(spec))
        return tmp_path

    def test_vec_spec_gives_predictor_the_vec_engine(self, tmp_path):
        from feature_engine_vec import GemmUniversalVecFeatureEngine
        from predict import Predictor

        names = GemmUniversalVecFeatureEngine().get_feature_names()
        d = self._model_dir(tmp_path, "GemmUniversalVecFeatureEngine", names)
        assert (
            type(Predictor(d).feature_engine).__name__
            == "GemmUniversalVecFeatureEngine"
        )

    def test_base_spec_gives_predictor_the_base_engine(self, tmp_path):
        from predict import Predictor

        names = GemmUniversalFeatureEngine().get_feature_names()
        d = self._model_dir(tmp_path, "GemmUniversalFeatureEngine", names)
        assert (
            type(Predictor(d).feature_engine).__name__ == "GemmUniversalFeatureEngine"
        )


class TestFeatureRemap:
    """Predictor remaps engine output onto the model's recorded feature_names so
    a model trained with a narrower or differently-ordered schema still loads.
    Every other test writes the engine's full list in order, which leaves
    _feature_indices None and the whole branch unexercised."""

    @staticmethod
    def _dir(tmp_path, feature_names, n_model_features):
        import lightgbm as lgb
        import numpy as np

        rs = np.random.RandomState(0)
        lgb.LGBMRegressor(n_estimators=3, verbose=-1).fit(
            rs.rand(40, n_model_features), rs.rand(40)
        ).booster_.save_model(str(tmp_path / "model_tflops.lgbm"))
        (tmp_path / "feature_spec.json").write_text(
            json.dumps(
                {
                    "feature_engine": "GemmUniversalVecFeatureEngine",
                    "feature_names": feature_names,
                }
            )
        )
        return tmp_path

    def test_a_reordered_subset_is_remapped(self, tmp_path):
        from feature_engine_vec import GemmUniversalVecFeatureEngine
        from predict import Predictor

        names = GemmUniversalVecFeatureEngine().get_feature_names()
        subset = list(reversed(names[:20]))
        p = Predictor(self._dir(tmp_path, subset, len(subset)))
        assert p._feature_indices is not None, "remap was not built"

        engine_order = {n: i for i, n in enumerate(names)}
        assert list(p._feature_indices) == [engine_order[n] for n in subset]

        X = np.arange(len(names), dtype=float).reshape(1, -1)
        np.testing.assert_array_equal(
            p._select_features(X)[0], [float(engine_order[n]) for n in subset]
        )

    def test_a_feature_the_engine_cannot_supply_raises(self, tmp_path):
        from predict import Predictor

        with pytest.raises(ValueError, match="cannot supply"):
            Predictor(self._dir(tmp_path, ["vec_a", "not_a_real_feature"], 2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
