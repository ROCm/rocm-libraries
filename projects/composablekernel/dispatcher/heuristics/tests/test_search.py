#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for search.py.

Covers: random search, DE search, config validity, result ordering,
budget compliance, and edge cases.
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
from search import SurrogateSearch


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal trained model."""
    fe = GemmUniversalFeatureEngine()
    n_features = len(fe.get_feature_names())
    np.random.seed(42)
    X = np.random.rand(200, n_features)
    y = np.random.rand(200) * 500
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))
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


class TestRandomSearch:
    def test_returns_results(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=50, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_results_sorted_descending(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=100, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_configs_are_valid(self, predictor):
        fe = GemmUniversalFeatureEngine()
        searcher = SurrogateSearch(predictor, feature_engine=fe, strategy="random")
        results = searcher.search(_problem(), budget=50, top_k=5)
        for cfg, _ in results:
            ps = fe.get_parameter_space()
            for k, v in cfg.items():
                if k in ps:
                    assert v in ps[k], f"{k}={v} not in {ps[k]}"

    def test_respects_top_k(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=100, top_k=3)
        assert len(results) <= 3

    def test_different_problems_produce_results(self, predictor):
        """Both problem sizes should produce valid search results."""
        searcher = SurrogateSearch(predictor, strategy="random", seed=42)
        r1 = searcher.search(
            {
                "m": 16,
                "n": 1536,
                "k": 7168,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=3,
        )
        searcher2 = SurrogateSearch(predictor, strategy="random", seed=42)
        r2 = searcher2.search(
            {
                "m": 20480,
                "n": 7168,
                "k": 256,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=3,
        )
        assert len(r1) > 0
        assert len(r2) > 0
        for _, score in r1 + r2:
            assert np.isfinite(score)

    def test_m1_corner_case(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(
            {
                "m": 1,
                "n": 4096,
                "k": 4096,
                "dtype": "fp8",
                "layout": "rcr",
                "split_k": 1,
            },
            budget=50,
            top_k=5,
        )
        assert len(results) > 0
        for _, score in results:
            assert np.isfinite(score)


class TestDESearch:
    def test_returns_results(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=100, top_k=5)
        assert len(results) > 0

    def test_results_sorted_descending(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=100, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_de_improves_over_initial(self, predictor):
        """DE should generally find at least as good as random initialization."""
        searcher_r = SurrogateSearch(predictor, strategy="random", seed=42)
        r_results = searcher_r.search(_problem(), budget=100, top_k=1)
        searcher_d = SurrogateSearch(predictor, strategy="de", seed=42)
        d_results = searcher_d.search(_problem(), budget=100, top_k=1)
        if r_results and d_results:
            assert d_results[0][1] >= r_results[0][1] * 0.9

    def test_small_budget(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="de")
        results = searcher.search(_problem(), budget=30, top_k=5)
        assert len(results) > 0


class TestSearchEdgeCases:
    def test_unknown_strategy_raises(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="unknown")
        with pytest.raises(ValueError):
            searcher.search(_problem(), budget=10)

    def test_zero_budget(self, predictor):
        searcher = SurrogateSearch(predictor, strategy="random")
        results = searcher.search(_problem(), budget=0, top_k=5)
        assert len(results) == 0

    def test_deterministic_with_same_seed(self, predictor):
        s1 = SurrogateSearch(predictor, strategy="random", seed=123)
        s2 = SurrogateSearch(predictor, strategy="random", seed=123)
        r1 = s1.search(_problem(), budget=50, top_k=5)
        r2 = s2.search(_problem(), budget=50, top_k=5)
        assert len(r1) == len(r2)
        for (c1, s1_), (c2, s2_) in zip(r1, r2):
            assert s1_ == pytest.approx(s2_)


class TestTheSearchEngineMatchesTheModel:
    """SurrogateSearch built its engine bare while the predictor scored with
    the model's own constants, so the candidate FILTER and the SCORER could
    disagree about the same hardware. get_constraints rejects on
    self._hw["lds_capacity"], so against a 160 KB model the filter discarded
    legal configs at the 64 KB default and they never reached the scorer.

    The existing model_dir fixture cannot show this: its spec records no
    `hardware` key, so both paths resolve to a default engine and the bug is
    invisible. This fixture records one."""

    #: Needs 128 KB of LDS ((256*256 + 256*256) bytes): rejected at the
    #: 64 KB default, accepted at 160 KB.
    BIG_TILE = dict(
        tile_m=256,
        tile_n=256,
        tile_k=256,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="compv3",
    )

    @pytest.fixture
    def big_lds_model(self, tmp_path):
        fe = GemmUniversalFeatureEngine()
        n = len(fe.get_feature_names())
        np.random.seed(0)
        model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
        model.fit(np.random.rand(50, n), np.random.rand(50))
        model.booster_.save_model(str(tmp_path / "model_tflops.lgbm"))
        (tmp_path / "feature_spec.json").write_text(
            json.dumps(
                {
                    "feature_names": fe.get_feature_names(),
                    "categorical_features": fe.get_categorical_features(),
                    "hardware": {"lds_capacity": 163840},
                }
            )
        )
        return tmp_path

    def test_the_config_discriminates(self):
        """Guard the guard: if BIG_TILE were legal at both capacities the test
        below would hold with the fix reverted."""
        assert not GemmUniversalFeatureEngine().validate_config(self.BIG_TILE)
        assert GemmUniversalFeatureEngine(lds_capacity=163840).validate_config(
            self.BIG_TILE
        )

    def test_the_search_filter_uses_the_models_constants(self, big_lds_model):
        search = SurrogateSearch(Predictor(big_lds_model))
        assert search._fe.validate_config(self.BIG_TILE), (
            "the search engine was built bare, so it filters candidates at the "
            "64 KB default while the predictor scores at the model's 160 KB"
        )

    def test_an_explicitly_passed_engine_is_not_replaced(self, big_lds_model):
        """A caller's engine must win even though the predictor now has one."""
        mine = GemmUniversalFeatureEngine()
        search = SurrogateSearch(Predictor(big_lds_model), feature_engine=mine)
        assert search._fe is mine

    def test_a_falsy_engine_is_still_honoured(self, big_lds_model):
        """Pins `is not None` rather than `or`, which the test above cannot.

        No engine defines __bool__ today, so every instance is truthy and the
        two spellings are behaviourally identical -- rewriting the resolution
        as a chain of `or` left the suite green. A deliberately falsy engine
        makes the distinction observable, so the claim in the production
        comment is guarded rather than merely asserted."""

        class FalsyEngine(GemmUniversalFeatureEngine):
            def __bool__(self):
                return False

        mine = FalsyEngine()
        assert not mine
        search = SurrogateSearch(Predictor(big_lds_model), feature_engine=mine)
        assert search._fe is mine, (
            "a falsy engine was discarded, so the resolution is testing "
            "truthiness instead of presence"
        )

    def test_a_falsy_engine_from_the_predictor_is_still_honoured(self):
        """The same distinction on the OTHER branch.

        The test above passes an engine explicitly, so it only exercises the
        caller-supplied branch; rewriting the predictor branch as `or` left it
        green. A stub predictor is enough -- SurrogateSearch reads only
        `feature_engine` from it during construction."""

        class FalsyEngine(GemmUniversalFeatureEngine):
            def __bool__(self):
                return False

        class StubPredictor:
            def __init__(self, engine):
                self.feature_engine = engine

        engine = FalsyEngine()
        search = SurrogateSearch(StubPredictor(engine))
        assert search._fe is engine, (
            "the predictor's engine was discarded because it is falsy, so the "
            "search filters on default constants while the model scores on its "
            "own"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
