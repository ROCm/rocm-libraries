# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Covers the regret metric: does the ranking this regressor induces pick well?

The model predicts TFLOPS and the ranking falls out of sorting that prediction, so
prediction error and selection quality are different questions. These tests pin the
difference, because a suite that only checked RMSE would accept a model that ranks
backwards.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from uhd_gen.train_uhd import evaluate_regret, induced_ranking_regret


def test_a_correct_pick_has_no_regret():
    measured = np.array([10.0, 50.0, 30.0])
    predicted = np.array([1.0, 9.0, 5.0])  # ranks the 50 first
    regret, hit = induced_ranking_regret(measured, predicted)
    assert regret == 0.0
    assert hit


def test_regret_is_the_throughput_given_up():
    # Picking the 30 when a 50 was available loses 40% of the achievable rate. The
    # metric is relative so it can be pooled across problems of different sizes.
    measured = np.array([10.0, 50.0, 30.0])
    predicted = np.array([1.0, 2.0, 9.0])
    regret, hit = induced_ranking_regret(measured, predicted)
    assert regret == pytest.approx(0.4)
    assert not hit


def test_a_confident_but_backwards_model_is_not_excused():
    # The point of the metric. This model's predictions are close to the measurements
    # in absolute terms -- a good RMSE -- and its ordering is exactly inverted.
    measured = np.array([10.0, 20.0, 30.0])
    predicted = np.array([30.0, 20.0, 10.0])
    regret, hit = induced_ranking_regret(measured, predicted)
    assert regret == pytest.approx(2.0 / 3.0)
    assert not hit


def test_ties_count_as_a_hit():
    # Two configurations at the same rate: picking either is a correct pick, and
    # counting one as a miss would penalise the model for a distinction that does not
    # exist in the measurement.
    measured = np.array([50.0, 50.0, 10.0])
    predicted = np.array([1.0, 9.0, 5.0])
    regret, hit = induced_ranking_regret(measured, predicted)
    assert regret == 0.0
    assert hit


def test_a_problem_that_measured_nothing_is_not_a_perfect_score():
    # All-zero measurements would divide by zero; returning 0 regret would report a
    # perfect pick for a problem where nothing ran.
    regret, hit = induced_ranking_regret(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    assert np.isnan(regret)
    assert not hit


def _corpus(n_problems: int = 40, seed: int = 0) -> pd.DataFrame:
    """A corpus whose best configuration depends on the problem, learnably."""
    rng = np.random.default_rng(seed)
    rows = []
    for problem in range(n_problems):
        m = int(rng.integers(1, 9)) * 128
        for tile in (32, 64, 128, 256):
            # Larger problems favour larger tiles; that is the signal the model can
            # find, and the reason regret should come out well below chance.
            rows.append(
                {
                    "q.M": m,
                    "q.N": 4096,
                    "kernel.tile_m": tile,
                    "tflops": 100.0 - abs(m / 8.0 - tile) * 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_regret_is_measured_out_of_fold_and_reports_its_population():
    corpus = _corpus()
    metrics = evaluate_regret(
        corpus,
        feature_cols=["q.M", "q.N", "kernel.tile_m"],
        target_col="tflops",
        problem_cols=["q.M", "q.N"],
        num_boost_round=40,
        n_splits=4,
    )

    # Every reported figure comes with the population it was computed over, so a
    # headline number cannot be read without knowing what was excluded.
    assert metrics["problems_scored"] > 0
    assert "problems_single_variant" in metrics
    assert "problems_unusable" in metrics
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_regret"] <= 1.0
    assert metrics["max_regret"] >= metrics["median_regret"]


def test_a_learnable_corpus_beats_picking_at_random():
    # Guards the metric itself: if evaluate_regret reported nonsense, this would not
    # separate a learnable signal from chance. Four configurations per problem, so a
    # random pick lands on the best 25% of the time.
    corpus = _corpus(n_problems=60, seed=3)
    metrics = evaluate_regret(
        corpus,
        feature_cols=["q.M", "q.N", "kernel.tile_m"],
        target_col="tflops",
        problem_cols=["q.M", "q.N"],
        num_boost_round=60,
        n_splits=4,
    )
    assert metrics["top1_accuracy"] > 0.25


def test_single_variant_problems_are_excluded_rather_than_scored_as_perfect():
    # A problem with one configuration has zero regret by construction. Counting it
    # would dilute the metric toward whatever fraction of the corpus is single-variant
    # and make a corpus look better by being less thoroughly measured.
    corpus = _corpus(n_problems=30)
    lonely = pd.DataFrame(
        [{"q.M": 99999, "q.N": 4096, "kernel.tile_m": 64, "tflops": 50.0}]
    )
    both = pd.concat([corpus, lonely], ignore_index=True)

    metrics = evaluate_regret(
        both,
        feature_cols=["q.M", "q.N", "kernel.tile_m"],
        target_col="tflops",
        problem_cols=["q.M", "q.N"],
        num_boost_round=40,
        n_splits=4,
    )
    assert metrics["problems_single_variant"] == 1


def test_a_corpus_too_small_to_split_is_refused():
    # Silently reducing the fold count would report a regret measured on a split
    # nobody asked for; refusing says the corpus cannot answer the question.
    tiny = pd.DataFrame(
        [
            {"q.M": 128, "q.N": 4096, "kernel.tile_m": 32, "tflops": 10.0},
            {"q.M": 128, "q.N": 4096, "kernel.tile_m": 64, "tflops": 20.0},
        ]
    )
    with pytest.raises(ValueError, match="fewer than"):
        evaluate_regret(
            tiny,
            feature_cols=["q.M", "q.N", "kernel.tile_m"],
            target_col="tflops",
            problem_cols=["q.M", "q.N"],
            n_splits=5,
        )


def test_negative_prediction_is_reported(caplog):
    """A model predicting a target its units cannot take must say so at training time.

    The runtime bounds this regardless, but by then the only recourse is to discard the score.
    Training is where it can still be fixed, and a model doing it on its own training data will
    do it worse in the field.
    """
    lgb = pytest.importorskip("lightgbm")
    np = pytest.importorskip("numpy")

    from uhd_gen.train_uhd import _report_out_of_range_predictions

    class _AlwaysNegative:
        """Stands in for a booster; the check only calls predict()."""

        @staticmethod
        def predict(features):
            return np.full(len(features), -0.75)

    features = np.zeros((4, 2))
    with caplog.at_level(logging.ERROR):
        count = _report_out_of_range_predictions(_AlwaysNegative(), features, "tflops")

    assert count == 4
    assert "negative tflops" in caplog.text
    assert "4 of 4" in caplog.text


def test_a_wholly_positive_model_reports_nothing(caplog):
    """The quiet path, so the check cannot become noise that gets ignored."""
    pytest.importorskip("lightgbm")
    np = pytest.importorskip("numpy")

    from uhd_gen.train_uhd import _report_out_of_range_predictions

    class _AlwaysPositive:
        @staticmethod
        def predict(features):
            return np.full(len(features), 1.5)

    with caplog.at_level(logging.ERROR):
        count = _report_out_of_range_predictions(_AlwaysPositive(), np.zeros((3, 2)), "tflops")

    assert count == 0
    assert caplog.text == ""
