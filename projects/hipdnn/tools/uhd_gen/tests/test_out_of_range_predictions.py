# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Training reports a model that predicts a target its units cannot take."""
from __future__ import annotations

import logging

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("lightgbm")

from uhd_gen.train_uhd import _report_out_of_range_predictions  # noqa: E402


def test_negative_prediction_is_reported(caplog):
    """A model predicting a target its units cannot take must say so at training time.

    The runtime bounds this regardless, but by then the only recourse is to discard the score.
    Training is where it can still be fixed, and a model doing it on its own training data will
    do it worse in the field.
    """

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

    class _AlwaysPositive:
        @staticmethod
        def predict(features):
            return np.full(len(features), 1.5)

    with caplog.at_level(logging.ERROR):
        count = _report_out_of_range_predictions(_AlwaysPositive(), np.zeros((3, 2)), "tflops")

    assert count == 0
    assert caplog.text == ""
