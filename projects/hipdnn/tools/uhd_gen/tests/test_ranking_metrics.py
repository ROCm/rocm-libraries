# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Metric values are valid as the runtime's `isValidMetricValue` decides them."""
import math

import pytest

from uhd_gen.ranking_metrics import RANKING_METRICS, is_valid_metric_value


@pytest.mark.parametrize("metric", sorted(RANKING_METRICS))
@pytest.mark.parametrize(
    "value,valid",
    [
        (1.5, True),
        (1e-9, True),
        # A zero is no measurement, of a time or a rate: the runtime refuses it, so an
        # engine reporting 0 TFLOPS is not measured.
        (0.0, False),
        (-1.0, False),
        (math.inf, False),
        (math.nan, False),
        (True, False),
        ("1.5", False),
    ],
)
def test_a_value_is_valid_only_when_finite_and_positive(metric, value, valid):
    assert is_valid_metric_value(metric, value) is valid


def test_an_unregistered_metric_is_an_error():
    with pytest.raises(ValueError, match="unregistered ranking metric 'latency'"):
        is_valid_metric_value("latency", 1.0)
