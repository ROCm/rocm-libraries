# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ranking-metric registry (RFC 0019 §4.4), mirrored from the data SDK.

Must match `hipdnn_data_sdk/utilities/RankingMetrics.hpp` row for row, or the loader
refuses emitted descriptors. `label` is the corpus column a model of that metric trains
on (RFC 0019 §13.4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingMetric:
    name: str
    units: str
    objective: str
    label: str


DEFAULT_RANKING_METRIC = "tflops"

RANKING_METRICS: dict[str, RankingMetric] = {
    "tflops": RankingMetric("tflops", "TFLOPS", "max", "tflops"),
    "time": RankingMetric("time", "ms", "min", "avgTimeMs"),
}


def ranking_metric(name: str) -> RankingMetric:
    """The registry entry for `name`; an unregistered name is an error, never a guess."""
    try:
        return RANKING_METRICS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unregistered ranking metric {name!r}; registered: {', '.join(RANKING_METRICS)}"
        ) from None


def is_valid_metric_value(name: str, value) -> bool:
    """`isValidMetricValue`: finite and positive, for every metric. A zero is no measurement,
    of a time or a rate."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return False
    ranking_metric(name)
    return value > 0
