# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ranking-metric registry (RFC 0019 §4.4), mirrored from the data SDK.

`hipdnn_data_sdk/utilities/RankingMetrics.hpp` is the registry the runtime enforces: a
UHD's `score.metric` must name an entry, its `objective` must be that entry's direction,
and a prediction value must be valid in that metric. This copy exists so a descriptor
the tool emits is one the loader accepts; it must match the C++ table row for row.

`label` is the tool's own column: the corpus field a model of this metric trains on.
RFC 0019 §13.4 fixes it per metric -- `tflops` is the binding layer's FLOPs over
`avgTimeMs`, `time` is `avgTimeMs` itself -- so an author names the metric and never
pairs a label with the wrong direction.
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
    #: Zero is the worst throughput but an impossible time, so validity is per metric.
    strictly_positive: bool


DEFAULT_RANKING_METRIC = "tflops"

RANKING_METRICS: dict[str, RankingMetric] = {
    "tflops": RankingMetric("tflops", "TFLOPS", "max", "tflops", strictly_positive=False),
    "time": RankingMetric("time", "ms", "min", "avgTimeMs", strictly_positive=True),
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
    """`isValidMetricValue`: finite, and nonnegative throughput or positive time."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value > 0 if ranking_metric(name).strictly_positive else value >= 0
