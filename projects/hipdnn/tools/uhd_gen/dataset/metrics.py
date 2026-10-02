# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deriving `tflops` and `gbs` from what an engine reported and a run measured.

RFC 0019.13 §8.3: the producer supplies a time; the work comes only from the engine's own
`<root>.flops`/`<root>.bytes` (§13.6), never from a corpus_gen declaration, whose parameter
names nothing checks against the engine's.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = [
    "reported",
    "derive_metrics",
]


def reported(query: Mapping[str, Any], name: str) -> float | None:
    """One engine-reported cost from a row, or None where the engine reported none.

    Blank and non-finite read as "not reported", never 0.0, which would yield a zero rate.
    """
    if name not in query:
        return None
    value = query[name]
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def derive_metrics(
    query: Mapping[str, Any],
    time_ms: float | None,
) -> dict[str, float | None]:
    """`tflops` and `gbs` for one row, or nulls where there is nothing to derive from.

    Null, never zero: a zero time would divide into an infinity that outranks every real
    measurement. Each metric is independent; `tflops` is null when no `flops` is reported.
    """
    absent: dict[str, float | None] = {"tflops": None, "gbs": None}

    # A zero time is impossible rather than fast: treat it as no measurement.
    if time_ms is None or not math.isfinite(time_ms) or time_ms <= 0.0:
        return absent

    seconds = time_ms / 1000.0
    derived: dict[str, float | None] = dict(absent)

    flops = reported(query, "flops")
    if flops is not None:
        derived["tflops"] = flops / seconds / 1e12

    moved = reported(query, "bytes")
    if moved is not None:
        derived["gbs"] = moved / seconds / 1e9

    return derived
