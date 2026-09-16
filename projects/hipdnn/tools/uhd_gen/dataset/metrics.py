# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deriving `tflops` and `gbs` from what an engine reported and a run measured.

RFC 0019.13 §8.3 makes both metrics derived rather than collected: a producer supplies a time,
and the engine that solved the problem supplies the work, so a corpus from any source can be
given its metrics rather than asked for them.

The work comes from the engine and from nowhere else. §13.6 has an engine log `<root>.flops`
and `<root>.bytes` beside the problem tokens it binds, which is the same engine, the same run
and the same vocabulary as every other column in the row. The alternative -- evaluating a
corpus_gen ``.opmeta.json`` declaration against the row -- was removed: a declaration names its
parameters in corpus_gen's vocabulary, nothing checks that an engine spells them the same (RFC
0020 §6), and the failure is silent in the worst direction, since a resolvable-but-wrong name
yields a plausible throughput rather than an error. It was also wrong in practice on measured
data: the sdpa declaration counts every tensor `heads` wide and so overstated a grouped-query
`bytes` by up to 7.8x, which the engine's own count gets right because it knows what it read.

Separate from the reader and the writer so it can be tested without either: everything here is
arithmetic over a row, which is what makes the failure it guards visible. A wrong count does not
raise, it produces a plausible throughput, and every model trained on it ranks kernels by a
number that is wrong by a constant factor nobody sees.
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

    Absent, blank and non-finite all read as "not reported" rather than as zero: a corpus is
    written with an empty restval, so a column an engine did not fill arrives as `''` or NaN,
    and either turning into a 0.0 would make a real measurement's rate exactly zero -- which
    sorts last and looks like a slow kernel rather than like a missing count.
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

    Null, not zero, in every absent case. A zero throughput sorts *last* and so is merely wrong;
    a zero *time* would divide into an infinity that outranks every real measurement. Keeping
    the absent case null throughout means no arithmetic here can manufacture a winner.

    Each metric is independent of the other, so an engine that reports one cost and not the
    other still gets the metric it can support. `tflops` is null for an operation that reports
    no `flops`. That is not an omission: a FLOP count for layernorm is a convention rather than
    a fact, and those operations are memory-bound anyway, so `gbs` is what ranks them.

    `bytes` rather than an element count and a dtype width: the engine already knows the width
    of every tensor it touched, including the mixed-width case a single `dtype` column cannot
    express.
    """
    absent: dict[str, float | None] = {"tflops": None, "gbs": None}

    # No measurement, or one that cannot yield a rate. A time of zero is impossible rather than
    # fast, so it is treated as no measurement rather than divided by.
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
