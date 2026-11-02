# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which stored measurements training reads: the usable ones, one per configuration per shape,
the best.

The dataset keeps every measurement, failures and repeats included (store.py). Training reads
one row per measured configuration on each graph and kind of device: rows that carry no usable
measurement are
removed, and of the repeats that remain the one with the lowest `avgTimeMs` (the statistic a
calibrated label is defined on, RFC 0019 §13.4) is kept, the newer on a tie. The store writes
this table as `training.parquet` after every change, and `generate` computes it the same way
when it trains on a subset.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

#: The statistic a repeat is judged by, lowest kept.
BEST_STATISTIC = "avgTimeMs"


def measurement_key(row: dict) -> tuple:
    """What a measurement is of: a configuration on a graph, on a kind of device.

    `benchmark` is the graph's content key, so one graph is one key however it was named or
    generated. `device` is the device's identity by its properties (RFC 0019.13 §4.1): the
    hash of its DeviceKey, the same for every GPU of one kind, so a repeat on another GPU of
    the same kind is the same measurement and a different kind of card is another. A collected
    row names its configuration by `kernel` and `knob_settings`; a published row, by its
    `kernel.*` fields. An L1 row carries no configuration, so its key is the graph and device.
    """
    published = tuple(
        (name, str(row[name])) for name in sorted(row) if name.startswith("kernel.")
    )
    return (
        str(row["benchmark"]),
        str(row["arch"]),
        str(row.get("device")),
        str(row.get("kernel")),
        str(row.get("knob_settings")),
        published if row.get("kernel") is None else (),
    )


def _false(value: Any) -> bool:
    return value is False or str(value).strip().lower() in ("false", "0")


def measurement_error(row: dict) -> str | None:
    """Why `row` cannot be a label, or None when it can: checked wrong (§13.2), not measured
    (`is_valid`), not run (`succeeded`), errored, or no finite positive `avgTimeMs`."""
    from ..correctness import VERDICT, numerical_verdict

    if numerical_verdict(row.get(VERDICT)) is False:
        return "numerically_wrong"
    if "is_valid" in row and _false(row["is_valid"]):
        return "invalid"
    if "succeeded" in row and (row["succeeded"] is None or _false(row["succeeded"])):
        return "not_run"
    error = row.get("error")
    if isinstance(error, str) and error.strip():
        return "errored"
    try:
        time = float(row.get(BEST_STATISTIC))
    except (TypeError, ValueError):
        return "unmeasured"
    if not math.isfinite(time) or time <= 0:
        return "unmeasured"
    return None


def best_per_shape(measured: list) -> tuple[list[int], int, dict[str, int]]:
    """Of `measured` -- `(session, row)` oldest first -- the indices training keeps, how many
    usable repeats were set aside, and how many rows were removed for each reason.

    A repeat under another engine binding is refused, not set aside, so a mix cannot hide.
    The binding's `provider_build` is diagnostic: a rebuilt provider's measurement competes.
    """
    from ..immediate import binding_identity

    best: dict = {}
    binding: dict = {}
    excluded: Counter = Counter()
    usable = 0
    for index, (_, row) in enumerate(measured):
        key = measurement_key(row)
        current = row.get("binding")
        if current is not None:
            current = binding_identity(
                json.loads(current) if isinstance(current, str) else current
            )
        if binding.setdefault(key, current) != current:
            raise ValueError(
                f"{key[0]} on {key[1]} was measured under two engine bindings; "
                "the selector or descriptor provenance changed between measurements"
            )
        reason = measurement_error(row)
        if reason is not None:
            excluded[reason] += 1
            continue
        usable += 1
        time = float(row[BEST_STATISTIC])
        held = best.get(key)
        if held is None or time <= held[0]:
            best[key] = (time, index)
    kept = sorted(index for _, index in best.values())
    return kept, usable - len(kept), dict(excluded)
