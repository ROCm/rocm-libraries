# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A measured row's numerical-correctness verdict, read the same way at every entrance.

RFC 0019 §13.2: "A timing is only a training label once the candidate is known correct",
and a candidate shown wrong "is written with its measurement suppressed and an explicit
invalid marker". The verdict is tri-state and sits beside `is_valid`, never folded into it:
`is_valid` says a measurement came back, `numerically_valid` says whether anything showed
the result to be wrong. True is checked-correct, False is checked-wrong, and None (null) is
"nothing could decide" -- the honest state of every corpus collected without a reference,
so it is admissible but never spelled as True. Its reason travels beside it as `validation`.

One reader for every entrance -- `generate`'s collection and label gate, direct `train`,
dataset publication, L1 immediate import, evaluation and promotion -- because a wrong-but-
fast kernel holds the best time in its group: any entrance that reads its timing as a label
trains the selector to prefer it.

Stdlib and pandas only: `uhd_gen.dataset` imports this without the training stack.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: The row's verdict column: True, False or null.
VERDICT = "numerically_valid"
#: The reason the verdict was reached (`output_mismatch: ...`, `no_reference: ...`).
REASON = "validation"
#: The measured timings a known-wrong row must not carry. Everything a label is derived
#: from (`tflops`, `gbs`) is derived from `avgTimeMs`, so suppressing these suppresses it.
SUPPRESSED_TIMINGS = ("robustMeanMs", "minTimeMs", "avgTimeMs", "stddevMs")
#: Labels derived from the timings above, cleared with them where a row already carries one.
DERIVED_LABELS = ("tflops", "gbs")

_SPELLINGS = {
    "true": True,
    "false": False,
    "": None,
    "none": None,
    "null": None,
    "nan": None,
}


def numerical_verdict(value) -> bool | None:
    """The tri-state verdict a row carries; ValueError for anything that is not one.

    Accepts the spellings a corpus actually arrives in: JSON booleans/null, NumPy booleans
    and NaN from a pandas frame, `pd.NA` from Parquet, and the text a CSV column holds.
    Never coerces a number: `1` is not a correctness verdict.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in _SPELLINGS:
        return _SPELLINGS[value.strip().lower()]
    raise ValueError(f"{VERDICT} must be true, false or null; got {value!r}")


def numerical_reason(value) -> str | None:
    """The verdict's reason as text, or None when the row records none."""
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    ):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{REASON} must be text; got {value!r}")
    return value.strip() or None


def known_wrong(frame: pd.DataFrame) -> pd.Series:
    """Rows a correctness check showed to compute the wrong answer.

    A corpus without the column was collected before there was a check; every row is then
    undecided, not wrong.
    """
    if VERDICT not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return (
        frame[VERDICT].map(lambda value: numerical_verdict(value) is False).astype(bool)
    )


def suppress_timings(row: dict) -> dict:
    """Null every timing and derived label on a known-wrong row; the row itself stays.

    Only the columns the row carries are touched, so a row that never had a derived rate
    is not given an explicit null for one.
    """
    for column in (*SUPPRESSED_TIMINGS, *DERIVED_LABELS):
        if column in row:
            row[column] = None
    return row
