# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A measured row's numerical-correctness verdict, read the same way at every entrance.

Tri-state and separate from `is_valid` (RFC 0019 §13.2): True checked-correct, False
checked-wrong (timings suppressed), None undecided. Every entrance must use this reader,
or a wrong-but-fast kernel becomes a label. Pandas only: `uhd_gen.dataset` imports it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: The row's verdict column: True, False or null.
VERDICT = "numerically_valid"
#: The reason the verdict was reached (`output_mismatch: ...`, `no_reference: ...`).
REASON = "validation"
#: Timings a known-wrong row must not carry; every derived label comes from these.
SUPPRESSED_TIMINGS = ("robustMeanMs", "minTimeMs", "avgTimeMs", "stddevMs")
#: Labels derived from the timings above, cleared with them when present.
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

    Accepts bool/NaN/pd.NA and CSV text spellings; never coerces a number.
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
    """Rows a correctness check showed wrong; none if the column is absent (undecided)."""
    if VERDICT not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return (
        frame[VERDICT].map(lambda value: numerical_verdict(value) is False).astype(bool)
    )


def suppress_timings(row: dict) -> dict:
    """Null every timing and derived label the known-wrong row carries; keep the row."""
    for column in (*SUPPRESSED_TIMINGS, *DERIVED_LABELS):
        if column in row:
            row[column] = None
    return row
