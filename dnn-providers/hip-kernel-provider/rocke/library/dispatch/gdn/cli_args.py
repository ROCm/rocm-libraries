# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small torch-free helpers the GDN/KDA decode tools share.

The host driver (``builders``), the sweep, the defaults report and the two
benchmarks (``benchmarks``) all need the same ``--batches`` parser,
``--rotate-mb`` rule, NaN-safe maximum, "fastest timed row" rule and, for a
file they cannot read, the same usage-error text. They live here, below both
layers, so none of those tools imports another one just to reuse a helper and
a rule has one definition.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, Optional


def parse_batches(text: str) -> tuple:
    """``"1,8,64"`` -> ``(1, 8, 64)``: comma-separated positive ints. Anything
    else (a non-integer, an empty item, zero, a negative number, no value) is
    an ``argparse.ArgumentTypeError``; every ``--batches`` flag of the GDN/KDA
    decode tools parses through here."""
    try:
        values = tuple(int(x) for x in str(text).split(","))
    except ValueError:
        values = ()
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive ints, got {text!r}"
        )
    return values


_ROTATE_MB_RULE = "must be >= 0 (0 = warm)"


def check_rotate_mb(
    value: Optional[int],
    parser: Optional[argparse.ArgumentParser] = None,
    name: str = "rotate_mb",
) -> None:
    """The one rule for a cold-memory rotation size: ``None`` (not given) or an
    int ``>= 0`` (``0`` = warm). A violation is ``parser.error`` (``--rotate-mb
    ...``, exit 2) when a parser is given, else a ``ValueError`` naming
    ``name`` and the value."""
    if value is None or value >= 0:
        return
    if parser is not None:
        parser.error(f"--rotate-mb {_ROTATE_MB_RULE}")
    raise ValueError(f"{name} {_ROTATE_MB_RULE}: {value}")


def usage_error_text(exc: Exception) -> str:
    """The text of a ``parser.error`` for a failed input read: ``str(exc)``,
    except an ``OSError`` that names a file reads ``path: reason`` with the path
    as the tool opened it (the argument as given, or the file a glob matched).
    ``str(OSError)`` shows the path through ``repr``, which doubles every
    backslash of a Windows path, so a message built from it no longer contains
    the path the user passed."""
    if isinstance(exc, OSError) and exc.filename is not None:
        return f"{exc.filename}: {exc.strerror or type(exc).__name__}"
    return str(exc)


def nan_safe_max(values: Iterable[float]) -> float:
    """The largest value; NaN when any value is NaN, so a NaN error can never
    hide behind a finite one (``max`` alone depends on argument order)."""
    values = tuple(values)
    if any(math.isnan(v) for v in values):
        return math.nan
    return max(values)


def _timing_key(row: dict) -> tuple:
    return (row["us"], row["index"])


def fastest_first(rows: Iterable[dict], ok: str = "ok") -> list:
    """The rows whose ``status`` is ``ok``, fastest first, ties broken by the
    lower ``index``. Every ordering of sweep ROWS by speed goes through this
    rule; a report that ranks merged per-configuration times, with no row
    index, breaks ties by ``tuning_id`` instead."""
    return sorted((r for r in rows if r["status"] == ok), key=_timing_key)


def fastest_timed(rows: Iterable[dict], ok: str = "ok") -> Optional[dict]:
    """The first of :func:`fastest_first`; ``None`` when no row is timed."""
    return min((r for r in rows if r["status"] == ok), key=_timing_key, default=None)
