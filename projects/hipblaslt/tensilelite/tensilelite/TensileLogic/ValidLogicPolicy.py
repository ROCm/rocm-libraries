# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tuning-policy checks for one logic library, run only when requested.

``filename_errors`` requires a library-type marker in the file name to agree
with ``LibraryType``. ``performance_errors`` requires every ``ExactLogic``
performance value to be 0. ``scripts/run_tensile_logic_check.py
--policy-checks`` runs them on the files a change touched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

from .ValidLibraryShape import _library_type

_NAME_MARKERS = (("Equality_", "Equality"), ("GridBased_", "GridBased"), ("FreeSize_", "FreeSize"))


def _field(data: Any, key: str, position: int) -> Any:
    """Return one header field from either logic format.

    Args:
        data: Parsed logic document.
        key: Dict-format key.
        position: List-format index.

    Returns:
        The value, or None when the document does not carry it.

    Raises:
        None.
    """
    if isinstance(data, dict):
        return data.get(key)
    if isinstance(data, list) and len(data) > position:
        return data[position]
    return None


def filename_errors(data: Any, path: Path) -> list[str]:
    """Require a library-type marker in the file name to match ``LibraryType``.

    Args:
        data: Parsed logic document.
        path: Logic file path.

    Returns:
        Error messages.

    Raises:
        None.
    """
    errors = []
    library_type = _library_type(data)
    for marker, marker_type in _NAME_MARKERS:
        if marker in path.name and library_type != marker_type:
            errors.append(f"File name says {marker_type} but LibraryType is {library_type}.")
    if "Origami" in path.name and library_type != "Prediction":
        errors.append(f"File name says Origami but LibraryType is {library_type}.")
    return errors


def _performance_entries(data: Any) -> Iterator[tuple[Any, float]]:
    """Yield each ``ExactLogic`` size with its numeric performance value.

    Args:
        data: Parsed logic document.

    Yields:
        ``(size, performance)`` pairs.

    Raises:
        None.
    """
    for entry in _field(data, "ExactLogic", 7) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        mapping = entry[1]
        if isinstance(mapping, (list, tuple)) and len(mapping) > 1:
            if isinstance(mapping[1], (int, float)) and not isinstance(mapping[1], bool):
                yield entry[0], float(mapping[1])


def performance_errors(data: Any) -> list[str]:
    """Require every ``ExactLogic`` performance value to be 0.

    Args:
        data: Parsed logic document.

    Returns:
        One message per size whose value is not 0.

    Raises:
        None.
    """
    return [
        f"ExactLogic size {size}: performance {value!r} is not 0."
        for size, value in _performance_entries(data)
        if math.isnan(value) or value != 0.0
    ]


def policy_errors(data: Any, path: Path) -> list[str]:
    """Run every tuning-policy check on one revision of one file.

    Args:
        data: Parsed logic document.
        path: Logic file path.

    Returns:
        Error messages.

    Raises:
        None.
    """
    return filename_errors(data, path) + performance_errors(data)
