# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""File-level shape checks for one logic library.

``dict_structure`` requires the dict keys TensileLogic reads.
``library_type_folder`` requires ``LibraryType`` to match the parent folder.
``Prediction`` is accepted only in an Origami folder, and an Origami folder
accepts only ``Prediction``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_FOLDERS = {"Equality", "Origami", "GridBased", "Range", "Prediction", "FreeSize"}
_MATCHING_FOLDERS = {"GridBased", "Equality", "Range"}


def _folder(path: Path) -> str:
    """Return the library-type directory in ``path``.

    Args:
        path: Logic file path.

    Returns:
        The first known folder name, or an empty string.

    Raises:
        No exceptions are raised.
    """
    for part in path.parts:
        if part in _FOLDERS:
            return part
    return ""


def _library_type(data: Any) -> str:
    """Return the library type stored in either logic format.

    Args:
        data: Parsed logic document.

    Returns:
        The type string, or an empty string when the document does not carry one.
        A dict with ``Library.distance`` uses that distance as its type.

    Raises:
        No exceptions are raised.
    """
    if isinstance(data, dict):
        library_type = str(data.get("LibraryType", ""))
        library = data.get("Library")
        if isinstance(library, dict) and library.get("distance"):
            return str(library["distance"])
        return library_type
    if isinstance(data, list) and len(data) > 11 and data[11] is not None:
        return str(data[11])
    return ""


def dict_structure_errors(data: Any) -> list[str]:
    """Require the dict keys a logic library is read from.

    Args:
        data: Parsed logic document. List documents are not checked.

    Returns:
        One message listing the missing keys, or an empty list.

    Raises:
        No exceptions are raised.
    """
    if not isinstance(data, dict):
        return []
    missing = [
        key
        for key, present in (
            ("ArchitectureName", bool(data.get("ArchitectureName"))),
            (
                "ProblemType",
                isinstance(data.get("ProblemType"), dict)
                and bool(data.get("ProblemType")),
            ),
            ("Solutions", isinstance(data.get("Solutions"), list)),
            ("LibraryType", bool(_library_type(data))),
        )
        if not present
    ]
    if not missing:
        return []
    return ["Dict library is missing " + ", ".join(missing) + "."]


def library_type_folder_errors(data: Any, path: Path) -> list[str]:
    """Require ``LibraryType`` to match the parent folder.

    Args:
        data: Parsed logic document.
        path: Logic file path. The folder is read from its parts.

    Returns:
        One message when the type and folder disagree, otherwise an empty list.
        A path with no known folder is not checked.

    Raises:
        No exceptions are raised.
    """
    folder = _folder(path)
    if not folder:
        return []
    library_type = _library_type(data)
    if library_type == "Prediction" or folder == "Origami":
        if library_type == "Prediction" and folder == "Origami":
            return []
        return [
            f"Prediction is accepted only in an Origami folder "
            f"(found LibraryType {library_type} in {folder})."
        ]
    if library_type == "Matching":
        if folder in _MATCHING_FOLDERS:
            return []
        return [f"Matching library in unexpected folder {folder}."]
    if library_type and library_type != folder:
        return [f"LibraryType {library_type} does not match folder {folder}."]
    return []


def library_shape_errors(data: Any, path: Path) -> list[str]:
    """Run both file-level shape checks.

    Args:
        data: Parsed logic document.
        path: Logic file path.

    Returns:
        The messages from both checks.

    Raises:
        No exceptions are raised.
    """
    return dict_structure_errors(data) + library_type_folder_errors(data, path)
