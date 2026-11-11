# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic tests for dict structure and library-type folder checks."""

from pathlib import Path

import pytest

from tensilelite.TensileLogic.ValidLibraryShape import library_shape_errors

pytestmark = pytest.mark.unit


def test_dict_library_must_carry_its_header_keys() -> None:
    """A dict without ProblemType fails, and a complete dict passes.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the two documents are classified the same way.
    """
    incomplete = {
        "ArchitectureName": "gfx950",
        "Solutions": [],
        "LibraryType": "Equality",
    }
    assert library_shape_errors(incomplete, Path("a.yaml"))
    complete = {
        "ArchitectureName": "gfx950",
        "ProblemType": {"OperationType": "GEMM"},
        "Solutions": [],
        "LibraryType": "Equality",
    }
    assert library_shape_errors(complete, Path("Equality/a.yaml")) == []


def test_prediction_belongs_in_origami_and_types_match_their_folder() -> None:
    """Prediction outside Origami fails. A matching folder passes.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a folder rule is reversed.
    """
    prediction = {
        "ArchitectureName": "gfx950",
        "ProblemType": {"OperationType": "GEMM"},
        "Solutions": [],
        "LibraryType": "Prediction",
    }
    assert library_shape_errors(prediction, Path("Equality/a.yaml"))
    assert library_shape_errors(prediction, Path("Origami/a.yaml")) == []
    grid = dict(prediction, LibraryType="GridBased")
    assert library_shape_errors(grid, Path("Equality/a.yaml"))
    assert library_shape_errors(grid, Path("GridBased/a.yaml")) == []


def test_distance_list_and_matching_types_follow_their_folders() -> None:
    """``Library.distance``, a list type, and ``Matching`` are each placed by folder.

    Returns:
        None.

    Raises:
        AssertionError: If one of the three type sources is misread.
    """
    base = {
        "ArchitectureName": "gfx950",
        "ProblemType": {"OperationType": "GEMM"},
        "Solutions": [],
        "LibraryType": "Equality",
    }
    distance = dict(base, Library={"distance": "GridBased"})
    assert library_shape_errors(distance, Path("GridBased/a.yaml")) == []
    assert library_shape_errors(distance, Path("Equality/a.yaml"))
    listed = [None] * 11 + ["Range"]
    assert library_shape_errors(listed, Path("Range/a.yaml")) == []
    assert library_shape_errors(listed, Path("Equality/a.yaml"))
    matching = dict(base, LibraryType="Matching")
    assert library_shape_errors(matching, Path("Range/a.yaml")) == []
    assert library_shape_errors(matching, Path("FreeSize/a.yaml")) == [
        "Matching library in unexpected folder FreeSize."
    ]


def test_short_list_document_is_not_a_shape_error() -> None:
    """A list that is not a full library is left to the reader that loads it.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the short list is rejected.
    """
    assert library_shape_errors(["schedule"], Path("Equality/a.yaml")) == []
