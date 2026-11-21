# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic tests for dict structure and library-type folder checks."""

from pathlib import Path

import pytest

from tensilelite.TensileLogic.ValidLibraryShape import (
    library_shape_errors,
    solution_index_errors,
)

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


def _indexed(indices: list, rows: list) -> dict:
    """Build a dict library with the given solution indices and ExactLogic rows.

    Args:
        indices: One ``SolutionIndex`` per solution. None leaves the key out.
        rows: ``ExactLogic`` rows.

    Returns:
        The document.

    Raises:
        None.
    """
    solutions = [{} if index is None else {"SolutionIndex": index} for index in indices]
    return {"Solutions": solutions, "ExactLogic": rows}


def test_repeated_solution_index_is_reported_once_per_index() -> None:
    """Each repeated index gives one message with its count.

    Returns:
        None.

    Raises:
        AssertionError: If a repeat is missed or reported per solution.
    """
    data = _indexed([0, 1, 1, 2, 2, 2], [])
    assert solution_index_errors(data) == [
        "SolutionIndex 1 is used by 2 solutions.",
        "SolutionIndex 2 is used by 3 solutions.",
    ]
    assert solution_index_errors(_indexed([0, 1, 2], [])) == []


def test_exact_logic_must_name_an_existing_solution_index() -> None:
    """A row whose index no solution has fails once per missing index.

    Returns:
        None.

    Raises:
        AssertionError: If a dangling row passes or a valid row fails.
    """
    rows = [[[1, 1, 1, 1], [0, 1.0]], [[2, 2, 1, 2], [7, 1.0]], [[3, 3, 1, 3], [7, 1.0]]]
    assert solution_index_errors(_indexed([0, 1], rows)) == [
        "ExactLogic refers to SolutionIndex 7, which no solution has."
    ]
    malformed = [[[1, 1, 1, 1]], [[1, 1, 1, 1], []], "row", [[1, 1, 1, 1], [[0], 1.0]]]
    assert solution_index_errors(_indexed([0], malformed)) == []


def test_rows_are_not_resolved_without_an_index_on_every_solution() -> None:
    """A solution without an index, or no solutions, skips the row check.

    Returns:
        None.

    Raises:
        AssertionError: If rows are resolved against an incomplete index set.
    """
    rows = [[[1, 1, 1, 1], [5, 1.0]]]
    assert solution_index_errors(_indexed([0, None], rows)) == []
    assert solution_index_errors(_indexed([], rows)) == []


def test_list_format_reads_solutions_and_exact_logic_by_position() -> None:
    """A list document reads solutions at 5 and ExactLogic at 7.

    Returns:
        None.

    Raises:
        AssertionError: If the list positions are misread.
    """
    listed = [None, None, None, None, None, [{"SolutionIndex": 0}, {"SolutionIndex": 0}]]
    listed += [None, [[[1, 1, 1, 1], [3, 1.0]]]]
    assert solution_index_errors(listed) == [
        "SolutionIndex 0 is used by 2 solutions.",
        "ExactLogic refers to SolutionIndex 3, which no solution has.",
    ]
    assert solution_index_errors("text") == []
