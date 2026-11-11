# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the opt-in tuning-policy checks on changed logic files."""

from pathlib import Path

import pytest
import yaml

import tensilelite.TensileLogic.ValidLogicChange as change
import tensilelite.TensileLogic.ValidLogicPolicy as policy

pytestmark = pytest.mark.unit

_PROBLEM = {
    "OperationType": "GEMM",
    "DataType": 4,
    "DestDataType": 0,
    "ComputeDataType": 0,
    "HighPrecisionAccumulate": True,
    "TransposeA": False,
    "TransposeB": False,
    "UseBeta": True,
    "Batched": True,
}


def _library(**fields: object) -> dict:
    """Build a dict-format library with an HSS ProblemType.

    Args:
        **fields: Keys that replace or extend the defaults.

    Returns:
        The document.

    Raises:
        None.
    """
    data = {
        "ScheduleName": "gfx950",
        "ArchitectureName": "gfx950",
        "ProblemType": dict(_PROBLEM),
        "Solutions": [],
        "LibraryType": "Equality",
        "ExactLogic": [[[1, 1, 1, 1], [0, 0.0]]],
    }
    data.update(fields)
    return data


def test_library_type_markers_in_the_name_must_agree() -> None:
    """``GridBased_`` and ``Origami`` in a name require the matching type.

    Returns:
        None.

    Raises:
        AssertionError: If a disagreeing marker passes or a plain name fails.
    """
    data = _library()
    assert policy.filename_errors(data, Path("GridBased_x.yaml")) == [
        "File name says GridBased but LibraryType is Equality."
    ]
    assert policy.filename_errors(data, Path("x_Origami.yaml")) == [
        "File name says Origami but LibraryType is Equality."
    ]
    assert policy.filename_errors(data, Path("Equality_x.yaml")) == []
    assert policy.filename_errors(data, Path("gfx950_Cijk_Ailk_Bljk_HHS.yaml")) == []


def test_exact_logic_performance_must_be_zero() -> None:
    """A nonzero or NaN value fails per size. Zero and malformed rows pass.

    Returns:
        None.

    Raises:
        AssertionError: If a value is misclassified.
    """
    rows = [
        [[1, 1, 1, 1], [0, 0.0]],
        [[2, 2, 1, 2], [3, 0.05]],
        [[3, 3, 1, 3], [4, float("nan")]],
        [[4, 4, 1, 4], [5, True]],
        [[5, 5, 1, 5], [6]],
        "malformed",
    ]
    assert policy.performance_errors(_library(ExactLogic=rows)) == [
        "ExactLogic size [2, 2, 1, 2]: performance 0.05 is not 0.",
        "ExactLogic size [3, 3, 1, 3]: performance nan is not 0.",
    ]
    assert policy.performance_errors(_library(ExactLogic=None)) == []
    listed = [{}, "gfx950", "gfx950", [], {}, [], [], rows]
    assert len(policy.performance_errors(listed)) == 2
    assert policy.performance_errors(["short"]) == []


def _changed(repo: Path, name: str, text: str) -> tuple:
    """Write one changed logic file and its change list.

    Args:
        repo: Temporary repository root.
        name: File name.
        text: YAML contents.

    Returns:
        ``(repository-relative path, list file)``.

    Raises:
        OSError: If a file cannot be written.
    """
    rel = f"projects/hipblaslt/library/{change._LOGIC_MARK}gfx950/Equality/{name}"
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)
    listed = repo / "changed.txt"
    listed.write_text(rel + "\n")
    return rel, listed


def _yaml(rows: int, perf: float) -> str:
    """Render a complete dict library whose ExactLogic has ``rows`` sizes.

    Args:
        rows: Number of ExactLogic sizes.
        perf: Performance value of every size.

    Returns:
        YAML text.

    Raises:
        None.
    """
    problem = yaml.safe_dump(_PROBLEM, default_flow_style=True).strip()
    header = (
        "ArchitectureName: gfx950\nScheduleName: gfx950\nLibraryType: Equality\n"
        f"ProblemType: {problem}\nSolutions: []\nDeviceNames: [Device 75a0]\nExactLogic:\n"
    )
    return header + "".join(f"- [[{i}, 1, 1, 1], [0, {perf}]]\n" for i in range(rows))


_NAME = "gfx950_Cijk_Ailk_Bljk_HSS_BH_UserArgs.yaml"


def test_policy_checks_run_only_when_requested(tmp_path: Path) -> None:
    """A new file with nonzero performance passes by default and fails on request.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the default run checks policy or the opt-in run does not.
    """
    rel, listed = _changed(tmp_path, _NAME, _yaml(1, 0.5))
    assert change.check_changed_logic(tmp_path, listed, {}.get) == []
    assert change.check_changed_logic(tmp_path, listed, {}.get, policy=True) == [
        f"{rel}: ExactLogic size [0, 1, 1, 1]: performance 0.5 is not 0."
    ]


def test_policy_reports_only_sizes_the_change_adds(tmp_path: Path) -> None:
    """An edited file fails only for its new nonzero sizes.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If an existing size is reported.
    """
    rel, listed = _changed(tmp_path, _NAME, _yaml(2, 0.5))
    errors = change.check_changed_logic(tmp_path, listed, {rel: _yaml(1, 0.5)}.get, policy=True)
    assert errors == [f"{rel}: ExactLogic size [1, 1, 1, 1]: performance 0.5 is not 0."]


def test_each_file_prints_a_bounded_number_of_errors(tmp_path: Path) -> None:
    """A file with many errors prints the cap and counts the rest.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If the output is not bounded.
    """
    cap = change.MAX_MESSAGES_PER_FILE
    rel, listed = _changed(tmp_path, _NAME, _yaml(cap + 5, 0.5))
    errors = change.check_changed_logic(tmp_path, listed, {}.get, policy=True)
    assert len(errors) == cap + 1
    assert errors[-1] == f"{rel}: 5 more errors not shown."
