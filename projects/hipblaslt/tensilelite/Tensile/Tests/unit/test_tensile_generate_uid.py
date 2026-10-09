# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`Tensile.TensileGenerateUID`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from Tensile import LibraryIO
from Tensile.Common.SolutionIdGen import (
    RANDOM_MASK,
    RANDOM_SHIFT,
    decode_solution_id,
    decode_solution_uid,
    encode_solution_uid,
)
from Tensile.TensileGenerateUID import (
    find_solution_by_index,
    main,
    regenerate_uid_for_solution,
)

pytestmark = pytest.mark.unit


def _sample_logic_data() -> dict:
    """Build minimal dict-format logic YAML content for tests.

    Returns:
        Dict-format library logic data with two solutions.
    """
    return {
        "MinimumRequiredVersion": "0.0.0",
        "ProblemType": {"OperationType": "GEMM"},
        "Solutions": [
            {
                "SolutionIndex": 0,
                "SolutionUID": "0u1c",
                "SolutionNameMin": "sol0",
                "KernelNameMin": "kern0",
                "NumThreads": 256,
            },
            {
                "SolutionIndex": 1,
                "SolutionUID": "0u3E",
                "SolutionNameMin": "sol1",
                "KernelNameMin": "kern1",
                "NumThreads": 128,
            },
        ],
    }


def test_find_solution_by_index_returns_matching_solution() -> None:
    """``find_solution_by_index`` locates the requested solution."""
    data = _sample_logic_data()
    solution = find_solution_by_index(data, 1)
    assert solution["SolutionUID"] == "0u3E"


def test_find_solution_by_index_raises_when_missing() -> None:
    """Missing ``SolutionIndex`` values raise ``ValueError``."""
    with pytest.raises(ValueError, match="SolutionIndex=99"):
        find_solution_by_index(_sample_logic_data(), 99)


def test_find_solution_by_index_raises_when_solutions_missing() -> None:
    """Logic without a ``Solutions`` list raises ``ValueError``.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a non-list ``Solutions`` value is accepted.
    """
    with pytest.raises(ValueError, match="Solutions list"):
        find_solution_by_index({"ProblemType": {}}, 0)
    with pytest.raises(ValueError, match="Solutions list"):
        find_solution_by_index({"Solutions": {"0": {}}}, 0)


def test_find_solution_by_index_skips_non_dict_and_missing_index() -> None:
    """Non-dict entries and missing ``SolutionIndex`` keys are skipped.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a later matching dict entry is not returned.
    """
    data = {
        "Solutions": [
            "not-a-solution",
            {"SolutionNameMin": "no-index"},
            {"SolutionIndex": 3, "SolutionUID": "0u4q"},
        ]
    }
    solution = find_solution_by_index(data, 3)
    assert solution["SolutionUID"] == "0u4q"


def test_regenerate_uid_for_solution_replaces_uid(tmp_path: Path) -> None:
    """Regeneration updates only the targeted solution UID."""
    yaml_path = tmp_path / "logic.yaml"
    data = _sample_logic_data()
    LibraryIO.writeYAML(str(yaml_path), data)

    new_uid = regenerate_uid_for_solution(yaml_path, 1, inplace=True)
    assert new_uid != 200

    updated = LibraryIO.readYAML(str(yaml_path))
    assert updated["Solutions"][0]["SolutionUID"] == "0u1c"
    assert updated["Solutions"][1]["SolutionUID"] == encode_solution_uid(new_uid)


def test_regenerate_uid_for_solution_without_inplace_does_not_write(
    tmp_path: Path,
) -> None:
    """Without ``--inplace`` the YAML file on disk is unchanged."""
    yaml_path = tmp_path / "logic.yaml"
    data = _sample_logic_data()
    LibraryIO.writeYAML(str(yaml_path), data)

    new_uid = regenerate_uid_for_solution(yaml_path, 0, inplace=False)
    assert new_uid != 100

    unchanged = LibraryIO.readYAML(str(yaml_path))
    assert unchanged["Solutions"][0]["SolutionUID"] == "0u1c"


def test_regenerate_uid_uses_40_plus_24_layout(tmp_path: Path) -> None:
    """New UIDs follow the 40+24 bit layout."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    new_uid = regenerate_uid_for_solution(yaml_path, 0, inplace=False)
    ms_part, random_part = decode_solution_id(new_uid)
    assert ms_part >= 0
    assert 0 <= random_part <= RANDOM_MASK
    assert new_uid == (ms_part << RANDOM_SHIFT) | random_part


def test_main_prints_uid_and_returns_zero(capsys, tmp_path: Path) -> None:
    """CLI main prints the new UID and exits successfully."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    rc = main([str(yaml_path), "--index", "0"])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.strip().startswith("0u")
    assert decode_solution_uid(captured.out.strip()) != 100


def test_main_inplace_updates_file(tmp_path: Path) -> None:
    """CLI ``--inplace`` persists the regenerated UID."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    rc = main([str(yaml_path), "--index", "1", "--inplace"])
    assert rc == 0

    updated = LibraryIO.readYAML(str(yaml_path))
    assert updated["Solutions"][1]["SolutionUID"] != "0u3E"


def test_main_returns_error_for_missing_index(tmp_path: Path) -> None:
    """CLI returns exit code 1 when the solution index is not found."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    rc = main([str(yaml_path), "--index", "42"])
    assert rc == 1


def test_regenerate_uid_for_solution_raises_for_missing_file(tmp_path: Path) -> None:
    """A missing YAML path raises ``FileNotFoundError``.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If a missing file is treated as success.
    """
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError, match="Logic YAML not found"):
        regenerate_uid_for_solution(missing, 0)


def test_regenerate_uid_for_solution_rejects_non_dict_yaml(tmp_path: Path) -> None:
    """List-format YAML is rejected as not dict-format logic.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If non-dict YAML is accepted.
    """
    yaml_path = tmp_path / "list.yaml"
    yaml_path.write_text("- not\n- a\n- dict\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dict-format logic YAML"):
        regenerate_uid_for_solution(yaml_path, 0)


def test_main_returns_error_for_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI reports ``FileNotFoundError`` on stderr and returns 1.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        capsys: Pytest stdout/stderr capture fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the CLI does not fail for a missing path.
    """
    missing = tmp_path / "missing.yaml"
    rc = main([str(missing), "--index", "0"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Error:" in captured.err
    assert "Logic YAML not found" in captured.err


def test_module_main_guard_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``python -m Tensile.TensileGenerateUID`` runs ``main`` via ``__main__``.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the module entry point does not exit 0.
    """
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())
    monkeypatch.setattr(
        sys,
        "argv",
        ["TensileGenerateUID.py", str(yaml_path), "--index", "0"],
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("Tensile.TensileGenerateUID", run_name="__main__")
    assert excinfo.value.code == 0
