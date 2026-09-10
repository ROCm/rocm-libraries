# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for :mod:`Tensile.TensileGenerateUID`."""

from __future__ import annotations

from pathlib import Path

import pytest

from Tensile import LibraryIO
from Tensile.Common.SolutionIdGen import decode_solution_id, RANDOM_MASK, RANDOM_SHIFT
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
                "SolutionUID": 100,
                "SolutionNameMin": "sol0",
                "KernelNameMin": "kern0",
                "NumThreads": 256,
            },
            {
                "SolutionIndex": 1,
                "SolutionUID": 200,
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
    assert solution["SolutionUID"] == 200


def test_find_solution_by_index_raises_when_missing() -> None:
    """Missing ``SolutionIndex`` values raise ``ValueError``."""
    with pytest.raises(ValueError, match="SolutionIndex=99"):
        find_solution_by_index(_sample_logic_data(), 99)


def test_regenerate_uid_for_solution_replaces_uid(tmp_path: Path) -> None:
    """Regeneration updates only the targeted solution UID."""
    yaml_path = tmp_path / "logic.yaml"
    data = _sample_logic_data()
    LibraryIO.writeYAML(str(yaml_path), data)

    new_uid = regenerate_uid_for_solution(yaml_path, 1, inplace=True)
    assert new_uid != 200

    updated = LibraryIO.readYAML(str(yaml_path))
    assert updated["Solutions"][0]["SolutionUID"] == 100
    assert updated["Solutions"][1]["SolutionUID"] == new_uid


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
    assert unchanged["Solutions"][0]["SolutionUID"] == 100


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
    assert captured.out.strip().isdigit()
    assert int(captured.out.strip()) != 100


def test_main_inplace_updates_file(tmp_path: Path) -> None:
    """CLI ``--inplace`` persists the regenerated UID."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    rc = main([str(yaml_path), "--index", "1", "--inplace"])
    assert rc == 0

    updated = LibraryIO.readYAML(str(yaml_path))
    assert updated["Solutions"][1]["SolutionUID"] != 200


def test_main_returns_error_for_missing_index(tmp_path: Path) -> None:
    """CLI returns exit code 1 when the solution index is not found."""
    yaml_path = tmp_path / "logic.yaml"
    LibraryIO.writeYAML(str(yaml_path), _sample_logic_data())

    rc = main([str(yaml_path), "--index", "42"])
    assert rc == 1
