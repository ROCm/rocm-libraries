# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for loading SolutionIdGen from the hipBLASLt checkout used by Geko."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geko.library.uid_uniqueness import (
    assign_fresh_solution_uids,
    uniquify_generated_logic,
)


def _write_solution_id_gen(directory: Path, source: str) -> Path:
    """Write a stand-in SolutionIdGen.py.

    Args:
        directory: Directory that will contain the stand-in module.
        source: Python source of the stand-in module.

    Returns:
        Path of the written module.

    Raises:
        OSError: If the file cannot be created.
    """
    module_path = directory / "SolutionIdGen.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(source, encoding="utf-8")
    return module_path


def test_assign_fresh_solution_uids_skips_an_empty_library(tmp_path: Path) -> None:
    """An empty library list does not look for SolutionIdGen.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call loads the codec or returns a count.
    """
    assert assign_fresh_solution_uids([], hipblaslt_path=tmp_path / "missing") == 0


def test_assign_fresh_solution_uids_rejects_a_missing_codec(tmp_path: Path) -> None:
    """A checkout without SolutionIdGen.py fails before a copied UID is written.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the missing codec is not reported.
    """
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        assign_fresh_solution_uids([library], hipblaslt_path=tmp_path / "missing")

    assert library.solutions[0]["SolutionUID"] == "0u5"


def test_assign_fresh_solution_uids_uses_the_given_checkout(tmp_path: Path) -> None:
    """An explicit hipBLASLt checkout supplies the codec.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the codec from that checkout is not used.
    """
    _write_solution_id_gen(
        tmp_path / "tensilelite" / "tensilelite" / "Common",
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uCHECKOUT'\n"
        "    seen.add('0uCHECKOUT')\n"
        "    return '0uCHECKOUT'\n",
    )
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    assert assign_fresh_solution_uids([library], hipblaslt_path=tmp_path) == 1
    assert library.solutions[0]["SolutionUID"] == "0uCHECKOUT"


def test_assign_fresh_solution_uids_uses_the_loaded_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present SolutionIdGen.py supplies the UID draw.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in UID is not stored.
    """
    module_path = _write_solution_id_gen(
        tmp_path,
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uTEST'\n"
        "    seen.add('0uTEST')\n"
        "    return '0uTEST'\n",
    )
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    count = assign_fresh_solution_uids([library])

    assert count == 1
    assert library.solutions[0]["SolutionUID"] == "0uTEST"


def test_load_rejects_a_module_without_a_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module spec without a loader is an import error.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the missing loader is not reported.
    """
    module_path = _write_solution_id_gen(tmp_path, "x = 1\n")
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    monkeypatch.setattr(
        "geko.library.uid_uniqueness.importlib.util.spec_from_file_location",
        lambda *_args, **_kwargs: SimpleNamespace(loader=None),
    )
    library = SimpleNamespace(solutions=[{}])
    with pytest.raises(ImportError, match="Could not load SolutionIdGen"):
        assign_fresh_solution_uids([library])


def test_uniquify_generated_logic_skips_empty_paths() -> None:
    """No logic files means no SolutionIdGen load.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call returns a rewrite count.
    """
    assert uniquify_generated_logic([]) == 0


def test_uniquify_generated_logic_rejects_a_missing_codec(tmp_path: Path) -> None:
    """A checkout without SolutionIdGen.py fails without rewriting logic files.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the logic file changes or no error is raised.
    """
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        uniquify_generated_logic([logic], hipblaslt_path=tmp_path / "missing")

    assert logic.read_text(encoding="utf-8") == "SolutionUID: 0u5\n"


def test_uniquify_generated_logic_uses_the_loaded_rewriter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collision rewriting uses the loaded SolutionIdGen.py.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in rewriter is not called.
    """
    module_path = _write_solution_id_gen(
        tmp_path,
        "def rewrite_colliding_solution_uids(paths):\n" "    return len(paths)\n",
    )
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda *_: module_path)
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    rewritten = uniquify_generated_logic([logic])

    assert rewritten == 1
