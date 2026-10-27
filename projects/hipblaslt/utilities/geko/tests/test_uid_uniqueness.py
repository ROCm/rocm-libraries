# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests that Geko loads SolutionIdGen from the hipBLASLt checkout path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geko.library.uid_uniqueness import (
    assign_fresh_solution_uids,
    uniquify_generated_logic,
)


def _write_solution_id_gen(hipblaslt_path: Path, source: str) -> Path:
    """Write a stand-in SolutionIdGen.py under a fake hipBLASLt checkout.

    Args:
        hipblaslt_path: Root treated as the hipBLASLt checkout.
        source: Python source of the stand-in module.

    Returns:
        Path of the written module.

    Raises:
        OSError: If the file cannot be created.
    """
    module_path = (
        hipblaslt_path / "tensilelite" / "tensilelite" / "Common" / "SolutionIdGen.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text(source, encoding="utf-8")
    return module_path


def test_assign_fresh_solution_uids_skips_an_empty_library(tmp_path: Path) -> None:
    """An empty library list does not look for SolutionIdGen.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call tries to load a missing checkout.
    """
    assert assign_fresh_solution_uids([], hipblaslt_path=tmp_path / "missing") == 0


def test_assign_fresh_solution_uids_loads_the_checkout_copy(tmp_path: Path) -> None:
    """The UID draw comes from the checkout path, not from Geko's install path.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in UID is not stored.
    """
    _write_solution_id_gen(
        tmp_path,
        "def claim_solution_uid(solution, seen):\n"
        "    solution['SolutionUID'] = '0uTEST'\n"
        "    seen.add('0uTEST')\n"
        "    return '0uTEST'\n",
    )
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    count = assign_fresh_solution_uids([library], hipblaslt_path=tmp_path)

    assert count == 1
    assert library.solutions[0]["SolutionUID"] == "0uTEST"


def test_assign_fresh_solution_uids_names_a_missing_checkout_file(tmp_path: Path) -> None:
    """A missing SolutionIdGen.py reports the checkout path.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the error does not name the missing file.
    """
    library = SimpleNamespace(solutions=[{}])
    with pytest.raises(FileNotFoundError, match="SolutionIdGen.py not found"):
        assign_fresh_solution_uids([library], hipblaslt_path=tmp_path)


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
    _write_solution_id_gen(tmp_path, "x = 1\n")
    monkeypatch.setattr(
        "geko.library.uid_uniqueness.importlib.util.spec_from_file_location",
        lambda *_args, **_kwargs: SimpleNamespace(loader=None),
    )
    library = SimpleNamespace(solutions=[{}])
    with pytest.raises(ImportError, match="Could not load SolutionIdGen"):
        assign_fresh_solution_uids([library], hipblaslt_path=tmp_path)


def test_uniquify_generated_logic_skips_empty_paths(tmp_path: Path) -> None:
    """No logic files means no SolutionIdGen load.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call tries to load a missing checkout.
    """
    assert uniquify_generated_logic([], hipblaslt_path=tmp_path / "missing") == 0


def test_uniquify_generated_logic_uses_the_checkout_rewriter(tmp_path: Path) -> None:
    """Collision rewriting uses the checkout's SolutionIdGen.py.

    Args:
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the stand-in rewriter is not called.
    """
    hipblaslt_path = tmp_path / "hip"
    _write_solution_id_gen(
        hipblaslt_path,
        "def rewrite_colliding_solution_uids(paths):\n"
        "    return len(paths)\n",
    )
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    rewritten = uniquify_generated_logic([logic], hipblaslt_path=hipblaslt_path)

    assert rewritten == 1
