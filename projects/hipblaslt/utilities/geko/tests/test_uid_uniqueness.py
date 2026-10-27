# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests that a missing SolutionIdGen warns and leaves UIDs unchanged."""

from __future__ import annotations

import logging
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


def test_assign_fresh_solution_uids_skips_an_empty_library(caplog: pytest.LogCaptureFixture) -> None:
    """An empty library list does not look for SolutionIdGen.

    Args:
        caplog: Pytest log-capture fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the empty call warns or returns a count.
    """
    with caplog.at_level(logging.WARNING, logger="GEKO"):
        assert assign_fresh_solution_uids([]) == 0
    assert "SolutionIdGen.py not found" not in caplog.text


def test_assign_fresh_solution_uids_warns_and_keeps_the_source_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing codec warns and does not clear a copied UID.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        caplog: Pytest log-capture fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the UID changes or the warning omits the editable install.
    """
    missing = tmp_path / "missing" / "SolutionIdGen.py"
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda: missing)
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    with caplog.at_level(logging.WARNING, logger="GEKO"):
        count = assign_fresh_solution_uids([library])

    assert count == 0
    assert library.solutions[0]["SolutionUID"] == "0u5"
    assert "pip install -e ." in caplog.text
    assert str(missing) in caplog.text


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
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda: module_path)
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
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda: module_path)
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


def test_uniquify_generated_logic_warns_and_leaves_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing codec warns and does not rewrite logic files.

    Args:
        tmp_path: Pytest temporary-directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
        caplog: Pytest log-capture fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the logic file changes.
    """
    monkeypatch.setattr(
        "geko.library.uid_uniqueness._solution_id_gen_path",
        lambda: tmp_path / "missing" / "SolutionIdGen.py",
    )
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="GEKO"):
        rewritten = uniquify_generated_logic([logic])

    assert rewritten == 0
    assert logic.read_text(encoding="utf-8") == "SolutionUID: 0u5\n"
    assert "pip install -e ." in caplog.text


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
    monkeypatch.setattr("geko.library.uid_uniqueness._solution_id_gen_path", lambda: module_path)
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\n", encoding="utf-8")

    rewritten = uniquify_generated_logic([logic])

    assert rewritten == 1
