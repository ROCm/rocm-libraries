# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for SolutionUID assignment through the installed tensilelite codec."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geko.library.uid_uniqueness import assign_fresh_solution_uids, uniquify_generated_logic
from tests.uid_codec import install_codec, remove_codec


def test_assign_fresh_solution_uids_skips_an_empty_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty library list does not import the codec.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If a UID is reported as assigned.
    """
    remove_codec(monkeypatch)

    assert assign_fresh_solution_uids([]) == 0
    assert assign_fresh_solution_uids([SimpleNamespace(solutions=[])]) == 0


def test_assign_fresh_solution_uids_rejects_a_missing_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tensilelite package fails before any UID is removed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the import error is not raised or the UID changed.
    """
    remove_codec(monkeypatch)
    library = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    with pytest.raises(ImportError, match="needs the tensilelite package"):
        assign_fresh_solution_uids([library])
    assert library.solutions == [{"SolutionUID": "0u5"}]


def test_assign_fresh_solution_uids_uses_the_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every solution loses its source UID and gets one shared-draw UID.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If a source UID survives or two solutions share a UID.
    """
    install_codec(monkeypatch)
    first = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}, {}])
    second = SimpleNamespace(solutions=[{"SolutionUID": "0u5"}])

    assert assign_fresh_solution_uids([first, second]) == 3
    assert [s["SolutionUID"] for s in first.solutions + second.solutions] == [
        "0u1",
        "0u2",
        "0u3",
    ]


def test_uniquify_generated_logic_skips_empty_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """No logic files means no codec import.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        AssertionError: If a rewrite is reported.
    """
    remove_codec(monkeypatch)

    assert uniquify_generated_logic([]) == 0


def test_uniquify_generated_logic_rejects_a_missing_codec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing tensilelite package fails without rewriting the file.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the import error is not raised or the file changed.
    """
    remove_codec(monkeypatch)
    logic = tmp_path / "logic.yaml"
    logic.write_text("SolutionUID: 0u5\nSolutionUID: 0u5\n")

    with pytest.raises(ImportError, match="needs the tensilelite package"):
        uniquify_generated_logic([logic])
    assert logic.read_text() == "SolutionUID: 0u5\nSolutionUID: 0u5\n"


def test_uniquify_generated_logic_uses_the_codec_rewriter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Collision rewriting is delegated to the codec with the given files.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary-directory fixture.

    Returns:
        None.

    Raises:
        AssertionError: If the rewriter is not called with the files.
    """
    calls: list[list[Path]] = []

    def _rewrite(paths: list[Path]) -> int:
        """Record the files passed to the rewriter.

        Args:
            paths: Logic files to rewrite.

        Returns:
            A fixed rewrite count.

        Raises:
            None.
        """
        calls.append(paths)
        return 4

    install_codec(monkeypatch, rewrite=_rewrite)
    logic = tmp_path / "logic.yaml"

    assert uniquify_generated_logic([logic]) == 4
    assert calls == [[logic]]
