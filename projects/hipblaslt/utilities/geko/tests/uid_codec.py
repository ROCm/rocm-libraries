# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Install a stand-in ``tensilelite.SolutionIdGen`` for Geko unit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest


def install_codec(
    monkeypatch: pytest.MonkeyPatch,
    rewrite: Callable[[list[Path]], int] | None = None,
) -> ModuleType:
    """Make ``from tensilelite import SolutionIdGen`` return a stand-in module.

    ``claim_solution_uid`` draws ``0u1``, ``0u2``, and so on.

    Args:
        monkeypatch: Pytest monkeypatch fixture. The modules are removed when
            the test ends.
        rewrite: ``rewrite_colliding_solution_uids`` replacement. Defaults to
            returning 0.

    Returns:
        The installed stand-in ``SolutionIdGen`` module.

    Raises:
        None.
    """
    draws = iter(range(1, 1_000_000))

    def _claim(solution: dict, seen: set[str]) -> str:
        """Store the next drawn UID on ``solution``.

        Args:
            solution: Solution updated in place.
            seen: UIDs already used in this draw.

        Returns:
            The stored UID.

        Raises:
            None.
        """
        uid = f"0u{next(draws)}"
        solution["SolutionUID"] = uid
        seen.add(uid)
        return uid

    codec = ModuleType("tensilelite.SolutionIdGen")
    codec.claim_solution_uid = _claim
    codec.rewrite_colliding_solution_uids = rewrite or (lambda _paths: 0)
    package = ModuleType("tensilelite")
    package.SolutionIdGen = codec
    monkeypatch.setitem(sys.modules, "tensilelite", package)
    monkeypatch.setitem(sys.modules, "tensilelite.SolutionIdGen", codec)
    return codec


def remove_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from tensilelite import SolutionIdGen`` raise ``ImportError``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        None.
    """
    monkeypatch.setitem(sys.modules, "tensilelite", None)
    monkeypatch.setitem(sys.modules, "tensilelite.SolutionIdGen", None)
