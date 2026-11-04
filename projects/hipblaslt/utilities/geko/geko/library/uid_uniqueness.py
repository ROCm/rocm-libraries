# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assign and retry SolutionUIDs for logic files written by one Geko run."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol


def _load_solution_id_gen() -> ModuleType:
    """Import the SolutionUID codec from the installed ``tensilelite`` package.

    A missing codec is an error: keeping a copied UID would produce a
    duplicate that only fails later, when the merged library is built.

    Args:
        None.

    Returns:
        The ``tensilelite.SolutionIdGen`` module.

    Raises:
        ImportError: If ``tensilelite`` is not installed.
    """
    try:
        from tensilelite import SolutionIdGen
    except ImportError as e:
        raise ImportError(
            "SolutionUID assignment needs the tensilelite package. Install it "
            f"from projects/hipblaslt/tensilelite. Error: {e}"
        ) from e
    return SolutionIdGen


class _SolutionLibrary(Protocol):
    """A library whose solutions are mutable dictionaries."""

    solutions: list[dict]


def assign_fresh_solution_uids(libraries: list[_SolutionLibrary]) -> int:
    """Give every solution a new UID before writing ``full_libs`` or ``final_libs``.

    The source UID is dropped, including a UID copied from an Origami solution.
    One draw is shared across the libraries so the written directory has no
    duplicate. ``TensileMergeLibrary`` still matches ``SolutionNameMin``: an
    existing Equality solution keeps its UID, and a new name keeps this one.

    Args:
        libraries: Libraries updated in place. Each solution dict loses
            ``SolutionUID`` and receives a new canonical value.

    Returns:
        Number of solutions assigned a new UID.

    Raises:
        ImportError: If ``tensilelite`` is not installed.
        RuntimeError: If a unique SolutionUID cannot be drawn.
        ValueError: If UID generation overflows the 40-bit timestamp.
    """
    solutions = [solution for library in libraries for solution in library.solutions]
    if not solutions:
        return 0
    solution_id_gen = _load_solution_id_gen()
    seen: set[str] = set()
    for solution in solutions:
        solution.pop("SolutionUID", None)
        solution_id_gen.claim_solution_uid(solution, seen)
    return len(solutions)


def uniquify_generated_logic(paths: list[Path]) -> int:
    """Make SolutionUIDs unique across logic files from one Geko run.

    Args:
        paths: ``3_LibraryLogic`` YAML files written by the Tensile jobs in
            this run. Other architectures are not read.

    Returns:
        Number of UID lines rewritten.

    Raises:
        ImportError: If ``tensilelite`` is not installed.
        RuntimeError: If a replacement UID cannot be drawn.
        OSError: If a logic file cannot be read or written.
    """
    if not paths:
        return 0
    return _load_solution_id_gen().rewrite_colliding_solution_uids(list(paths))
