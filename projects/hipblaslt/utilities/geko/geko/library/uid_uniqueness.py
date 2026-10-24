# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assign and retry SolutionUIDs for logic files written by one Geko run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Protocol


def _load_solution_id_gen() -> ModuleType:
    """Load SolutionIdGen without importing ``tensilelite.Common``.

    ``tensilelite.Common`` imports rocisa. Geko's uniqueness retry only needs
    the UID codec and the line rewriter.

    Args:
        None.

    Returns:
        The loaded ``SolutionIdGen`` module.

    Raises:
        FileNotFoundError: If ``SolutionIdGen.py`` is missing.
        ImportError: If the module spec or loader cannot be created.
    """
    module_path = (
        Path(__file__).resolve().parents[4]
        / "tensilelite"
        / "tensilelite"
        / "Common"
        / "SolutionIdGen.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"SolutionIdGen.py not found: {module_path}")
    spec = importlib.util.spec_from_file_location("geko_solution_id_gen", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SolutionIdGen from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        FileNotFoundError: If ``SolutionIdGen.py`` cannot be located.
        ImportError: If ``SolutionIdGen.py`` cannot be loaded.
        RuntimeError: If a unique SolutionUID cannot be drawn.
        ValueError: If UID generation overflows the 40-bit timestamp.
    """
    solutions = [solution for library in libraries for solution in library.solutions]
    if not solutions:
        return 0
    claim_solution_uid = _load_solution_id_gen().claim_solution_uid
    seen: set[str] = set()
    for solution in solutions:
        solution.pop("SolutionUID", None)
        claim_solution_uid(solution, seen)
    return len(solutions)


def uniquify_generated_logic(paths: list[Path]) -> int:
    """Make SolutionUIDs unique across logic files from one Geko run.

    Args:
        paths: ``3_LibraryLogic`` YAML files written by the Tensile jobs in
            this generation. Other architectures are not read.

    Returns:
        Number of UID lines rewritten.

    Raises:
        FileNotFoundError: If ``SolutionIdGen.py`` cannot be located.
        RuntimeError: If a replacement UID cannot be drawn.
        OSError: If a logic file cannot be read or written.
    """
    if not paths:
        return 0
    solution_id_gen = _load_solution_id_gen()
    return solution_id_gen.rewrite_colliding_solution_uids(list(paths))
