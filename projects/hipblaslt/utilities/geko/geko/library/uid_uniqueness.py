# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Assign and retry SolutionUIDs for logic files written by one Geko run."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from typing import Protocol

logger = logging.getLogger("GEKO")


def _solution_id_gen_path() -> Path:
    """Return ``SolutionIdGen.py`` in the hipBLASLt checkout that contains Geko.

    ``parents[4]`` is the hipBLASLt root when this file is still in the checkout,
    including an editable install. A non-editable ``pip install .`` copies Geko
    into ``site-packages``, and this path is then inside the virtualenv.

    Args:
        None.

    Returns:
        Path to ``tensilelite/tensilelite/Common/SolutionIdGen.py``.

    Raises:
        None.
    """
    return (
        Path(__file__).resolve().parents[4]
        / "tensilelite"
        / "tensilelite"
        / "Common"
        / "SolutionIdGen.py"
    )


def _load_solution_id_gen() -> ModuleType | None:
    """Load SolutionIdGen without importing ``tensilelite.Common``.

    ``tensilelite.Common`` imports rocisa. Geko's uniqueness retry only needs
    the UID codec and the line rewriter. When the file is missing, Geko warns
    and the caller leaves existing UIDs unchanged. Install from the hipBLASLt
    checkout with ``pip install -e .`` so this path stays inside that checkout.

    Args:
        None.

    Returns:
        The loaded ``SolutionIdGen`` module, or None when the file is missing.

    Raises:
        ImportError: If the module spec or loader cannot be created.
    """
    module_path = _solution_id_gen_path()
    if not module_path.is_file():
        logger.warning(
            "SolutionIdGen.py not found: %s. "
            "Install Geko from the hipBLASLt checkout with 'pip install -e .'. "
            "A non-editable 'pip install .' copies Geko out of that checkout, "
            "so copied SolutionUIDs are left unchanged.",
            module_path,
        )
        return None
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
    When ``SolutionIdGen.py`` cannot be found, the solutions are left unchanged.

    Args:
        libraries: Libraries updated in place. Each solution dict loses
            ``SolutionUID`` and receives a new canonical value when the codec loads.

    Returns:
        Number of solutions assigned a new UID. Zero when the codec file is missing.

    Raises:
        ImportError: If ``SolutionIdGen.py`` cannot be loaded.
        RuntimeError: If a unique SolutionUID cannot be drawn.
        ValueError: If UID generation overflows the 40-bit timestamp.
    """
    solutions = [solution for library in libraries for solution in library.solutions]
    if not solutions:
        return 0
    solution_id_gen = _load_solution_id_gen()
    if solution_id_gen is None:
        return 0
    seen: set[str] = set()
    for solution in solutions:
        solution.pop("SolutionUID", None)
        solution_id_gen.claim_solution_uid(solution, seen)
    return len(solutions)


def uniquify_generated_logic(paths: list[Path]) -> int:
    """Make SolutionUIDs unique across logic files from one Geko run.

    When ``SolutionIdGen.py`` cannot be found, the files are left unchanged.

    Args:
        paths: ``3_LibraryLogic`` YAML files written by the Tensile jobs in
            this generation. Other architectures are not read.

    Returns:
        Number of UID lines rewritten. Zero when the codec file is missing.

    Raises:
        ImportError: If ``SolutionIdGen.py`` cannot be loaded.
        RuntimeError: If a replacement UID cannot be drawn.
        OSError: If a logic file cannot be read or written.
    """
    if not paths:
        return 0
    solution_id_gen = _load_solution_id_gen()
    if solution_id_gen is None:
        return 0
    return solution_id_gen.rewrite_colliding_solution_uids(list(paths))
