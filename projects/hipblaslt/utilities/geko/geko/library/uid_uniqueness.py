# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Retry SolutionUID collisions in logic files from one Geko generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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
