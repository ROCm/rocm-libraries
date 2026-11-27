# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The source files a Python script's imports can reach, found without running it.

One home for "which files can this script import": a check that runs a script
is affected by a change to any of them. The search reads each file's bytecode,
so every import statement counts, function-level and ``TYPE_CHECKING`` ones
included. It cannot follow a module named only at run time
(``importlib.import_module`` of a computed name, a file loaded by path) or a
file read as data. An import it cannot resolve in the search roots, the
standard library or the interpreter's built-in modules is reported by its
top-level name, so a caller can tell an installed package from code the search
did not reach (a module found through a ``sys.path`` entry the script adds at
run time, for one).
"""

from __future__ import annotations

import modulefinder
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ImportClosure", "script_import_closure"]


@dataclass(frozen=True)
class ImportClosure:
    """What a script's imports reach: the files found, as resolved paths, and
    the top-level names of the imports found nowhere."""

    files: frozenset[Path]
    unresolved: frozenset[str]


def _is_relative_import_name(
    name: str, callers: Iterable[str], modules: Mapping[str, object]
) -> bool:
    """Whether ModuleFinder listed ``name`` for a relative from-import: it lists
    ``from .core import x`` in package ``rocke`` as "core.x", which resolves
    against a package that holds each caller."""
    first = name.split(".", 1)[0]
    for caller in callers:
        parts = caller.split(".")
        if not any(
            ".".join([*parts[:depth], first]) in modules
            for depth in range(1, len(parts) + 1)
        ):
            return False
    return True


def _unresolved_names(finder: modulefinder.ModuleFinder) -> frozenset[str]:
    missing, _ = finder.any_missing_maybe()
    unresolved = set()
    for name in missing:
        if _is_relative_import_name(name, finder.badmodules[name], finder.modules):
            continue
        top = name.split(".", 1)[0]
        if top not in sys.stdlib_module_names and top not in sys.builtin_module_names:
            unresolved.add(top)
    return frozenset(unresolved)


def script_import_closure(script: Path, search_roots: Iterable[Path]) -> ImportClosure:
    """What the imports of ``script`` reach, transitively.

    An import resolves as it does for ``python script``: in the script's own
    directory first, then in ``search_roots`` in order. The files are
    ``script`` and every file found there; an import of the standard library
    is neither a file nor unresolved."""
    roots = [script.resolve().parent, *(Path(root).resolve() for root in search_roots)]
    finder = modulefinder.ModuleFinder(path=[str(root) for root in roots])
    finder.run_script(str(script))
    files = frozenset(
        Path(module.__file__).resolve()
        for module in finder.modules.values()
        if module.__file__
    )
    return ImportClosure(files, _unresolved_names(finder))
