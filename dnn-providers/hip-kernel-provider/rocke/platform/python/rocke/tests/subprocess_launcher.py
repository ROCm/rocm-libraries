# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic launch helpers for tests that start a Python interpreter."""

from __future__ import annotations

import importlib.machinery
import os
import runpy
import site
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = ["rocke_python_command", "rocke_subprocess_env"]

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
_EXPLICIT_PATHS_ENV = "ROCKE_TEST_SUBPROCESS_PATHS"


def rocke_python_command(
    *arguments: str | os.PathLike[str], interpreter_flags: Sequence[str] = ()
) -> list[str]:
    """Build a Python command that imports the caller's ``rocke`` package first.

    The standalone launcher inserts the selected package parent ahead of the
    child's script directory or current working directory before it executes
    ``-c`` code, a ``-m`` module, or a script path.
    """
    return [
        sys.executable,
        *interpreter_flags,
        "-S",
        str(Path(__file__).resolve()),
        *(os.fspath(argument) for argument in arguments),
    ]


def rocke_subprocess_env(
    base: Mapping[str, str] | None = None,
    *,
    python_paths: Iterable[str | os.PathLike[str]] = (),
) -> dict[str, str]:
    """Return a child environment with selected paths ahead of ambient ones.

    The launcher controls package identity in the immediate child. ``PYTHONPATH``
    makes the package and caller-supplied roots available to grandchildren, which
    must reuse the launcher when they also require identity rather than availability.
    """
    env = dict(os.environ if base is None else base)
    explicit = [str(Path(path).resolve()) for path in python_paths]
    paths = [str(_PACKAGE_PARENT), *explicit]
    if inherited := env.get("PYTHONPATH"):
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if explicit:
        env[_EXPLICIT_PATHS_ENV] = os.pathsep.join(explicit)
    else:
        env.pop(_EXPLICIT_PATHS_ENV, None)
    return env


def _set_import_path(natural_entry: Path, explicit: list[str]) -> None:
    """Put rocKE first, then explicit and normal roots, and retain the rest."""
    selected = str(_PACKAGE_PARENT)
    natural = str(natural_entry.resolve())
    launcher = str(Path(__file__).resolve().parent)
    prioritized = {selected, natural, launcher, *explicit}
    remainder = [entry for entry in sys.path if entry and entry not in prioritized]
    sys.path[:] = [selected, *explicit, natural, *remainder]


class _SelectedRockeFinder:
    """Resolve the top-level package from the selected root during site hooks."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname != "rocke":
            return None
        return importlib.machinery.PathFinder.find_spec(
            fullname, [str(_PACKAGE_PARENT)]
        )


def _run(arguments: list[str]) -> None:
    """Execute normal Python ``-c``, ``-m``, or file-script arguments."""
    if not arguments:
        raise SystemExit("expected -c CODE, -m MODULE, or a script path")

    mode, *remaining = arguments
    if mode == "-c":
        if not remaining:
            raise SystemExit("expected code after -c")
        code, *code_arguments = remaining
        natural = Path.cwd()
        sys.argv = ["-c", *code_arguments]
    elif mode == "-m":
        if not remaining:
            raise SystemExit("expected a module after -m")
        module, *module_arguments = remaining
        natural = Path.cwd()
        sys.argv = [module, *module_arguments]
    else:
        script = Path(mode)
        if not script.is_file():
            raise SystemExit(f"expected a Python script file, got {mode!r}")
        natural = script.parent
        sys.argv = [mode, *remaining]

    explicit = [
        entry
        for entry in os.environ.pop(_EXPLICIT_PATHS_ENV, "").split(os.pathsep)
        if entry
    ]
    _set_import_path(natural, explicit)
    finder = _SelectedRockeFinder()
    sys.meta_path.insert(0, finder)
    try:
        site.main()
    finally:
        sys.meta_path.remove(finder)
    _set_import_path(natural, explicit)

    if mode == "-c":
        namespace = {
            "__name__": "__main__",
            "__doc__": None,
            "__package__": None,
            "__loader__": None,
            "__spec__": None,
            "__annotations__": {},
        }
        exec(compile(code, "<string>", "exec", dont_inherit=True), namespace)
        return
    if mode == "-m":
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    runpy.run_path(mode, run_name="__main__")


if __name__ == "__main__":
    _run(sys.argv[1:])
