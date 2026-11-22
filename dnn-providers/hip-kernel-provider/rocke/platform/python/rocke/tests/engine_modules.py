# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Whether an installed test tree has the C++ engine's Python modules it was built with.

A build with ``ROCKE_BUILD_PYBIND`` makes the engine's Python modules
(``rocke_engine`` and, beside the library tree, ``rocke_library_engine``) and
installs them beside the ``rocke`` package. Every test that runs the C++ engine,
or compares it with the Python engine, skips where they do not import, and a
run's pass count does not show that. So the configure records which modules the
build makes in a marker file in the installed tests tree, and the installed
guard holds the tree to it::

    {"modules": ["rocke_engine", "rocke_library_engine"]}
    {"modules": [], "absent_reason": "the build's Python had no pybind11"}

Run as ``python -m rocke.tests.engine_modules <marker>``, the guard:

- exits :data:`SKIP_EXIT_CODE` and prints the marker's reason when the build
  made no module, so CTest reports the entry as skipped, with why;
- otherwise imports every module the marker names and prints where each came
  from and its engine build id, then lowers a probe kernel through
  ``lower_kernel_to_llvm`` and prints which engine lowered it. It exits 0 when
  every module imports and the ``cpp`` backend lowered through the C++ engine,
  and 1 with the reason otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..core.backend import (
    BACKEND_PYTHON,
    BackendError,
    cpp_fallbacks,
    import_engine,
    resolve_backend,
)
from ..core.ir import IRBuilder
from ..core.lower_llvm import lower_kernel_to_llvm

__all__ = [
    "SKIP_EXIT_CODE",
    "EngineModule",
    "EngineModulesError",
    "EngineModulesMarker",
    "import_engine_modules",
    "lowering_engine",
    "main",
    "read_engine_modules_marker",
]

# CTest reports an entry that exits with this code as skipped (SKIP_RETURN_CODE).
SKIP_EXIT_CODE = 77
# The target the probe kernel is lowered for: one every engine build lowers.
_PROBE_ARCH = "gfx942"


class EngineModulesError(RuntimeError):
    """The installed tree lacks an engine module its build made, or does not
    lower through it."""


@dataclasses.dataclass(frozen=True)
class EngineModulesMarker:
    """What the configure recorded: the engine modules the build makes, or why
    it makes none."""

    modules: tuple[str, ...]
    absent_reason: str

    def __post_init__(self) -> None:
        for name in self.modules:
            if not name.isidentifier():
                raise ValueError(
                    f"EngineModulesMarker failed - module={name!r}, expected a "
                    "Python module name"
                )
        if bool(self.modules) == bool(self.absent_reason.strip()):
            raise ValueError(
                f"EngineModulesMarker failed - modules={self.modules!r}, "
                f"absent_reason={self.absent_reason!r}, expected either modules "
                "or the reason the build made none"
            )


@dataclasses.dataclass(frozen=True)
class EngineModule:
    """One engine module as this process imported it."""

    name: str
    path: str
    build_id: str


def read_engine_modules_marker(path: Path) -> EngineModulesMarker:
    """The marker the configure wrote at ``path``."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"read_engine_modules_marker failed - path={path}, expected the "
            f"marker the rocKE configure installs: {error}"
        ) from error
    return EngineModulesMarker(
        modules=tuple(document.get("modules", ())),
        absent_reason=document.get("absent_reason", ""),
    )


def import_engine_modules(marker: EngineModulesMarker) -> tuple[EngineModule, ...]:
    """Import every module ``marker`` names."""
    imported = []
    for name in marker.modules:
        try:
            module = import_engine(name)
        except BackendError as error:
            raise EngineModulesError(
                f"the build made {name}, but this tree's Python "
                f"({sys.version.split()[0]}) cannot import it: {error}"
            ) from error
        imported.append(EngineModule(name, module.__file__, module.build_id()))
    return tuple(imported)


def lowering_engine() -> str:
    """The backend ``lower_kernel_to_llvm`` lowers with in this process
    (``"python"``, ``"cpp"`` or ``"both"``). Unless it is ``"python"``, a probe
    kernel is lowered first, and :class:`EngineModulesError` is raised when the
    lowering fell back to the Python engine."""
    if resolve_backend() == BACKEND_PYTHON:
        return BACKEND_PYTHON
    builder = IRBuilder("rocke_engine_modules_probe")
    builder.ret()
    fallbacks_before = len(cpp_fallbacks())
    lower_kernel_to_llvm(builder.kernel, arch=_PROBE_ARCH)
    fallbacks = cpp_fallbacks()[fallbacks_before:]
    if fallbacks:
        raise EngineModulesError(
            f"backend {resolve_backend()!r} fell back to the Python engine: "
            + "; ".join(reason for _, reason in fallbacks)
        )
    return resolve_backend()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("marker", type=Path, help="the installed marker file")
    arguments = parser.parse_args(argv)

    marker = read_engine_modules_marker(arguments.marker)
    if not marker.modules:
        print(f"SKIP: no C++ engine Python modules to check: {marker.absent_reason}")
        return SKIP_EXIT_CODE
    try:
        modules = import_engine_modules(marker)
        engine = lowering_engine()
    except EngineModulesError as error:
        print(f"FAIL: {error}")
        return 1
    for module in modules:
        print(f"{module.name}: {module.path} (build id {module.build_id})")
    print(f"lower_kernel_to_llvm lowers with the {engine} engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
