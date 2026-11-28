# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Whether an installed test tree has the C++ engine pieces it was built with.

The tests reach the C++ engine two ways. Its family-agnostic lowering goes
through the shared library (``librocke``, loaded by ``rocke.core.native_engine``
through its C ABI), which a provider build installs beside the ``rocke``
package. Its kernel-family builders go through its Python modules
(``rocke_engine`` and, beside the library tree, ``rocke_library_engine``),
which a build with ``ROCKE_BUILD_PYBIND`` installs there too. Every test that
runs the C++ engine, or compares it with the Python engine, skips where the
piece it needs does not load, and a run's pass count does not show that. So the
configure records what the build made in a marker file in the installed tests
tree, and the installed guard holds the tree to it::

    {"modules": ["rocke_engine", "rocke_library_engine"],
     "shared_engine": "librocke.so"}
    {"modules": [], "absent_reason": "the build had ROCKE_BUILD_PYBIND off",
     "shared_engine": "", "shared_engine_absent_reason": "..."}

Run as ``python -m rocke.tests.engine_modules <marker>``, the guard:

- exits :data:`SKIP_EXIT_CODE` and prints the marker's reasons when the build
  made neither the modules nor the shared engine, so CTest reports the entry as
  skipped, with why;
- otherwise imports every module the marker names and prints where each came
  from and its engine build id, loads the shared engine the marker names from
  beside the ``rocke`` package and prints its path, and, when the build made
  the shared engine, lowers a probe kernel through ``lower_kernel_to_llvm`` and
  prints which engine lowered it. It exits 0 when everything the build made
  loads and the ``cpp`` backend lowered through the C++ engine, and 1 with the
  reason otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..core import native_engine
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
    "load_shared_engine",
    "lowering_engine",
    "main",
    "read_engine_modules_marker",
]

# CTest reports an entry that exits with this code as skipped (SKIP_RETURN_CODE).
SKIP_EXIT_CODE = 77
# The target the probe kernel is lowered for: one every engine build lowers.
_PROBE_ARCH = "gfx942"


class EngineModulesError(RuntimeError):
    """The installed tree lacks an engine piece its build made, or does not
    lower through it."""


@dataclasses.dataclass(frozen=True)
class EngineModulesMarker:
    """What the configure recorded: the engine modules the build makes, or why
    it makes none, and the shared engine it installs beside the package, or why
    it installs none there."""

    modules: tuple[str, ...]
    absent_reason: str
    shared_engine: str
    shared_engine_absent_reason: str

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
        if self.shared_engine and Path(self.shared_engine).name != self.shared_engine:
            raise ValueError(
                f"EngineModulesMarker failed - shared_engine={self.shared_engine!r}, "
                "expected a file name"
            )
        if bool(self.shared_engine) == bool(self.shared_engine_absent_reason.strip()):
            raise ValueError(
                f"EngineModulesMarker failed - shared_engine={self.shared_engine!r}, "
                f"shared_engine_absent_reason={self.shared_engine_absent_reason!r}, "
                "expected either the shared engine or the reason the build "
                "installs none beside the package"
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
        shared_engine=document.get("shared_engine", ""),
        shared_engine_absent_reason=document.get("shared_engine_absent_reason", ""),
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


def load_shared_engine(marker: EngineModulesMarker) -> Path | None:
    """Load the shared engine ``marker`` names from beside the ``rocke``
    package and return its path; None when the marker names none."""
    if not marker.shared_engine:
        return None
    installed = native_engine.installed_shared_engine_path()
    if installed is None or installed.name != marker.shared_engine:
        raise EngineModulesError(
            f"the build made {marker.shared_engine}, but it is not installed "
            "beside the rocke package"
        )
    try:
        native_engine.require_native_lowering()
    except (OSError, RuntimeError) as error:
        raise EngineModulesError(
            f"the build made {marker.shared_engine}, but it does not load: {error}"
        ) from error
    loaded = native_engine.loaded_shared_engine_path()
    if loaded is None or loaded.resolve() != installed:
        raise EngineModulesError(
            f"the build made {installed}, but this process loaded {loaded} "
            f"({native_engine.SHARED_ENGINE_ENV} names another library)"
        )
    return installed


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
    if not marker.modules and not marker.shared_engine:
        print(
            "SKIP: no C++ engine to check: no Python modules "
            f"({marker.absent_reason}); no shared engine "
            f"({marker.shared_engine_absent_reason})"
        )
        return SKIP_EXIT_CODE
    try:
        modules = import_engine_modules(marker)
        shared_engine = load_shared_engine(marker)
        engine = lowering_engine() if shared_engine is not None else None
    except EngineModulesError as error:
        print(f"FAIL: {error}")
        return 1
    for module in modules:
        print(f"{module.name}: {module.path} (build id {module.build_id})")
    if not modules:
        print(f"Python modules: none ({marker.absent_reason})")
    if shared_engine is None:
        print(f"shared engine: none ({marker.shared_engine_absent_reason})")
    else:
        print(f"shared engine: {shared_engine}")
    if engine is not None:
        print(f"lower_kernel_to_llvm lowers with the {engine} engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
