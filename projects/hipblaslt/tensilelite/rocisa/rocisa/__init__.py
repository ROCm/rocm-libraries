# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import types

from pathlib import Path as _Path

if any(_Path(__file__).parent.glob("_rocisa.abi3.*")) and sys.version_info < (3, 12):
    raise ImportError(
        f"rocisa stable-ABI extension requires Python >= 3.12 "
        f"(running {sys.version_info.major}.{sys.version_info.minor}). "
        f"Install a non-stable-ABI build or upgrade Python."
    )
del _Path


def _candidate_dll_dirs(dep_dlls, environ, ext_dir):
    """Ordered, de-duplicated directories to search for _rocisa's dependent DLLs.

    In resolution order: the directories of the build-supplied dependency DLLs
    (origami, HIP runtime, comgr, stinkytofu — scattered across per-subproject
    dirs in a source/integrated build), then the installed ROCm SDK bin
    (HIP_PATH/ROCM_PATH), then the extension's own directory. Pure and
    host-agnostic (no filesystem or os.add_dll_directory side effects) so it can
    be unit-tested off Windows. Extracted from _register_win_dll_dirs.
    """
    import os

    dirs = [os.path.dirname(p) for p in dep_dlls if p]
    for var in ("HIP_PATH", "ROCM_PATH"):
        root = environ.get(var)
        if root:
            dirs.append(os.path.join(root, "bin"))
    dirs.append(ext_dir)
    ordered = []
    seen = set()
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def _register_win_dll_dirs() -> None:
    """Register _candidate_dll_dirs via os.add_dll_directory on Windows.

    Since Python 3.8 the loader resolves an extension module's dependent DLLs
    only from the system directories, the directory containing the .pyd, and
    directories added via os.add_dll_directory() -- PATH is ignored. Mirrors the
    runtime resolver in
    dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/runtime/hip_module.py.
    """
    import os

    try:
        # Source/integrated build: CMake emits the resolved dependency DLL paths.
        from ._dll_dirs import DEP_DLLS
    except ImportError:
        DEP_DLLS = []  # Installed package: resolve via ROCm SDK / merged bin.
    for d in _candidate_dll_dirs(DEP_DLLS, os.environ, os.path.dirname(__file__)):
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


# Must run BEFORE the _rocisa import below: it registers the directories of
# _rocisa's dependent DLLs so the extension loads on Windows (WinError 126).
if sys.platform == "win32":
    _register_win_dll_dirs()

from ._rocisa import *
from . import _rocisa

# Register nanobind submodules under the rocisa.* namespace so that
# `from rocisa.enum import X` and `import rocisa.instruction as ri` work.
for _name, _obj in vars(_rocisa).items():
    if isinstance(_obj, types.ModuleType) and not _name.startswith("_"):
        sys.modules.setdefault(f"rocisa.{_name}", _obj)

# Staleness check: only active in source builds.
# Pre-built packages (wheels, apt) lack _build_info.py — the import is
# silently skipped. Catching ImportError (not just ModuleNotFoundError)
# because Python 3.10 raises ImportError for missing relative submodules.
# The intentional staleness ImportError is raised outside the try/except
# so it is never swallowed.
_bi = None
try:
    from . import _build_info as _bi
except ImportError:
    pass  # Pre-built package — no source tree, skip check

def _find_stale_sources(so_path, source_roots, build_dir):
    """Return source files newer than so_path, excluding files under build_dir.

    Extracted from the module-level staleness check so it can be unit-tested
    without requiring a real _rocisa.so or touching actual source files.
    """
    from pathlib import Path

    so_mtime = Path(so_path).stat().st_mtime
    build_dir = Path(build_dir).resolve()
    stale = []
    for root in source_roots:
        for pattern in ("*.[ch]pp", "*.h", "*.def", "*.inc"):
            for p in Path(root).rglob(pattern):
                if p.stat().st_mtime > so_mtime and not p.resolve().is_relative_to(build_dir):
                    stale.append(str(p))
    return stale


if _bi is not None:
    from pathlib import Path

    _so = Path(_rocisa.__file__)
    # Scan rocisa sources and, while stinkytofu is compiled into _rocisa.so,
    # stinkytofu sources too. STINKYTOFU_SOURCE_ROOT is removed once rocisa
    # and stinkytofu are loaded independently.
    # Both roots are populated by CMake; an empty one signals a malformed
    # _build_info.py. Warn (rather than scan Path("") == the CWD) and skip it,
    # so a regression surfaces instead of silently disabling the check.
    _roots = []
    for _name, _root in (("rocisa", _bi.SOURCE_ROOT), ("stinkytofu", _bi.STINKYTOFU_SOURCE_ROOT)):
        if _root:
            _roots.append(Path(_root))
        else:
            import warnings

            warnings.warn(
                f"rocisa staleness check: {_name} source root is unset in "
                f"_build_info.py; skipping it. Rebuild with: invoke rocisa",
                stacklevel=2,
            )
    _stale = _find_stale_sources(_so, _roots, _bi.BUILD_DIR)
    if _stale:
        _preview = _stale[:3] + (["..."] if len(_stale) > 3 else [])
        raise ImportError(
            "rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.\n"
            f"  Modified: {', '.join(_preview)}\n"
            "  Rebuild:  invoke rocisa"
        )
    del _bi, _so, _stale, _roots, _name, _root, Path


def hasStinkyTofuBackend() -> bool:
    """Return True if rocisa was built with StinkyTofu backend support."""
    return hasattr(_rocisa, "isSupportedByStinkyTofu")
