# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import types
import warnings

from pathlib import Path as _Path

if any(_Path(__file__).parent.glob("_rocisa.abi3.*")) and sys.version_info < (3, 12):
    raise ImportError(
        f"rocisa stable-ABI extension requires Python >= 3.12 "
        f"(running {sys.version_info.major}.{sys.version_info.minor}). "
        f"Install a non-stable-ABI build or upgrade Python."
    )
del _Path


def _candidate_dll_dirs(dep_dlls, ext_dir):
    """Ordered, de-duplicated directories to search for _rocisa's dependent DLLs.

    In resolution order: the directories of the build-supplied dependency DLLs
    (origami, HIP runtime, comgr, stinkytofu -- scattered across per-subproject
    dirs in a source/integrated build), then the extension's own directory.
    Pure and host-agnostic (no filesystem or os.add_dll_directory side effects)
    so it can be unit-tested off Windows. Extracted from _register_win_dll_dirs.
    """
    import os

    dirs = [os.path.dirname(p) for p in dep_dlls if p]
    dirs.append(ext_dir)
    ordered = []
    seen = set()
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def _installed_dll_dirs(ext_dir):
    """Windows dependency dirs for an installed _rocisa (no _dll_dirs.py present).

    A build tree gets exact dependency paths from _dll_dirs.py; an install does
    not, and since Python 3.8 the extension loader ignores PATH. The installed
    extension sits at <prefix>/lib/hipblaslt/rocisa while its dependency DLLs
    (HIP runtime, comgr, origami, stinkytofu) live in the merged ROCm
    <prefix>/bin. Return that bin/ (three levels up) plus the bin/ of any standard
    ROCM_PATH/HIP_PATH/ROCM_HOME, mirroring hipdnn_frontend. Callers guard each
    entry with os.path.isdir, so a wrong guess is a harmless no-op. Pure and
    host-agnostic so it can be unit-tested off Windows.
    """
    import os

    dirs = [
        os.path.normpath(
            os.path.join(ext_dir, os.pardir, os.pardir, os.pardir, "bin")
        )
    ]
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            dirs.append(os.path.join(root, "bin"))
    return dirs


def _register_win_dll_dirs() -> None:
    """Register _candidate_dll_dirs via os.add_dll_directory on Windows.

    Since Python 3.8 the loader resolves an extension module's dependent DLLs
    only from the system directories, the directory containing the .pyd, and
    directories added via os.add_dll_directory() -- PATH is ignored. This is the
    standard CPython 3.8+ pattern for loading an extension's dependent DLLs.
    """
    import os

    ext_dir = os.path.dirname(__file__)
    try:
        # Source/integrated build: CMake emits the resolved dependency DLL paths.
        from ._dll_dirs import DEP_DLLS
    except ImportError:
        DEP_DLLS = []  # Installed package: deps resolve via the merged layout.
    dll_dirs = _candidate_dll_dirs(DEP_DLLS, ext_dir)
    if not DEP_DLLS:
        dll_dirs = dll_dirs + _installed_dll_dirs(ext_dir)
    for d in dll_dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def _import_rocisa():
    """Import the _rocisa extension, registering its DLL dirs first on Windows.

    Registration and import are bound in one scope so their order is
    inseparable: _register_win_dll_dirs() must run before the loader resolves
    _rocisa's dependent DLLs, and no reorder of module-level imports can split
    them (a split silently reintroduces WinError 126 on Windows). For the same
    reason there is no module-level `from ._rocisa import *` -- that would be a
    second, reorderable trigger of the load; the public names are bound below.
    """
    if sys.platform == "win32":
        _register_win_dll_dirs()
    from . import _rocisa

    return _rocisa


def _find_stale_sources(so_path, source_roots, build_dir):
    """Return source files newer than so_path, excluding files under build_dir.

    Extracted from the module-level staleness check so it can be unit-tested
    without requiring a real _rocisa.so or touching actual source files.
    """
    import math
    from pathlib import Path

    # cmake --install truncates the installed .so's mtime to a whole second, so
    # a source edited in that same second would read as stale forever. The real
    # install time is in [t, t+1); compare against the end of that interval.
    so_mtime = math.floor(Path(so_path).stat().st_mtime) + 1
    build_dir = Path(build_dir).resolve()
    stale = []
    for root in source_roots:
        for pattern in ("*.[ch]pp", "*.h", "*.def", "*.inc"):
            for p in Path(root).rglob(pattern):
                if p.stat().st_mtime > so_mtime and not p.resolve().is_relative_to(build_dir):
                    stale.append(str(p))
    return stale


# Original nanobind bindings (Windows-safe import).
_rocisa = _import_rocisa()

# Reorder-safe equivalent of `from ._rocisa import *`: binding the
# extension's public API here keeps the DLL load confined to
# _import_rocisa() above.
_all = getattr(_rocisa, "__all__", None)
_public = (
    list(_all)
    if _all is not None
    else [_n for _n in dir(_rocisa) if not _n.startswith("_")]
)
globals().update({_n: getattr(_rocisa, _n) for _n in _public})
del _all, _public

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

if _bi is not None:
    from pathlib import Path

    _so = Path(_rocisa.__file__)
    # Scan exactly the sources compiled into _rocisa.so so its scan set
    # matches what the binary is built from:
    #   - rocisa bindings:  <SOURCE_ROOT>  (rocisa/rocisa/**, the _rocisa TUs)
    #   - stinkytofu:       the sources that make up libstinkytofu.so (which
    #     _rocisa.so links for toStinkyTofuModule / emitAssembly) plus the
    #     src/conversion/rocisa glue compiled directly into _rocisa.so:
    #       src/        libstinkytofu core + rocisa glue (src/conversion)
    #       include/    headers
    #       hardware/, tools/tablegen/   generators whose generated output
    #                   is compiled into libstinkytofu
    # Not scanned (never compiled into _rocisa.so): tests/, examples/,
    # python_module/ (-> _stinkytofu.so only) and the standalone tools/
    # (stinkytofu-opt, -check, -cfg, intrinsic-compiler, waitcnt-check).
    # Roots come from CMake; an empty one signals a malformed _build_info.py,
    # so warn (rather than scan Path("") == the CWD) and skip it.
    _roots = []
    _rocisa_root = Path(_bi.SOURCE_ROOT) if _bi.SOURCE_ROOT else None
    if _rocisa_root:
        _roots.append(_rocisa_root)
    else:
        warnings.warn(
            "rocisa staleness check: rocisa source root is unset in "
            "_build_info.py; skipping it. Rebuild with: invoke rocisa",
            stacklevel=2,
        )
    _st_root = Path(_bi.STINKYTOFU_SOURCE_ROOT) if _bi.STINKYTOFU_SOURCE_ROOT else None
    if _st_root:
        _roots.extend(
            d
            for d in (
                _st_root / "src",
                _st_root / "include",
                _st_root / "hardware",
                _st_root / "tools" / "tablegen",
            )
            if d.is_dir()
        )
    else:
        warnings.warn(
            "rocisa staleness check: stinkytofu source root is unset in "
            "_build_info.py; skipping it. Rebuild with: invoke rocisa",
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
    del _bi, _so, _stale, _roots, _rocisa_root, _st_root, Path


def hasStinkyTofuBackend() -> bool:
    """Return True if rocisa was built with StinkyTofu backend support."""
    _mod = globals().get("_rocisa")
    return _mod is not None and hasattr(_mod, "isSupportedByStinkyTofu")


if not hasStinkyTofuBackend():
    def isSupportedByStinkyTofu(version) -> bool:
        return False

    def isMnemonicSupportedByStinkyTofu(mnemonic, version) -> bool:
        # No StinkyTofu to lower through, so nothing here can reject a mnemonic.
        return True

    def parseRegisterAllocationConfig(value):
        # No StinkyTofu, so no allocation settings to read: only a bare mode parses.
        if type(value) is int and 0 <= value <= 3:
            return (str(value), value)
        raise ValueError(
            f"register allocation config {value!r} needs rocisa built with StinkyTofu")
