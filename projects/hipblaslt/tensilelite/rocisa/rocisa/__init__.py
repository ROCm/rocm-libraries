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


def _register_win_dll_dirs() -> None:
    """Register _candidate_dll_dirs via os.add_dll_directory on Windows.

    Since Python 3.8 the loader resolves an extension module's dependent DLLs
    only from the system directories, the directory containing the .pyd, and
    directories added via os.add_dll_directory() -- PATH is ignored. This is the
    standard CPython 3.8+ pattern for loading an extension's dependent DLLs.
    """
    import os

    try:
        # Source/integrated build: CMake emits the resolved dependency DLL paths.
        from ._dll_dirs import DEP_DLLS
    except ImportError:
        DEP_DLLS = []  # Installed package: deps resolve via the merged layout.
    for d in _candidate_dll_dirs(DEP_DLLS, os.path.dirname(__file__)):
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


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
# ``ROCISA_BACKEND=stinkytofu`` redirects ``import rocisa`` to the
# ``rocisa_stinkytofu_adaptor`` shim (a rocisa-shaped facade backed by the
# stinkytofu Python binding ``_stinkytofu.so``). Anything else (or unset)
# keeps the original nanobind bindings in ``_rocisa``.

_BACKEND = os.environ.get("ROCISA_BACKEND", "").strip().lower()

_ADAPTER_PKG = "rocisa_stinkytofu_adaptor"


def _load_stinkytofu_adapter() -> "tuple[bool, str]":
    """Try to install the rocisa_stinkytofu_adaptor as the ``rocisa`` module.

    Returns ``(True, "")`` iff we successfully rewired sys.modules; on any
    failure returns ``(False, <reason>)`` so the caller can surface why the
    stinkytofu backend could not be loaded (instead of falling back silently).
    """

    # Locate ``<repo_root>/projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor``
    # by walking up from this file until we find an ancestor whose basename
    # is ``projects``; the adapter then lives at
    # ``<that_parent>/projects/hipblaslt/tensilelite/<_ADAPTER_PKG>/``.
    #
    # The adapter is a sibling of ``tensilelite/rocisa/`` on purpose — it is
    # a *consumer* of the stinkytofu Python binding (a tensilelite-internal
    # alternative backend), not a piece of ``shared/stinkytofu/`` itself.
    # Sibling-of-rocisa keeps ``ROCISA_BACKEND=stinkytofu`` purely a
    # tensilelite concern.
    #
    # Works for both the source tree (``<repo>/projects/hipblaslt/
    # tensilelite/rocisa/rocisa/__init__.py``) and CMake-staged copies
    # under ``<repo>/projects/hipblaslt/tensilelite/<build_dir>/tensilelite/
    # rocisa/rocisa/__init__.py`` because in either case walking up from
    # ``__file__`` eventually hits the ``projects`` directory.
    repo_root = None
    cur = os.path.dirname(os.path.abspath(__file__))
    adapter_rel = os.path.join("projects", "hipblaslt", "tensilelite", _ADAPTER_PKG)
    while True:
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            break
        if os.path.basename(cur) == "projects":
            candidate = parent
            if os.path.isdir(os.path.join(candidate, adapter_rel)):
                repo_root = candidate
                break
        cur = parent
    if repo_root is None:
        return False, (
            f"adapter package not found: no ancestor 'projects' dir containing "
            f"{adapter_rel}"
        )
    adapter_parent = os.path.join(repo_root, adapter_rel)

    if not os.path.isdir(os.path.join(adapter_parent, _ADAPTER_PKG)):
        return False, f"adapter package directory missing under {adapter_parent}"

    if adapter_parent not in sys.path:
        sys.path.insert(0, adapter_parent)

    try:
        import rocisa_stinkytofu_adaptor as _adapter  # noqa: F401
    except Exception as exc:
        return False, f"import failed: {exc!r}"

    # Install the adapter as ``rocisa`` and re-export each
    # ``rocisa_stinkytofu_adaptor.*`` submodule under ``rocisa.*`` in
    # ``sys.modules``.
    sys.modules["rocisa"] = _adapter
    _prefix = f"{_ADAPTER_PKG}."
    for _name, _obj in vars(_adapter).items():
        if isinstance(_obj, types.ModuleType) and _obj.__name__.startswith(_prefix):
            short = _obj.__name__[len(_prefix):]
            sys.modules[f"rocisa.{short}"] = _obj

    return True, ""


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


_FALLBACK = "; falling back to native rocisa backend."


def _stinkytofu_available() -> "tuple[bool, str]":
    """Probe the standalone stinkytofu binding (_stinkytofu.so) by importing it.

    Returns ``(True, "")`` iff ``import stinkytofu`` succeeds; otherwise
    ``(False, <cause-specific message>)`` so ``_resolve_backend`` can emit a
    *distinct* warning per failure mode rather than one generic message.

    The adapter's logical-IR path (``lower_logical_module`` -> ``StinkyAsmModule``
    -> ``emitAssembly``) is backed entirely by the standalone ``_stinkytofu.so``
    (built from ``shared/stinkytofu/python_module``), NOT by ``_rocisa.so``'s
    in-process ``init_stinkytofu`` bindings.

    We must *actually import* stinkytofu here rather than use
    ``importlib.util.find_spec("stinkytofu")``, for two reasons:
      1. ``find_spec`` only locates the package's ``__init__.py`` *source*; it
         does not execute it, so it returns truthy even when ``_stinkytofu.so``
         was never built (the source tree is always present).
      2. Importing the adapter package does NOT surface a missing
         ``_stinkytofu.so`` either -- ``rocisa_stinkytofu_adaptor/__init__.py``
         swaps ``StinkyAsmModule`` for a dummy on ImportError. So the switch
         would appear to succeed and only fail later, mid-lowering.
    ``import stinkytofu`` eagerly loads ``_stinkytofu.so`` (see that package's
    ``from ._stinkytofu import *``). This probe runs only when
    ``ROCISA_BACKEND=stinkytofu`` is requested."""
    import importlib

    _req = "ROCISA_BACKEND=stinkytofu requested but "
    try:
        importlib.import_module("stinkytofu")
    except ModuleNotFoundError as exc:
        # The stinkytofu package or its ``_stinkytofu.so`` extension is not on
        # the path at all -- i.e. the standalone binding was never built.
        return False, (
            f"{_req}the standalone stinkytofu binding is not built/importable "
            f"(_stinkytofu.so missing: {exc}){_FALLBACK}"
        )
    except ImportError as exc:
        # ImportError (but not ModuleNotFoundError): either the deliberate
        # staleness guard in stinkytofu/__init__.py (sources newer than the
        # built .so) or a .so that is present but failed to load (missing
        # symbols / ABI mismatch). Distinguish so the developer knows whether
        # to *rebuild* or to investigate a broken binding.
        _msg = str(exc)
        if "newer than the built _stinkytofu.so" in _msg or "bindings are stale" in _msg:
            return False, (
                f"{_req}the stinkytofu binding is stale and must be rebuilt "
                f"({_msg}){_FALLBACK}"
            )
        return False, (
            f"{_req}the stinkytofu binding is present but failed to load "
            f"({exc!r}){_FALLBACK}"
        )
    except Exception as exc:  # noqa: BLE001 -- any import-time error disables it
        return False, (
            f"{_req}importing the stinkytofu binding raised an unexpected error "
            f"({exc!r}){_FALLBACK}"
        )
    return True, ""


def _resolve_backend(requested, available_fn, load_fn, warn=warnings.warn) -> bool:
    """Decide whether to use the stinkytofu adapter (True) or native rocisa (False).

    Emits a warning *only* when the stinkytofu backend was explicitly requested
    but we have to fall back to native — so an unnoticed silent fallback becomes
    visible, with a cause-specific reason attached. ``available_fn`` and
    ``load_fn`` share the same ``(ok, reason)`` contract; the reason is surfaced
    verbatim so each distinct failure mode produces its own warning. Requesting
    anything else (or unset) selects native without touching the probes and
    without warning.
    """
    if requested != "stinkytofu":
        return False
    available, avail_reason = available_fn()
    if not available:
        warn(avail_reason, stacklevel=2)
        return False
    ok, reason = load_fn()
    if not ok:
        warn(
            f"ROCISA_BACKEND=stinkytofu requested but the adapter failed to load "
            f"({reason}){_FALLBACK}",
            stacklevel=2,
        )
        return False
    return True


if _resolve_backend(_BACKEND, _stinkytofu_available, _load_stinkytofu_adapter):
    # stinkytofu adapter active; wiring done inside _load_stinkytofu_adapter.
    pass
else:
    # Default path: original nanobind bindings (Windows-safe import).
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
