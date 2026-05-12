"""StinkyTofu: High-Level IR for AMDGPU Assembly Generation"""

# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import types
from ._stinkytofu import *
from . import _stinkytofu

# Register submodules under the stinkytofu.* namespace
for _name, _obj in vars(_stinkytofu).items():
    if isinstance(_obj, types.ModuleType) and not _name.startswith("_"):
        sys.modules.setdefault(f"stinkytofu.{_name}", _obj)

# Runtime intrinsic data
_sigs = {}
_lib = None
_init = False

try:
    reg = _stinkytofu.IntrinsicRegistry.instance()
    if reg.is_initialized():
        _init = True
        _lib = reg.get_library()
        for name in _lib.get_intrinsic_names():
            _sigs[name] = [arg.name for arg in _lib.get_arguments(name)]
except Exception:
    pass


def list_intrinsics():
    return list(_sigs.keys()) if _init else []


def get_intrinsic_signature(name):
    return _sigs.get(name)


def get_intrinsic_info(name):
    if not _init or name not in _sigs:
        return None
    return {"signature": _sigs[name], "comment": _lib.get_comment(name) if _lib else ""}


def Intrinsic(name, **kwargs):
    """Create intrinsic with validation and auto-reordering (ORDER DOESN'T MATTER!)"""
    if not _init:
        raise RuntimeError("Intrinsics not loaded")
    if name not in _sigs:
        raise ValueError(
            f"Unknown: '{name}' (available: {', '.join(sorted(_sigs.keys()))})"
        )

    expected = _sigs[name]
    provided = set(kwargs.keys())
    missing = set(expected) - provided
    extra = provided - set(expected)

    if missing:
        raise ValueError(f"'{name}' missing: {sorted(missing)} (expected: {expected})")
    if extra:
        raise ValueError(f"'{name}' unexpected: {sorted(extra)} (expected: {expected})")

    ordered_kwargs = {arg: kwargs[arg] for arg in expected}
    return _stinkytofu.Intrinsic(name, **ordered_kwargs)


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

    _so = Path(_stinkytofu.__file__)
    _so_mtime = _so.stat().st_mtime
    _build_dir = Path(_bi.BUILD_DIR).resolve()
    _stale = [
        str(p)
        for _pattern in ("*.[ch]pp", "*.h", "*.def", "*.inc")
        for p in Path(_bi.SOURCE_ROOT).rglob(_pattern)
        if p.stat().st_mtime > _so_mtime and not p.resolve().is_relative_to(_build_dir)
    ]
    if _stale:
        _preview = _stale[:3] + (["..."] if len(_stale) > 3 else [])
        raise ImportError(
            "stinkytofu C++ sources are newer than the built _stinkytofu.so — bindings are stale.\n"
            f"  Modified: {', '.join(_preview)}\n"
            "  Rebuild:  cmake --build <build_dir> --target stinkytofu_python"
        )
    del _bi, _so, _so_mtime, _stale, _build_dir, Path
