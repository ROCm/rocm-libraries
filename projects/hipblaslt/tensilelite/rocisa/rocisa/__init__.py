# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import types
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

if _bi is not None:
    from pathlib import Path

    _so = Path(_rocisa.__file__)
    _so_mtime = _so.stat().st_mtime
    # Scan rocisa sources and, while stinkytofu is compiled into _rocisa.so,
    # stinkytofu sources too. STINKYTOFU_SOURCE_ROOT is removed once rocisa
    # and stinkytofu are loaded independently.
    _roots = [Path(_bi.SOURCE_ROOT), Path(_bi.STINKYTOFU_SOURCE_ROOT)]
    _build_dir = Path(_bi.BUILD_DIR).resolve()
    _stale = [
        str(p)
        for _root in _roots
        for _pattern in ("*.[ch]pp", "*.h", "*.def", "*.inc")
        for p in _root.rglob(_pattern)
        if p.stat().st_mtime > _so_mtime and not p.resolve().is_relative_to(_build_dir)
    ]
    if _stale:
        _preview = _stale[:3] + (["..."] if len(_stale) > 3 else [])
        raise ImportError(
            "rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.\n"
            f"  Modified: {', '.join(_preview)}\n"
            "  Rebuild:  cmake --build <build_dir> --target _rocisa"
        )
    del _bi, _so, _so_mtime, _stale, _roots, _build_dir, Path


def hasStinkyTofuBackend() -> bool:
    """Return True if rocisa was built with StinkyTofu backend support."""
    return hasattr(_rocisa, "isSupportedByStinkyTofu")
