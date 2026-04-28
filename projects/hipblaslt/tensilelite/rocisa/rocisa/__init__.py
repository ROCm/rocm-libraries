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
# Pre-built packages (wheels, apt) lack _build_info.py, so the import fails
# silently and the check is skipped.
try:
    from . import _build_info as _bi
    from pathlib import Path

    _so = Path(_rocisa.__file__)
    _so_mtime = _so.stat().st_mtime
    _stale = [
        str(p) for p in Path(_bi.SOURCE_ROOT).rglob("*.[ch]pp")
        if p.stat().st_mtime > _so_mtime
    ]
    if _stale:
        _preview = _stale[:3] + (["..."] if len(_stale) > 3 else [])
        raise ImportError(
            "rocisa C++ sources are newer than the built _rocisa.so — bindings are stale.\n"
            f"  Modified: {', '.join(_preview)}\n"
            "  Rebuild:  cmake --build <build_dir> --target _rocisa"
        )
    del _bi, _so, _so_mtime, _stale, Path
except ModuleNotFoundError:
    pass  # Pre-built package — no source tree, skip check
